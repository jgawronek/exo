"""Draft/verify speculative decoding for the batched MLX engine.

A small draft model greedily proposes a short continuation; the target model
scores the pending token plus all proposals in a single forward pass. Every
emitted token is still sampled from the target model's logits for its true
prefix, so output is identical in distribution to plain decoding - only the
number of sequential target forward passes changes.

The batched engine's patched ``GenerationBatch._step`` returns exactly one
token per call. A speculative round therefore banks the extra accepted tokens
in a queue that subsequent ``_step`` calls drain without running the model.

Invariants between ``_step`` calls (mirroring plain decode):

- ``batch._next_tokens`` holds the single pending token, which is neither in
  ``batch.tokens`` nor in the target KV cache.
- The target KV cache covers exactly ``batch.tokens[0]`` plus, while the
  queue is non-empty, the queued tokens (they were verified ahead of time).
  ``flush_speculative_tokens`` trims that overhang to restore the plain
  invariant before any batch mutation (new prompt merge or cancellation).
"""

import os
from dataclasses import dataclass, field
from typing import Any, cast, final

import mlx.core as mx
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from exo.shared.environment import get_compatible_environment_value
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger

# Keep draft rounds short: MLX quantized matmuls lose their single-row fast
# path on multi-token inputs, so verifying k+1 tokens costs roughly
# (1 + 0.25*k) target passes rather than ~1. Measured on an M3 Ultra with
# Qwen3-32B-8bit + Qwen3-0.6B-4bit (67% first-token acceptance): k=1 and k=2
# give +23% decode TPS while k=4 is a 23% regression.
DRAFT_TOKENS_PER_ROUND: int = int(
    get_compatible_environment_value(os.environ, "EXO_DRAFT_TOKENS", "2")
)
_DRAFT_CATCH_UP_CHUNK_SIZE = 512
_ACCEPTANCE_LOG_INTERVAL_ROUNDS = 200

_SPECULATIVE_ATTRIBUTE = "_exo_speculative_state"


@final
@dataclass
class SpeculativeState:
    """Speculative-decoding state for a single generation row (one uid).

    ``draft_position`` counts how many tokens of the row's full history
    (``prompt_prefix`` + ``batch.tokens[0]`` + pending/proposed tokens) the
    draft KV cache currently covers. The draft cache lazily catches up at the
    start of each round, so prefix-cache hits and remote prefill need no
    special handling.
    """

    draft_model: Model
    uid: int
    prompt_prefix: list[int]
    draft_cache: list[Any] = field(init=False)
    draft_position: int = 0
    queue: list[tuple[int, mx.array]] = field(default_factory=list)
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0

    def __post_init__(self) -> None:
        self.draft_cache = cast(list[Any], make_prompt_cache(self.draft_model))


def attach_speculative_state(batch: GenerationBatch, state: SpeculativeState) -> None:
    setattr(batch, _SPECULATIVE_ATTRIBUTE, state)


def get_speculative_state(batch: GenerationBatch) -> SpeculativeState | None:
    return cast(SpeculativeState | None, getattr(batch, _SPECULATIVE_ATTRIBUTE, None))


def detach_speculative_state(batch: GenerationBatch) -> None:
    """Flush any verified-ahead tokens and stop speculating on this batch.

    Must be called before the generation batch is mutated (a new prompt is
    merged in or a row is removed) so the target KV cache matches
    ``batch.tokens`` again. Safe to call when nothing is attached.
    """
    state = get_speculative_state(batch)
    if state is None:
        return
    setattr(batch, _SPECULATIVE_ATTRIBUTE, None)
    if batch.uids == [state.uid]:
        flush_speculative_tokens(batch, state)
    else:
        # The row is gone (finished or cancelled); its cache rows were
        # dropped with it, so there is nothing to trim.
        state.queue.clear()
    if state.rounds > 0:
        _log_acceptance(state)


def flush_speculative_tokens(batch: GenerationBatch, state: SpeculativeState) -> None:
    """Trim verified-ahead tokens so the caches match committed history."""
    if state.queue:
        trim_prompt_cache(batch.prompt_cache, len(state.queue))
        state.queue.clear()
    committed_length = len(state.prompt_prefix) + len(batch.tokens[0])
    draft_overhang = state.draft_position - committed_length
    if draft_overhang > 0:
        trim_prompt_cache(state.draft_cache, draft_overhang)
        state.draft_position = committed_length


def speculative_step(
    batch: GenerationBatch, state: SpeculativeState
) -> tuple[list[int], list[mx.array]]:
    """Drop-in replacement for the patched ``GenerationBatch._step``.

    Returns the pending token (sampled on a previous call) and either drains
    one queued verified token or runs a fresh draft/verify round to refill
    the queue.
    """
    batch._current_tokens = batch._next_tokens
    batch._current_logprobs = batch._next_logprobs
    inputs = batch._current_tokens
    assert inputs is not None, "speculative_step requires initialized _next_tokens"

    if state.queue:
        queued_token, queued_logprobs = state.queue.pop(0)
        batch._next_tokens = mx.array([queued_token], dtype=mx.uint32)
        batch._next_logprobs = queued_logprobs
    else:
        _run_speculative_round(batch, state, pending_token=int(inputs.item()))

    current_logprobs = batch._current_logprobs
    if isinstance(current_logprobs, mx.array):
        mx.eval(inputs, current_logprobs)
    elif current_logprobs:
        mx.eval(inputs, *current_logprobs)
    else:
        mx.eval(inputs)

    token_list = cast(list[int], inputs.tolist())
    batch.tokens[0].append(token_list[0])

    if isinstance(current_logprobs, mx.array):
        current_logprobs = list(current_logprobs)
    return token_list, current_logprobs


def _run_speculative_round(
    batch: GenerationBatch, state: SpeculativeState, pending_token: int
) -> None:
    """Propose with the draft model, verify with the target, trim overshoot.

    On entry the target cache covers exactly ``batch.tokens[0]`` and
    ``pending_token`` is not yet in it. On exit the first verified token is
    the new pending token, the rest sit in ``state.queue``, and both caches
    cover their histories plus the queued (verified-ahead) tokens.
    """
    committed = batch.tokens[0]
    proposals = _draft_proposals(state, committed, pending_token)
    verify_tokens = [pending_token, *proposals]

    logits = batch.model(mx.array([verify_tokens]), cache=batch.prompt_cache)

    sampler = (
        batch.samplers[0]
        if batch.samplers and batch.samplers[0] is not None
        else batch.fallback_sampler
    )
    processors = batch.logits_processors[0] if batch.logits_processors else []

    # Position i of `logits` is the distribution for the token following
    # verify_tokens[i]. Walk forward sampling from the target; a proposal
    # mismatch invalidates every later position.
    verified: list[tuple[int, mx.array]] = []
    matched_proposals = 0
    for position in range(len(verify_tokens)):
        position_logits = logits[:, position, :]
        if processors:
            context = mx.array(committed + verify_tokens[:position])
            for processor in processors:
                position_logits = processor(context, position_logits)
        position_logprobs = position_logits - mx.logsumexp(
            position_logits, axis=-1, keepdims=True
        )
        sampled_token = int(sampler(position_logprobs).item())
        verified.append((sampled_token, position_logprobs))
        if position == len(proposals) or proposals[position] != sampled_token:
            break
        matched_proposals += 1

    # The verify pass advanced the target cache over every proposal; keep
    # only the matched prefix (which equals the tokens that will be emitted
    # before the next round).
    target_overshoot = len(proposals) - matched_proposals
    if target_overshoot > 0:
        trim_prompt_cache(batch.prompt_cache, target_overshoot)

    history_with_pending = len(state.prompt_prefix) + len(committed) + 1
    draft_overshoot = state.draft_position - (history_with_pending + matched_proposals)
    if draft_overshoot > 0:
        trim_prompt_cache(state.draft_cache, draft_overshoot)
        state.draft_position -= draft_overshoot

    first_token, first_logprobs = verified[0]
    batch._next_tokens = mx.array([first_token], dtype=mx.uint32)
    batch._next_logprobs = first_logprobs
    state.queue = verified[1:]

    state.rounds += 1
    state.drafted_tokens += len(proposals)
    state.accepted_tokens += matched_proposals
    if state.rounds % _ACCEPTANCE_LOG_INTERVAL_ROUNDS == 0:
        _log_acceptance(state)


def _log_acceptance(state: SpeculativeState) -> None:
    acceptance = state.accepted_tokens / max(1, state.drafted_tokens)
    tokens_per_round = (state.accepted_tokens + state.rounds) / state.rounds
    logger.info(
        f"speculative decode: {acceptance:.0%} draft acceptance, "
        f"{tokens_per_round:.2f} tokens/target-pass over {state.rounds} rounds"
    )


def _draft_proposals(
    state: SpeculativeState, committed: list[int], pending_token: int
) -> list[int]:
    """Greedily draft ``DRAFT_TOKENS_PER_ROUND`` continuation tokens.

    First feeds the draft cache any history it has not seen yet (its own
    prefill), then autoregressively proposes. The final proposal is never fed
    to the draft cache; whether it enters the history is decided by
    verification.
    """
    full_history = state.prompt_prefix + committed
    catch_up_tokens = full_history[state.draft_position :] + [pending_token]

    logits: mx.array | None = None
    for start in range(0, len(catch_up_tokens), _DRAFT_CATCH_UP_CHUNK_SIZE):
        chunk = catch_up_tokens[start : start + _DRAFT_CATCH_UP_CHUNK_SIZE]
        logits = state.draft_model(mx.array([chunk]), cache=state.draft_cache)
    assert logits is not None
    state.draft_position = len(full_history) + 1

    proposals: list[int] = []
    proposal = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    proposals.append(proposal)
    for _ in range(DRAFT_TOKENS_PER_ROUND - 1):
        logits = state.draft_model(mx.array([[proposal]]), cache=state.draft_cache)
        state.draft_position += 1
        proposal = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        proposals.append(proposal)
    return proposals
