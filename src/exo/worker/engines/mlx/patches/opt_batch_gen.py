from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch

from exo.worker.engines.mlx.auto_parallel import (
    get_active_relay_context,
    relay_sampled_tokens,
)
from exo.worker.engines.mlx.generator.speculative import (
    get_speculative_state,
    speculative_step,
)

_PRECOMPUTE_TOP_K = 20


@dataclass
class BatchTopKLogprobs:
    uids: list[int] = field(default_factory=list)
    indices: mx.array | None = None
    values: mx.array | None = None
    selected: mx.array | None = None
    _uid_to_row: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._uid_to_row = {uid: i for i, uid in enumerate(self.uids)}

    def for_uid(self, uid: int) -> tuple[list[int], list[float], float] | None:
        if self.indices is None or self.values is None or self.selected is None:
            return None
        row = self._uid_to_row.get(uid)
        if row is None:
            return None
        return (
            cast(list[int], self.indices[row].tolist()),
            cast(list[float], self.values[row].tolist()),
            float(self.selected[row].item()),
        )


@dataclass
class _TopKBuffer:
    needs_topk: bool = False
    pending: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)
    ready: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)


def _get_buffer(batch: GenerationBatch) -> _TopKBuffer:
    buf = getattr(batch, "_topk_buffer", None)
    if buf is None:
        buf = _TopKBuffer()
        batch._topk_buffer = buf  # pyright: ignore[reportAttributeAccessIssue]
    return buf


def set_needs_topk(batch: GenerationBatch, needed: bool) -> None:
    _get_buffer(batch).needs_topk = needed


def take_ready_topk(batch: GenerationBatch) -> BatchTopKLogprobs:
    return _get_buffer(batch).ready


def make_non_last_relay_outputs(
    logits: mx.array, batch_size: int, needs_topk: bool
) -> tuple[mx.array, mx.array]:
    """Return constant-size placeholders for a non-last pipeline rank.

    Raises:
        ValueError: Propagates to the runner if token relay is incorrectly
            enabled for a request that needs rank-local log probabilities.
    """
    if needs_topk:
        raise ValueError("token relay does not support rank-local logprobs")
    logprobs = mx.zeros((batch_size, 1), dtype=logits.dtype)
    sampled = mx.zeros((batch_size,), dtype=mx.int32)
    return logprobs, sampled


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    speculative = get_speculative_state(self)
    if speculative is not None:
        # The engine attaches speculative state only for a single-row batch
        # with no logprobs request and no pending prompts, so the draft/verify
        # path can fully replace the plain decode step.
        return speculative_step(self, speculative)

    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens
    assert inputs is not None, "_step requires initialized _next_tokens"

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]

    relay_context = get_active_relay_context(self.model)
    if relay_context is not None and not relay_context.is_last_rank:
        logprobs, sampled = make_non_last_relay_outputs(
            logits, len(self.uids), buf.needs_topk
        )
    else:
        if self.logits_processors is not None and any(self.logits_processors):
            processed_logits: list[mx.array] = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        if self.samplers is not None and any(self.samplers):
            all_samples: list[mx.array] = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                all_samples.append(sample_sampler(logprobs[e : e + 1]))
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

    if relay_context is not None:
        # Token-relay decode: only the last pipeline rank sampled from real
        # logits; circulate its token ids so every rank stays in lockstep.
        sampled = relay_sampled_tokens(sampled, relay_context)

    self._next_tokens = sampled
    self._next_logprobs = logprobs

    if buf.needs_topk:
        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(batch_size), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            pending_indices,
            pending_values,
            pending_selected,
        )
    else:
        mx.async_eval(self._next_tokens, self._next_logprobs)

    current_lp = self._current_logprobs
    if isinstance(current_lp, mx.array):
        mx.eval(inputs, current_lp)
    elif current_lp:
        mx.eval(inputs, *current_lp)
    else:
        mx.eval(inputs)

    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)

    if isinstance(current_lp, mx.array):
        current_lp = list(current_lp)
    return token_list, current_lp


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step
