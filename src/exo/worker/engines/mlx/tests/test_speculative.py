# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportArgumentType=false
# pyright: reportInvalidCast=false
"""Speculative decoding must emit exactly the same tokens as plain decode.

Uses a deterministic "arithmetic" model whose next-token logits are exact
one-hots computed from (last token, cache position). That makes the plain
and speculative decode paths bit-exact comparable with no downloads and no
floating-point flakiness, while still exercising the real mlx-lm
BatchGenerator machinery, KV-cache trimming, and the patched _step hook.
"""

from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import BatchGenerator as MlxBatchGenerator

from exo.worker.engines.mlx.generator.speculative import (
    SpeculativeState,
    attach_speculative_state,
    detach_speculative_state,
)
from exo.worker.engines.mlx.patches import apply_mlx_patches
from exo.worker.engines.mlx.types import Model

apply_mlx_patches()

_VOCABULARY_SIZE = 97


def _cache_position(cache_entry: object) -> int:
    """Current sequence position for row 0, before the next cache update."""
    offset = cache_entry.offset  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(offset, mx.array):
        return int(offset[0].item())
    return int(offset)


class _ArithmeticModel(nn.Module):
    """Deterministic model: next token = (multiplier*token + position) % vocab.

    Emits exact one-hot logits, so greedy decode is bit-exact regardless of
    how many tokens are processed per forward pass. The KV cache is only used
    for position tracking, mirroring how real models depend on cache offsets.
    """

    def __init__(self, multiplier: int) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.layers: list[nn.Module] = [nn.Identity()]

    def __call__(
        self,
        x: mx.array,
        cache: list[object] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        assert cache is not None
        cache_entry = cache[0]
        start_position = _cache_position(cache_entry)
        batch_size, sequence_length = x.shape

        placeholder = mx.zeros((batch_size, 1, sequence_length, 1))
        cache_entry.update_and_fetch(placeholder, placeholder)  # pyright: ignore[reportAttributeAccessIssue]

        positions = start_position + mx.arange(sequence_length)[None, :]
        next_ids = (self.multiplier * x.astype(mx.int32) + positions) % (
            _VOCABULARY_SIZE
        )
        vocabulary = mx.arange(_VOCABULARY_SIZE)[None, None, :]
        one_hot = cast(mx.array, vocabulary == next_ids[..., None])
        return 10.0 * one_hot.astype(mx.float32)


@dataclass
class _DecodeRun:
    tokens: list[int]
    state: SpeculativeState | None


def _context_shift_processor(context: mx.array, logits: mx.array) -> mx.array:
    """Deterministically boost a token derived from the context history.

    Depends on both the context length and the last context token, so it
    fails loudly if speculative verification passes the wrong per-position
    context to processors.
    """
    last_token = int(context[-1].item()) if context.size > 0 else 0
    boosted = (7 * context.size + last_token) % _VOCABULARY_SIZE
    return logits + 100.0 * (mx.arange(_VOCABULARY_SIZE)[None, :] == boosted)


def _decode(
    target: _ArithmeticModel,
    prompt: list[int],
    total_tokens: int,
    draft: _ArithmeticModel | None,
    detach_after: int | None = None,
    with_processor: bool = False,
) -> _DecodeRun:
    """Decode with the real BatchGenerator, optionally speculating."""
    generator = MlxBatchGenerator(model=cast(nn.Module, target))
    uids = generator.insert(
        prompts=[prompt],
        max_tokens=[total_tokens + 10],
        logits_processors=[[_context_shift_processor]] if with_processor else None,
    )
    uid = uids[0]

    state: SpeculativeState | None = None
    produced: list[int] = []
    while len(produced) < total_tokens:
        generation_batch = generator._generation_batch
        if draft is not None and generation_batch.uids == [uid]:
            if state is None:
                state = SpeculativeState(
                    draft_model=cast(Model, draft), uid=uid, prompt_prefix=[]
                )
            if detach_after is not None and len(produced) >= detach_after:
                detach_speculative_state(generation_batch)
            else:
                attach_speculative_state(generation_batch, state)
        _, responses = generator.next()
        produced.extend(response.token for response in responses)
    return _DecodeRun(tokens=produced[:total_tokens], state=state)


_PROMPT = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26]


def test_perfect_draft_matches_plain_decode_and_accepts_everything() -> None:
    target = _ArithmeticModel(multiplier=3)
    plain = _decode(target, _PROMPT, total_tokens=40, draft=None)
    speculative = _decode(
        target, _PROMPT, total_tokens=40, draft=_ArithmeticModel(multiplier=3)
    )
    assert speculative.tokens == plain.tokens
    assert speculative.state is not None
    assert speculative.state.drafted_tokens > 0
    assert speculative.state.accepted_tokens == speculative.state.drafted_tokens


def test_wrong_draft_still_matches_plain_decode() -> None:
    target = _ArithmeticModel(multiplier=3)
    plain = _decode(target, _PROMPT, total_tokens=40, draft=None)
    speculative = _decode(
        target, _PROMPT, total_tokens=40, draft=_ArithmeticModel(multiplier=5)
    )
    assert speculative.tokens == plain.tokens
    assert speculative.state is not None
    # A draft with a different rule should have proposals rejected.
    assert speculative.state.accepted_tokens < speculative.state.drafted_tokens


def test_logits_processors_see_identical_context_under_speculation() -> None:
    target = _ArithmeticModel(multiplier=3)
    plain = _decode(target, _PROMPT, total_tokens=40, draft=None, with_processor=True)
    speculative = _decode(
        target,
        _PROMPT,
        total_tokens=40,
        draft=_ArithmeticModel(multiplier=3),
        with_processor=True,
    )
    assert speculative.tokens == plain.tokens


def test_detach_mid_stream_flushes_and_resumes_plain_decode() -> None:
    target = _ArithmeticModel(multiplier=3)
    plain = _decode(target, _PROMPT, total_tokens=40, draft=None)
    speculative = _decode(
        target,
        _PROMPT,
        total_tokens=40,
        draft=_ArithmeticModel(multiplier=3),
        detach_after=13,
    )
    assert speculative.tokens == plain.tokens
