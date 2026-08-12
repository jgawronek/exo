from typing import cast

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator.generate import thinking_budget_processor

VOCAB = 32
THINK_START = 10
THINK_END = 11
PLAIN = 5


class _StubTokenizer:
    think_start_tokens: tuple[int, ...] = (THINK_START,)
    think_end_tokens: tuple[int, ...] = (THINK_END,)


class _TaglessTokenizer:
    think_start_tokens = None
    think_end_tokens = None


def _tokenizer() -> TokenizerWrapper:
    return cast(TokenizerWrapper, cast(object, _StubTokenizer()))


def _logits() -> mx.array:
    return mx.zeros((1, VOCAB))


def _forced_token(logits: mx.array) -> int:
    return int(mx.argmax(logits, axis=-1).item())


def test_returns_none_without_think_tags() -> None:
    tokenizer = cast(TokenizerWrapper, cast(object, _TaglessTokenizer()))
    assert thinking_budget_processor(tokenizer, 256, starts_in_thinking=True) is None


def test_forces_think_end_when_budget_exhausted() -> None:
    process = thinking_budget_processor(_tokenizer(), 3, starts_in_thinking=True)
    assert process is not None

    history = [THINK_START]
    for _ in range(2):
        out = process(mx.array(history), _logits())
        assert mx.array_equal(out, _logits())
        history.append(PLAIN)

    forced = process(mx.array(history), _logits())
    assert _forced_token(forced) == THINK_END


def test_passes_through_after_forced_close() -> None:
    process = thinking_budget_processor(_tokenizer(), 1, starts_in_thinking=True)
    assert process is not None

    forced = process(mx.array([THINK_START]), _logits())
    assert _forced_token(forced) == THINK_END

    after = process(mx.array([THINK_START, THINK_END]), _logits())
    assert mx.array_equal(after, _logits())


def test_natural_think_end_resets_budget() -> None:
    process = thinking_budget_processor(_tokenizer(), 2, starts_in_thinking=True)
    assert process is not None

    _ = process(mx.array([THINK_START]), _logits())
    # Model closes thinking on its own; the processor must stop counting.
    out = process(mx.array([THINK_START, THINK_END]), _logits())
    assert mx.array_equal(out, _logits())
    for _ in range(4):
        out = process(mx.array([THINK_START, THINK_END, PLAIN]), _logits())
        assert mx.array_equal(out, _logits())


def test_never_forces_outside_thinking() -> None:
    process = thinking_budget_processor(_tokenizer(), 1, starts_in_thinking=False)
    assert process is not None

    for _ in range(5):
        out = process(mx.array([PLAIN, PLAIN]), _logits())
        assert mx.array_equal(out, _logits())


def test_detects_thinking_started_mid_stream() -> None:
    process = thinking_budget_processor(_tokenizer(), 2, starts_in_thinking=False)
    assert process is not None

    assert mx.array_equal(process(mx.array([PLAIN]), _logits()), _logits())
    # Previous sampled token opened a thinking block.
    assert mx.array_equal(process(mx.array([PLAIN, THINK_START]), _logits()), _logits())
    forced = process(mx.array([PLAIN, THINK_START, PLAIN]), _logits())
    assert _forced_token(forced) == THINK_END
