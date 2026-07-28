from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import GenerationBatch, SequenceStateMachine

from exo.worker.engines.mlx.auto_parallel import PipelineRelayContext
from exo.worker.engines.mlx.patches import opt_batch_gen


@dataclass
class _CallCounts:
    processor: int = 0
    sampler: int = 0


class _MockModel(nn.Module):
    def __init__(self, batch_size: int, vocabulary_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size

    def __call__(self, inputs: mx.array, *, cache: list[object]) -> mx.array:
        assert inputs.shape == (self.batch_size, 1)
        assert cache == []
        return mx.ones((self.batch_size, 1, self.vocabulary_size))


def _generation_batch(
    call_counts: _CallCounts, batch_size: int, vocabulary_size: int
) -> GenerationBatch:
    def processor(context: mx.array, logits: mx.array) -> mx.array:
        call_counts.processor += 1
        return logits

    def sampler(logprobs: mx.array) -> mx.array:
        call_counts.sampler += 1
        return mx.argmax(logprobs, axis=-1)

    return GenerationBatch(
        model=_MockModel(batch_size, vocabulary_size),
        uids=list(range(batch_size)),
        inputs=mx.arange(batch_size, dtype=mx.int32),
        prompt_cache=[],
        tokens=[[] for _ in range(batch_size)],
        samplers=[sampler for _ in range(batch_size)],
        fallback_sampler=sampler,
        logits_processors=[[processor] for _ in range(batch_size)],
        state_machines=[SequenceStateMachine() for _ in range(batch_size)],
        max_tokens=[4 for _ in range(batch_size)],
    )


def test_non_last_pipeline_rank_uses_constant_size_placeholders() -> None:
    batch_size = 2
    vocabulary_size = 32_768
    logits = mx.ones((batch_size, vocabulary_size), dtype=mx.bfloat16)

    logprobs, sampled = opt_batch_gen.make_non_last_relay_outputs(
        logits=logits,
        batch_size=batch_size,
        needs_topk=False,
    )
    mx.eval(logprobs, sampled)

    assert logprobs.shape == (batch_size, 1)
    assert logprobs.dtype == logits.dtype
    assert sampled.shape == (batch_size,)
    assert sampled.dtype == mx.int32
    assert sampled.tolist() == [0, 0]


def test_non_last_pipeline_rank_rejects_topk_logprobs() -> None:
    with pytest.raises(ValueError, match="rank-local logprobs"):
        opt_batch_gen.make_non_last_relay_outputs(
            logits=mx.ones((1, 128)),
            batch_size=1,
            needs_topk=True,
        )


def test_non_last_pipeline_rank_skips_vocabulary_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 2
    vocabulary_size = 256
    call_counts = _CallCounts()
    relay_context = PipelineRelayContext(
        group=cast(mx.distributed.Group, object()),
        device_rank=0,
        world_size=2,
    )

    def active_relay_context(_model: nn.Module) -> PipelineRelayContext:
        return relay_context

    def relay_sampled_tokens(
        sampled: mx.array, _relay_context: PipelineRelayContext
    ) -> mx.array:
        assert sampled.tolist() == [0, 0]
        return mx.array([17, 23], dtype=mx.int32)

    monkeypatch.setattr(opt_batch_gen, "get_active_relay_context", active_relay_context)
    monkeypatch.setattr(opt_batch_gen, "relay_sampled_tokens", relay_sampled_tokens)
    opt_batch_gen.apply_batch_gen_patch()

    responses = _generation_batch(call_counts, batch_size, vocabulary_size).next()

    assert [response.token for response in responses] == [17, 23]
    assert all(response.logprobs.shape == (1,) for response in responses)
    assert call_counts == _CallCounts()


def test_last_pipeline_rank_preserves_vocabulary_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 2
    vocabulary_size = 256
    call_counts = _CallCounts()
    relay_context = PipelineRelayContext(
        group=cast(mx.distributed.Group, object()),
        device_rank=1,
        world_size=2,
    )

    def active_relay_context(_model: nn.Module) -> PipelineRelayContext:
        return relay_context

    def relay_sampled_tokens(
        sampled: mx.array, _relay_context: PipelineRelayContext
    ) -> mx.array:
        return sampled

    monkeypatch.setattr(opt_batch_gen, "get_active_relay_context", active_relay_context)
    monkeypatch.setattr(opt_batch_gen, "relay_sampled_tokens", relay_sampled_tokens)
    opt_batch_gen.apply_batch_gen_patch()

    responses = _generation_batch(call_counts, batch_size, vocabulary_size).next()

    assert all(response.logprobs.shape == (vocabulary_size,) for response in responses)
    assert call_counts == _CallCounts(processor=4, sampler=4)
