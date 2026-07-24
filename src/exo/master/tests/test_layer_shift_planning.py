import pytest

from exo.master.placement_utils import plan_pipeline_layer_shift_steps
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata

_TOTAL_LAYERS = 6

_MODEL_CARD = ModelCard(
    model_id=ModelId("test-model"),
    storage_size=Memory.from_mb(1000),
    n_layers=_TOTAL_LAYERS,
    hidden_size=2048,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)


def _shards(layer_counts: list[int]) -> dict[RunnerId, PipelineShardMetadata]:
    world_size = len(layer_counts)
    shards: dict[RunnerId, PipelineShardMetadata] = {}
    start_layer = 0
    for rank, layer_count in enumerate(layer_counts):
        shards[RunnerId(f"runner-{rank}")] = PipelineShardMetadata(
            model_card=_MODEL_CARD,
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
            n_layers=sum(layer_counts),
        )
        start_layer += layer_count
    return shards


def _layer_counts_by_rank(
    step: dict[RunnerId, PipelineShardMetadata],
) -> list[int]:
    ordered = sorted(step.values(), key=lambda shard: shard.device_rank)
    return [shard.end_layer - shard.start_layer for shard in ordered]


def _assert_contiguous(step: dict[RunnerId, PipelineShardMetadata]) -> None:
    ordered = sorted(step.values(), key=lambda shard: shard.device_rank)
    assert ordered[0].start_layer == 0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        assert previous.end_layer == current.start_layer
    assert ordered[-1].end_layer == ordered[-1].n_layers


def test_single_boundary_move_is_one_step() -> None:
    current = _shards([4, 2])
    target = dict(
        zip(sorted(current, key=lambda r: current[r].device_rank), [3, 3], strict=True)
    )
    steps = plan_pipeline_layer_shift_steps(current, target)
    assert len(steps) == 1
    assert _layer_counts_by_rank(steps[0]) == [3, 3]
    _assert_contiguous(steps[0])


def test_matching_target_needs_no_steps() -> None:
    current = _shards([3, 3])
    target = dict(
        zip(sorted(current, key=lambda r: current[r].device_rank), [3, 3], strict=True)
    )
    assert plan_pipeline_layer_shift_steps(current, target) == []


def test_multi_hop_moves_one_layer_per_step_and_keeps_every_rank_nonempty() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda r: current[r].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))

    steps = plan_pipeline_layer_shift_steps(current, target)
    # Boundaries move from [4, 5] to [1, 2]: 3 + 3 single-layer steps.
    assert len(steps) == 6

    previous_counts = _layer_counts_by_rank(
        {runner_id: shard for runner_id, shard in current.items()}
    )
    for step in steps:
        _assert_contiguous(step)
        counts = _layer_counts_by_rank(step)
        assert all(count >= 1 for count in counts)
        moved = sum(abs(a - b) for a, b in zip(counts, previous_counts, strict=True))
        # Exactly one layer crosses exactly one boundary per step.
        assert moved == 2
        previous_counts = counts
    assert previous_counts == [1, 1, 4]


def test_final_step_matches_target_exactly() -> None:
    current = _shards([1, 4, 1])
    ranked = sorted(current, key=lambda r: current[r].device_rank)
    target = dict(zip(ranked, [2, 1, 3], strict=True))

    steps = plan_pipeline_layer_shift_steps(current, target)
    assert _layer_counts_by_rank(steps[-1]) == [2, 1, 3]
    for runner_id, shard in steps[-1].items():
        assert shard.device_rank == current[runner_id].device_rank
        assert shard.world_size == current[runner_id].world_size
        assert shard.n_layers == current[runner_id].n_layers


def test_rejects_mismatched_runner_set() -> None:
    current = _shards([3, 3])
    with pytest.raises(ValueError, match="exactly the instance's runners"):
        plan_pipeline_layer_shift_steps(current, {RunnerId("other"): _TOTAL_LAYERS})


def test_rejects_counts_not_summing_to_total() -> None:
    current = _shards([3, 3])
    ranked = sorted(current, key=lambda r: current[r].device_rank)
    with pytest.raises(ValueError, match="sum to"):
        plan_pipeline_layer_shift_steps(current, dict(zip(ranked, [3, 4], strict=True)))


def test_rejects_empty_rank() -> None:
    current = _shards([3, 3])
    ranked = sorted(current, key=lambda r: current[r].device_rank)
    with pytest.raises(ValueError, match="at least one layer"):
        plan_pipeline_layer_shift_steps(current, dict(zip(ranked, [6, 0], strict=True)))
