import anyio
import pytest

from exo.master.main import Master, recover_pipeline_layer_shift_steps
from exo.master.placement_utils import plan_pipeline_layer_shift_steps
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import IndexedEvent, InstanceDeleted, TaskDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.tasks import ShiftLayers, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.state_replica import StateReplica

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


def _restore_master(
    *,
    current_shards: dict[RunnerId, PipelineShardMetadata],
    task: ShiftLayers,
    additional_tasks: tuple[ShiftLayers, ...] = (),
    node_memory_by_rank: tuple[MemoryUsage, ...] = (),
) -> Master:
    instance_id = task.instance_id
    node_to_runner = {
        NodeId(f"node-{index}"): runner_id
        for index, runner_id in enumerate(current_shards)
    }
    effective_node_memory = node_memory_by_rank or tuple(
        MemoryUsage.from_bytes(
            ram_total=10_000_000_000,
            ram_available=10_000_000_000,
            swap_total=0,
            swap_available=0,
        )
        for _ in node_to_runner
    )
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=_MODEL_CARD.model_id,
            runner_to_shard=current_shards,
            node_to_runner=node_to_runner,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    session = SessionId(master_node_id=NodeId("master"), election_clock=1)
    master = object.__new__(Master)
    master.state_replica = StateReplica(
        session=session,
        initial_state=State(
            instances={instance_id: instance},
            node_memory=dict(zip(node_to_runner, effective_node_memory, strict=True)),
            tasks={
                replicated_task.task_id: replicated_task
                for replicated_task in (task, *additional_tasks)
            },
        ),
        ready=True,
    )
    master._layer_shift_plans = {}  # pyright: ignore[reportPrivateUsage]
    master._shift_task_instance = {}  # pyright: ignore[reportPrivateUsage]
    master._recovered_shift_updates = []  # pyright: ignore[reportPrivateUsage]
    master._recovered_shift_deletions = set()  # pyright: ignore[reportPrivateUsage]
    master._recovered_instance_deletions = set()  # pyright: ignore[reportPrivateUsage]
    master._restore_layer_shift_plans()  # pyright: ignore[reportPrivateUsage]
    return master


def test_master_quarantines_recovered_shift_that_exceeds_live_memory() -> None:
    current = _shards([4, 2])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [5, 1], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )
    memory = (
        MemoryUsage.from_bytes(
            ram_total=1000,
            ram_available=240,
            swap_total=0,
            swap_available=0,
        ),
        MemoryUsage.from_bytes(
            ram_total=1000,
            ram_available=500,
            swap_total=0,
            swap_available=0,
        ),
    )

    master = _restore_master(
        current_shards=current,
        task=task,
        node_memory_by_rank=memory,
    )

    assert master._recovered_instance_deletions == {  # pyright: ignore[reportPrivateUsage]
        task.instance_id
    }
    assert master._recovered_shift_deletions == {  # pyright: ignore[reportPrivateUsage]
        task.task_id
    }
    assert task.instance_id not in master._layer_shift_plans  # pyright: ignore[reportPrivateUsage]


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


def test_pass_through_rank_sheds_before_gaining() -> None:
    # A mid-pipeline rank whose layer count is unchanged, but through which
    # layers flow rightward, must never transiently exceed
    # max(current, target) — a rank at its live-shift memory ceiling would
    # otherwise fail step validation before shedding to its neighbour.
    current = _shards([3, 37, 10, 10])
    ranked = sorted(current, key=lambda r: current[r].device_rank)
    target_counts = [1, 37, 7, 15]
    target = dict(zip(ranked, target_counts, strict=True))

    steps = plan_pipeline_layer_shift_steps(current, target)

    current_counts = _layer_counts_by_rank(current)
    for step in steps:
        _assert_contiguous(step)
        for rank, count in enumerate(_layer_counts_by_rank(step)):
            assert count <= max(current_counts[rank], target_counts[rank])
    assert _layer_counts_by_rank(steps[-1]) == target_counts


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


def test_rejects_inconsistent_replicated_boundaries() -> None:
    current = _shards([3, 3])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    current[ranked[1]] = current[ranked[1]].model_copy(update={"start_layer": 4})
    target = dict(zip(ranked, [3, 3], strict=True))

    with pytest.raises(ValueError, match="boundaries must be contiguous"):
        plan_pipeline_layer_shift_steps(current, target)


def test_shift_task_persists_full_plan_progress() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    plan_id = TaskId()

    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        plan_id=plan_id,
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )

    assert task.current_step == 1
    assert task.total_steps == 6
    assert task.plan_id == plan_id
    assert task.target_layer_counts == target


def test_recovers_remaining_shift_steps_from_replicated_task() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )

    recovered = recover_pipeline_layer_shift_steps(current, task)

    assert recovered == steps


def test_rejects_inconsistent_replicated_shift_progress() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[1],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )

    with pytest.raises(ValueError, match="next replicated step"):
        recover_pipeline_layer_shift_steps(current, task)


def test_recovers_after_completed_step_was_committed_before_failover() -> None:
    initial = _shards([4, 1, 1])
    ranked = sorted(initial, key=lambda runner_id: initial[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(initial, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Complete,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )

    recovered = recover_pipeline_layer_shift_steps(steps[0], task)

    assert recovered[0] == steps[0]
    assert recovered[1:] == steps[1:]


def test_master_restores_active_plan_and_task_mapping() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )

    master = _restore_master(current_shards=current, task=task)

    assert (
        list(
            master._layer_shift_plans[task.instance_id]  # pyright: ignore[reportPrivateUsage]
        )
        == steps
    )
    assert (
        master._shift_task_instance[task.task_id]  # pyright: ignore[reportPrivateUsage]
        == task.instance_id
    )


def test_master_quarantines_instance_when_recovery_is_unsafe() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )
    current[ranked[1]] = current[ranked[1]].model_copy(update={"start_layer": 5})

    master = _restore_master(current_shards=current, task=task)

    deleted_instances = (
        master._recovered_instance_deletions  # pyright: ignore[reportPrivateUsage]
    )
    assert deleted_instances == {task.instance_id}


def test_master_restores_highest_completed_step_from_same_plan() -> None:
    initial = _shards([4, 1, 1])
    ranked = sorted(initial, key=lambda runner_id: initial[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(initial, target)
    plan_id = TaskId()
    first_task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Complete,
        new_shards=steps[0],
        plan_id=plan_id,
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )
    second_task = ShiftLayers(
        instance_id=first_task.instance_id,
        task_status=TaskStatus.Complete,
        new_shards=steps[1],
        plan_id=plan_id,
        target_layer_counts=target,
        current_step=2,
        total_steps=len(steps),
    )

    master = _restore_master(
        current_shards=steps[1],
        task=first_task,
        additional_tasks=(second_task,),
    )

    assert (
        master._recovered_instance_deletions  # pyright: ignore[reportPrivateUsage]
        == set()
    )
    assert (
        master._recovered_shift_deletions  # pyright: ignore[reportPrivateUsage]
        == {first_task.task_id}
    )
    updates = master._recovered_shift_updates  # pyright: ignore[reportPrivateUsage]
    assert [update.task_id for update in updates] == [second_task.task_id]


@pytest.mark.asyncio
async def test_master_waits_for_recovered_state_cleanup() -> None:
    current = _shards([4, 1, 1])
    ranked = sorted(current, key=lambda runner_id: current[runner_id].device_rank)
    target = dict(zip(ranked, [1, 1, 4], strict=True))
    steps = plan_pipeline_layer_shift_steps(current, target)
    task = ShiftLayers(
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        new_shards=steps[0],
        target_layer_counts=target,
        current_step=1,
        total_steps=len(steps),
    )
    current[ranked[1]] = current[ranked[1]].model_copy(update={"start_layer": 5})
    master = _restore_master(current_shards=current, task=task)
    wait_finished = False

    async def wait_for_cleanup() -> None:
        nonlocal wait_finished
        await master._wait_for_shift_recovery_reconciliation()  # pyright: ignore[reportPrivateUsage]
        wait_finished = True

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(wait_for_cleanup)
        await anyio.sleep(0)
        assert not wait_finished
        master.state_replica.apply(
            IndexedEvent(
                idx=0,
                event=InstanceDeleted(instance_id=task.instance_id),
            )
        )
        master.state_replica.apply(
            IndexedEvent(
                idx=1,
                event=TaskDeleted(task_id=task.task_id),
            )
        )

    assert wait_finished
