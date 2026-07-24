from exo.shared.apply import apply_instance_shard_assignments_updated
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceShardAssignmentsUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import StageTiming
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata

_MODEL_CARD = ModelCard(
    model_id=ModelId("test-model"),
    storage_size=Memory.from_mb(1000),
    n_layers=8,
    hidden_size=2048,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)


def _shard(rank: int, start_layer: int, end_layer: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_MODEL_CARD,
        device_rank=rank,
        world_size=2,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=8,
    )


def _assignments(
    node_ids: list[NodeId],
    runner_ids: list[RunnerId],
    boundary: int,
) -> ShardAssignments:
    return ShardAssignments(
        model_id=ModelId("test-model"),
        node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
        runner_to_shard={
            runner_ids[0]: _shard(0, 0, boundary),
            runner_ids[1]: _shard(1, boundary, 8),
        },
    )


def _instance(assignments: ShardAssignments) -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=assignments,
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_shard_assignments_replaced_and_stage_timings_cleared() -> None:
    node_ids = [NodeId(), NodeId()]
    runner_ids = [RunnerId(), RunnerId()]
    instance = _instance(_assignments(node_ids, runner_ids, boundary=6))
    stale_timing = StageTiming(
        layers_held=6,
        compute_ms_per_token=10.0,
        communication_ms_per_token=1.0,
        tokens_measured=64,
    )
    state = State(
        instances={instance.instance_id: instance},
        instance_stage_timings={instance.instance_id: {node_ids[0]: stale_timing}},
    )

    shifted = _assignments(node_ids, runner_ids, boundary=5)
    new_state = apply_instance_shard_assignments_updated(
        InstanceShardAssignmentsUpdated(
            instance_id=instance.instance_id, shard_assignments=shifted
        ),
        state,
    )

    updated_instance = new_state.instances[instance.instance_id]
    assert updated_instance.shard_assignments == shifted
    assert updated_instance.instance_id == instance.instance_id
    assert new_state.instance_stage_timings == {}


def test_update_for_unknown_instance_is_ignored() -> None:
    node_ids = [NodeId(), NodeId()]
    runner_ids = [RunnerId(), RunnerId()]
    state = State()
    new_state = apply_instance_shard_assignments_updated(
        InstanceShardAssignmentsUpdated(
            instance_id=InstanceId(),
            shard_assignments=_assignments(node_ids, runner_ids, boundary=4),
        ),
        state,
    )
    assert new_state == state
