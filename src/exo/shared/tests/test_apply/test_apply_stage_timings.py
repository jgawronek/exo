from exo.shared.apply import (
    apply_instance_deleted,
    apply_stage_timings_updated,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, StageTimingsUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import StageTiming
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _instance(node_id: NodeId) -> Instance:
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_mb(1000),
            n_layers=32,
            hidden_size=2048,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={node_id: runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _timing(compute_ms_per_token: float) -> StageTiming:
    return StageTiming(
        layers_held=32,
        compute_ms_per_token=compute_ms_per_token,
        communication_ms_per_token=1.0,
        tokens_measured=64,
    )


def test_stage_timing_recorded_per_instance_and_node() -> None:
    node_id = NodeId()
    instance = _instance(node_id)
    state = State(instances={instance.instance_id: instance})

    timing = _timing(compute_ms_per_token=12.5)
    new_state = apply_stage_timings_updated(
        StageTimingsUpdated(
            instance_id=instance.instance_id, node_id=node_id, timing=timing
        ),
        state,
    )
    assert new_state.instance_stage_timings == {instance.instance_id: {node_id: timing}}


def test_stage_timing_replaces_previous_measurement_for_node() -> None:
    node_id = NodeId()
    instance = _instance(node_id)
    state = State(
        instances={instance.instance_id: instance},
        instance_stage_timings={instance.instance_id: {node_id: _timing(20.0)}},
    )

    updated = _timing(compute_ms_per_token=10.0)
    new_state = apply_stage_timings_updated(
        StageTimingsUpdated(
            instance_id=instance.instance_id, node_id=node_id, timing=updated
        ),
        state,
    )
    assert new_state.instance_stage_timings[instance.instance_id][node_id] == updated


def test_stage_timing_for_unknown_instance_is_ignored() -> None:
    state = State()
    new_state = apply_stage_timings_updated(
        StageTimingsUpdated(
            instance_id=InstanceId(), node_id=NodeId(), timing=_timing(5.0)
        ),
        state,
    )
    assert new_state.instance_stage_timings == {}


def test_instance_deleted_clears_stage_timings() -> None:
    node_id = NodeId()
    instance = _instance(node_id)
    state = State(
        instances={instance.instance_id: instance},
        instance_stage_timings={instance.instance_id: {node_id: _timing(5.0)}},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id), state
    )
    assert new_state.instance_stage_timings == {}
