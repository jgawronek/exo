from exo.shared.apply import (
    apply_expert_activations_updated,
    apply_instance_deleted,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import ExpertActivationsUpdated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import LayerExpertActivity
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


def _activity(activations: list[int], tokens: int = 64) -> LayerExpertActivity:
    unique = sum(1 for count in activations if count > 0)
    return LayerExpertActivity(
        num_experts=len(activations),
        tokens_measured=tokens,
        unique_experts_activated=unique,
        effective_experts=float(unique),
        top_activations={
            str(index): count for index, count in enumerate(activations) if count > 0
        },
    )


def test_expert_activity_merges_layers_across_reports() -> None:
    node_id = NodeId()
    instance = _instance(node_id)
    state = State(instances={instance.instance_id: instance})

    state = apply_expert_activations_updated(
        ExpertActivationsUpdated(
            instance_id=instance.instance_id,
            node_id=node_id,
            layers={"3": _activity([4, 0, 2, 0])},
        ),
        state,
    )
    state = apply_expert_activations_updated(
        ExpertActivationsUpdated(
            instance_id=instance.instance_id,
            node_id=node_id,
            layers={"5": _activity([0, 6, 0, 0]), "3": _activity([1, 1, 1, 1])},
        ),
        state,
    )

    activity = state.instance_expert_activity[instance.instance_id]
    assert set(activity) == {"3", "5"}
    # The newest report for a layer replaces the previous window.
    assert activity["3"].top_activations == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert activity["3"].unique_experts_activated == 4
    assert activity["5"].unique_experts_activated == 1


def test_expert_activity_for_unknown_instance_is_ignored() -> None:
    state = State()
    updated = apply_expert_activations_updated(
        ExpertActivationsUpdated(
            instance_id=InstanceId(),
            node_id=NodeId(),
            layers={"0": _activity([1])},
        ),
        state,
    )
    assert updated.instance_expert_activity == {}


def test_instance_deletion_drops_expert_activity() -> None:
    node_id = NodeId()
    instance = _instance(node_id)
    state = State(instances={instance.instance_id: instance})
    state = apply_expert_activations_updated(
        ExpertActivationsUpdated(
            instance_id=instance.instance_id,
            node_id=node_id,
            layers={"0": _activity([2, 2])},
        ),
        state,
    )

    state = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id), state
    )

    assert state.instance_expert_activity == {}
