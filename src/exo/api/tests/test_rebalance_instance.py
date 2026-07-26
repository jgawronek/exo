import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


async def test_rebalance_rejects_unsafe_intermediate_step_as_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = (NodeId(), NodeId(), NodeId())
    runners = (RunnerId(), RunnerId(), RunnerId())
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(600),
        n_layers=10,
        hidden_size=128,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    current_counts = (4, 1, 5)
    start_layer = 0
    shards: dict[RunnerId, PipelineShardMetadata] = {}
    for rank, (runner_id, layer_count) in enumerate(
        zip(runners, current_counts, strict=True)
    ):
        shards[runner_id] = PipelineShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=3,
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
            n_layers=10,
        )
        start_layer += layer_count

    instance_id = InstanceId()
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(nodes, runners, strict=True)),
        ),
        hosts_by_node={},
        ephemeral_port=50_000,
    )
    api = object.__new__(API)
    api.state_replica = None
    api.state = State(
        instances={instance_id: instance},
        node_memory={
            nodes[0]: _memory_usage(total_bytes=1000, available_bytes=1000),
            nodes[1]: _memory_usage(total_bytes=320, available_bytes=300),
            nodes[2]: _memory_usage(total_bytes=1000, available_bytes=1000),
        },
    )

    # nodes[1]'s ceiling is 3 layers (three-quarters of 320 bytes minus one
    # 60-byte transient layer), so an allocation growing it to 4 has an
    # unsafe step no matter how the planner orders the moves.
    def unsafe_allocation(**_: object) -> dict[NodeId, int]:
        return dict(zip(nodes, (1, 4, 5), strict=True))

    monkeypatch.setattr(
        "exo.api.main.allocate_layers_by_measured_speed",
        unsafe_allocation,
    )

    with pytest.raises(HTTPException) as raised:
        await api.rebalance_instance(instance_id)

    assert raised.value.status_code == 400
    assert "step 3" in str(raised.value.detail)


def _memory_usage(*, total_bytes: int, available_bytes: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=total_bytes,
        ram_available=available_bytes,
        swap_total=0,
        swap_available=0,
    )
