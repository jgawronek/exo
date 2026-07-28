import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.master.placement_utils import pipeline_safe_max_layers
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, StageTiming
from exo.shared.types.state import State
from exo.shared.types.tasks import ShiftLayers as ShiftLayersTask
from exo.shared.types.tasks import TaskStatus
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


def _pipeline_instance(
    nodes: tuple[NodeId, ...],
    runners: tuple[RunnerId, ...],
    layer_counts: tuple[int, ...],
    model_card: ModelCard,
) -> MlxRingInstance:
    shards: dict[RunnerId, PipelineShardMetadata] = {}
    start_layer = 0
    for rank, (runner_id, layer_count) in enumerate(
        zip(runners, layer_counts, strict=True)
    ):
        shards[runner_id] = PipelineShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=len(runners),
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
            n_layers=model_card.n_layers,
        )
        start_layer += layer_count
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(nodes, runners, strict=True)),
        ),
        hosts_by_node={},
        ephemeral_port=50_000,
    )


def _model_card(storage_bytes: int, n_layers: int) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=n_layers,
        hidden_size=128,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _stage_timing(layers_held: int, compute_ms_per_token: float) -> StageTiming:
    return StageTiming(
        layers_held=layers_held,
        compute_ms_per_token=compute_ms_per_token,
        communication_ms_per_token=1.0,
        tokens_measured=32,
    )


def _api_with_instance(
    instance: MlxRingInstance,
    nodes: tuple[NodeId, ...],
    memories: tuple[MemoryUsage, ...],
    timings: dict[NodeId, StageTiming],
) -> tuple[API, list[object]]:
    api = API.__new__(API)
    api.state_replica = None
    api.state = State(
        instances={instance.instance_id: instance},
        node_memory=dict(zip(nodes, memories, strict=True)),
        instance_stage_timings={instance.instance_id: timings},
    )
    sent: list[object] = []

    async def record(command: object) -> None:
        sent.append(command)

    setattr(api, "_send", record)  # noqa: B010 - stub the send channel
    return api, sent


async def test_dry_run_matches_applied_allocation_and_sends_nothing() -> None:
    """The preview must be exactly what the applier would do — the whole
    point of the change, replacing the dashboard's divergent estimate."""
    nodes = (NodeId(), NodeId())
    runners = (RunnerId(), RunnerId())
    card = _model_card(storage_bytes=1000, n_layers=10)
    instance = _pipeline_instance(nodes, runners, (5, 5), card)
    memories = (
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
    )
    # Node 0 is four times faster per layer, so layers should move to it.
    timings = {
        nodes[0]: _stage_timing(layers_held=5, compute_ms_per_token=5.0),
        nodes[1]: _stage_timing(layers_held=5, compute_ms_per_token=20.0),
    }

    api, sent = _api_with_instance(instance, nodes, memories, timings)
    preview = await api.rebalance_instance(instance.instance_id, dry_run=True)

    assert preview.dry_run is True
    assert preview.command_id is None
    assert sent == []

    applied = await api.rebalance_instance(instance.instance_id)
    assert applied.node_layers == preview.node_layers
    assert applied.steps == preview.steps
    assert applied.command_id is not None
    assert len(sent) == 1


async def test_dry_run_reports_no_gain_when_already_optimal() -> None:
    nodes = (NodeId(), NodeId())
    runners = (RunnerId(), RunnerId())
    card = _model_card(storage_bytes=1000, n_layers=10)
    instance = _pipeline_instance(nodes, runners, (5, 5), card)
    memories = (
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
    )
    # Identical rates: an even split is already the optimum.
    timings = {
        nodes[0]: _stage_timing(layers_held=5, compute_ms_per_token=10.0),
        nodes[1]: _stage_timing(layers_held=5, compute_ms_per_token=10.0),
    }

    api, sent = _api_with_instance(instance, nodes, memories, timings)
    preview = await api.rebalance_instance(instance.instance_id, dry_run=True)

    assert preview.node_layers == preview.current_layers
    assert preview.steps == 0
    assert preview.projected_compute_speedup == 0.0
    assert "already matches" in preview.message
    assert sent == []


async def test_dry_run_projection_respects_physical_memory_ceiling() -> None:
    """The case the dashboard got wrong: a node reporting plenty of available
    memory is still capped by the three-quarters-of-physical budget."""
    nodes = (NodeId(), NodeId())
    runners = (RunnerId(), RunnerId())
    card = _model_card(storage_bytes=1000, n_layers=10)
    instance = _pipeline_instance(nodes, runners, (5, 5), card)
    # Fast node claims huge availability but has little physical memory, so
    # only the physical ceiling binds.
    memories = (
        _memory_usage(total_bytes=800, available_bytes=100_000),
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
    )
    timings = {
        nodes[0]: _stage_timing(layers_held=5, compute_ms_per_token=5.0),
        nodes[1]: _stage_timing(layers_held=5, compute_ms_per_token=20.0),
    }

    api, _ = _api_with_instance(instance, nodes, memories, timings)
    preview = await api.rebalance_instance(instance.instance_id, dry_run=True)

    expected_cap = pipeline_safe_max_layers(
        model_card=card,
        memory_usage=memories[0],
        current_layer_count=5,
    )
    assert preview.node_layers[nodes[0]] <= expected_cap
    # An uncapped fill would have handed the fast node all ten layers.
    assert preview.node_layers[nodes[0]] < card.n_layers


async def test_dry_run_reports_400_without_stage_timings() -> None:
    nodes = (NodeId(), NodeId())
    runners = (RunnerId(), RunnerId())
    card = _model_card(storage_bytes=1000, n_layers=10)
    instance = _pipeline_instance(nodes, runners, (5, 5), card)
    memories = (
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
    )

    api, _ = _api_with_instance(instance, nodes, memories, {})
    with pytest.raises(HTTPException) as raised:
        _ = await api.rebalance_instance(instance.instance_id, dry_run=True)
    assert raised.value.status_code == 400


async def test_dry_run_allowed_while_migration_in_progress() -> None:
    nodes = (NodeId(), NodeId())
    runners = (RunnerId(), RunnerId())
    card = _model_card(storage_bytes=1000, n_layers=10)
    instance = _pipeline_instance(nodes, runners, (5, 5), card)
    memories = (
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
        _memory_usage(total_bytes=100_000, available_bytes=100_000),
    )
    timings = {
        nodes[0]: _stage_timing(layers_held=5, compute_ms_per_token=5.0),
        nodes[1]: _stage_timing(layers_held=5, compute_ms_per_token=20.0),
    }
    api, sent = _api_with_instance(instance, nodes, memories, timings)
    in_flight = ShiftLayersTask(
        instance_id=instance.instance_id,
        task_status=TaskStatus.Running,
        new_shards={
            runner_id: shard
            for runner_id, shard in instance.shard_assignments.runner_to_shard.items()
            if isinstance(shard, PipelineShardMetadata)
        },
    )
    api.state = api.state.model_copy(update={"tasks": {in_flight.task_id: in_flight}})

    preview = await api.rebalance_instance(instance.instance_id, dry_run=True)
    assert preview.dry_run is True
    assert sent == []

    with pytest.raises(HTTPException) as raised:
        _ = await api.rebalance_instance(instance.instance_id)
    assert raised.value.status_code == 409
