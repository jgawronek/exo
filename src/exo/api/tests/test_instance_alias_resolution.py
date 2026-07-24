"""Tests for resolving `model@instance-prefix` aliases to pinned instances."""

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _model_card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(1000),
        n_layers=8,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _instance(instance_id: str, model_id: str) -> Instance:
    runner_id = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(instance_id),
        shard_assignments=ShardAssignments(
            model_id=ModelId(model_id),
            node_to_runner={NodeId(): runner_id},
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=_model_card(model_id),
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=8,
                    n_layers=8,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _make_api_with_instances(instances: dict[InstanceId, Instance]) -> API:
    api = object.__new__(API)
    api.state_replica = None
    api.state = State(instances=instances)
    return api


async def test_alias_resolves_to_matching_instance() -> None:
    instance = _instance("abcd1234-instance", "org/test-model")
    api = _make_api_with_instances({instance.instance_id: instance})

    model_id, pinned = await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
        "org/test-model@abcd1234"
    )

    assert model_id == ModelId("org/test-model")
    assert pinned == instance.instance_id


async def test_plain_model_id_is_not_pinned() -> None:
    instance = _instance("abcd1234-instance", "org/test-model")
    api = _make_api_with_instances({instance.instance_id: instance})

    model_id, pinned = await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
        "org/test-model"
    )

    assert model_id == ModelId("org/test-model")
    assert pinned is None


async def test_alias_with_unknown_prefix_raises_404() -> None:
    instance = _instance("abcd1234-instance", "org/test-model")
    api = _make_api_with_instances({instance.instance_id: instance})

    with pytest.raises(HTTPException) as exc_info:
        await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
            "org/test-model@ffff0000"
        )
    assert exc_info.value.status_code == 404


async def test_alias_with_wrong_model_raises_404() -> None:
    instance = _instance("abcd1234-instance", "org/test-model")
    api = _make_api_with_instances({instance.instance_id: instance})

    with pytest.raises(HTTPException) as exc_info:
        await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
            "org/other-model@abcd1234"
        )
    assert exc_info.value.status_code == 404


async def test_ambiguous_alias_prefix_raises_400() -> None:
    first = _instance("abcd1111-instance", "org/test-model")
    second = _instance("abcd2222-instance", "org/test-model")
    api = _make_api_with_instances(
        {first.instance_id: first, second.instance_id: second}
    )

    with pytest.raises(HTTPException) as exc_info:
        await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
            "org/test-model@abcd"
        )
    assert exc_info.value.status_code == 400


async def test_distinct_prefixes_pin_distinct_instances() -> None:
    first = _instance("abcd1111-instance", "org/test-model")
    second = _instance("abcd2222-instance", "org/test-model")
    api = _make_api_with_instances(
        {first.instance_id: first, second.instance_id: second}
    )

    _, pinned_first = await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
        "org/test-model@abcd1111"
    )
    _, pinned_second = await api._resolve_model_and_instance(  # pyright: ignore[reportPrivateUsage]
        "org/test-model@abcd2222"
    )

    assert pinned_first == first.instance_id
    assert pinned_second == second.instance_id
