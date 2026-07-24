import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import ShardAssignments
from exo.worker.runner.bootstrap import resolve_metal_fast_synchronization


def make_instance(*, uses_jaccl_backend: bool) -> Instance:
    shard_assignments = ShardAssignments(
        model_id=ModelId("test-model"),
        runner_to_shard={},
        node_to_runner={},
    )
    if uses_jaccl_backend:
        return MlxJacclInstance(
            instance_id=InstanceId("jaccl-instance"),
            shard_assignments=shard_assignments,
            jaccl_devices=[],
            jaccl_coordinators={},
        )
    return MlxRingInstance(
        instance_id=InstanceId("ring-instance"),
        shard_assignments=shard_assignments,
        hosts_by_node={},
        ephemeral_port=50000,
    )


@pytest.mark.parametrize(
    ("uses_jaccl_backend", "override", "expected_value"),
    [
        (False, None, "0"),
        (True, None, "1"),
        (False, "true", "1"),
        (True, "false", "0"),
    ],
)
def test_metal_fast_synchronization_defaults_to_jaccl_only(
    *,
    uses_jaccl_backend: bool,
    override: str | None,
    expected_value: str,
) -> None:
    assert (
        resolve_metal_fast_synchronization(
            make_instance(uses_jaccl_backend=uses_jaccl_backend),
            override,
        )
        == expected_value
    )
