"""A departing node, or a deleted instance, must not leave state behind.

Node IDs are regenerated on every process start (``get_node_zid`` returns
random bytes while persistence is disabled), so a node that merely restarts
rejoins under a new id. Any state map that is not pruned when the old id times
out therefore grows without bound: four machines had accumulated 31
``node_identities`` and 97 ``runners`` before this was caught.
"""

from datetime import datetime, timezone
from typing import cast

from exo.shared.apply import apply_instance_deleted, apply_node_timed_out
from exo.shared.models.model_cards import ModelId
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, NodeTimedOut
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, ShardAssignments
from exo.worker.tests.unittests.conftest import get_pipeline_shard_metadata

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

GONE = NodeId("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
STAYS = NodeId("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")


def _state_with_two_nodes() -> tuple[State, RunnerId, RunnerId, InstanceId]:
    gone_runner = RunnerId()
    stays_runner = RunnerId()
    instance_id = InstanceId()

    shard = get_pipeline_shard_metadata(
        model_id=ModelId("test-org/test-model"), device_rank=0
    )
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=shard.model_card.model_id,
            runner_to_shard={gone_runner: shard, stays_runner: shard},
            node_to_runner={GONE: gone_runner, STAYS: stays_runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )

    state = State(
        instances={instance_id: instance},
        runners={gone_runner: RunnerIdle(), stays_runner: RunnerIdle()},
        node_identities={
            GONE: NodeIdentity(friendly_name="departing"),
            STAYS: NodeIdentity(friendly_name="staying"),
        },
        node_backends={GONE: [Backend.MlxMetal], STAYS: [Backend.MlxMetal]},
        last_seen={GONE: _EPOCH, STAYS: _EPOCH},
    )
    return state, gone_runner, stays_runner, instance_id


def test_node_timeout_prunes_identity_backends_and_runners():
    state, gone_runner, stays_runner, _ = _state_with_two_nodes()

    new_state = apply_node_timed_out(NodeTimedOut(node_id=GONE), state)

    assert GONE not in new_state.node_identities
    assert GONE not in new_state.node_backends
    assert gone_runner not in new_state.runners

    # The surviving node keeps everything.
    assert STAYS in new_state.node_identities
    assert STAYS in new_state.node_backends
    assert stays_runner in new_state.runners


def test_node_timeout_leaves_no_trace_in_any_node_keyed_map():
    """Generic sweep so a newly added NodeId-keyed map cannot be forgotten.

    Every map above was pruned individually; node_identities, node_backends and
    runners were simply left off that list. This fails if that happens again.
    """
    state, _, _, _ = _state_with_two_nodes()

    new_state = apply_node_timed_out(NodeTimedOut(node_id=GONE), state)

    dumped: dict[str, object] = new_state.model_dump()
    leaked = [
        field
        for field, value in dumped.items()
        if isinstance(value, dict) and GONE in cast(dict[str, object], value)
    ]
    assert leaked == [], f"departed node still present in: {leaked}"


def test_instance_deletion_prunes_its_runners():
    state, gone_runner, stays_runner, instance_id = _state_with_two_nodes()

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=instance_id), state)

    assert instance_id not in new_state.instances
    # Both runners belonged to the deleted instance, so neither should survive
    # as an ownerless RunnerShuttingDown record.
    assert gone_runner not in new_state.runners
    assert stays_runner not in new_state.runners


def test_instance_deletion_keeps_runners_of_other_instances():
    state, _, _, instance_id = _state_with_two_nodes()
    unrelated_runner = RunnerId()
    state = state.model_copy(
        update={"runners": {**state.runners, unrelated_runner: RunnerIdle()}}
    )

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=instance_id), state)

    assert unrelated_runner in new_state.runners
