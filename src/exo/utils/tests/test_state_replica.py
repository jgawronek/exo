import pytest

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import IndexedEvent, TestEvent
from exo.shared.types.state import State
from exo.utils.state_replica import ReplicaSequenceError, StateReplica


def make_session(master_node_id: str, election_clock: int) -> SessionId:
    return SessionId(
        master_node_id=NodeId(master_node_id),
        election_clock=election_clock,
    )


def make_indexed_test_event(index: int) -> IndexedEvent:
    return IndexedEvent(idx=index, event=TestEvent())


def test_replica_applies_events_in_strict_order() -> None:
    replica = StateReplica(
        session=make_session("first-master", 1),
        initial_state=State(),
    )

    first_event = make_indexed_test_event(0)
    second_event = make_indexed_test_event(1)

    assert replica.apply(first_event).last_event_applied_idx == 0
    assert replica.apply(second_event).last_event_applied_idx == 1
    assert replica.state.last_event_applied_idx == 1


def test_replica_rejects_event_that_skips_an_index() -> None:
    replica = StateReplica(
        session=make_session("first-master", 1),
        initial_state=State(),
    )

    with pytest.raises(ReplicaSequenceError, match="expected index 0"):
        replica.apply(make_indexed_test_event(1))


def test_replica_treats_identical_duplicate_as_idempotent() -> None:
    replica = StateReplica(
        session=make_session("first-master", 1),
        initial_state=State(),
    )
    event = make_indexed_test_event(0)

    state_after_first_apply = replica.apply(event)
    state_after_duplicate = replica.apply(event)

    assert state_after_duplicate is state_after_first_apply
    assert replica.state.last_event_applied_idx == 0


def test_replica_rejects_conflicting_duplicate_index() -> None:
    replica = StateReplica(
        session=make_session("first-master", 1),
        initial_state=State(),
    )
    replica.apply(make_indexed_test_event(0))

    with pytest.raises(ReplicaSequenceError, match="conflicting event"):
        replica.apply(make_indexed_test_event(0))


def test_session_rebase_retains_state_and_resets_event_cursor() -> None:
    old_session = make_session("first-master", 1)
    new_session = make_session("promoted-follower", 2)
    replica = StateReplica(session=old_session, initial_state=State())
    replica.apply(make_indexed_test_event(0))
    replica.apply(make_indexed_test_event(1))

    state_before_rebase = replica.state
    replica.rebase_session(new_session)

    assert replica.session == new_session
    assert replica.state == state_before_rebase.model_copy(
        update={"last_event_applied_idx": -1}
    )
    assert replica.apply(make_indexed_test_event(0)).last_event_applied_idx == 0
