from pathlib import Path

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import IndexedEvent, TestEvent
from exo.shared.types.state import State
from exo.utils.replica_store import ReplicaStore
from exo.utils.state_replica import StateReplica
from exo.utils.state_snapshot import decode_state_snapshot, encode_state_snapshot


def make_session() -> SessionId:
    return SessionId(master_node_id=NodeId("master"), election_clock=7)


def make_indexed_test_events(count: int) -> list[IndexedEvent]:
    return [IndexedEvent(idx=index, event=TestEvent()) for index in range(count)]


def append_and_apply(
    store: ReplicaStore,
    replica: StateReplica,
    indexed_event: IndexedEvent,
) -> None:
    store.append(replica.session, indexed_event)
    replica.apply(indexed_event)


def test_store_recovers_checkpoint_and_complete_journal_tail(
    tmp_path: Path,
) -> None:
    session = make_session()
    store = ReplicaStore(tmp_path / "replica")
    replica = StateReplica(session=session, initial_state=State())
    events = make_indexed_test_events(5)

    for event in events[:3]:
        append_and_apply(store, replica, event)
    store.write_checkpoint(session, replica.state)
    for event in events[3:]:
        append_and_apply(store, replica, event)

    recovered_replica = ReplicaStore(tmp_path / "replica").recover()

    assert recovered_replica is not None
    assert recovered_replica.session == session
    assert recovered_replica.state.model_dump_json() == replica.state.model_dump_json()
    assert recovered_replica.state.last_event_applied_idx == 4


def test_store_ignores_and_truncates_only_torn_final_journal_record(
    tmp_path: Path,
) -> None:
    session = make_session()
    directory = tmp_path / "replica"
    store = ReplicaStore(directory)
    replica = StateReplica(session=session, initial_state=State())
    events = make_indexed_test_events(2)

    append_and_apply(store, replica, events[0])
    complete_record_size = store.journal_path.stat().st_size
    append_and_apply(store, replica, events[1])
    journal_data = store.journal_path.read_bytes()
    store.journal_path.write_bytes(journal_data[:-3])

    recovered_store = ReplicaStore(directory)
    recovered_replica = recovered_store.recover()

    assert recovered_replica is not None
    assert recovered_replica.state.last_event_applied_idx == 0
    assert recovered_store.journal_path.stat().st_size == complete_record_size


def test_snapshot_plus_tail_produces_same_state_as_full_replay() -> None:
    session = make_session()
    events = make_indexed_test_events(8)
    full_replay = StateReplica(session=session, initial_state=State())
    for event in events:
        full_replay.apply(event)

    snapshot_source = StateReplica(session=session, initial_state=State())
    for event in events[:5]:
        snapshot_source.apply(event)
    encoded_snapshot = encode_state_snapshot(snapshot_source.state)
    restored_snapshot = decode_state_snapshot(
        encoded_snapshot,
        max_compressed_size=1_000_000,
        max_uncompressed_size=4_000_000,
    )
    snapshot_and_tail = StateReplica(
        session=session,
        initial_state=restored_snapshot,
    )
    for event in events[5:]:
        snapshot_and_tail.apply(event)

    assert (
        snapshot_and_tail.state.model_dump_json() == full_replay.state.model_dump_json()
    )


def test_compacted_checkpoint_retains_journal_events_after_snapshot_index(
    tmp_path: Path,
) -> None:
    session = make_session()
    store = ReplicaStore(tmp_path / "replica")
    replica = StateReplica(session=session, initial_state=State())
    events = make_indexed_test_events(4)
    for event in events[:3]:
        append_and_apply(store, replica, event)

    checkpoint_state = replica.state
    store.append(session, events[3])
    store.write_compacted_checkpoint(session, checkpoint_state)

    recovered_replica = ReplicaStore(tmp_path / "replica").recover()

    assert recovered_replica is not None
    assert recovered_replica.state.last_event_applied_idx == 3
