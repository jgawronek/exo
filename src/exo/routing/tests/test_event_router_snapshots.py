from collections.abc import Iterable
from hashlib import sha256

import anyio
import pytest
from anyio import WouldBlock

from exo.routing.event_router import (
    EventRouter,
    ReplicatedEventDelivery,
    SnapshotTransport,
)
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    TestEvent,
)
from exo.shared.types.state import State
from exo.shared.types.state_snapshots import (
    EncodedStateSnapshot,
    SnapshotRequestId,
    StateSnapshotChunk,
    StateSnapshotManifest,
    StateSnapshotRequest,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.state_replica import StateReplica
from exo.utils.state_snapshot import (
    decode_state_snapshot,
    encode_state_snapshot,
    split_snapshot_payload,
)

_CHANNEL_CAPACITY = 16
_MAX_COMPRESSED_SIZE = 1_000_000
_MAX_UNCOMPRESSED_SIZE = 4_000_000


def make_session(
    *,
    master_node_id: NodeId | None = None,
    election_clock: int = 7,
) -> SessionId:
    return SessionId(
        master_node_id=master_node_id or NodeId("master"),
        election_clock=election_clock,
    )


def make_snapshot_transport(
    *,
    request_timeout_seconds: float = 0.05,
) -> tuple[
    SnapshotTransport,
    Sender[StateSnapshotRequest],
    Receiver[StateSnapshotRequest],
    Sender[StateSnapshotManifest],
    Receiver[StateSnapshotManifest],
    Sender[StateSnapshotChunk],
    Receiver[StateSnapshotChunk],
]:
    request_outbound, emitted_requests = channel[StateSnapshotRequest](
        _CHANNEL_CAPACITY
    )
    incoming_requests, request_inbound = channel[StateSnapshotRequest](
        _CHANNEL_CAPACITY
    )
    manifest_outbound, emitted_manifests = channel[StateSnapshotManifest](
        _CHANNEL_CAPACITY
    )
    incoming_manifests, manifest_inbound = channel[StateSnapshotManifest](
        _CHANNEL_CAPACITY
    )
    chunk_outbound, emitted_chunks = channel[StateSnapshotChunk](_CHANNEL_CAPACITY)
    incoming_chunks, chunk_inbound = channel[StateSnapshotChunk](_CHANNEL_CAPACITY)
    transport = SnapshotTransport(
        request_sender=request_outbound,
        request_receiver=request_inbound,
        manifest_sender=manifest_outbound,
        manifest_receiver=manifest_inbound,
        chunk_sender=chunk_outbound,
        chunk_receiver=chunk_inbound,
        request_timeout_seconds=request_timeout_seconds,
        chunk_size=16,
        max_compressed_size=_MAX_COMPRESSED_SIZE,
        max_uncompressed_size=_MAX_UNCOMPRESSED_SIZE,
    )
    return (
        transport,
        incoming_requests,
        emitted_requests,
        incoming_manifests,
        emitted_manifests,
        incoming_chunks,
        emitted_chunks,
    )


def make_event_router(
    *,
    node_id: NodeId,
    session: SessionId,
    replica: StateReplica,
    snapshot_transport: SnapshotTransport,
) -> tuple[
    EventRouter,
    Sender[GlobalForwarderEvent],
    Receiver[LocalForwarderEvent],
    Receiver[ForwarderCommand],
]:
    command_sender, emitted_commands = channel[ForwarderCommand](_CHANNEL_CAPACITY)
    incoming_global_events, global_event_receiver = channel[GlobalForwarderEvent](
        _CHANNEL_CAPACITY
    )
    local_event_sender, emitted_local_events = channel[LocalForwarderEvent](
        _CHANNEL_CAPACITY
    )
    router = EventRouter(
        session_id=session,
        node_id=node_id,
        state_replica=replica,
        snapshot_transport=snapshot_transport,
        command_sender=command_sender,
        external_inbound=global_event_receiver,
        external_outbound=local_event_sender,
    )
    return router, incoming_global_events, emitted_local_events, emitted_commands


def make_snapshot_response(
    *,
    request: StateSnapshotRequest,
    state: State,
    chunk_size: int = 16,
) -> tuple[StateSnapshotManifest, tuple[StateSnapshotChunk, ...]]:
    encoded_snapshot = encode_state_snapshot(state)
    return split_snapshot_payload(
        encoded_snapshot.payload,
        request_id=request.request_id,
        session=request.session,
        source_node_id=request.target_node_id,
        target_node_id=request.source_node_id,
        snapshot_index=state.last_event_applied_idx,
        uncompressed_size=encoded_snapshot.uncompressed_size,
        chunk_size=chunk_size,
    )


async def send_snapshot_response(
    *,
    manifest_sender: Sender[StateSnapshotManifest],
    chunk_sender: Sender[StateSnapshotChunk],
    manifest: StateSnapshotManifest,
    chunks: Iterable[StateSnapshotChunk],
) -> None:
    await manifest_sender.send(manifest)
    for chunk in chunks:
        await chunk_sender.send(chunk)


@pytest.mark.asyncio
async def test_not_ready_follower_installs_snapshot_then_applies_buffered_tail() -> (
    None
):
    session = make_session()
    follower_node_id = NodeId("follower")
    snapshot_height = 4
    snapshot_state = State(last_event_applied_idx=snapshot_height)
    replica = StateReplica(session=session, initial_state=State())
    (
        snapshot_transport,
        _incoming_requests,
        emitted_requests,
        incoming_manifests,
        _emitted_manifests,
        incoming_chunks,
        _emitted_chunks,
    ) = make_snapshot_transport()
    router, incoming_events, _emitted_local_events, _emitted_commands = (
        make_event_router(
            node_id=follower_node_id,
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )
    emitted_events = router.receiver()
    buffered_tail = TestEvent()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        with anyio.fail_after(0.1):
            request = await emitted_requests.receive()

        assert request.session == session
        assert request.source_node_id == follower_node_id
        assert request.target_node_id == session.master_node_id

        await incoming_events.send(
            GlobalForwarderEvent(
                origin_idx=snapshot_height + 1,
                origin=session.master_node_id,
                session=session,
                event=buffered_tail,
            )
        )
        await anyio.sleep(0)
        with pytest.raises(WouldBlock):
            emitted_events.receive_nowait()

        manifest, chunks = make_snapshot_response(
            request=request,
            state=snapshot_state,
        )
        await send_snapshot_response(
            manifest_sender=incoming_manifests,
            chunk_sender=incoming_chunks,
            manifest=manifest,
            chunks=reversed(chunks),
        )

        with anyio.fail_after(0.1):
            released_tail = await emitted_events.receive()
        assert released_tail == ReplicatedEventDelivery(
            indexed_event=IndexedEvent(
                idx=snapshot_height + 1,
                event=buffered_tail,
            ),
            state_after_event=replica.state,
        )
        assert replica.state.last_event_applied_idx == snapshot_height + 1
        assert router.event_buffer.next_idx_to_release == snapshot_height + 2
        assert not router.event_buffer.store
        router.shutdown()


@pytest.mark.asyncio
async def test_snapshot_tail_gap_requests_replay_before_replica_becomes_ready() -> None:
    session = make_session()
    follower_node_id = NodeId("follower")
    snapshot_height = 4
    replica = StateReplica(session=session, initial_state=State())
    (
        snapshot_transport,
        _incoming_requests,
        emitted_requests,
        incoming_manifests,
        _emitted_manifests,
        incoming_chunks,
        _emitted_chunks,
    ) = make_snapshot_transport()
    router, incoming_events, _emitted_local_events, emitted_commands = (
        make_event_router(
            node_id=follower_node_id,
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )
    emitted_events = router.receiver()
    missing_event = TestEvent()
    buffered_event = TestEvent()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        with anyio.fail_after(0.1):
            request = await emitted_requests.receive()
        await incoming_events.send(
            GlobalForwarderEvent(
                origin_idx=snapshot_height + 2,
                origin=session.master_node_id,
                session=session,
                event=buffered_event,
            )
        )
        manifest, chunks = make_snapshot_response(
            request=request,
            state=State(last_event_applied_idx=snapshot_height),
        )
        await send_snapshot_response(
            manifest_sender=incoming_manifests,
            chunk_sender=incoming_chunks,
            manifest=manifest,
            chunks=chunks,
        )

        with anyio.fail_after(1):
            replay_request = await emitted_commands.receive()
        assert isinstance(replay_request.command, RequestEventLog)
        assert replay_request.command.since_idx == snapshot_height + 1
        assert not replica.ready

        await incoming_events.send(
            GlobalForwarderEvent(
                origin_idx=snapshot_height + 1,
                origin=session.master_node_id,
                session=session,
                event=missing_event,
            )
        )
        with anyio.fail_after(0.1):
            first_delivery = await emitted_events.receive()
            assert first_delivery.indexed_event == IndexedEvent(
                idx=snapshot_height + 1, event=missing_event
            )
            assert first_delivery.state_after_event.last_event_applied_idx == (
                snapshot_height + 1
            )
            second_delivery = await emitted_events.receive()
            assert second_delivery.indexed_event == IndexedEvent(
                idx=snapshot_height + 2, event=buffered_event
            )
            assert second_delivery.state_after_event is replica.state
        assert replica.ready
        router.shutdown()


@pytest.mark.asyncio
async def test_ready_master_responds_only_to_targeted_current_session_request() -> None:
    master_node_id = NodeId("master")
    session = make_session(master_node_id=master_node_id)
    replica = StateReplica(
        session=session,
        initial_state=State(last_event_applied_idx=8),
    )
    (
        snapshot_transport,
        incoming_requests,
        _emitted_requests,
        _incoming_manifests,
        emitted_manifests,
        _incoming_chunks,
        emitted_chunks,
    ) = make_snapshot_transport()
    router, _incoming_events, _emitted_local_events, _emitted_commands = (
        make_event_router(
            node_id=master_node_id,
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )
    follower_node_id = NodeId("follower")
    wrong_target_request = StateSnapshotRequest(
        request_id=SnapshotRequestId(),
        session=session,
        source_node_id=follower_node_id,
        target_node_id=NodeId("another-master"),
    )
    stale_session_request = StateSnapshotRequest(
        request_id=SnapshotRequestId(),
        session=make_session(election_clock=session.election_clock - 1),
        source_node_id=follower_node_id,
        target_node_id=master_node_id,
    )
    valid_request = StateSnapshotRequest(
        request_id=SnapshotRequestId(),
        session=session,
        source_node_id=follower_node_id,
        target_node_id=master_node_id,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await incoming_requests.send(wrong_target_request)
        await incoming_requests.send(stale_session_request)
        await incoming_requests.send(valid_request)

        with anyio.fail_after(0.1):
            manifest = await emitted_manifests.receive()
            chunks = [
                await emitted_chunks.receive() for _ in range(manifest.chunk_count)
            ]

        assert manifest.request_id == valid_request.request_id
        assert manifest.session == session
        assert manifest.source_node_id == master_node_id
        assert manifest.target_node_id == follower_node_id
        assert all(chunk.request_id == valid_request.request_id for chunk in chunks)
        assert all(chunk.target_node_id == follower_node_id for chunk in chunks)
        with pytest.raises(WouldBlock):
            emitted_manifests.receive_nowait()
        with pytest.raises(WouldBlock):
            emitted_chunks.receive_nowait()

        encoded_snapshot = EncodedStateSnapshot(
            payload=b"".join(chunk.payload for chunk in chunks),
            payload_sha256=manifest.payload_sha256,
            compressed_size=manifest.compressed_size,
            uncompressed_size=manifest.uncompressed_size,
        )
        decoded_state = decode_state_snapshot(
            encoded_snapshot,
            max_compressed_size=_MAX_COMPRESSED_SIZE,
            max_uncompressed_size=_MAX_UNCOMPRESSED_SIZE,
        )
        assert decoded_state.model_dump_json() == replica.state.model_dump_json()
        router.shutdown()


@pytest.mark.asyncio
async def test_snapshot_timeout_requests_legacy_event_log_from_zero() -> None:
    session = make_session()
    replica = StateReplica(session=session, initial_state=State())
    (
        snapshot_transport,
        _incoming_requests,
        emitted_requests,
        _incoming_manifests,
        _emitted_manifests,
        _incoming_chunks,
        _emitted_chunks,
    ) = make_snapshot_transport(request_timeout_seconds=0.01)
    router, _incoming_events, _emitted_local_events, emitted_commands = (
        make_event_router(
            node_id=NodeId("follower"),
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        with anyio.fail_after(0.1):
            await emitted_requests.receive()
            fallback = await emitted_commands.receive()
        assert isinstance(fallback.command, RequestEventLog)
        assert fallback.command.since_idx == 0
        router.shutdown()


@pytest.mark.asyncio
async def test_invalid_snapshot_response_requests_legacy_event_log_from_zero() -> None:
    session = make_session()
    replica = StateReplica(session=session, initial_state=State())
    (
        snapshot_transport,
        _incoming_requests,
        emitted_requests,
        incoming_manifests,
        _emitted_manifests,
        incoming_chunks,
        _emitted_chunks,
    ) = make_snapshot_transport()
    router, _incoming_events, _emitted_local_events, emitted_commands = (
        make_event_router(
            node_id=NodeId("follower"),
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        with anyio.fail_after(0.1):
            request = await emitted_requests.receive()
        manifest, chunks = make_snapshot_response(request=request, state=State())
        invalid_chunk = chunks[0].model_copy(
            update={"payload_sha256": sha256(b"different payload").hexdigest()}
        )
        await incoming_manifests.send(manifest)
        await incoming_chunks.send(invalid_chunk)

        with anyio.fail_after(0.1):
            fallback = await emitted_commands.receive()
        assert isinstance(fallback.command, RequestEventLog)
        assert fallback.command.since_idx == 0
        router.shutdown()


@pytest.mark.asyncio
async def test_follower_ignores_wrong_target_and_session_snapshot_responses() -> None:
    session = make_session()
    follower_node_id = NodeId("follower")
    replica = StateReplica(session=session, initial_state=State())
    (
        snapshot_transport,
        _incoming_requests,
        emitted_requests,
        incoming_manifests,
        _emitted_manifests,
        incoming_chunks,
        _emitted_chunks,
    ) = make_snapshot_transport()
    router, _incoming_events, _emitted_local_events, emitted_commands = (
        make_event_router(
            node_id=follower_node_id,
            session=session,
            replica=replica,
            snapshot_transport=snapshot_transport,
        )
    )
    emitted_events = router.receiver()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        with anyio.fail_after(0.1):
            request = await emitted_requests.receive()
        manifest, chunks = make_snapshot_response(
            request=request,
            state=State(last_event_applied_idx=3),
        )
        wrong_target_manifest = manifest.model_copy(
            update={"target_node_id": NodeId("other-follower")}
        )
        wrong_session_chunk = chunks[0].model_copy(
            update={"session": make_session(election_clock=session.election_clock + 1)}
        )

        await incoming_manifests.send(wrong_target_manifest)
        await incoming_chunks.send(wrong_session_chunk)
        await anyio.sleep(0)
        await anyio.sleep(0)

        assert replica.state.last_event_applied_idx == -1
        with pytest.raises(WouldBlock):
            emitted_events.receive_nowait()
        with pytest.raises(WouldBlock):
            emitted_commands.receive_nowait()
        router.shutdown()
