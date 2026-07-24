from dataclasses import dataclass, field
from random import random

import anyio
from anyio import BrokenResourceError, ClosedResourceError, WouldBlock, to_thread
from anyio.abc import CancelScope
from loguru import logger

from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    EventId,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
)
from exo.shared.types.state import State
from exo.shared.types.state_snapshots import (
    EncodedStateSnapshot,
    SnapshotRequestId,
    StateSnapshotChunk,
    StateSnapshotManifest,
    StateSnapshotRequest,
    StateSnapshotUnavailable,
)
from exo.utils import channels
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.utils.replica_store import ReplicaStore, ReplicaStoreError
from exo.utils.state_replica import StateReplica
from exo.utils.state_snapshot import (
    SnapshotError,
    SnapshotTargetError,
    SnapshotUnavailableError,
    StateSnapshotAssembler,
    decode_state_snapshot,
    encode_state_snapshot,
    split_snapshot_payload,
)
from exo.utils.task_group import TaskGroup


class EventRouterClosedResourceError(ClosedResourceError):
    pass


class EventRouterBrokenResourceError(BrokenResourceError):
    pass


# Event Router is created and destroyed before consumers of its channels are,
# hence its nice to have tagged errors for event-router channels being closed
#
# so consumers can catch specifically these errors, rather than the generic ones
_ERROR_CFG = channels.ErrorOverride(
    closed_resource_error=EventRouterClosedResourceError,
    broken_resource_error=EventRouterBrokenResourceError,
)


@dataclass(frozen=True)
class SnapshotTransport:
    request_sender: Sender[StateSnapshotRequest]
    request_receiver: Receiver[StateSnapshotRequest]
    manifest_sender: Sender[StateSnapshotManifest]
    manifest_receiver: Receiver[StateSnapshotManifest]
    chunk_sender: Sender[StateSnapshotChunk]
    chunk_receiver: Receiver[StateSnapshotChunk]
    unavailable_sender: Sender[StateSnapshotUnavailable] | None = None
    unavailable_receiver: Receiver[StateSnapshotUnavailable] | None = None
    request_timeout_seconds: float = 2.0
    chunk_size: int = 512 * 1024
    max_compressed_size: int = 64 * 1024 * 1024
    max_uncompressed_size: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ValueError("Snapshot request timeout must be positive")
        if self.chunk_size <= 0:
            raise ValueError("Snapshot chunk size must be positive")
        if self.max_compressed_size <= 0 or self.max_uncompressed_size <= 0:
            raise ValueError("Snapshot size limits must be positive")


@dataclass(frozen=True)
class ReplicatedEventDelivery:
    indexed_event: IndexedEvent
    state_after_event: State


@dataclass
class EventRouter:
    session_id: SessionId
    command_sender: Sender[ForwarderCommand]
    external_inbound: Receiver[GlobalForwarderEvent]
    external_outbound: Sender[LocalForwarderEvent]
    node_id: NodeId
    state_replica: StateReplica
    snapshot_transport: SnapshotTransport
    replica_store: ReplicaStore | None = None
    allow_legacy_fallback: bool = True
    _system_id: SystemId = field(init=False, default_factory=SystemId)
    internal_outbound: list[Sender[ReplicatedEventDelivery]] = field(
        init=False, default_factory=list
    )
    event_buffer: OrderedBuffer[Event] = field(
        init=False, default_factory=OrderedBuffer
    )
    out_for_delivery: dict[EventId, tuple[float, LocalForwarderEvent]] = field(
        init=False, default_factory=dict
    )
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    _nack_cancel_scope: CancelScope | None = field(init=False, default=None)
    _nack_attempts: int = field(init=False, default=0)
    _nack_base_seconds: float = field(init=False, default=0.5)
    _nack_cap_seconds: float = field(init=False, default=10.0)
    _response_drainers_started: bool = field(init=False, default=False)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)
    _catching_up_snapshot_tail: bool = field(init=False, default=False)
    _served_snapshot_requests: set[SnapshotRequestId] = field(
        init=False,
        default_factory=set,
    )
    _cached_snapshot: tuple[int, EncodedStateSnapshot] | None = field(
        init=False,
        default=None,
    )

    async def run(self):
        if self.state_replica.ready:
            self.event_buffer.bootstrap(
                snapshot_index=self.state_replica.state.last_event_applied_idx
            )
        try:
            async with self._tg as tg:
                tg.start_soon(self._run_ext_in)
                tg.start_soon(self._simple_retry)
                tg.start_soon(self._serve_snapshot_requests)
                if self.replica_store is not None:
                    tg.start_soon(self._checkpoint_replica)
                if not self.state_replica.ready:
                    tg.start_soon(self._request_snapshot)
                else:
                    self._start_response_drainers()
        finally:
            self.external_outbound.close()
            self.snapshot_transport.request_receiver.close()
            self.snapshot_transport.manifest_receiver.close()
            self.snapshot_transport.chunk_receiver.close()
            if self.snapshot_transport.unavailable_receiver is not None:
                self.snapshot_transport.unavailable_receiver.close()
            for send in self.internal_outbound:
                send.close()
            self._stopped.set()

    # can make this better in future
    async def _simple_retry(self):
        while True:
            await anyio.sleep(1 + random())
            # list here is a shallow clone for shared mutation
            for e_id, (time, event) in list(self.out_for_delivery.items()):
                if anyio.current_time() > time + 5:
                    self.out_for_delivery[e_id] = (anyio.current_time(), event)
                    await self.external_outbound.send(event)

    def sender(self) -> Sender[Event]:
        send, recv = channel[Event](error_override_config=_ERROR_CFG)
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, SystemId(), recv)
        else:
            self._tg.queue(self._ingest, SystemId(), recv)
        return send

    def receiver(self) -> Receiver[ReplicatedEventDelivery]:
        send, recv = channel[ReplicatedEventDelivery](error_override_config=_ERROR_CFG)
        self.internal_outbound.append(send)
        return recv

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()

    async def _ingest(self, system_id: SystemId, recv: Receiver[Event]):
        idx = 0
        with recv as events:
            async for event in events:
                f_ev = LocalForwarderEvent(
                    origin_idx=idx,
                    origin=system_id,
                    session=self.session_id,
                    event=event,
                )
                idx += 1
                await self.external_outbound.send(f_ev)
                self.out_for_delivery[event.event_id] = (anyio.current_time(), f_ev)

    async def _run_ext_in(self):
        with self.external_inbound as events:
            async for event in events:
                if event.session != self.session_id:
                    continue
                if event.origin != self.session_id.master_node_id:
                    continue

                self.event_buffer.ingest(event.origin_idx, event.event)
                event_id = event.event.event_id
                if event_id in self.out_for_delivery:
                    self.out_for_delivery.pop(event_id)

                if not self.state_replica.ready and not self._catching_up_snapshot_tail:
                    continue

                drained = self.event_buffer.drain_indexed()
                if drained:
                    self._nack_attempts = 0
                    if self._nack_cancel_scope:
                        self._nack_cancel_scope.cancel()

                if not drained and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    # Request the next index.
                    self._tg.start_soon(
                        self._nack_request,
                        self.event_buffer.next_idx_to_release,
                    )
                    continue

                await self._release_drained(
                    drained,
                    mark_replica_ready=not self._catching_up_snapshot_tail,
                )
                if self._catching_up_snapshot_tail and not self.event_buffer.store:
                    self._catching_up_snapshot_tail = False
                    self.state_replica.mark_ready()
                elif self.event_buffer.store and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    self._tg.start_soon(
                        self._nack_request,
                        self.event_buffer.next_idx_to_release,
                    )

    async def _release_drained(
        self,
        drained: list[tuple[int, Event]],
        *,
        mark_replica_ready: bool = True,
    ) -> None:
        for idx, event in drained:
            indexed_event = IndexedEvent(idx=idx, event=event)
            current_index = self.state_replica.state.last_event_applied_idx
            if idx > current_index:
                if self.replica_store is not None:
                    self.replica_store.append(self.session_id, indexed_event)
                self.state_replica.apply(
                    indexed_event,
                    mark_ready=mark_replica_ready,
                )
            elif idx == current_index:
                self.state_replica.apply(
                    indexed_event,
                    mark_ready=mark_replica_ready,
                )
            delivery = ReplicatedEventDelivery(
                indexed_event=indexed_event,
                state_after_event=self.state_replica.state,
            )
            to_clear = set[int]()
            for sender_index, sender in enumerate(self.internal_outbound):
                try:
                    await sender.send(delivery)
                except (ClosedResourceError, BrokenResourceError):
                    to_clear.add(sender_index)
            for sender_index in sorted(to_clear, reverse=True):
                self.internal_outbound.pop(sender_index)

    async def _serve_snapshot_requests(self) -> None:
        transport = self.snapshot_transport
        with transport.request_receiver as requests:
            async for request in requests:
                if request.target_node_id != self.node_id:
                    continue
                if request.session != self.session_id:
                    continue
                if self.node_id != self.session_id.master_node_id:
                    continue
                if request.request_id in self._served_snapshot_requests:
                    continue
                if len(self._served_snapshot_requests) >= 1024:
                    self._served_snapshot_requests.pop()
                self._served_snapshot_requests.add(request.request_id)
                if not self.state_replica.ready:
                    if transport.unavailable_sender is not None:
                        await transport.unavailable_sender.send(
                            StateSnapshotUnavailable(
                                request_id=request.request_id,
                                session=self.session_id,
                                source_node_id=self.node_id,
                                target_node_id=request.source_node_id,
                                reason="not_ready",
                            )
                        )
                    continue

                captured_state = self.state_replica.state
                snapshot_index = captured_state.last_event_applied_idx
                if (
                    self._cached_snapshot is not None
                    and self._cached_snapshot[0] == snapshot_index
                ):
                    encoded_snapshot = self._cached_snapshot[1]
                else:
                    encoded_snapshot = await to_thread.run_sync(
                        encode_state_snapshot,
                        captured_state,
                    )
                    self._cached_snapshot = (snapshot_index, encoded_snapshot)
                if (
                    encoded_snapshot.compressed_size > transport.max_compressed_size
                    or encoded_snapshot.compressed_size > transport.chunk_size * 128
                    or encoded_snapshot.uncompressed_size
                    > transport.max_uncompressed_size
                ):
                    if transport.unavailable_sender is not None:
                        await transport.unavailable_sender.send(
                            StateSnapshotUnavailable(
                                request_id=request.request_id,
                                session=self.session_id,
                                source_node_id=self.node_id,
                                target_node_id=request.source_node_id,
                                reason="snapshot_too_large",
                            )
                        )
                    continue
                manifest, chunks = split_snapshot_payload(
                    encoded_snapshot.payload,
                    request_id=request.request_id,
                    session=self.session_id,
                    source_node_id=self.node_id,
                    target_node_id=request.source_node_id,
                    snapshot_index=snapshot_index,
                    chunk_size=transport.chunk_size,
                    uncompressed_size=encoded_snapshot.uncompressed_size,
                )
                await transport.manifest_sender.send(manifest)
                for chunk in chunks:
                    await transport.chunk_sender.send(chunk)

    async def _request_snapshot(self) -> None:
        transport = self.snapshot_transport
        request = StateSnapshotRequest(
            request_id=SnapshotRequestId(),
            session=self.session_id,
            source_node_id=self.node_id,
            target_node_id=self.session_id.master_node_id,
        )
        await transport.request_sender.send(request)
        assembler = StateSnapshotAssembler(
            request_id=request.request_id,
            session=self.session_id,
            local_node_id=self.node_id,
            expected_source_node_id=self.session_id.master_node_id,
            max_compressed_size=transport.max_compressed_size,
        )

        try:
            with anyio.fail_after(transport.request_timeout_seconds):
                manifest = await self._receive_snapshot_manifest(assembler)
                while True:
                    chunk = await transport.chunk_receiver.receive()
                    try:
                        assembler.accept_chunk(chunk)
                    except SnapshotTargetError:
                        continue
                    if assembler.received_chunk_count == manifest.chunk_count:
                        break

            compressed_payload = assembler.assemble()
            encoded_snapshot = EncodedStateSnapshot(
                payload=compressed_payload,
                payload_sha256=manifest.payload_sha256,
                compressed_size=manifest.compressed_size,
                uncompressed_size=manifest.uncompressed_size,
            )
            restored_state = await to_thread.run_sync(
                lambda: decode_state_snapshot(
                    encoded_snapshot,
                    max_compressed_size=transport.max_compressed_size,
                    max_uncompressed_size=transport.max_uncompressed_size,
                )
            )
        except (SnapshotError, TimeoutError):
            if self.allow_legacy_fallback:
                await self._fallback_to_event_log()
            else:
                await anyio.sleep(0.5)
                self._tg.start_soon(self._request_snapshot)
            return

        self.state_replica.prepare_snapshot(self.session_id, restored_state)
        if self.replica_store is not None:
            try:
                await to_thread.run_sync(
                    self.replica_store.write_compacted_checkpoint,
                    self.session_id,
                    restored_state,
                )
            except (OSError, ReplicaStoreError) as exception:
                logger.opt(exception=exception).warning(
                    "Failed to persist installed state snapshot"
                )
        self.event_buffer.bootstrap(
            snapshot_index=restored_state.last_event_applied_idx
        )
        self._catching_up_snapshot_tail = True
        while drained := self.event_buffer.drain_indexed():
            await self._release_drained(
                drained,
                mark_replica_ready=False,
            )
        if not self.event_buffer.store:
            self._catching_up_snapshot_tail = False
            self.state_replica.mark_ready()
        else:
            self._tg.start_soon(
                self._nack_request,
                self.event_buffer.next_idx_to_release,
            )
        self._start_response_drainers()

    async def _receive_snapshot_manifest(
        self,
        assembler: StateSnapshotAssembler,
    ) -> StateSnapshotManifest:
        while True:
            unavailable_receiver = self.snapshot_transport.unavailable_receiver
            if unavailable_receiver is not None:
                try:
                    unavailable = unavailable_receiver.receive_nowait()
                except WouldBlock:
                    pass
                else:
                    if (
                        unavailable.request_id == assembler.request_id
                        and unavailable.session == self.session_id
                        and unavailable.source_node_id == self.session_id.master_node_id
                        and unavailable.target_node_id == self.node_id
                    ):
                        raise SnapshotUnavailableError(
                            f"Master cannot provide snapshot: {unavailable.reason}"
                        )
                    continue
            manifest: StateSnapshotManifest | None = None
            with anyio.move_on_after(0.05):
                manifest = await self.snapshot_transport.manifest_receiver.receive()
            if manifest is None:
                continue
            try:
                assembler.accept_manifest(manifest)
            except SnapshotTargetError:
                continue
            return manifest

    async def _fallback_to_event_log(self) -> None:
        self.state_replica.prepare_snapshot(
            self.session_id,
            self.state_replica.state,
        )
        self.event_buffer.bootstrap(
            snapshot_index=self.state_replica.state.last_event_applied_idx
        )
        self.state_replica.mark_ready()
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=RequestEventLog(
                    since_idx=self.event_buffer.next_idx_to_release,
                ),
            )
        )
        self._start_response_drainers()

    async def _checkpoint_replica(self) -> None:
        replica_store = self.replica_store
        assert replica_store is not None
        while True:
            await anyio.sleep(5)
            if self.state_replica.ready:
                captured_session = self.state_replica.session
                captured_state = self.state_replica.state
                try:
                    await to_thread.run_sync(
                        replica_store.write_compacted_checkpoint,
                        captured_session,
                        captured_state,
                    )
                except (OSError, ReplicaStoreError) as exception:
                    logger.opt(exception=exception).warning(
                        "Failed to write periodic state-replica checkpoint"
                    )

    def _start_response_drainers(self) -> None:
        if self._response_drainers_started:
            return
        self._response_drainers_started = True
        self._tg.start_soon(
            self._discard_snapshot_manifests,
            self.snapshot_transport.manifest_receiver,
        )
        self._tg.start_soon(
            self._discard_snapshot_chunks,
            self.snapshot_transport.chunk_receiver,
        )
        if self.snapshot_transport.unavailable_receiver is not None:
            self._tg.start_soon(
                self._discard_snapshot_unavailable,
                self.snapshot_transport.unavailable_receiver,
            )

    async def _discard_snapshot_manifests(
        self,
        receiver: Receiver[StateSnapshotManifest],
    ) -> None:
        with receiver:
            async for _manifest in receiver:
                pass

    async def _discard_snapshot_chunks(
        self,
        receiver: Receiver[StateSnapshotChunk],
    ) -> None:
        with receiver:
            async for _chunk in receiver:
                pass

    async def _discard_snapshot_unavailable(
        self,
        receiver: Receiver[StateSnapshotUnavailable],
    ) -> None:
        with receiver:
            async for _unavailable in receiver:
                pass

    async def _nack_request(self, since_idx: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self._nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self._nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    f"Nack attempt {self._nack_attempts}: Requesting Event Log from {since_idx}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=RequestEventLog(since_idx=since_idx),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None
