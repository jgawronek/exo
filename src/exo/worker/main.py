import hashlib
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import anyio
from anyio import fail_after, move_on_after, to_thread
from loguru import logger

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.download.shared_models_dir import (
    load_persisted_shared_models_dir,
    persist_shared_models_dir,
    set_shared_models_dir,
    validate_shared_models_directory,
)
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
    ReplicatedEventDelivery,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelId, card_cache
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.events import (
    Event,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeSharedDirectoryStatusUpdated,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerStatus
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.state_replica import StateReplica
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor

# How long to wait for a runner to acknowledge a submitted task before
# treating it as wedged. Acknowledgement normally happens within
# milliseconds, but a runner mid-prefill only drains its queue afterwards,
# so this must exceed the prefill finish timeout (300s). A runner stuck in
# an abandoned distributed collective never acknowledges at all — observed
# after a generation was cancelled mid-prefill — and previously blocked the
# worker's whole plan loop forever.
RUNNER_TASK_ACK_TIMEOUT_SECONDS = 330
RUNNER_CANCEL_TIMEOUT_SECONDS = 10
# How often each worker restates its runners' statuses. Ring formation is
# sequenced on cluster-wide runner status — the last rank only dials once
# every peer reports connecting — so a single dropped status event would
# otherwise deadlock the whole instance permanently. Restating makes that
# state self-correcting instead.
RUNNER_STATUS_REPUBLISH_SECONDS = 5


def diverged_runner_statuses(
    *,
    local_statuses: Mapping[RunnerId, RunnerStatus],
    replicated_statuses: Mapping[RunnerId, RunnerStatus],
) -> dict[RunnerId, RunnerStatus]:
    """Local runner statuses that cluster state does not yet agree with."""
    return {
        runner_id: status
        for runner_id, status in local_statuses.items()
        if replicated_statuses.get(runner_id) != status
    }


# Backstop for waiting on a shutting-down runner's process to exit before its
# instance may create a replacement.
RUNNER_REAP_TIMEOUT_SECONDS = 60


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[ReplicatedEventDelivery],
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
        state_replica: StateReplica | None = None,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port
        self.state_replica = state_replica

        self._state = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        # Runners whose shutdown has been initiated but whose OS process may
        # still be alive. A replacement runner for the same instance must not
        # be created until these are reaped: the ring listener port is fixed
        # per instance, so an early replacement hits EADDRINUSE and dies,
        # which recycles into an endless bind-fail loop.
        self.terminating_runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[Base64ImageHash, Base64Image] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._stopped: anyio.Event = anyio.Event()

    @property
    def state(self) -> State:
        if self.state_replica is not None:
            return self.state_replica.state
        return self._state

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send, api_port=self.api_port)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._reconcile_custom_cards)
                tg.start_soon(self._reconcile_shared_models_dir)
                tg.start_soon(self._republish_runner_statuses)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                await self.event_sender.send(
                    NodeGatheredInfo(
                        node_id=self.node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=info,
                    )
                )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for delivery in events:
                indexed_event = delivery.indexed_event
                if self.state_replica is None:
                    self._state = apply(self._state, event=indexed_event)
                event = indexed_event.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )
                    if (
                        len(self.input_chunk_buffer[cmd_id])
                        == self.input_chunk_counts[cmd_id]
                    ):
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in self.input_chunk_buffer[cmd_id].values():
                            per_image[chunk.image_index].append(chunk)
                        for chunks_for_image in per_image.values():
                            sorted_chunks = sorted(
                                chunks_for_image, key=lambda c: c.chunk_index
                            )
                            img = Base64Image("".join(c.data for c in sorted_chunks))
                            self.image_cache[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img

    async def _reconcile_shared_models_dir(self) -> None:
        """Track the cluster's shared models directory setting.

        Validates the configured path locally, installs it as the preferred
        models directory when usable, reports the outcome to the cluster, and
        persists the setting so it survives restarts (any node can become
        master and re-announce it).
        """
        # Preload the locally persisted value so models on the share resolve
        # before the master re-announces the setting after a restart.
        persisted = load_persisted_shared_models_dir()
        if persisted is not None:
            preload_status = await to_thread.run_sync(
                validate_shared_models_directory, persisted
            )
            if preload_status.valid:
                set_shared_models_dir(Path(persisted).expanduser())

        applied: str | None = persisted
        reported: str | None = None
        while True:
            await anyio.sleep(1)
            target = self.state.shared_models_dir
            if target is not None and target != reported:
                status = await to_thread.run_sync(
                    validate_shared_models_directory, target
                )
                set_shared_models_dir(
                    Path(target).expanduser() if status.valid else None
                )
                await to_thread.run_sync(persist_shared_models_dir, target)
                applied = target
                reported = target
                logger.info(
                    f"Shared models directory '{target}': "
                    f"{'valid' if status.valid else f'invalid ({status.error})'}"
                )
                await self.event_sender.send(
                    NodeSharedDirectoryStatusUpdated(
                        node_id=self.node_id, status=status
                    )
                )
            elif (
                target is None
                and applied is not None
                # Only treat None as an explicit clear once a master has
                # announced itself; before that, state is still recovering.
                and self.state.master_node_id is not None
            ):
                set_shared_models_dir(None)
                await to_thread.run_sync(persist_shared_models_dir, None)
                applied = None
                reported = None
                logger.info("Shared models directory cleared")

    async def _reconcile_custom_cards(self) -> None:
        while True:
            await anyio.sleep(1)
            target = dict(self.state.custom_model_cards)
            for model_id, card in target.items():
                if card_cache.get(model_id) == card:
                    continue
                await card_cache.save(card)

            for card in await card_cache.list_all():
                if card.model_id not in target:
                    await card_cache.pop(card.model_id)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self.image_cache,
                self._instance_backoff,
                self._download_backoff,
                self._terminating_instance_ids(),
            )
            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= EXO_MAX_INSTANCE_RETRIES:
                    logger.warning(
                        f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(instance_id=iid),
                        )
                    )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    await self._create_supervisor(task)
                    self._instance_backoff.record_attempt(task.instance_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id, shard.model_card
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    self.terminating_runners[runner_id] = runner
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()
                        self._tg.start_soon(self._reap_runner, runner_id, runner)
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    cancel_target = self.runners.get(runner_id)
                    if cancel_target is not None:
                        try:
                            with fail_after(RUNNER_CANCEL_TIMEOUT_SECONDS):
                                await cancel_target.cancel_task(cancelled_task_id)
                        except TimeoutError:
                            await self._fail_unresponsive_runner(
                                runner_id,
                                f"Runner {runner_id} did not process a cancel "
                                f"within {RUNNER_CANCEL_TIMEOUT_SECONDS}s; "
                                "treating it as wedged",
                            )
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    resolved_images = [
                        self.image_cache[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)

                    await self._start_runner_task(task)
                case task:
                    await self._start_runner_task(task)

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is None:
            return
        runner_id = instance.shard_assignments.node_to_runner[self.node_id]
        runner = self.runners.get(runner_id)
        if runner is None:
            return
        try:
            with fail_after(RUNNER_TASK_ACK_TIMEOUT_SECONDS):
                await runner.start_task(task)
        except TimeoutError:
            await self._fail_unresponsive_runner(
                runner_id,
                f"Runner {runner_id} did not acknowledge task {task.task_id} "
                f"within {RUNNER_TASK_ACK_TIMEOUT_SECONDS}s; treating it as "
                "wedged",
            )

    async def _fail_unresponsive_runner(self, runner_id: RunnerId, reason: str):
        """Report a wedged runner as failed and tear its process down.

        Publishing RunnerFailed makes every node's planner recycle its runner
        for the instance, and the supervisor shutdown escalates through
        SIGTERM to SIGKILL, so a runner stuck in an abandoned collective
        cannot hold the node (or its memory) hostage.
        """
        runner = self.runners.pop(runner_id, None)
        if runner is None:
            return
        logger.error(reason)
        await self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=reason, diagnostics=[]),
            )
        )
        runner.shutdown()

    async def _republish_runner_statuses(self) -> None:
        """Periodically restate local runner statuses so state self-heals.

        Runner status is published once per transition. Ring formation reads
        those statuses cluster-wide, so one lost event strands an instance in
        a half-connected state forever with nothing to retry it. Re-sending
        the current status is idempotent — apply just overwrites with the
        same value — and bounds any such gap to one interval.
        """
        while True:
            await anyio.sleep(RUNNER_STATUS_REPUBLISH_SECONDS)
            for runner_id, status in diverged_runner_statuses(
                local_statuses={
                    runner_id: runner.status
                    for runner_id, runner in self.runners.items()
                },
                replicated_statuses=self.state.runners,
            ).items():
                await self.event_sender.send(
                    RunnerStatusUpdated(runner_id=runner_id, runner_status=status)
                )

    def _terminating_instance_ids(self) -> frozenset[InstanceId]:
        return frozenset(
            runner.bound_instance.instance.instance_id
            for runner in self.terminating_runners.values()
        )

    async def _reap_runner(self, runner_id: RunnerId, runner: RunnerSupervisor) -> None:
        """Release an instance for a replacement once its process is gone.

        The supervisor's teardown escalates SIGTERM to SIGKILL, so this always
        completes; the timeout is a backstop so a pathological process cannot
        block the instance forever.
        """
        try:
            with move_on_after(RUNNER_REAP_TIMEOUT_SECONDS):
                _ = await runner.runner_process.wait()
        except Exception as exception:
            # Never let reaping a dead runner escape into the worker's task
            # group: that would take the whole worker down and strand the
            # node, which is far worse than an unreaped bookkeeping entry.
            logger.opt(exception=exception).warning(
                f"Failed while awaiting runner {runner_id} termination"
            )
        finally:
            _ = self.terminating_runners.pop(runner_id, None)

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    def _peer_api_port(self, node_id: NodeId) -> int:
        network_info = self.state.node_network.get(node_id)
        return network_info.api_port if network_info is not None else self.api_port

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                peer_api_port = self._peer_api_port(nid)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{peer_api_port}")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{peer_api_port}"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != self._peer_api_port(conn.sink):
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
