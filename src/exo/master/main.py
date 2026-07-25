import os
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import anyio
from loguru import logger

from exo.download.shared_models_dir import load_persisted_shared_models_dir
from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import (
    find_ip_prioritised,
    plan_pipeline_layer_shift_steps,
)
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.environment import get_compatible_environment_value
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    SetSharedModelsDirectory,
    ShiftInstanceLayers,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    InstanceShardAssignmentsUpdated,
    LocalForwarderEvent,
    MasterAnnounced,
    NodeGatheredInfo,
    NodeTimedOut,
    SharedModelsDirectorySet,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    ShiftLayers as ShiftLayersTask,
)
from exo.shared.types.tasks import (
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.replica_store import ReplicaStore
from exo.utils.state_replica import StateReplica
from exo.utils.task_group import TaskGroup

# A generation task that produces no chunk (token, prefill progress, or
# error) for this long is considered wedged: healthy decode emits chunks
# every ~100ms and prefill reports progress per chunk, so even the largest
# prompts tick far more often than this. Pipeline ranks stuck in a
# distributed collective cannot recover on their own; the watchdog fails the
# task, notifies the client, and tears the instance down so it can be
# relaunched cleanly.
GENERATION_STALL_TIMEOUT = timedelta(
    seconds=float(
        get_compatible_environment_value(
            os.environ,
            "EXO_STALL_TIMEOUT_SECONDS",
            "120",
        )
    )
)


def find_stalled_generation_tasks(
    tasks: Mapping[TaskId, Task],
    last_progress: Mapping[TaskId, datetime],
    now: datetime,
    stall_timeout: timedelta,
) -> list[TextGenerationTask]:
    """Running text-generation tasks whose last observed progress is too old.

    Tasks without a recorded progress timestamp are skipped; the caller seeds
    one when it first observes the task, so a fresh master never kills tasks
    it has not watched for a full timeout window.
    """
    stalled: list[TextGenerationTask] = []
    for task_id, task in tasks.items():
        if not isinstance(task, TextGenerationTask):
            continue
        if task.task_status != TaskStatus.Running:
            continue
        observed = last_progress.get(task_id)
        if observed is not None and now - observed > stall_timeout:
            stalled.append(task)
    return stalled


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def recover_pipeline_layer_shift_steps(
    current_shards: Mapping[RunnerId, PipelineShardMetadata],
    task: ShiftLayersTask,
) -> list[dict[RunnerId, PipelineShardMetadata]]:
    """Rebuild a replicated layer-shift plan after master failover.

    Raises ``ValueError`` when persisted progress is missing or inconsistent;
    ``Master._restore_layer_shift_plans`` handles it by leaving that malformed
    migration inactive rather than advancing an unsafe boundary sequence.
    """
    target_layer_counts = task.target_layer_counts
    if target_layer_counts is None:
        raise ValueError("Replicated layer-shift task has no final target")
    if (
        task.task_status == TaskStatus.Complete
        and dict(current_shards) == task.new_shards
    ):
        remaining_steps = plan_pipeline_layer_shift_steps(
            current_shards,
            target_layer_counts,
        )
        expected_remaining_steps = task.total_steps - task.current_step
        if len(remaining_steps) != expected_remaining_steps:
            raise ValueError(
                "Replicated layer-shift progress does not match the committed plan"
            )
        return [dict(current_shards), *remaining_steps]
    steps = plan_pipeline_layer_shift_steps(current_shards, target_layer_counts)
    expected_remaining_steps = task.total_steps - task.current_step + 1
    if len(steps) != expected_remaining_steps:
        raise ValueError(
            "Replicated layer-shift progress does not match the remaining plan"
        )
    if not steps or steps[0] != task.new_shards:
        raise ValueError("Replicated task is not the next replicated step")
    return steps


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
        state_replica: StateReplica | None = None,
        replica_store: ReplicaStore | None = None,
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state_replica = state_replica or StateReplica(
            session=session_id,
            initial_state=State(),
            ready=True,
        )
        self.replica_store = replica_store
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        # Watchdog bookkeeping: last time each generation task showed progress
        # (any chunk reaching the master), seeded when the task is first seen.
        self._task_last_progress: dict[TaskId, datetime] = {}
        # Live layer rebalancing: remaining single-layer boundary shifts per
        # instance, executed one task at a time; the head of each deque is the
        # step currently in flight.
        self._layer_shift_plans: dict[
            InstanceId, deque[dict[RunnerId, PipelineShardMetadata]]
        ] = {}
        self._shift_task_instance: dict[TaskId, InstanceId] = {}
        self._recovered_shift_updates: list[TaskStatusUpdated] = []
        self._recovered_shift_deletions: set[TaskId] = set()
        self._recovered_instance_deletions: set[InstanceId] = set()
        self._restore_layer_shift_plans()

    @property
    def state(self) -> State:
        return self.state_replica.state

    def _restore_layer_shift_plans(self) -> None:
        tasks_by_instance: dict[InstanceId, list[tuple[TaskId, ShiftLayersTask]]] = {}
        for task_id, task in self.state.tasks.items():
            if not isinstance(task, ShiftLayersTask):
                continue
            tasks_by_instance.setdefault(task.instance_id, []).append((task_id, task))

        active_statuses = {TaskStatus.Pending, TaskStatus.Running}
        for instance_id, instance_tasks in tasks_by_instance.items():
            active_plan_ids = {
                task.plan_id or task_id
                for task_id, task in instance_tasks
                if task.task_status in active_statuses
            }
            if len(active_plan_ids) > 1:
                self._quarantine_shift_instance(instance_id, instance_tasks)
                continue
            if active_plan_ids:
                selected_plan_id = next(iter(active_plan_ids))
                selected_tasks = [
                    (task_id, task)
                    for task_id, task in instance_tasks
                    if (task.plan_id or task_id) == selected_plan_id
                ]
            else:
                selected_tasks = [
                    (task_id, task)
                    for task_id, task in instance_tasks
                    if task.task_status == TaskStatus.Complete
                ]
                if not selected_tasks:
                    self._recovered_shift_deletions.update(
                        task_id for task_id, _ in instance_tasks
                    )
                    continue
                completed_plan_ids = {
                    task.plan_id or task_id for task_id, task in selected_tasks
                }
                if len(completed_plan_ids) != 1:
                    self._quarantine_shift_instance(instance_id, instance_tasks)
                    continue
            selected_tasks.sort(key=lambda item: item[1].current_step, reverse=True)
            task_id, task = selected_tasks[0]
            stale_tasks = [
                (stale_task_id, stale_task)
                for stale_task_id, stale_task in instance_tasks
                if stale_task_id != task_id
            ]
            if any(
                stale_task.task_status in active_statuses
                for _, stale_task in stale_tasks
            ):
                self._quarantine_shift_instance(instance_id, instance_tasks)
                continue
            self._recovered_shift_deletions.update(
                stale_task_id for stale_task_id, _ in stale_tasks
            )
            instance = self.state.instances.get(task.instance_id)
            if instance is None:
                self._quarantine_shift_instance(instance_id, instance_tasks)
                continue
            if task.task_status not in active_statuses | {TaskStatus.Complete}:
                self._shift_task_instance[task_id] = instance_id
                self._recovered_shift_updates.append(
                    TaskStatusUpdated(
                        task_id=task_id,
                        task_status=task.task_status,
                    )
                )
                continue
            current_shards = {
                runner_id: shard
                for runner_id, shard in instance.shard_assignments.runner_to_shard.items()
                if isinstance(shard, PipelineShardMetadata)
            }
            if len(current_shards) != len(
                instance.shard_assignments.runner_to_shard
            ):
                self._quarantine_shift_instance(instance_id, instance_tasks)
                continue
            try:
                remaining_steps = recover_pipeline_layer_shift_steps(
                    current_shards,
                    task,
                )
            except ValueError as error:
                logger.warning(
                    f"Cannot restore layer shift for instance {task.instance_id}: "
                    f"{error}"
                )
                self._quarantine_shift_instance(instance_id, instance_tasks)
                continue
            self._layer_shift_plans[task.instance_id] = deque(remaining_steps)
            self._shift_task_instance[task_id] = task.instance_id
            if task.task_status == TaskStatus.Complete:
                self._recovered_shift_updates.append(
                    TaskStatusUpdated(
                        task_id=task_id,
                        task_status=task.task_status,
                    )
                )
            logger.info(
                f"Restored live layer shift for instance {task.instance_id}: "
                f"step {task.current_step}/{task.total_steps}"
            )

    def _quarantine_shift_instance(
        self,
        instance_id: InstanceId,
        instance_tasks: list[tuple[TaskId, ShiftLayersTask]],
    ) -> None:
        logger.error(
            f"Deleting instance {instance_id} because its replicated layer-shift "
            "plan cannot be recovered safely"
        )
        self._recovered_instance_deletions.add(instance_id)
        self._recovered_shift_deletions.update(
            task_id for task_id, _ in instance_tasks
        )
        self._layer_shift_plans.pop(instance_id, None)
        self._shift_task_instance = {
            task_id: mapped_instance_id
            for task_id, mapped_instance_id in self._shift_task_instance.items()
            if mapped_instance_id != instance_id
        }

    async def _wait_for_shift_recovery_reconciliation(self) -> None:
        recovered_task_ids = self._recovered_shift_deletions | {
            update.task_id for update in self._recovered_shift_updates
        }
        with anyio.fail_after(10):
            while any(
                instance_id in self.state.instances
                for instance_id in self._recovered_instance_deletions
            ) or any(task_id in self.state.tasks for task_id in recovered_task_ids):
                await anyio.sleep(0.01)

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                await self.event_sender.send(MasterAnnounced(node_id=self.node_id))
                persisted_shared_dir = load_persisted_shared_models_dir()
                if persisted_shared_dir is not None:
                    await self.event_sender.send(
                        SharedModelsDirectorySet(path=persisted_shared_dir)
                    )
                for instance_id in sorted(self._recovered_instance_deletions):
                    await self.event_sender.send(InstanceDeleted(instance_id=instance_id))
                for task_id in sorted(self._recovered_shift_deletions):
                    await self.event_sender.send(TaskDeleted(task_id=task_id))
                for update in self._recovered_shift_updates:
                    await self._advance_layer_shift(update)
                await self._wait_for_shift_recovery_reconciliation()
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            # set-difference => prefill-only nodes
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            for instance in self.state.instances.values():
                                # NON-prefill-only instances matching the model ID
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                    and instance.instance_id not in prefill_only
                                ):
                                    # count in-flight tasks of that instance
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            # there are no NON-prefill-only instances matching this model ID
                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            if command.pinned_instance_id is not None:
                                if (
                                    command.pinned_instance_id
                                    not in instance_task_counts
                                ):
                                    raise ValueError(
                                        f"Pinned instance {command.pinned_instance_id} is not "
                                        f"serving model {command.task_params.model}"
                                    )
                                decode_instance_id = command.pinned_instance_id
                            else:
                                decode_instance_id = min(
                                    instance_task_counts,
                                    key=lambda instance_id: instance_task_counts[
                                        instance_id
                                    ],
                                )
                            task_id = TaskId()
                            params = command.task_params.model_copy(
                                update={
                                    "prefill_endpoint": _prefill_endpoint_for(
                                        self.state, decode_instance_id
                                    ),
                                }
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=decode_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=params,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ShiftInstanceLayers():
                            generated_events.extend(self._begin_layer_shift(command))
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                                node_identities=self.state.node_identities,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case SetSharedModelsDirectory():
                            generated_events.append(
                                SharedModelsDirectorySet(path=command.path)
                            )
                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_indexed_event(
                                    IndexedEvent(idx=i, event=event)
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                except Exception as e:
                    logger.opt(exception=e).warning("Error in command processor")

    def _begin_layer_shift(self, command: ShiftInstanceLayers) -> list[Event]:
        """Plan a live rebalance and emit the task for its first step.

        Raises ValueError on invalid targets; handled by the command
        processor's catch-all, which logs it (the API validates the same
        preconditions before sending the command).
        """
        instance = self.state.instances.get(command.instance_id)
        if instance is None:
            raise ValueError(f"Instance {command.instance_id} not found")
        if command.instance_id in self._layer_shift_plans:
            raise ValueError(
                f"Instance {command.instance_id} already has a layer shift in progress"
            )
        assignments = instance.shard_assignments
        current_shards: dict[RunnerId, PipelineShardMetadata] = {}
        for runner_id, shard in assignments.runner_to_shard.items():
            if not isinstance(shard, PipelineShardMetadata):
                raise ValueError("Live layer shifts require pipeline sharding")
            current_shards[runner_id] = shard
        target_layer_counts: dict[RunnerId, int] = {}
        for node_id, layer_count in command.node_layers.items():
            runner_id = assignments.node_to_runner.get(node_id)
            if runner_id is None:
                raise ValueError(
                    f"Node {node_id} is not part of instance {command.instance_id}"
                )
            target_layer_counts[runner_id] = layer_count

        steps = plan_pipeline_layer_shift_steps(current_shards, target_layer_counts)
        if not steps:
            logger.info(
                f"Instance {command.instance_id} already matches the requested layout"
            )
            return []
        logger.info(
            f"Beginning live layer shift for instance {command.instance_id}: "
            f"{len(steps)} single-layer steps"
        )
        self._layer_shift_plans[command.instance_id] = deque(steps)
        plan_id = TaskId()
        return [
            self._create_shift_task(
                command.instance_id,
                steps[0],
                target_layer_counts,
                plan_id=plan_id,
                current_step=1,
                total_steps=len(steps),
            )
        ]

    def _create_shift_task(
        self,
        instance_id: InstanceId,
        new_shards: dict[RunnerId, PipelineShardMetadata],
        target_layer_counts: dict[RunnerId, int],
        *,
        plan_id: TaskId,
        current_step: int,
        total_steps: int,
    ) -> TaskCreated:
        task_id = TaskId()
        self._shift_task_instance[task_id] = instance_id
        return TaskCreated(
            task_id=task_id,
            task=ShiftLayersTask(
                task_id=task_id,
                instance_id=instance_id,
                task_status=TaskStatus.Pending,
                new_shards=new_shards,
                plan_id=plan_id,
                target_layer_counts=target_layer_counts,
                current_step=current_step,
                total_steps=total_steps,
            ),
        )

    async def _advance_layer_shift(self, event: TaskStatusUpdated) -> None:
        """Commit a finished shift step and launch the next one.

        Every rank reports completion independently; only the first report
        advances the plan (the task id is untracked afterwards). Any failure
        aborts the remaining steps — the instance keeps serving with the
        boundaries committed so far.
        """
        instance_id = self._shift_task_instance.get(event.task_id)
        if instance_id is None:
            return
        if event.task_status in (TaskStatus.Pending, TaskStatus.Running):
            return
        del self._shift_task_instance[event.task_id]

        follow_up_events: list[Event] = []
        plan = self._layer_shift_plans.get(instance_id)
        instance = self.state.instances.get(instance_id)
        shift_task = self.state.tasks.get(event.task_id)
        if event.task_status == TaskStatus.Complete and plan and instance is not None:
            if not isinstance(shift_task, ShiftLayersTask):
                self._layer_shift_plans.pop(instance_id, None)
                logger.warning(
                    f"Layer shift task {event.task_id} is missing; "
                    "aborting the remaining plan"
                )
                await self.event_sender.send(TaskDeleted(task_id=event.task_id))
                return
            committed_shards = plan.popleft()
            new_assignments = instance.shard_assignments.model_copy(
                update={"runner_to_shard": committed_shards}
            )
            follow_up_events.append(
                InstanceShardAssignmentsUpdated(
                    instance_id=instance_id, shard_assignments=new_assignments
                )
            )
            if plan:
                target_layer_counts = shift_task.target_layer_counts
                if target_layer_counts is None:
                    self._layer_shift_plans.pop(instance_id, None)
                    logger.warning(
                        f"Layer shift task {event.task_id} has no final target; "
                        "aborting the remaining plan"
                    )
                else:
                    follow_up_events.append(
                        self._create_shift_task(
                            instance_id,
                            plan[0],
                            target_layer_counts,
                            plan_id=shift_task.plan_id or shift_task.task_id,
                            current_step=shift_task.current_step + 1,
                            total_steps=shift_task.total_steps,
                        )
                    )
            else:
                del self._layer_shift_plans[instance_id]
                logger.info(f"Live layer shift for instance {instance_id} complete")
        else:
            self._layer_shift_plans.pop(instance_id, None)
            logger.warning(
                f"Live layer shift step for instance {instance_id} ended with "
                f"{event.task_status}; aborting the remaining plan"
            )
        follow_up_events.append(TaskDeleted(task_id=event.task_id))
        for follow_up in follow_up_events:
            await self.event_sender.send(follow_up)

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            # kill broken instances
            connected_node_ids = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break

            # time out dead nodes
            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > timedelta(seconds=30):
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

            await self._fail_stalled_tasks()

            await anyio.sleep(10)

    def _record_task_progress(self, command_id: CommandId) -> None:
        for task_id, task in self.state.tasks.items():
            if isinstance(task, TextGenerationTask) and task.command_id == command_id:
                self._task_last_progress[task_id] = datetime.now(tz=timezone.utc)
                return

    async def _fail_stalled_tasks(self) -> None:
        """Fail generation tasks that stopped making progress and tear down
        their instances.

        Pipeline ranks wedged in a distributed collective can neither finish
        nor cancel, so the stream would hang forever. The instance is deleted
        (workers kill its runners) and the client receives a clean error.
        """
        now = datetime.now(tz=timezone.utc)

        # Seed and prune bookkeeping so a fresh master watches every running
        # task for a full window before judging it, and entries don't leak.
        running_task_ids = {
            task_id
            for task_id, task in self.state.tasks.items()
            if task.task_status == TaskStatus.Running
        }
        for task_id in running_task_ids:
            _ = self._task_last_progress.setdefault(task_id, now)
        for task_id in list(self._task_last_progress):
            if task_id not in running_task_ids:
                del self._task_last_progress[task_id]

        for task in find_stalled_generation_tasks(
            self.state.tasks, self._task_last_progress, now, GENERATION_STALL_TIMEOUT
        ):
            logger.error(
                f"Task {task.task_id} made no progress for "
                f"{GENERATION_STALL_TIMEOUT.total_seconds():.0f}s; failing it and "
                f"restarting instance {task.instance_id}"
            )
            await self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=task.task_params.model,
                        error_message=(
                            "Generation stalled: no progress for "
                            f"{GENERATION_STALL_TIMEOUT.total_seconds():.0f}s. "
                            "The model instance was restarted; please retry."
                        ),
                    ),
                )
            )
            await self.event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Failed)
            )
            await self.event_sender.send(InstanceDeleted(instance_id=task.instance_id))

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")

                    event = event.model_copy(
                        update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
                    )
                    if isinstance(event, NodeGatheredInfo):
                        event = event.model_copy(
                            update={"when": str(datetime.now(tz=timezone.utc))}
                        )

                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    if self.replica_store is not None:
                        self.replica_store.append(self.session_id, indexed)
                    self.state_replica.apply(indexed)

                    if isinstance(event, ChunkGenerated):
                        self._record_task_progress(event.command_id)

                    self._event_log.append(event)
                    await self._send_indexed_event(indexed)

                    if isinstance(event, TaskStatusUpdated):
                        await self._advance_layer_shift(event)
                    elif isinstance(event, InstanceDeleted):
                        self._layer_shift_plans.pop(event.instance_id, None)
                        self._shift_task_instance = {
                            task_id: instance_id
                            for task_id, instance_id in self._shift_task_instance.items()
                            if instance_id != event.instance_id
                        }

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
