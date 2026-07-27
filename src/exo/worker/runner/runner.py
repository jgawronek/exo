import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from anyio import ClosedResourceError, EndOfStream

from exo.shared.constants import ENABLE_DISAGGREGATION
from exo.shared.environment import get_compatible_environment_value
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    ExpertActivationsUpdated,
    RunnerStatusUpdated,
    StageTimingsUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.profiling import StageTiming
from exo.shared.types.tasks import (
    ConnectToGroup,
    GenerationTask,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    ShiftLayers,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.server import (
    PrefillRequest,
    PrefillServer,
)
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger

PREFILL_PICKUP_TIMEOUT_SECONDS = 3
PREFILL_FINISH_TIMEOUT_SECONDS = 300

# How many engine steps to wait between stage-timing and expert-activation
# publications while a generation is running (each step is roughly one decode
# token). Smaller windows update the dashboard's activity display faster but
# sample fewer routed dispatches per report; overridable for tuning.
DECODE_TIMING_PUBLISH_INTERVAL_STEPS = int(
    get_compatible_environment_value(
        os.environ,
        "EXO_TIMING_PUBLISH_STEPS",
        "32",
    )
)


@dataclass
class PrefillTask:
    request: PrefillRequest
    wfile: BinaryIO
    started: threading.Event
    done: threading.Event


class _TaskStreamClosed:
    pass


WorkItem = Task | PrefillTask | _TaskStreamClosed


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        builder: Builder,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | Engine = builder

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            GenerationTask | ShiftLayers,
        ] = {}

        self._prefill_server: PrefillServer | None = None
        self._prefill_server_port: int | None = None
        self._work_queue: queue.Queue[WorkItem] = queue.Queue()
        self._task_reader_thread: threading.Thread | None = None
        self._steps_since_timing_publish = 0

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _start_prefill_server(self) -> int | None:
        if not ENABLE_DISAGGREGATION:
            return None
        if self.device_rank != 0:
            return None
        if self._prefill_server_port is not None:
            return self._prefill_server_port

        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:
            req = PrefillTask(
                request=request,
                wfile=wfile,
                started=threading.Event(),
                done=threading.Event(),
            )
            self._work_queue.put(req)
            if not req.started.wait(timeout=PREFILL_PICKUP_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} not picked up within "
                    f"{PREFILL_PICKUP_TIMEOUT_SECONDS}s — runner busy"
                )
                return False
            if not req.done.wait(timeout=PREFILL_FINISH_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} did not finish within "
                    f"{PREFILL_FINISH_TIMEOUT_SECONDS}s"
                )
            return True

        port = random_ephemeral_port()
        self._prefill_server = PrefillServer(resolve=resolve, host="0.0.0.0", port=port)
        self._prefill_server_port = port
        return self._prefill_server_port

    def _start_task_reader(self) -> None:
        if self._task_reader_thread is not None:
            return

        def loop() -> None:
            try:
                with self.task_receiver:
                    for task in self.task_receiver:
                        self._work_queue.put(task)
            except (EndOfStream, ClosedResourceError):
                pass
            finally:
                self._work_queue.put(_TaskStreamClosed())

        self._task_reader_thread = threading.Thread(target=loop, name="task-reader")
        self._task_reader_thread.start()

    def _serve_prefill(self, req: PrefillTask) -> None:
        req.started.set()
        try:
            assert isinstance(self.generator, Engine)
            self.generator.serve_prefill(req.request, req.wfile)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to serve prefill request {req.request.request_id}"
            )
        finally:
            req.done.set()

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        self._start_task_reader()
        try:
            while True:
                item = self._work_queue.get()
                if isinstance(item, _TaskStreamClosed):
                    break
                if isinstance(item, PrefillTask):
                    self._serve_prefill(item)
                    continue
                if item.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(item.task_id)
                self.handle_first_task(item)
                if isinstance(self.current_status, RunnerShutdown):
                    break
        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.connect(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                isinstance(self.current_status, (RunnerConnected, RunnerIdle))
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                for load_progress in self.generator.load(self.bound_instance):
                    self.update_status(
                        RunnerLoading(
                            layers_loaded=load_progress.layers_loaded,
                            total_layers=load_progress.total,
                        )
                    )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, Engine)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self._start_prefill_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(
                    RunnerReady(prefill_server_port=self._prefill_server_port)
                )
                logger.info("runner ready")

            case TextGeneration() | ImageEdits() | ImageGeneration() | ShiftLayers() if isinstance(
                self.current_status, RunnerReady
            ):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        self.generator.close()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_generation(self, task: GenerationTask | ShiftLayers):
        assert isinstance(self.generator, Engine)
        if isinstance(task, ShiftLayers):
            if not self.generator.submit_shard_update(task):
                logger.warning(
                    "engine does not support live layer shifts; failing task"
                )
                self.send_task_status(task.task_id, TaskStatus.Failed)
                return
            self.active_tasks[task.task_id] = task
            return
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def handle_generation_tasks(self, starting_task: GenerationTask | ShiftLayers):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, Engine)

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_generation(starting_task)

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case CancelledResponse():
                        finished.append(task_id)
                    case FinishedResponse():
                        self._finalize_task(task_id)
                        finished.append(task_id)
                    case other:
                        active_task = self.active_tasks[task_id]
                        assert not isinstance(active_task, ShiftLayers), (
                            "shard updates produce no generation chunks"
                        )
                        self.send_chunk(other, active_task.command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            self._steps_since_timing_publish += 1
            if (
                finished
                or self._steps_since_timing_publish
                >= DECODE_TIMING_PUBLISH_INTERVAL_STEPS
            ):
                self._publish_stage_timing()

            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                continue
            if isinstance(item, _TaskStreamClosed):
                return ExitCode.Shutdown
            if isinstance(item, PrefillTask):
                self._serve_prefill(item)
                continue
            if item.task_id in self.seen:
                logger.warning("repeat task - potential error")
                continue
            self.seen.add(item.task_id)
            match item:
                case TextGeneration() | ImageGeneration() | ImageEdits() | ShiftLayers():
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                case Shutdown():
                    self.shutdown(item)
                    return ExitCode.Shutdown
                case _:
                    raise ValueError(
                        f"Received {item.__class__.__name__} outside of state machine in {self.current_status=}"
                    )

        self.update_status(RunnerReady(prefill_server_port=self._prefill_server_port))
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def _finalize_task(self, task_id: TaskId) -> None:
        """Report completion and absorb side-effects of special tasks.

        A completed layer shift changes which layers this runner holds, so the
        published stage timings must be attributed to the new range.
        """
        finished_task = self.active_tasks.get(task_id)
        if isinstance(finished_task, ShiftLayers):
            new_shard = finished_task.new_shards.get(self.runner_id)
            if new_shard is not None:
                self.shard_metadata = new_shard
        self.send_task_status(task_id, TaskStatus.Complete)

    def send_chunk(
        self,
        chunk: Chunk,
        command_id: CommandId,
    ):
        assert isinstance(self.generator, Engine)
        self.event_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))

    def _publish_stage_timing(self) -> None:
        self._steps_since_timing_publish = 0
        assert isinstance(self.generator, Engine)
        expert_layers = self.generator.poll_expert_activity()
        if expert_layers:
            self.event_sender.send(
                ExpertActivationsUpdated(
                    instance_id=self.instance.instance_id,
                    node_id=self.bound_instance.bound_node_id,
                    layers={
                        str(layer_index): activity
                        for layer_index, activity in expert_layers.items()
                    },
                )
            )
        sample = self.generator.poll_decode_timing()
        if sample is None:
            return
        self.event_sender.send(
            StageTimingsUpdated(
                instance_id=self.instance.instance_id,
                node_id=self.bound_instance.bound_node_id,
                timing=StageTiming(
                    layers_held=self.shard_metadata.end_layer
                    - self.shard_metadata.start_layer,
                    compute_ms_per_token=sample.compute_ms_per_token,
                    communication_ms_per_token=sample.communication_ms_per_token,
                    tokens_measured=sample.tokens_measured,
                ),
            )
        )
