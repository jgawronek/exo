from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from typing import BinaryIO

from exo.shared.types.chunks import Chunk
from exo.shared.types.profiling import DecodeTimingSample, LayerExpertActivity
from exo.shared.types.tasks import CANCEL_ALL_TASKS, GenerationTask, ShiftLayers, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.worker.disaggregated.server import PrefillRequest


class Engine(ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: GenerationTask,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None: ...

    def poll_decode_timing(self) -> DecodeTimingSample | None:
        """Drain per-stage decode timing averaged since the last poll.

        Engines that do not measure pipeline stage timings (image engines,
        tensor sharding, single-node instances) return None.
        """
        return None

    def poll_expert_activity(self) -> Mapping[int, LayerExpertActivity]:
        """Drain per-layer MoE expert activation counts since the last poll.

        Keys are absolute decoder layer indices. Engines without MoE
        instrumentation (image engines, dense models) return an empty map.
        """
        return {}

    def submit_shard_update(self, task: ShiftLayers) -> bool:
        """Queue a live pipeline layer-boundary shift.

        The engine applies the shift once every rank has agreed on it and the
        current batch has drained, then reports it through ``step()`` as a
        ``FinishedResponse`` for the task. Returns False when the engine
        cannot re-shard (image engines, tensor sharding, single-node
        instances); the runner fails the task in that case.
        """
        return False


class Builder(ABC):
    @abstractmethod
    def connect(self, bound_instance: BoundInstance) -> None: ...

    @abstractmethod
    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]: ...

    @abstractmethod
    def build(self) -> Engine: ...

    @abstractmethod
    def close(self) -> None: ...
