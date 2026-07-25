from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.types.common import CommandId, Id
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    pass


CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")


class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"
    Cancelled = "Cancelled"


class BaseTask(TaggedModel):
    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata


class LoadModel(BaseTask):  # emitted by Worker
    pass


class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


class StartWarmup(BaseTask):  # emitted by Worker
    pass


class TextGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: TextGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class CancelTask(BaseTask):
    cancelled_task_id: TaskId
    runner_id: RunnerId


class ImageGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    command_id: CommandId
    task_params: ImageEditsTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


class ShiftLayers(BaseTask):  # emitted by Master
    """Move pipeline layer boundaries of a running instance by one step.

    Every rank of the instance receives this task and applies its own entry
    from ``new_shards`` in lockstep (the engine's task-agreement collective
    guarantees identical ordering across ranks). Ranks whose boundaries are
    unchanged apply a no-op but still participate in the agreement.
    """

    new_shards: dict[RunnerId, PipelineShardMetadata]
    plan_id: TaskId | None = None
    target_layer_counts: dict[RunnerId, int] | None = None
    current_step: int = Field(default=1, ge=1)
    total_steps: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.current_step > self.total_steps:
            raise ValueError("Current layer-shift step cannot exceed total steps")
        if not self.new_shards:
            raise ValueError("Layer-shift task must contain at least one runner")
        ranked_shards = sorted(
            self.new_shards.values(),
            key=lambda shard: shard.device_rank,
        )
        reference_shard = ranked_shards[0]
        total_layers = reference_shard.n_layers
        if [shard.device_rank for shard in ranked_shards] != list(
            range(len(ranked_shards))
        ):
            raise ValueError("Layer-shift shard ranks must be contiguous")
        if any(
            shard.n_layers != total_layers
            or shard.world_size != len(ranked_shards)
            or shard.model_card.model_id != reference_shard.model_card.model_id
            for shard in ranked_shards
        ):
            raise ValueError("Layer-shift shards must describe the same complete model")
        previous_end = 0
        for shard in ranked_shards:
            if shard.start_layer != previous_end or shard.end_layer <= shard.start_layer:
                raise ValueError(
                    "Layer-shift shard boundaries must be contiguous and nonempty"
                )
            previous_end = shard.end_layer
        if previous_end != total_layers:
            raise ValueError("Layer-shift shards must cover the complete model")
        target_layer_counts = self.target_layer_counts
        if target_layer_counts is None:
            return self
        if set(target_layer_counts) != set(self.new_shards):
            raise ValueError("Layer-shift target must cover exactly the task's runners")
        if any(layer_count < 1 for layer_count in target_layer_counts.values()):
            raise ValueError("Every target pipeline rank must keep at least one layer")
        if sum(target_layer_counts.values()) != total_layers:
            raise ValueError(
                "Layer-shift target counts must sum to the model's layer count"
            )
        return self


Task = (
    CreateRunner
    | DownloadModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | TextGeneration
    | CancelTask
    | ImageGeneration
    | ImageEdits
    | Shutdown
    | ShiftLayers
)
TextTask = TextGeneration
ImageTask = ImageGeneration | ImageEdits
GenerationTask = TextTask | ImageTask
