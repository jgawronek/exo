from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.storage import SharedStorage
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams
    # When set, the master routes the request to this exact instance instead
    # of load-balancing across all instances serving the model.
    pinned_instance_id: InstanceId | None = None


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    node_layers: dict[NodeId, int] | None = None
    # Explicit pipeline ring order (device ranks 0..N-1). When set, placement
    # skips auto link-speed reordering and uses this sequence instead.
    node_order: list[NodeId] | None = None
    max_context_length: int | None = None
    prefill_step_size: int | None = None


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class ShiftInstanceLayers(BaseCommand):
    """Live-rebalance a pipeline instance by moving one layer at a time.

    The master decomposes the move into single-layer boundary shifts between
    adjacent ranks and executes them sequentially while the instance keeps
    serving requests.
    """

    instance_id: InstanceId
    node_layers: dict[NodeId, int]


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class SetSharedModelsDirectory(BaseCommand):
    """Set (or clear, with ``None``) the cluster-wide shared models directory."""

    path: str | None


class SetSharedStorage(BaseCommand):
    """Set (or clear, with ``None``) the named shared storage.

    Carries each node's own local path for the share, so nodes are never asked
    to agree on one path.
    """

    storage: SharedStorage | None


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


class SetInstanceLink(BaseCommand):
    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class DeleteInstanceLink(BaseCommand):
    link_id: InstanceLinkId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | ShiftInstanceLayers
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | SetSharedModelsDirectory
    | SetSharedStorage
    | AddCustomModelCard
    | DeleteCustomModelCard
    | SetInstanceLink
    | DeleteInstanceLink
)


class ForwarderCommand(FrozenModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    origin: SystemId
    command: DownloadCommand
