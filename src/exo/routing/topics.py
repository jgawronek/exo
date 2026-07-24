from dataclasses import dataclass
from enum import Enum

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from exo.shared.types.state_snapshots import (
    StateSnapshotChunk,
    StateSnapshotManifest,
    StateSnapshotRequest,
    StateSnapshotUnavailable,
)
from exo.utils.pydantic_ext import FrozenModel


class PublishPolicy(str, Enum):
    Never = "Never"
    """Never publish to the network - this is a local message"""
    Minimal = "Minimal"
    """Only publish when there is no local receiver for this type of message"""
    Always = "Always"
    """Always publish to the network"""


@dataclass  # (frozen=True)
class TypedTopic[T: FrozenModel]:
    topic: str
    publish_policy: PublishPolicy

    model_type: type[
        T
    ]  # This can be worked around with evil type hacking, see https://stackoverflow.com/a/71720366 - I don't think it's necessary here.
    max_payload_size: int = 16 * 1024 * 1024

    @staticmethod
    def serialize(t: T) -> bytes:
        return t.model_dump_json().encode("utf-8")

    def deserialize(self, b: bytes) -> T:
        return self.model_type.model_validate_json(b.decode("utf-8"))


GLOBAL_EVENTS = TypedTopic("global_events", PublishPolicy.Always, GlobalForwarderEvent)
LOCAL_EVENTS = TypedTopic("local_events", PublishPolicy.Always, LocalForwarderEvent)
COMMANDS = TypedTopic("commands", PublishPolicy.Always, ForwarderCommand)
ELECTION_MESSAGES = TypedTopic(
    "election_messages", PublishPolicy.Always, ElectionMessage
)
CONNECTION_MESSAGES = TypedTopic(
    "connection_messages", PublishPolicy.Never, ConnectionMessage
)
DOWNLOAD_COMMANDS = TypedTopic(
    "download_commands", PublishPolicy.Always, ForwarderDownloadCommand
)
STATE_SNAPSHOT_REQUESTS = TypedTopic(
    "state_snapshot_requests",
    PublishPolicy.Always,
    StateSnapshotRequest,
    max_payload_size=64 * 1024,
)
STATE_SNAPSHOT_MANIFESTS = TypedTopic(
    "state_snapshot_manifests",
    PublishPolicy.Always,
    StateSnapshotManifest,
    max_payload_size=64 * 1024,
)
STATE_SNAPSHOT_CHUNKS = TypedTopic(
    "state_snapshot_chunks",
    PublishPolicy.Always,
    StateSnapshotChunk,
    max_payload_size=1024 * 1024,
)
STATE_SNAPSHOT_UNAVAILABLE = TypedTopic(
    "state_snapshot_unavailable",
    PublishPolicy.Always,
    StateSnapshotUnavailable,
    max_payload_size=64 * 1024,
)
