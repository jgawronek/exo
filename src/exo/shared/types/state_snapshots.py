from typing import Literal, final
from uuid import UUID, uuid4

from pydantic import Field
from pydantic_core import core_schema

from exo.shared.types.common import NodeId, SessionId
from exo.utils.pydantic_ext import FrozenModel


@final
class SnapshotRequestId(UUID):
    """Version-four UUID identifying one snapshot transfer."""

    def __init__(self, value: str | UUID | None = None) -> None:
        generated_value = uuid4() if value is None else UUID(str(value))
        if generated_value.version != 4:
            raise ValueError("Snapshot request IDs must be UUID version 4")
        super().__init__(hex=generated_value.hex)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: object,
        _handler: object,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.uuid_schema(version=4),
        )


@final
class EncodedStateSnapshot(FrozenModel):
    model_config = FrozenModel.model_config | {
        "ser_json_bytes": "base64",
        "val_json_bytes": "base64",
    }

    compression: Literal["zstd"] = "zstd"
    format_version: Literal[1] = 1
    payload: bytes
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)


@final
class StateSnapshotManifest(FrozenModel):
    request_id: SnapshotRequestId
    session: SessionId
    source_node_id: NodeId
    target_node_id: NodeId
    snapshot_index: int = Field(ge=-1)
    chunk_count: int = Field(ge=1, le=128)
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(ge=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compression: Literal["zstd"] = "zstd"
    format_version: Literal[1] = 1


@final
class StateSnapshotChunk(FrozenModel):
    model_config = FrozenModel.model_config | {
        "ser_json_bytes": "base64",
        "val_json_bytes": "base64",
    }

    request_id: SnapshotRequestId
    session: SessionId
    source_node_id: NodeId
    target_node_id: NodeId
    chunk_index: int = Field(ge=0, le=127)
    payload: bytes
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@final
class StateSnapshotRequest(FrozenModel):
    request_id: SnapshotRequestId = Field(default_factory=SnapshotRequestId)
    session: SessionId
    source_node_id: NodeId
    target_node_id: NodeId


@final
class StateSnapshotUnavailable(FrozenModel):
    request_id: SnapshotRequestId
    session: SessionId
    source_node_id: NodeId
    target_node_id: NodeId
    reason: Literal["not_ready", "snapshot_too_large", "unsupported_version"]
