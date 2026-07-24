from hashlib import sha256
from io import BytesIO
from typing import final

import zstandard
from pydantic import ValidationError

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.shared.types.state_snapshots import (
    EncodedStateSnapshot,
    SnapshotRequestId,
    StateSnapshotChunk,
    StateSnapshotManifest,
)


class SnapshotError(ValueError):
    pass


class SnapshotIntegrityError(SnapshotError):
    pass


class SnapshotSizeLimitError(SnapshotError):
    pass


class SnapshotTargetError(SnapshotError):
    pass


class SnapshotUnavailableError(SnapshotError):
    pass


def validate_state_snapshot_json(serialized_state: bytes) -> State:
    return State.model_validate_json(
        serialized_state,
        strict=True,
        context={"skip_local_model_enrichment": True},
    )


def encode_state_snapshot(state: State) -> EncodedStateSnapshot:
    serialized_state = state.model_dump_json().encode()
    payload = zstandard.ZstdCompressor().compress(serialized_state)
    return EncodedStateSnapshot(
        payload=payload,
        payload_sha256=sha256(payload).hexdigest(),
        compressed_size=len(payload),
        uncompressed_size=len(serialized_state),
    )


def decode_state_snapshot(
    encoded_snapshot: EncodedStateSnapshot,
    *,
    max_compressed_size: int,
    max_uncompressed_size: int,
) -> State:
    payload = encoded_snapshot.payload
    if (
        len(payload) > max_compressed_size
        or encoded_snapshot.compressed_size > max_compressed_size
    ):
        raise SnapshotSizeLimitError("Snapshot exceeds the compressed size limit")
    if encoded_snapshot.uncompressed_size > max_uncompressed_size:
        raise SnapshotSizeLimitError("Snapshot exceeds the uncompressed size limit")
    if len(payload) != encoded_snapshot.compressed_size:
        raise SnapshotIntegrityError("Snapshot compressed size does not match")
    if sha256(payload).hexdigest() != encoded_snapshot.payload_sha256:
        raise SnapshotIntegrityError("Snapshot payload hash does not match")

    try:
        with zstandard.ZstdDecompressor(
            max_window_size=(max_uncompressed_size + 1023) // 1024
        ).stream_reader(BytesIO(payload)) as reader:
            serialized_state = reader.read(max_uncompressed_size + 1)
    except zstandard.ZstdError as error:
        raise SnapshotIntegrityError("Snapshot zstandard payload is invalid") from error

    if len(serialized_state) > max_uncompressed_size:
        raise SnapshotSizeLimitError("Snapshot exceeds the uncompressed size limit")
    if len(serialized_state) != encoded_snapshot.uncompressed_size:
        raise SnapshotIntegrityError("Snapshot uncompressed size does not match")
    try:
        return validate_state_snapshot_json(serialized_state)
    except ValidationError as error:
        raise SnapshotIntegrityError(
            "Snapshot does not contain a valid State"
        ) from error


def split_snapshot_payload(
    payload: bytes,
    *,
    request_id: SnapshotRequestId,
    session: SessionId,
    source_node_id: NodeId,
    target_node_id: NodeId,
    snapshot_index: int,
    chunk_size: int,
    uncompressed_size: int = 0,
) -> tuple[StateSnapshotManifest, tuple[StateSnapshotChunk, ...]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    payload_hash = sha256(payload).hexdigest()
    payload_parts = tuple(
        payload[offset : offset + chunk_size]
        for offset in range(0, len(payload), chunk_size)
    )
    if not payload_parts:
        payload_parts = (b"",)
    manifest = StateSnapshotManifest(
        request_id=request_id,
        session=session,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        snapshot_index=snapshot_index,
        chunk_count=len(payload_parts),
        compressed_size=len(payload),
        uncompressed_size=uncompressed_size,
        payload_sha256=payload_hash,
    )
    chunks = tuple(
        StateSnapshotChunk(
            request_id=request_id,
            session=session,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            chunk_index=chunk_index,
            payload=payload_part,
            payload_sha256=sha256(payload_part).hexdigest(),
        )
        for chunk_index, payload_part in enumerate(payload_parts)
    )
    return manifest, chunks


@final
class StateSnapshotAssembler:
    def __init__(
        self,
        *,
        request_id: SnapshotRequestId,
        session: SessionId,
        local_node_id: NodeId,
        expected_source_node_id: NodeId,
        max_compressed_size: int,
    ) -> None:
        self._request_id = request_id
        self._session = session
        self._local_node_id = local_node_id
        self._expected_source_node_id = expected_source_node_id
        self._max_compressed_size = max_compressed_size
        self._manifest: StateSnapshotManifest | None = None
        self._chunks: dict[int, bytes] = {}

    @property
    def received_chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def request_id(self) -> SnapshotRequestId:
        return self._request_id

    def _validate_envelope(
        self,
        *,
        request_id: SnapshotRequestId,
        session: SessionId,
        source_node_id: NodeId,
        target_node_id: NodeId,
    ) -> None:
        if target_node_id != self._local_node_id:
            raise SnapshotTargetError("Snapshot response targets another node")
        if source_node_id != self._expected_source_node_id:
            raise SnapshotTargetError(
                "Snapshot response came from an unexpected source"
            )
        if request_id != self._request_id or session != self._session:
            raise SnapshotTargetError("Snapshot response does not match this request")

    def accept_manifest(self, manifest: StateSnapshotManifest) -> None:
        self._validate_envelope(
            request_id=manifest.request_id,
            session=manifest.session,
            source_node_id=manifest.source_node_id,
            target_node_id=manifest.target_node_id,
        )
        if manifest.compressed_size > self._max_compressed_size:
            raise SnapshotSizeLimitError("Snapshot exceeds the compressed size limit")
        if self._manifest is not None and self._manifest != manifest:
            raise SnapshotIntegrityError("Conflicting snapshot manifests received")
        if any(chunk_index >= manifest.chunk_count for chunk_index in self._chunks):
            raise SnapshotIntegrityError("Snapshot chunk index exceeds the manifest")
        if sum(map(len, self._chunks.values())) > manifest.compressed_size:
            raise SnapshotIntegrityError("Snapshot chunks exceed the manifest size")
        self._manifest = manifest

    def accept_chunk(self, chunk: StateSnapshotChunk) -> None:
        self._validate_envelope(
            request_id=chunk.request_id,
            session=chunk.session,
            source_node_id=chunk.source_node_id,
            target_node_id=chunk.target_node_id,
        )
        if sha256(chunk.payload).hexdigest() != chunk.payload_sha256:
            raise SnapshotIntegrityError("Snapshot chunk hash does not match")
        if len(chunk.payload) > self._max_compressed_size:
            raise SnapshotSizeLimitError(
                "Snapshot chunk exceeds the compressed size limit"
            )
        if (
            self._manifest is not None
            and chunk.chunk_index >= self._manifest.chunk_count
        ):
            raise SnapshotIntegrityError("Snapshot chunk index exceeds the manifest")
        existing_payload = self._chunks.get(chunk.chunk_index)
        if existing_payload is not None and existing_payload != chunk.payload:
            raise SnapshotIntegrityError("Conflicting snapshot chunks received")
        size_without_existing = sum(map(len, self._chunks.values())) - (
            len(existing_payload) if existing_payload is not None else 0
        )
        if size_without_existing + len(chunk.payload) > self._max_compressed_size:
            raise SnapshotSizeLimitError(
                "Snapshot chunks exceed the compressed size limit"
            )
        self._chunks[chunk.chunk_index] = chunk.payload

    def assemble(self) -> bytes:
        manifest = self._manifest
        if manifest is None:
            raise SnapshotIntegrityError("Snapshot manifest has not been received")
        expected_indices = set(range(manifest.chunk_count))
        if set(self._chunks) != expected_indices:
            raise SnapshotIntegrityError("Snapshot chunks are incomplete")
        payload = b"".join(self._chunks[index] for index in range(manifest.chunk_count))
        if len(payload) != manifest.compressed_size:
            raise SnapshotIntegrityError("Snapshot compressed size does not match")
        if sha256(payload).hexdigest() != manifest.payload_sha256:
            raise SnapshotIntegrityError("Snapshot payload hash does not match")
        return payload
