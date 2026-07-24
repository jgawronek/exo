import base64
import json
from hashlib import sha256
from typing import cast

import pytest

from exo.routing.topics import (
    STATE_SNAPSHOT_CHUNKS,
    STATE_SNAPSHOT_MANIFESTS,
    STATE_SNAPSHOT_REQUESTS,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state_snapshots import (
    SnapshotRequestId,
    StateSnapshotChunk,
    StateSnapshotManifest,
    StateSnapshotRequest,
)


def make_session() -> SessionId:
    return SessionId(master_node_id=NodeId("master"), election_clock=11)


def test_snapshot_request_topic_json_roundtrip() -> None:
    request = StateSnapshotRequest(
        request_id=SnapshotRequestId(),
        session=make_session(),
        source_node_id=NodeId("follower"),
        target_node_id=NodeId("master"),
    )

    serialized_request = STATE_SNAPSHOT_REQUESTS.serialize(request)

    assert STATE_SNAPSHOT_REQUESTS.deserialize(serialized_request) == request


def test_snapshot_manifest_topic_json_roundtrip() -> None:
    manifest = StateSnapshotManifest(
        request_id=SnapshotRequestId(),
        session=make_session(),
        source_node_id=NodeId("master"),
        target_node_id=NodeId("follower"),
        snapshot_index=27,
        chunk_count=2,
        compressed_size=19,
        uncompressed_size=103,
        payload_sha256=sha256(b"compressed snapshot").hexdigest(),
    )

    serialized_manifest = STATE_SNAPSHOT_MANIFESTS.serialize(manifest)

    assert STATE_SNAPSHOT_MANIFESTS.deserialize(serialized_manifest) == manifest


@pytest.mark.parametrize(
    "compressed_bytes",
    [
        b"\x00\xff\xfe\x80",
        bytes(range(256)),
        b"\x28\xb5\x2f\xfd\x00\x00\x00\x00not utf-8 \xf0\x80",
    ],
)
def test_snapshot_chunk_topic_json_uses_base64_for_arbitrary_compressed_bytes(
    compressed_bytes: bytes,
) -> None:
    chunk = StateSnapshotChunk(
        request_id=SnapshotRequestId(),
        session=make_session(),
        source_node_id=NodeId("master"),
        target_node_id=NodeId("follower"),
        chunk_index=0,
        payload=compressed_bytes,
        payload_sha256=sha256(compressed_bytes).hexdigest(),
    )

    serialized_chunk = STATE_SNAPSHOT_CHUNKS.serialize(chunk)
    serialized_object = cast(dict[str, object], json.loads(serialized_chunk))
    serialized_payload = serialized_object["payload"]

    assert isinstance(serialized_payload, str)
    assert base64.urlsafe_b64decode(serialized_payload) == compressed_bytes
    assert STATE_SNAPSHOT_CHUNKS.deserialize(serialized_chunk) == chunk
