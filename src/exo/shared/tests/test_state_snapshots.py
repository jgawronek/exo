from hashlib import sha256

import pytest
from pydantic import ValidationError

import exo.shared.models.model_cards as model_cards_module
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.state_snapshots import (
    SnapshotRequestId,
    StateSnapshotChunk,
    StateSnapshotManifest,
)
from exo.utils.state_snapshot import (
    SnapshotIntegrityError,
    SnapshotSizeLimitError,
    SnapshotTargetError,
    StateSnapshotAssembler,
    decode_state_snapshot,
    encode_state_snapshot,
    split_snapshot_payload,
)


def make_session(master_node_id: str = "master") -> SessionId:
    return SessionId(master_node_id=NodeId(master_node_id), election_clock=3)


def test_snapshot_codec_compression_roundtrip_preserves_strict_state() -> None:
    state = State(last_event_applied_idx=41)

    encoded_snapshot = encode_state_snapshot(state)
    decoded_state = decode_state_snapshot(
        encoded_snapshot,
        max_compressed_size=1_000_000,
        max_uncompressed_size=4_000_000,
    )

    assert encoded_snapshot.compression == "zstd"
    assert encoded_snapshot.format_version == 1
    assert encoded_snapshot.uncompressed_size > encoded_snapshot.compressed_size
    assert decoded_state.model_dump_json() == state.model_dump_json()
    assert type(decoded_state.last_event_applied_idx) is int


def test_snapshot_codec_rejects_payload_with_invalid_hash() -> None:
    encoded_snapshot = encode_state_snapshot(State(last_event_applied_idx=2))
    corrupted_payload = encoded_snapshot.payload[:-1] + bytes(
        [encoded_snapshot.payload[-1] ^ 0xFF]
    )
    corrupted_snapshot = encoded_snapshot.model_copy(
        update={"payload": corrupted_payload}
    )

    with pytest.raises(SnapshotIntegrityError, match="hash"):
        decode_state_snapshot(
            corrupted_snapshot,
            max_compressed_size=1_000_000,
            max_uncompressed_size=4_000_000,
        )


@pytest.mark.parametrize(
    ("compressed_limit", "uncompressed_limit"),
    [(1, 4_000_000), (1_000_000, 1)],
)
def test_snapshot_codec_rejects_payload_over_size_limit(
    compressed_limit: int,
    uncompressed_limit: int,
) -> None:
    encoded_snapshot = encode_state_snapshot(State())

    with pytest.raises(SnapshotSizeLimitError):
        decode_state_snapshot(
            encoded_snapshot,
            max_compressed_size=compressed_limit,
            max_uncompressed_size=uncompressed_limit,
        )


def test_snapshot_protocol_models_reject_non_strict_chunk_values() -> None:
    with pytest.raises(ValidationError):
        StateSnapshotChunk.model_validate(
            {
                "request_id": str(SnapshotRequestId()),
                "session": make_session().model_dump(),
                "source_node_id": "master",
                "target_node_id": "follower",
                "chunk_index": "0",
                "payload": b"payload",
                "payload_sha256": sha256(b"payload").hexdigest(),
            }
        )


def test_snapshot_payload_splits_and_assembles_out_of_order() -> None:
    request_id = SnapshotRequestId()
    session = make_session()
    source_node_id = NodeId("master")
    target_node_id = NodeId("follower")
    payload = b"0123456789abcdefghijklmn"

    manifest, chunks = split_snapshot_payload(
        payload,
        request_id=request_id,
        session=session,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        snapshot_index=17,
        chunk_size=5,
    )

    assert isinstance(manifest, StateSnapshotManifest)
    assert len(chunks) == 5
    assert [len(chunk.payload) for chunk in chunks] == [5, 5, 5, 5, 4]
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    assembler = StateSnapshotAssembler(
        request_id=request_id,
        session=session,
        local_node_id=target_node_id,
        expected_source_node_id=source_node_id,
        max_compressed_size=1_000,
    )
    assembler.accept_manifest(manifest)
    for chunk in reversed(chunks):
        assembler.accept_chunk(chunk)

    assert assembler.assemble() == payload


def test_snapshot_assembler_rejects_response_targeted_to_another_node() -> None:
    request_id = SnapshotRequestId()
    session = make_session()
    manifest, _ = split_snapshot_payload(
        b"snapshot",
        request_id=request_id,
        session=session,
        source_node_id=NodeId("master"),
        target_node_id=NodeId("other-follower"),
        snapshot_index=0,
        chunk_size=4,
    )
    assembler = StateSnapshotAssembler(
        request_id=request_id,
        session=session,
        local_node_id=NodeId("follower"),
        expected_source_node_id=NodeId("master"),
        max_compressed_size=1_000,
    )

    with pytest.raises(SnapshotTargetError):
        assembler.accept_manifest(manifest)


def test_snapshot_restore_does_not_read_local_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_card = ModelCard(
        model_id=ModelId("custom/model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    state = State(custom_model_cards={model_card.model_id: model_card})
    encoded_snapshot = encode_state_snapshot(state)

    def fail_on_local_read(*_arguments: object) -> None:
        raise AssertionError("snapshot restore attempted local model enrichment")

    monkeypatch.setattr(
        model_cards_module,
        "detect_vision_from_config",
        fail_on_local_read,
    )
    monkeypatch.setattr(
        model_cards_module,
        "load_local_config_data",
        fail_on_local_read,
    )

    restored_state = decode_state_snapshot(
        encoded_snapshot,
        max_compressed_size=1_000_000,
        max_uncompressed_size=4_000_000,
    )

    assert restored_state.model_dump_json() == state.model_dump_json()
