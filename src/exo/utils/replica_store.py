import json
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import TypeVar, final

import msgspec
from pydantic import TypeAdapter, ValidationError

from exo.shared.types.common import SessionId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.state import State
from exo.shared.types.state_snapshots import EncodedStateSnapshot
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.state_replica import ReplicaSequenceError, StateReplica
from exo.utils.state_snapshot import (
    SnapshotError,
    decode_state_snapshot,
    encode_state_snapshot,
)

_LENGTH_SIZE = 4
_CHECKSUM_SIZE = 32
_MAX_RECORD_SIZE = 256 * 1024 * 1024
_MAX_JOURNAL_SIZE = 64 * 1024 * 1024


class ReplicaStoreError(ValueError):
    pass


@final
class _JournalRecord(FrozenModel):
    session: SessionId
    indexed_event: IndexedEvent


@final
class _CheckpointRecord(FrozenModel):
    session: SessionId
    encoded_snapshot: EncodedStateSnapshot


_JOURNAL_RECORD_ADAPTER = TypeAdapter(_JournalRecord)
_CHECKPOINT_RECORD_ADAPTER = TypeAdapter(_CheckpointRecord)
_Record = TypeVar("_Record")


def _encode_record(record: FrozenModel) -> bytes:
    payload = msgspec.msgpack.encode(record.model_dump(mode="json"))
    if len(payload) > _MAX_RECORD_SIZE:
        raise ReplicaStoreError("Replica record exceeds size limit")
    return (
        len(payload).to_bytes(_LENGTH_SIZE, byteorder="big")
        + sha256(payload).digest()
        + payload
    )


def _decode_record_payload(
    payload: bytes,
    adapter: TypeAdapter[_Record],
) -> _Record:
    try:
        unpacked_record = msgspec.msgpack.decode(payload, type=dict[str, object])
        serialized_record = json.dumps(unpacked_record)
        return adapter.validate_json(
            serialized_record,
            strict=True,
            context={"skip_local_model_enrichment": True},
        )
    except (msgspec.DecodeError, TypeError, ValidationError) as error:
        raise ReplicaStoreError("Replica record payload is invalid") from error


@final
class ReplicaStore:
    """Persists process-crash recovery state.

    Journal appends rely on the operating-system page cache; the atomically
    replaced checkpoint is the durable power-loss recovery boundary.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self.journal_path = directory / "journal.msgpack"
        self.checkpoint_path = directory / "checkpoint.msgpack"
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            for path in (self.journal_path, self.checkpoint_path):
                path.unlink(missing_ok=True)

    def append(self, session: SessionId, indexed_event: IndexedEvent) -> None:
        encoded_record = _encode_record(
            _JournalRecord(session=session, indexed_event=indexed_event)
        )
        with self._lock, self.journal_path.open("ab") as journal:
            journal.write(encoded_record)

    def write_checkpoint(
        self,
        session: SessionId,
        state: State,
        *,
        compact_journal: bool = False,
    ) -> None:
        encoded_record = _encode_record(
            _CheckpointRecord(
                session=session,
                encoded_snapshot=encode_state_snapshot(state),
            )
        )
        with self._lock:
            temporary_path = self.checkpoint_path.with_suffix(".tmp")
            with temporary_path.open("wb") as checkpoint:
                checkpoint.write(encoded_record)
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
            temporary_path.replace(self.checkpoint_path)
            if compact_journal:
                self._compact_journal(
                    session=session,
                    checkpoint_index=state.last_event_applied_idx,
                )
            directory_descriptor = os.open(self._directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)

    def write_compacted_checkpoint(self, session: SessionId, state: State) -> None:
        self.write_checkpoint(session, state, compact_journal=True)

    def _compact_journal(
        self,
        *,
        session: SessionId,
        checkpoint_index: int,
    ) -> None:
        retained_records = tuple(
            record
            for record in self._read_journal()
            if record.session == session and record.indexed_event.idx > checkpoint_index
        )
        temporary_path = self.journal_path.with_suffix(".tmp")
        with temporary_path.open("wb") as journal:
            for record in retained_records:
                journal.write(_encode_record(record))
            journal.flush()
            os.fsync(journal.fileno())
        temporary_path.replace(self.journal_path)

    def _read_checkpoint(self) -> _CheckpointRecord | None:
        if not self.checkpoint_path.exists():
            return None
        if self.checkpoint_path.stat().st_size > (
            _LENGTH_SIZE + _CHECKSUM_SIZE + _MAX_RECORD_SIZE
        ):
            raise ReplicaStoreError("Checkpoint file exceeds size limit")
        checkpoint_data = self.checkpoint_path.read_bytes()
        minimum_size = _LENGTH_SIZE + _CHECKSUM_SIZE
        if len(checkpoint_data) < minimum_size:
            raise ReplicaStoreError("Checkpoint record is torn")
        payload_size = int.from_bytes(checkpoint_data[:_LENGTH_SIZE], byteorder="big")
        if payload_size > _MAX_RECORD_SIZE:
            raise ReplicaStoreError("Checkpoint record exceeds size limit")
        expected_size = minimum_size + payload_size
        if len(checkpoint_data) != expected_size:
            raise ReplicaStoreError("Checkpoint record length is invalid")
        expected_checksum = checkpoint_data[_LENGTH_SIZE:minimum_size]
        payload = checkpoint_data[minimum_size:]
        if sha256(payload).digest() != expected_checksum:
            raise ReplicaStoreError("Checkpoint record checksum does not match")
        return _decode_record_payload(payload, _CHECKPOINT_RECORD_ADAPTER)

    def _read_journal(self) -> list[_JournalRecord]:
        if not self.journal_path.exists():
            return []
        if self.journal_path.stat().st_size > _MAX_JOURNAL_SIZE:
            raise ReplicaStoreError("Replica journal exceeds size limit")
        records: list[_JournalRecord] = []
        with self.journal_path.open("r+b") as journal:
            while True:
                record_start = journal.tell()
                length_bytes = journal.read(_LENGTH_SIZE)
                if not length_bytes:
                    break
                if len(length_bytes) != _LENGTH_SIZE:
                    journal.truncate(record_start)
                    journal.flush()
                    os.fsync(journal.fileno())
                    break
                payload_size = int.from_bytes(length_bytes, byteorder="big")
                if payload_size > _MAX_RECORD_SIZE:
                    raise ReplicaStoreError(
                        f"Journal record at offset {record_start} exceeds size limit"
                    )
                expected_checksum = journal.read(_CHECKSUM_SIZE)
                payload = journal.read(payload_size)
                if (
                    len(expected_checksum) != _CHECKSUM_SIZE
                    or len(payload) != payload_size
                ):
                    journal.truncate(record_start)
                    journal.flush()
                    os.fsync(journal.fileno())
                    break
                if sha256(payload).digest() != expected_checksum:
                    raise ReplicaStoreError(
                        f"Journal record checksum does not match at offset {record_start}"
                    )
                records.append(_decode_record_payload(payload, _JOURNAL_RECORD_ADAPTER))
        return records

    def recover(self) -> StateReplica | None:
        with self._lock:
            try:
                checkpoint = self._read_checkpoint()
                journal_records = self._read_journal()
                if checkpoint is None and not journal_records:
                    return None

                if checkpoint is None:
                    session = journal_records[0].session
                    initial_state = State()
                else:
                    session = checkpoint.session
                    initial_state = decode_state_snapshot(
                        checkpoint.encoded_snapshot,
                        max_compressed_size=_MAX_RECORD_SIZE,
                        max_uncompressed_size=_MAX_RECORD_SIZE,
                    )
                replica = StateReplica(session=session, initial_state=initial_state)

                for record in journal_records:
                    if record.session != session:
                        raise ReplicaStoreError(
                            "Replica journal session does not match"
                        )
                    if record.indexed_event.idx <= initial_state.last_event_applied_idx:
                        continue
                    replica.apply(record.indexed_event)
                return replica
            except (ReplicaSequenceError, SnapshotError) as error:
                raise ReplicaStoreError("Replica recovery failed") from error
