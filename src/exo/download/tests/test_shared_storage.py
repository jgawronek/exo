"""Tests for share-based shared storage resolution.

The point of a share is that nodes agree on the *store*, never on a path: a
Linux node reading an NFS mount at ``/mnt/models`` and a Mac reading the same
export at ``/Volumes/AIModels`` are both serving one share. These tests pin
that nothing compares one node's path against another's.
"""

from pathlib import Path

from exo.download.shared_models_dir import (
    load_persisted_shared_storage,
    persist_shared_storage,
    resolve_shared_models_path,
    validate_shared_models_directory,
)
from exo.shared.types.common import NodeId
from exo.shared.types.storage import SharedStorage

LINUX_NODE = NodeId("1111111111111111")
MAC_NODE = NodeId("2222222222222222")
UNMAPPED_NODE = NodeId("3333333333333333")


def _share() -> SharedStorage:
    return SharedStorage(
        share_id="models",
        mounts={
            LINUX_NODE: "/mnt/lexar4tb/exo-models",
            MAC_NODE: "/Volumes/AIModels",
        },
        source="nfs://10.0.10.44/mnt/lexar4tb/exo-models",
    )


class TestResolveSharedModelsPath:
    def test_each_node_gets_its_own_path(self) -> None:
        share = _share()
        assert (
            resolve_shared_models_path(share, None, LINUX_NODE)
            == "/mnt/lexar4tb/exo-models"
        )
        assert resolve_shared_models_path(share, None, MAC_NODE) == "/Volumes/AIModels"

    def test_paths_are_allowed_to_differ(self) -> None:
        """The whole point: no agreement between nodes is required."""
        share = _share()
        linux = resolve_shared_models_path(share, None, LINUX_NODE)
        mac = resolve_shared_models_path(share, None, MAC_NODE)
        assert linux != mac
        assert linux is not None and mac is not None

    def test_node_without_an_entry_has_no_shared_storage(self) -> None:
        assert resolve_shared_models_path(_share(), None, UNMAPPED_NODE) is None

    def test_share_wins_over_the_legacy_single_path(self) -> None:
        resolved = resolve_shared_models_path(_share(), "/legacy/models", LINUX_NODE)
        assert resolved == "/mnt/lexar4tb/exo-models"

    def test_unmapped_node_does_not_fall_back_to_the_legacy_path(self) -> None:
        """Falling back would silently point one node at a different store."""
        assert (
            resolve_shared_models_path(_share(), "/legacy/models", UNMAPPED_NODE)
            is None
        )

    def test_legacy_path_still_applies_without_a_share(self) -> None:
        assert (
            resolve_shared_models_path(None, "/legacy/models", LINUX_NODE)
            == "/legacy/models"
        )

    def test_nothing_configured_resolves_to_nothing(self) -> None:
        assert resolve_shared_models_path(None, None, LINUX_NODE) is None


class TestSharedStoragePersistence:
    def test_round_trips_through_disk(self, tmp_path: Path, monkeypatch) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        target = tmp_path / "shared_storage.json"
        monkeypatch.setattr(  # pyright: ignore[reportUnknownMemberType]
            "exo.download.shared_models_dir.EXO_SHARED_STORAGE_FILE", target
        )
        share = _share()
        persist_shared_storage(share)
        assert load_persisted_shared_storage() == share

    def test_clearing_removes_the_file(self, tmp_path: Path, monkeypatch) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        target = tmp_path / "shared_storage.json"
        monkeypatch.setattr(  # pyright: ignore[reportUnknownMemberType]
            "exo.download.shared_models_dir.EXO_SHARED_STORAGE_FILE", target
        )
        persist_shared_storage(_share())
        persist_shared_storage(None)
        assert not target.exists()
        assert load_persisted_shared_storage() is None

    def test_corrupt_file_does_not_stop_the_node(
        self,
        tmp_path: Path,
        monkeypatch,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    ) -> None:
        target = tmp_path / "shared_storage.json"
        target.write_text("{ not json")
        monkeypatch.setattr(  # pyright: ignore[reportUnknownMemberType]
            "exo.download.shared_models_dir.EXO_SHARED_STORAGE_FILE", target
        )
        assert load_persisted_shared_storage() is None


class TestStatusCarriesResolvedPath:
    def test_valid_status_reports_the_probed_path(self, tmp_path: Path) -> None:
        status = validate_shared_models_directory(str(tmp_path))
        assert status.valid
        assert status.path == str(tmp_path)

    def test_invalid_status_still_reports_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        status = validate_shared_models_directory(str(missing))
        assert not status.valid
        assert status.path == str(missing)
        assert status.error == "Path does not exist"
