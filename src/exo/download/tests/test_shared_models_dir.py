"""Tests for the runtime shared models directory: validation and path preference."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest

from exo.download.download_utils import (
    build_model_path,
    resolve_existing_model,
    select_download_dir,
)
from exo.download.shared_models_dir import (
    _BROWSE_ENTRY_LIMIT,  # pyright: ignore[reportPrivateUsage]
    browse_shared_models_directories,
    get_shared_models_dir,
    list_network_volumes,
    set_shared_models_dir,
    validate_shared_models_directory,
)
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()


class _Partition(NamedTuple):
    """Stands in for ``psutil.disk_partitions()`` entries."""

    device: str
    mountpoint: str
    fstype: str


def _create_complete_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {"layer.weight": "model.safetensors"}
    index = {"metadata": {"total_size": 1024}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text('{"model_type": "test"}')


@pytest.fixture(autouse=True)
def reset_shared_models_dir() -> Iterator[None]:
    previous = get_shared_models_dir()
    yield
    set_shared_models_dir(previous)


class TestValidateSharedModelsDirectory:
    def test_missing_path_is_invalid(self, tmp_path: Path) -> None:
        status = validate_shared_models_directory(str(tmp_path / "missing"))
        assert not status.valid
        assert status.error == "Path does not exist"

    def test_file_is_invalid(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file"
        file_path.write_text("not a directory")
        status = validate_shared_models_directory(str(file_path))
        assert not status.valid
        assert status.error == "Path is not a directory"

    def test_writable_directory_is_valid(self, tmp_path: Path) -> None:
        status = validate_shared_models_directory(str(tmp_path))
        assert status.valid
        assert status.error is None
        assert status.free_bytes is not None and status.free_bytes > 0

    def test_leaves_no_probe_files_behind(self, tmp_path: Path) -> None:
        validate_shared_models_directory(str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestSharedDirPreference:
    def test_resolve_prefers_shared_dir(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        _create_complete_model(shared / NORMALIZED)
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == shared / NORMALIZED

    def test_resolve_falls_back_to_local_dirs(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == writable / NORMALIZED

    def test_build_model_path_targets_shared_dir_for_new_models(
        self, tmp_path: Path
    ) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", writable),
        ):
            assert build_model_path(MODEL_ID) == shared / NORMALIZED

    def test_build_model_path_uses_default_without_shared_dir(
        self, tmp_path: Path
    ) -> None:
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(None)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", writable),
        ):
            assert build_model_path(MODEL_ID) == writable / NORMALIZED

    def test_select_download_dir_prefers_shared_dir(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(shared)
        with patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)):
            assert select_download_dir(required_bytes=1) == shared


class TestBrowseSharedModelsDirectories:
    def test_lists_child_directories(self, tmp_path: Path) -> None:
        (tmp_path / "caches").mkdir()
        (tmp_path / "readme.txt").write_text("not a directory")
        (tmp_path / ".hidden").mkdir()
        result = browse_shared_models_directories(str(tmp_path))
        assert result.error is None
        assert result.path == str(tmp_path.resolve())
        assert [entry.name for entry in result.entries] == ["caches"]
        assert result.entries[0].path == str((tmp_path / "caches").resolve())

    def test_missing_path_returns_error(self, tmp_path: Path) -> None:
        result = browse_shared_models_directories(str(tmp_path / "missing"))
        assert result.entries == ()
        assert result.error == "Path does not exist"

    def test_empty_path_returns_roots_without_error(self) -> None:
        result = browse_shared_models_directories(None)
        assert result.path == ""
        assert result.error is None
        assert isinstance(result.entries, tuple)

    def test_shortcuts_lead_with_the_filesystem_root(self) -> None:
        """Every folder on the node has to be reachable, so / comes first."""
        result = browse_shared_models_directories(None)
        anchor = Path(Path.home().anchor or "/")
        assert result.entries[0].path == str(anchor)

    def test_hidden_directories_are_listed_on_request(self, tmp_path: Path) -> None:
        (tmp_path / "visible").mkdir()
        (tmp_path / ".cache").mkdir()

        without_hidden = browse_shared_models_directories(str(tmp_path))
        assert [entry.name for entry in without_hidden.entries] == ["visible"]

        with_hidden = browse_shared_models_directories(
            str(tmp_path), include_hidden=True
        )
        assert [entry.name for entry in with_hidden.entries] == [".cache", "visible"]
        assert [entry.hidden for entry in with_hidden.entries] == [True, False]

    def test_overlong_listing_reports_truncation(self, tmp_path: Path) -> None:
        for index in range(_BROWSE_ENTRY_LIMIT + 5):
            (tmp_path / f"dir-{index:04d}").mkdir()
        result = browse_shared_models_directories(str(tmp_path))
        assert result.truncated
        assert len(result.entries) == _BROWSE_ENTRY_LIMIT

    def test_listing_within_the_limit_is_not_flagged_truncated(
        self, tmp_path: Path
    ) -> None:
        for index in range(3):
            (tmp_path / f"dir-{index}").mkdir()
        result = browse_shared_models_directories(str(tmp_path))
        assert not result.truncated
        assert len(result.entries) == 3


class _PathRootedAt:
    """Stands in for Path so /run/user/<uid>/gvfs lands under the tmp dir."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def __call__(self, raw: str) -> Path:
        if raw.startswith("/run/user/"):
            return self._root / "gvfs"
        return Path(raw)


class TestListNetworkVolumes:
    def test_keeps_network_filesystems_and_drops_local_ones(
        self, tmp_path: Path
    ) -> None:
        mounted = tmp_path / "huggingface"
        mounted.mkdir()
        partitions = [
            _Partition("/dev/disk3s1s1", "/", "apfs"),
            _Partition("10.0.10.44:/export/models", str(mounted), "nfs"),
            _Partition("//jay@nas/media", "/Volumes/media", "smbfs"),
            _Partition("tmpfs", "/run", "tmpfs"),
        ]
        with patch(
            "exo.download.shared_models_dir.psutil.disk_partitions",
            return_value=partitions,
        ):
            volumes = list_network_volumes()

        by_path = {volume.path: volume for volume in volumes}
        assert set(by_path) == {str(mounted), "/Volumes/media"}
        assert by_path[str(mounted)].filesystem == "nfs"
        assert by_path["/Volumes/media"].filesystem == "smbfs"
        assert by_path[str(mounted)].source == "10.0.10.44:/export/models"
        assert by_path[str(mounted)].label == "models · 10.0.10.44"
        assert by_path["/Volumes/media"].label == "media · jay@nas".replace(
            "jay@", ""
        )  # //jay@nas/media -> "media · nas"
        assert by_path[str(mounted)].reachable
        # The SMB mount point does not exist in the sandbox, so it reads as down.
        assert not by_path["/Volumes/media"].reachable

    def test_unreadable_partition_table_yields_nothing(self) -> None:
        with patch(
            "exo.download.shared_models_dir.psutil.disk_partitions",
            side_effect=OSError("nope"),
        ):
            assert list_network_volumes() == ()

    def test_gvfs_smb_shares_are_listed_as_network_volumes(
        self, tmp_path: Path
    ) -> None:
        """gvfs hides shares inside one FUSE mount, invisible to fstype filters."""
        gvfs_root = tmp_path / "gvfs"
        gvfs_root.mkdir()
        share_dir = gvfs_root / "smb-share:server=10.0.10.44,share=aimodels"
        share_dir.mkdir()
        (gvfs_root / "not-a-share").mkdir()

        with (
            patch(
                "exo.download.shared_models_dir.psutil.disk_partitions",
                return_value=[],
            ),
            patch("exo.download.shared_models_dir.os.getuid", return_value=0),
            patch(
                "exo.download.shared_models_dir.Path",
                new=_PathRootedAt(tmp_path),
            ),
        ):
            volumes = list_network_volumes()

        assert [volume.source for volume in volumes] == ["//10.0.10.44/aimodels"]
        assert volumes[0].filesystem == "smb (gvfs)"
        assert volumes[0].label == "aimodels · 10.0.10.44"
        assert volumes[0].reachable is True  # exists and stats fine
        assert volumes[0].path.endswith("share=aimodels")

    def test_browse_result_carries_volumes_even_on_error(self, tmp_path: Path) -> None:
        partitions = [_Partition("10.0.10.44:/export", "/mnt/models", "nfs4")]
        with patch(
            "exo.download.shared_models_dir.psutil.disk_partitions",
            return_value=partitions,
        ):
            result = browse_shared_models_directories(str(tmp_path / "missing"))

        assert result.error == "Path does not exist"
        assert [volume.path for volume in result.network_volumes] == ["/mnt/models"]

    def test_stale_mount_root_does_not_sink_the_listing(self, tmp_path: Path) -> None:
        """A dead NFS/SMB mount raises from is_dir(); the other roots survive it."""
        mount_root = tmp_path / "mnt"
        mount_root.mkdir()
        (mount_root / "healthy").mkdir()
        stale = mount_root / "stale"
        stale.mkdir()

        real_is_dir = Path.is_dir

        def is_dir_with_stale_mount(
            self: Path, *args: object, **kwargs: object
        ) -> bool:
            if self == stale:
                raise OSError(116, "Stale file handle")
            return real_is_dir(self, *args, **kwargs)

        with (
            patch(
                "exo.download.shared_models_dir._BROWSE_ROOT_CANDIDATES",
                (mount_root,),
            ),
            patch.object(Path, "is_dir", is_dir_with_stale_mount),
        ):
            result = browse_shared_models_directories(None)

        names = [entry.name for entry in result.entries]
        assert "healthy" in names
        assert "stale" not in names

    def test_unreachable_path_reports_error_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        def exists_raising_stale(self: Path, *args: object, **kwargs: object) -> bool:
            raise OSError(116, "Stale file handle")

        with patch.object(Path, "exists", exists_raising_stale):
            result = browse_shared_models_directories(str(tmp_path))

        assert result.entries == ()
        assert result.error is not None
        assert "Stale file handle" in result.error
