"""Runtime-configurable shared models directory.

The models search directories in :mod:`exo.shared.constants` are import-time
constants; the shared directory configured from the dashboard must instead be
changeable while the node runs. This module holds that one mutable piece of
state behind a small effect-handler class, plus the pure helpers to validate
and persist the setting.

Runner subprocesses cannot observe in-process mutations, so the setter also
exports the value through the ``XEO_SHARED_MODELS_DIR`` environment variable,
which newly spawned runners read at import time.
"""

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import final

import psutil
from pydantic import ValidationError

from exo.shared.constants import (
    EXO_SHARED_MODELS_DIR_FILE,
    EXO_SHARED_STORAGE_FILE,
)
from exo.shared.environment import get_compatible_environment_value
from exo.shared.types.common import NodeId
from exo.shared.types.storage import SharedDirectoryStatus, SharedStorage
from exo.utils.pydantic_ext import FrozenModel

_SHARED_MODELS_DIR_ENVIRONMENT_NAME = "XEO_SHARED_MODELS_DIR"
_SHARED_MODELS_DIR_READ_ONLY_ENVIRONMENT_NAME = "XEO_SHARED_MODELS_DIR_READ_ONLY"
_BROWSE_ENTRY_LIMIT = 200
_BROWSE_ROOT_CANDIDATES = (
    Path("/Volumes"),
    Path("/mnt"),
    Path("/media"),
    Path("/run/media"),
)
# Filesystem types served over a network rather than by a local device. Listed
# per-platform spelling: Linux reports "nfs4"/"cifs", macOS "nfs"/"smbfs".
_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afpfs",
        "afs",
        "beegfs",
        "ceph",
        "cifs",
        "davfs",
        "davfs2",
        "ftp",
        "fuse.sshfs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb2",
        "smb3",
        "smbfs",
        "sshfs",
        "webdav",
    }
)


class SharedModelsDirectoryBrowseEntry(FrozenModel):
    name: str
    path: str
    hidden: bool = False


class NetworkVolume(FrozenModel):
    """A mounted network filesystem on this node."""

    path: str
    source: str
    filesystem: str
    reachable: bool


class SharedModelsDirectoryBrowseResult(FrozenModel):
    path: str
    parent_path: str | None
    entries: tuple[SharedModelsDirectoryBrowseEntry, ...]
    error: str | None = None
    truncated: bool = False
    network_volumes: tuple[NetworkVolume, ...] = ()


@final
class _SharedModelsDirectoryHolder:
    """Process-wide holder for the validated shared models directory."""

    def __init__(self) -> None:
        environment_value = get_compatible_environment_value(
            os.environ, "EXO_SHARED_MODELS_DIR"
        )
        self._path: Path | None = (
            Path(environment_value).expanduser() if environment_value else None
        )
        self._writable: bool = (
            os.environ.get(_SHARED_MODELS_DIR_READ_ONLY_ENVIRONMENT_NAME) != "1"
        )

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def writable_path(self) -> Path | None:
        return self._path if self._writable else None

    def set(self, path: Path | None, writable: bool = True) -> None:
        self._path = path
        self._writable = writable
        # Export for runner subprocesses spawned after this point.
        if path is None:
            os.environ.pop(_SHARED_MODELS_DIR_ENVIRONMENT_NAME, None)
            os.environ.pop(_SHARED_MODELS_DIR_READ_ONLY_ENVIRONMENT_NAME, None)
        else:
            os.environ[_SHARED_MODELS_DIR_ENVIRONMENT_NAME] = str(path)
            if writable:
                os.environ.pop(_SHARED_MODELS_DIR_READ_ONLY_ENVIRONMENT_NAME, None)
            else:
                os.environ[_SHARED_MODELS_DIR_READ_ONLY_ENVIRONMENT_NAME] = "1"


_holder = _SharedModelsDirectoryHolder()


def get_shared_models_dir() -> Path | None:
    """The validated shared models directory for this process, if any."""
    return _holder.path


def set_shared_models_dir(path: Path | None, writable: bool = True) -> None:
    """Install (or clear) the validated shared models directory.

    Only call this with a path that passed
    :func:`validate_shared_models_directory` on this node.
    """
    _holder.set(path, writable)


def get_writable_shared_models_dir() -> Path | None:
    """The shared directory, only when downloads may be written to it."""
    return _holder.writable_path


def validate_shared_models_directory(path_text: str) -> SharedDirectoryStatus:
    """Probe a shared models directory path on the local filesystem.

    Checks that the path exists, is a directory, and is writable (by creating
    and removing a probe file). Returns a status instead of raising so callers
    can report the outcome to the cluster.
    """
    path = Path(path_text).expanduser()
    reported = str(path)
    if not _is_listable_directory(path):
        # exists() and is_dir() both raise on a dead mount, so distinguish
        # "missing" from "unreachable" without letting either escape.
        try:
            exists = path.exists()
        except OSError as stat_error:
            return SharedDirectoryStatus(
                valid=False,
                error=f"Cannot reach path: {stat_error.strerror or stat_error}",
                path=reported,
            )
        if not exists:
            return SharedDirectoryStatus(
                valid=False, error="Path does not exist", path=reported
            )
        return SharedDirectoryStatus(
            valid=False, error="Path is not a directory", path=reported
        )

    probe = path / f".exo-write-probe-{uuid.uuid4().hex}"
    writable = True
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        # Read-only is a normal state for a share of models: valid to load
        # from, excluded from download targets.
        writable = False

    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError:
        free_bytes = None
    return SharedDirectoryStatus(
        valid=True, free_bytes=free_bytes, path=reported, writable=writable
    )


def persist_shared_models_dir(path_text: str | None) -> None:
    """Persist the configured path so the setting survives restarts."""
    if path_text is None:
        EXO_SHARED_MODELS_DIR_FILE.unlink(missing_ok=True)
        return
    EXO_SHARED_MODELS_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXO_SHARED_MODELS_DIR_FILE.write_text(path_text)


def load_persisted_shared_models_dir() -> str | None:
    """Read the persisted path, if the node has one."""
    try:
        content = EXO_SHARED_MODELS_DIR_FILE.read_text().strip()
    except OSError:
        return None
    return content or None


def persist_shared_storage(storage: SharedStorage | None) -> None:
    """Persist the share definition so a master restart can re-announce it."""
    if storage is None:
        EXO_SHARED_STORAGE_FILE.unlink(missing_ok=True)
        return
    EXO_SHARED_STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXO_SHARED_STORAGE_FILE.write_text(storage.model_dump_json())


def load_persisted_shared_storage() -> SharedStorage | None:
    """Read the persisted share definition, if this node has one.

    A file written by a newer or broken version must not stop the node from
    starting, so anything unreadable is treated as "no share configured".
    """
    try:
        content = EXO_SHARED_STORAGE_FILE.read_text().strip()
    except OSError:
        return None
    if not content:
        return None
    try:
        return SharedStorage.model_validate_json(content)
    except ValidationError:
        return None


def resolve_shared_models_path(
    storage: SharedStorage | None,
    legacy_path: str | None,
    node_id: NodeId,
) -> str | None:
    """The path *this* node should use for shared models.

    A configured share is authoritative: the node uses its own entry, and a
    node with no entry simply has no shared storage. Falling back to the legacy
    single path there would silently point one node somewhere else, which is the
    class of mismatch the share exists to prevent. Without a share, the legacy
    path applies unchanged.
    """
    if storage is not None:
        return storage.path_for(node_id)
    return legacy_path


def _is_listable_directory(path: Path) -> bool:
    """Whether the path is a directory, treating unreachable mounts as not one.

    A dead network mount raises rather than answering — ESTALE on NFS, ENOTCONN
    on a dropped SMB share — and one such mount must not sink the whole listing.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def list_network_volumes() -> tuple[NetworkVolume, ...]:
    """Network filesystems mounted on this node, newest-style names included.

    These are the shares the node can actually reach — a share sitting on the
    LAN but not mounted here is invisible, because only a mount gives it a path
    the cluster can be pointed at.
    """
    try:
        partitions = psutil.disk_partitions(all=True)
    except OSError:
        return ()

    volumes: list[NetworkVolume] = []
    for partition in partitions:
        filesystem = partition.fstype.lower()
        if filesystem not in _NETWORK_FILESYSTEMS:
            continue
        mount_path = Path(partition.mountpoint)
        volumes.append(
            NetworkVolume(
                path=partition.mountpoint,
                source=partition.device,
                filesystem=partition.fstype,
                reachable=_is_listable_directory(mount_path),
            )
        )
    volumes.extend(_gvfs_network_volumes())
    return tuple(sorted(volumes, key=lambda volume: volume.path.lower()))


_GVFS_SMB_DIRECTORY = re.compile(
    r"smb-share:server=(?P<server>[^,]+),share=(?P<share>.+)"
)


def _gvfs_network_volumes() -> list[NetworkVolume]:
    """SMB shares attached through gvfs, which hide from the mount table.

    gvfs exposes one ``fuse.gvfsd-fuse`` mount per user and hangs every
    attached share underneath it as a directory, so a filesystem-type filter
    alone reports a node's own auto-mounted shares as not existing.
    """
    gvfs_root = Path(f"/run/user/{os.getuid()}/gvfs")
    try:
        children = list(gvfs_root.iterdir())
    except OSError:
        return []
    volumes: list[NetworkVolume] = []
    for child in children:
        match = _GVFS_SMB_DIRECTORY.fullmatch(child.name)
        if match is None:
            continue
        volumes.append(
            NetworkVolume(
                path=str(child),
                source=f"//{match.group('server')}/{match.group('share')}",
                filesystem="smb (gvfs)",
                reachable=_is_listable_directory(child),
            )
        )
    return volumes


def _browse_shortcut_entries() -> tuple[SharedModelsDirectoryBrowseEntry, ...]:
    """Shortcuts the picker opens on: the filesystem root, home, and mounts.

    These are a starting point, not a boundary — the root is listed first so
    every directory on the node is reachable by walking down from it, and the
    rest are shortcuts to where network drives usually land.
    """
    entries: list[SharedModelsDirectoryBrowseEntry] = []
    seen: set[str] = set()

    def add(path: Path, name: str | None = None) -> None:
        resolved = str(path)
        if resolved in seen or not _is_listable_directory(path):
            return
        seen.add(resolved)
        entries.append(
            SharedModelsDirectoryBrowseEntry(
                name=name or path.name or resolved,
                path=resolved,
                hidden=path.name.startswith("."),
            )
        )

    filesystem_root = Path(Path.home().anchor or "/")
    add(filesystem_root, name=f"{filesystem_root} (whole filesystem)")
    add(Path.home(), name=f"{Path.home().name} (home)")
    for candidate in _BROWSE_ROOT_CANDIDATES:
        if not _is_listable_directory(candidate):
            continue
        add(candidate)
        try:
            children = sorted(candidate.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                continue
            add(child)
            if len(entries) >= _BROWSE_ENTRY_LIMIT:
                return tuple(entries)
    return tuple(entries)


def browse_shared_models_directories(
    path_text: str | None = None,
    include_hidden: bool = False,
) -> SharedModelsDirectoryBrowseResult:
    """List child directories for the shared-models folder picker.

    An empty path returns shortcuts — the filesystem root first, then home and
    any mount points — from which any directory on the node can be reached by
    walking down. Hidden directories are listed only when ``include_hidden`` is
    set, so caches like ``~/.cache/huggingface`` stay reachable without
    cluttering the common case. Only directory names are returned — never file
    contents.
    """
    network_volumes = list_network_volumes()
    if path_text is None or path_text.strip() == "":
        return SharedModelsDirectoryBrowseResult(
            path="",
            parent_path=None,
            entries=_browse_shortcut_entries(),
            network_volumes=network_volumes,
        )

    path = Path(path_text).expanduser()
    try:
        path = path.resolve(strict=False)
    except OSError as resolve_error:
        return SharedModelsDirectoryBrowseResult(
            path=path_text,
            parent_path=None,
            entries=(),
            error=f"Invalid path: {resolve_error}",
            network_volumes=network_volumes,
        )

    try:
        path_exists = path.exists()
    except OSError as stat_error:
        return SharedModelsDirectoryBrowseResult(
            path=str(path),
            parent_path=str(path.parent) if path.parent != path else None,
            entries=(),
            error=f"Cannot reach path: {stat_error.strerror or stat_error}",
            network_volumes=network_volumes,
        )

    if not path_exists:
        return SharedModelsDirectoryBrowseResult(
            path=str(path),
            parent_path=str(path.parent) if path.parent != path else None,
            entries=(),
            error="Path does not exist",
            network_volumes=network_volumes,
        )
    if not _is_listable_directory(path):
        return SharedModelsDirectoryBrowseResult(
            path=str(path),
            parent_path=str(path.parent) if path.parent != path else None,
            entries=(),
            error="Path is not a directory",
            network_volumes=network_volumes,
        )

    entries: list[SharedModelsDirectoryBrowseEntry] = []
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError as list_error:
        return SharedModelsDirectoryBrowseResult(
            path=str(path),
            parent_path=str(path.parent) if path.parent != path else None,
            entries=(),
            error=f"Cannot list directory: {list_error.strerror or list_error}",
            network_volumes=network_volumes,
        )

    truncated = False
    for child in children:
        hidden = child.name.startswith(".")
        if hidden and not include_hidden:
            continue
        if not _is_listable_directory(child):
            continue
        if len(entries) >= _BROWSE_ENTRY_LIMIT:
            truncated = True
            break
        entries.append(
            SharedModelsDirectoryBrowseEntry(
                name=child.name, path=str(child), hidden=hidden
            )
        )

    parent = path.parent
    parent_path = str(parent) if parent != path else None
    return SharedModelsDirectoryBrowseResult(
        path=str(path),
        parent_path=parent_path,
        entries=tuple(entries),
        truncated=truncated,
        network_volumes=network_volumes,
    )
