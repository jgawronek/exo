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
import shutil
import uuid
from pathlib import Path
from typing import final

from exo.shared.constants import EXO_SHARED_MODELS_DIR_FILE
from exo.shared.environment import get_compatible_environment_value
from exo.shared.types.storage import SharedDirectoryStatus

_SHARED_MODELS_DIR_ENVIRONMENT_NAME = "XEO_SHARED_MODELS_DIR"


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

    @property
    def path(self) -> Path | None:
        return self._path

    def set(self, path: Path | None) -> None:
        self._path = path
        # Export for runner subprocesses spawned after this point.
        if path is None:
            os.environ.pop(_SHARED_MODELS_DIR_ENVIRONMENT_NAME, None)
        else:
            os.environ[_SHARED_MODELS_DIR_ENVIRONMENT_NAME] = str(path)


_holder = _SharedModelsDirectoryHolder()


def get_shared_models_dir() -> Path | None:
    """The validated shared models directory for this process, if any."""
    return _holder.path


def set_shared_models_dir(path: Path | None) -> None:
    """Install (or clear) the validated shared models directory.

    Only call this with a path that passed
    :func:`validate_shared_models_directory` on this node.
    """
    _holder.set(path)


def validate_shared_models_directory(path_text: str) -> SharedDirectoryStatus:
    """Probe a shared models directory path on the local filesystem.

    Checks that the path exists, is a directory, and is writable (by creating
    and removing a probe file). Returns a status instead of raising so callers
    can report the outcome to the cluster.
    """
    path = Path(path_text).expanduser()
    if not path.exists():
        return SharedDirectoryStatus(valid=False, error="Path does not exist")
    if not path.is_dir():
        return SharedDirectoryStatus(valid=False, error="Path is not a directory")

    probe = path / f".exo-write-probe-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as write_error:
        return SharedDirectoryStatus(
            valid=False, error=f"Not writable: {write_error.strerror or write_error}"
        )

    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError:
        free_bytes = None
    return SharedDirectoryStatus(valid=True, free_bytes=free_bytes)


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
