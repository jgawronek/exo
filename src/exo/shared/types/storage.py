from typing import final

from exo.utils.pydantic_ext import FrozenModel


@final
class SharedDirectoryStatus(FrozenModel):
    """One node's verdict on the cluster-wide shared models directory.

    Each node probes the configured path locally (exists, is a directory,
    writable) and reports the outcome so the dashboard can show where the
    shared mount is usable.
    """

    valid: bool
    error: str | None = None
    free_bytes: int | None = None
