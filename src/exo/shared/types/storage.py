from collections.abc import Mapping
from typing import final

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel


@final
class SharedDirectoryStatus(FrozenModel):
    """One node's verdict on the shared models storage.

    Each node probes the path it resolved locally (exists, is a directory,
    writable) and reports the outcome so the dashboard can show where the
    shared storage is usable. ``path`` is what this node actually probed,
    which under a share differs from node to node.
    """

    valid: bool
    error: str | None = None
    free_bytes: int | None = None
    path: str | None = None
    # Readable-but-not-writable is a normal state for a network share of
    # models: fine to load from, never a download target. Defaults True so
    # events from older nodes keep their old meaning.
    writable: bool = True


@final
class SharedStorage(FrozenModel):
    """A named store of models that every node reaches by its own local path.

    The cluster agrees on the *share*, not on a path. Each node maps the share
    to wherever it is mounted locally, so a Linux node reading an NFS mount at
    ``/mnt/models`` and a Mac reading the same export at ``/Volumes/AIModels``
    are both serving this one share. Requiring a single identical path across
    nodes is what makes shared storage unusable on mixed fleets, so nothing
    here compares one node's path to another's.

    ``source`` records where the share comes from (``nfs://host/export``,
    ``smb://host/share``) for display, and as the input a future version can
    mount automatically. It is advisory: a node with a ``mounts`` entry works
    whether or not the source is set.
    """

    share_id: str
    mounts: Mapping[NodeId, str] = {}
    source: str | None = None
    label: str | None = None
    # Monotonic edit counter. Nodes persist the share so any of them can
    # re-announce it after winning an election — but without an ordering, a
    # node holding last week's file can resurrect it over yesterday's edit.
    # Higher revision wins, everywhere.
    revision: int = 0

    def path_for(self, node_id: NodeId) -> str | None:
        """This node's local path for the share, if it has one."""
        return self.mounts.get(node_id)
