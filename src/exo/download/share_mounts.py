"""Attach network shares to this node without configuration.

Selecting a shared folder should be enough — no usernames, passwords, sudo,
fstab entries, or per-node paths. Both platforms can mount a guest SMB share
unprivileged (``mount_smbfs`` on macOS, gvfs via ``gio`` on Linux), so a node
handed a share URI makes it reachable by itself. Only guest shares are
supported: they are the only kind mountable non-interactively without storing
credentials anywhere.

Discovery is best-effort — ``avahi-browse`` where present — and a host can
always be entered manually, so a server that does not announce itself is still
reachable.
"""

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import final

from exo.utils.pydantic_ext import FrozenModel

_COMMAND_TIMEOUT_SECONDS = 45
# Administrative shares (print$, IPC$) are never model storage.
_HIDDEN_SHARE_SUFFIX = "$"


@final
class ShareMountError(Exception):
    """A share could not be attached or enumerated."""


class NetworkServer(FrozenModel):
    host: str
    name: str


class NetworkShare(FrozenModel):
    name: str
    uri: str
    protocol: str = "smb"


def parse_smb_uri(uri: str) -> tuple[str, str]:
    """Split ``smb://host/share`` into ``(host, share)``.

    Extra slashes after the scheme are tolerated: mount tables report SMB
    sources as ``//host/share``, so a naive ``smb://`` prefix yields
    ``smb:////host/share``.
    """
    match = re.fullmatch(r"smb:/+([^/@]+)/([^/]+)/?", uri.strip())
    if match is None:
        raise ShareMountError(
            f"Not a share exo can attach: {uri!r} (expected smb://host/share)"
        )
    return match.group(1), match.group(2)


def _is_listable(path: Path) -> bool:
    try:
        next(iter(path.iterdir()), None)
        return path.is_dir()
    except OSError:
        return False


def _run(
    command: list[str],
    env: dict[str, str] | None = None,
    timeout: float = _COMMAND_TIMEOUT_SECONDS,
) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as run_error:
        raise ShareMountError(f"{command[0]}: {run_error}") from run_error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ShareMountError(detail or f"{command[0]} failed")
    return result.stdout


def _gvfs_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault(
        "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"
    )
    return environment


def _gvfs_mount_point(host: str, share: str) -> Path:
    # gvfs lowercases the share in its mount directory name.
    return Path(
        f"/run/user/{os.getuid()}/gvfs/smb-share:server={host},share={share.lower()}"
    )


def _mount_linux(host: str, share: str) -> Path:
    mount_point = _gvfs_mount_point(host, share)
    if _is_listable(mount_point):
        return mount_point
    try:
        _run(
            ["gio", "mount", "--anonymous", f"smb://{host}/{share}"],
            env=_gvfs_environment(),
        )
    except ShareMountError:
        # "Location is already mounted" and friends: trust the filesystem, not
        # the exit code.
        if not _is_listable(mount_point):
            raise
    if _is_listable(mount_point):
        return mount_point
    raise ShareMountError(f"Mounted smb://{host}/{share} but {mount_point} is empty")


def _mount_darwin(host: str, share: str) -> Path:
    mount_point = Path.home() / ".exo" / "mounts" / share
    if os.path.ismount(mount_point) and _is_listable(mount_point):
        return mount_point
    mount_point.mkdir(parents=True, exist_ok=True)
    _run(["/sbin/mount_smbfs", f"//guest:@{host}/{share}", str(mount_point)])
    if _is_listable(mount_point):
        return mount_point
    raise ShareMountError(f"Mounted //{host}/{share} but {mount_point} is empty")


def ensure_share_mounted(uri: str) -> Path:
    """Make the share reachable on this node and return its local path.

    Idempotent: an already-attached share returns its existing path without
    touching anything.
    """
    host, share = parse_smb_uri(uri)
    if sys.platform == "darwin":
        return _mount_darwin(host, share)
    return _mount_linux(host, share)


def _decode_avahi_escapes(value: str) -> str:
    # ``avahi-browse -p`` escapes each *byte* as a decimal ``\032`` sequence,
    # so multi-byte UTF-8 characters arrive as several escapes that must be
    # reassembled into bytes before decoding.
    with_bytes = re.sub(r"\\(\d{3})", lambda match: chr(int(match.group(1))), value)
    return with_bytes.encode("latin-1", "ignore").decode("utf-8", "replace")


def parse_avahi_smb_output(output: str) -> tuple[NetworkServer, ...]:
    """Extract IPv4 SMB servers from ``avahi-browse -prt _smb._tcp`` output."""
    servers: dict[str, NetworkServer] = {}
    for line in output.splitlines():
        fields = line.split(";")
        if len(fields) < 9 or fields[0] != "=" or fields[2] != "IPv4":
            continue
        address = fields[7]
        if address not in servers:
            servers[address] = NetworkServer(
                host=address, name=_decode_avahi_escapes(fields[3])
            )
    return tuple(sorted(servers.values(), key=lambda server: server.name.lower()))


def parse_smbutil_view_output(output: str, host: str) -> tuple[NetworkShare, ...]:
    """Extract disk shares from macOS ``smbutil view`` output."""
    shares: list[NetworkShare] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] != "Disk":
            continue
        name = parts[0]
        if name.endswith(_HIDDEN_SHARE_SUFFIX):
            continue
        shares.append(NetworkShare(name=name, uri=f"smb://{host}/{name}"))
    return tuple(sorted(shares, key=lambda share: share.name.lower()))


def discover_smb_servers() -> tuple[NetworkServer, ...]:
    """File servers announcing themselves on the LAN. Best effort."""
    if sys.platform == "darwin":
        # dns-sd never terminates on its own and browsing proved unreliable;
        # manual host entry covers macOS-served dashboards.
        return ()
    try:
        output = _run(["avahi-browse", "-prt", "_smb._tcp"])
    except ShareMountError:
        return ()
    return parse_avahi_smb_output(output)


def parse_showmount_output(output: str, host: str) -> tuple[NetworkShare, ...]:
    """Extract NFS exports from ``showmount -e`` output on either platform."""
    exports: list[NetworkShare] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("/"):
            continue  # headers and blank lines
        export_path = stripped.split()[0]
        name = export_path.rstrip("/").rsplit("/", 1)[-1] or export_path
        exports.append(
            NetworkShare(name=name, uri=f"nfs://{host}{export_path}", protocol="nfs")
        )
    return tuple(sorted(exports, key=lambda share: share.name.lower()))


def list_nfs_exports(host: str) -> tuple[NetworkShare, ...]:
    """NFS exports a server offers, whether or not anything mounts them."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
        raise ShareMountError(f"Not a hostname or address: {host!r}")
    showmount = "/usr/bin/showmount" if sys.platform == "darwin" else "showmount"
    output = _run([showmount, "-e", host], timeout=5)
    return parse_showmount_output(output, host)


def list_lan_shares(host: str) -> tuple[NetworkShare, ...]:
    """Everything a server offers: SMB shares and NFS exports together.

    Either protocol may be absent (no SMB service, no NFS server); the other
    still lists, and only both failing is an error worth surfacing.
    """
    shares: list[NetworkShare] = []
    errors: list[str] = []
    for lister in (list_smb_shares, list_nfs_exports):
        try:
            shares.extend(lister(host))
        except ShareMountError as list_error:
            errors.append(str(list_error))
    if not shares and errors:
        raise ShareMountError("; ".join(errors))
    return tuple(sorted(shares, key=lambda share: (share.name.lower(), share.protocol)))


def list_smb_shares(host: str) -> tuple[NetworkShare, ...]:
    """The guest-visible disk shares a server offers."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
        raise ShareMountError(f"Not a hostname or address: {host!r}")
    if sys.platform == "darwin":
        output = _run(["/usr/bin/smbutil", "view", "-g", f"//{host}"], timeout=10)
        return parse_smbutil_view_output(output, host)
    environment = _gvfs_environment()
    # Often already mounted; the listing below is the real test. A server
    # demanding credentials can stall the mount attempt, so keep it short —
    # the picker is waiting.
    with contextlib.suppress(ShareMountError):
        _run(
            ["gio", "mount", "--anonymous", f"smb://{host}/"],
            env=environment,
            timeout=10,
        )
    output = _run(["gio", "list", f"smb://{host}/"], env=environment, timeout=10)
    shares = [
        NetworkShare(name=name, uri=f"smb://{host}/{name}")
        for name in (line.strip() for line in output.splitlines())
        if name and not name.endswith(_HIDDEN_SHARE_SUFFIX)
    ]
    return tuple(sorted(shares, key=lambda share: share.name.lower()))
