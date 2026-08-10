"""Tests for zero-configuration share attachment.

The subprocess-facing pieces are exercised on the cluster; these pin the pure
parts — URI parsing, gvfs path construction, and the parsers fed by real
captured output from avahi-browse and smbutil.
"""

import os
from pathlib import Path

import pytest

from exo.download.share_mounts import (
    ShareMountError,
    _gvfs_mount_point,  # pyright: ignore[reportPrivateUsage]
    parse_avahi_smb_output,
    parse_smb_uri,
    parse_smbutil_view_output,
)

AVAHI_OUTPUT = """\
+;enP7s7;IPv4;HOME;Microsoft Windows Network;local
=;enP7s7;IPv4;truenas;Microsoft Windows Network;local;truenas.local;10.0.10.191;445;
=;enP7s7;IPv6;HOME;Microsoft Windows Network;local;home.local;fe80::9ab7:85ff:fe25:6c1d;445;
=;enP7s7;IPv4;HOME;Microsoft Windows Network;local;home.local;10.0.10.44;445;
=;wlP9s9;IPv4;HOME;Microsoft Windows Network;local;home.local;10.0.10.44;445;
"""

SMBUTIL_OUTPUT = """\
Share                                           Type    Comments
-------------------------------
print$                                          Disk    Printer Drivers
models                                          Disk
IPC$                                            Pipe    IPC Service (Samba)
nobody                                          Disk    Home Directories
AIModels                                        Disk
"""


class TestParseSmbUri:
    def test_splits_host_and_share(self) -> None:
        assert parse_smb_uri("smb://10.0.10.44/AIModels") == (
            "10.0.10.44",
            "AIModels",
        )

    def test_tolerates_a_trailing_slash(self) -> None:
        assert parse_smb_uri("smb://home.local/models/") == ("home.local", "models")

    @pytest.mark.parametrize(
        "uri",
        [
            "nfs://10.0.10.44/export",  # exo cannot mount NFS unprivileged
            "smb://10.0.10.44",  # no share named
            "smb://user:pass@host/share",  # credentials are not supported
            "/mnt/models",  # a path is not a share
        ],
    )
    def test_rejects_what_cannot_be_auto_mounted(self, uri: str) -> None:
        with pytest.raises(ShareMountError):
            parse_smb_uri(uri)


class TestGvfsMountPoint:
    def test_lowercases_the_share_like_gvfs_does(self) -> None:
        point = _gvfs_mount_point("10.0.10.44", "AIModels")
        assert point == Path(
            f"/run/user/{os.getuid()}/gvfs/smb-share:server=10.0.10.44,share=aimodels"
        )


class TestParseAvahi:
    def test_finds_ipv4_servers_once_each(self) -> None:
        servers = parse_avahi_smb_output(AVAHI_OUTPUT)
        assert [(server.name, server.host) for server in servers] == [
            ("HOME", "10.0.10.44"),
            ("truenas", "10.0.10.191"),
        ]

    def test_decodes_escaped_names(self) -> None:
        line = "=;eth0;IPv4;Time\\032Capsule;Microsoft Windows Network;local;tc.local;10.0.0.9;445;"
        servers = parse_avahi_smb_output(line)
        assert servers[0].name == "Time Capsule"

    def test_garbage_yields_nothing(self) -> None:
        assert parse_avahi_smb_output("not avahi output\n\n") == ()


class TestParseSmbutilView:
    def test_keeps_disk_shares_and_drops_admin_ones(self) -> None:
        shares = parse_smbutil_view_output(SMBUTIL_OUTPUT, "10.0.10.44")
        assert [share.name for share in shares] == ["AIModels", "models", "nobody"]
        assert shares[0].uri == "smb://10.0.10.44/AIModels"
