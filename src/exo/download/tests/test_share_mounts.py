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
    parse_nfs_uri,
    parse_showmount_output,
    parse_smb_mount_table,
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

    def test_tolerates_extra_slashes_from_a_prefixed_mount_source(self) -> None:
        # A mount table reports the source as //host/share; a naive smb://
        # prefix produced this exact value in saved configs.
        assert parse_smb_uri("smb:////10.0.10.44/aimodels") == (
            "10.0.10.44",
            "aimodels",
        )

    @pytest.mark.parametrize(
        "uri",
        [
            "nfs://10.0.10.44/export",  # wrong scheme for the SMB parser
            "smb://10.0.10.44",  # no share named
            "smb://user:pass@host/share",  # credentials are not supported
            "/mnt/models",  # a path is not a share
        ],
    )
    def test_rejects_what_cannot_be_auto_mounted(self, uri: str) -> None:
        with pytest.raises(ShareMountError):
            parse_smb_uri(uri)


class TestParseNfsUri:
    def test_splits_host_and_export(self) -> None:
        assert parse_nfs_uri("nfs://10.0.10.44/mnt/lexar4tb/exo-models") == (
            "10.0.10.44",
            "/mnt/lexar4tb/exo-models",
        )

    def test_tolerates_the_mount_table_colon_and_trailing_slash(self) -> None:
        # deriveShareSource prepends nfs:// to the mount table's host:/export.
        assert parse_nfs_uri("nfs://10.0.10.44:/mnt/lexar4tb/exo-models/") == (
            "10.0.10.44",
            "/mnt/lexar4tb/exo-models",
        )

    @pytest.mark.parametrize(
        "uri",
        [
            "nfs://10.0.10.44",  # no export named
            "smb://10.0.10.44/share",  # wrong scheme for the NFS parser
            "/mnt/models",  # a path is not a share
        ],
    )
    def test_rejects_what_it_cannot_resolve(self, uri: str) -> None:
        with pytest.raises(ShareMountError):
            parse_nfs_uri(uri)


LINUX_MOUNT_OUTPUT = """\
sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)
//10.0.10.44/aimodels on /mnt/aimodels type cifs (ro,relatime,vers=3.1.1)
10.0.10.44:/mnt/lexar4tb/exo-models on /mnt/exo-models type nfs4 (ro,relatime)
"""

DARWIN_MOUNT_OUTPUT = """\
/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
//guest:@10.0.10.44/AIModels on /Users/jayg/.exo/mounts/aimodels (smbfs, nodev)
"""


class TestParseSmbMountTable:
    def test_finds_a_linux_cifs_mount(self) -> None:
        found = parse_smb_mount_table(LINUX_MOUNT_OUTPUT, "10.0.10.44", "aimodels")
        assert found == Path("/mnt/aimodels")

    def test_finds_a_darwin_smbfs_mount_despite_user_and_case(self) -> None:
        found = parse_smb_mount_table(DARWIN_MOUNT_OUTPUT, "10.0.10.44", "aimodels")
        assert found == Path("/Users/jayg/.exo/mounts/aimodels")

    def test_other_shares_do_not_match(self) -> None:
        assert (
            parse_smb_mount_table(LINUX_MOUNT_OUTPUT, "10.0.10.44", "models") is None
        )


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

    def test_reassembles_multibyte_utf8_names(self) -> None:
        # A curly apostrophe arrives as three byte escapes.
        line = "=;en0;IPv4;jay\\226\\128\\153s Mac;Microsoft Windows Network;local;m.local;10.0.10.195;445;"
        servers = parse_avahi_smb_output(line)
        assert servers[0].name == "jay\u2019s Mac"

    def test_decodes_escaped_names(self) -> None:
        line = "=;eth0;IPv4;Time\\032Capsule;Microsoft Windows Network;local;tc.local;10.0.0.9;445;"
        servers = parse_avahi_smb_output(line)
        assert servers[0].name == "Time Capsule"

    def test_garbage_yields_nothing(self) -> None:
        assert parse_avahi_smb_output("not avahi output\n\n") == ()


SHOWMOUNT_OUTPUT = """\
Exports list on 10.0.10.44:
/mnt/lexar4tb/exo-models            10.0.10.0/24
/mnt/lexar4tb/huggingface/models    10.0.10.0/24
/mnt/lexar4tb/huggingface/incoming  10.0.10.0/24
"""


class TestParseShowmount:
    def test_extracts_exports_with_friendly_names(self) -> None:
        exports = parse_showmount_output(SHOWMOUNT_OUTPUT, "10.0.10.44")
        assert [(e.name, e.protocol) for e in exports] == [
            ("exo-models", "nfs"),
            ("incoming", "nfs"),
            ("models", "nfs"),
        ]
        assert exports[0].uri == "nfs://10.0.10.44/mnt/lexar4tb/exo-models"

    def test_headers_and_noise_are_ignored(self) -> None:
        assert parse_showmount_output("no exports here\n\n", "h") == ()


class TestParseSmbutilView:
    def test_keeps_disk_shares_and_drops_admin_ones(self) -> None:
        shares = parse_smbutil_view_output(SMBUTIL_OUTPUT, "10.0.10.44")
        assert [share.name for share in shares] == ["AIModels", "models", "nobody"]
        assert shares[0].uri == "smb://10.0.10.44/AIModels"
