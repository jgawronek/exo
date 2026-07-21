import json
from typing import cast

from exo.shared.types.common import Host
from exo.worker.engines.mlx.utils_mlx import build_ring_hostfile_json


def test_single_connection_hostfile_stays_flat() -> None:
    hosts = [
        Host(ip="0.0.0.0", port=50000),
        Host(ip="10.0.0.2", port=50000),
    ]
    parsed = cast(
        object, json.loads(build_ring_hostfile_json(hosts, connections_per_host=1))
    )
    assert parsed == ["0.0.0.0:50000", "10.0.0.2:50000"]


def test_multi_connection_hostfile_expands_consecutive_ports() -> None:
    hosts = [
        Host(ip="0.0.0.0", port=50000),
        Host(ip="10.0.0.2", port=50000),
        Host(ip="198.51.100.1", port=0),
    ]
    parsed = cast(
        object, json.loads(build_ring_hostfile_json(hosts, connections_per_host=3))
    )
    assert parsed == [
        ["0.0.0.0:50000", "0.0.0.0:50001", "0.0.0.0:50002"],
        ["10.0.0.2:50000", "10.0.0.2:50001", "10.0.0.2:50002"],
        ["198.51.100.1:0", "198.51.100.1:0", "198.51.100.1:0"],
    ]
