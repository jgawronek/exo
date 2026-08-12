from exo.master.placement_utils import find_ip_prioritised
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.topology import Connection, SocketConnection
from exo.utils.info_gatherer.system_info import parse_interface_types_from_networksetup

_STUDIO_STYLE_PORTS = """
Hardware Port: Ethernet
Device: en0
Ethernet Address: 11:22:33:44:55:66

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: 01:23:45:67:89:ab

Hardware Port: Thunderbolt 2
Device: en3
Ethernet Address: 01:23:45:67:89:ac

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 01:23:45:67:89:ad

Hardware Port: iPhone USB
Device: en7
Ethernet Address: 01:23:45:67:89:ae
"""


def test_thunderbolt_en2_plus_keeps_thunderbolt_type() -> None:
    types = parse_interface_types_from_networksetup(_STUDIO_STYLE_PORTS)

    assert types["en0"] == "ethernet"
    assert types["en1"] == "wifi"
    assert types["en2"] == "thunderbolt"
    assert types["en3"] == "thunderbolt"


def test_unidentified_en2_plus_is_maybe_ethernet() -> None:
    types = parse_interface_types_from_networksetup(_STUDIO_STYLE_PORTS)

    # iPhone USB is not Wi‑Fi / Ethernet / Thunderbolt → demoted.
    assert types["en7"] == "maybe_ethernet"


def test_ring_prefers_thunderbolt_nominal_over_maybe_ethernet() -> None:
    """Mac Studio TB (en2, 169.254) must beat a 10GbE-class maybe_ethernet hop."""
    thunderbolt_ip, lan_ip = "169.254.10.2", "10.0.0.2"
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    # Insert LAN socket edge first so discovery-order ties would formerly
    # prefer 10GbE when both scored as maybe_ethernet (10_000).
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address=f"/ip4/{lan_ip}/tcp/52415")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address=f"/ip4/{thunderbolt_ip}/tcp/52415")
            ),
        )
    )

    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address=lan_ip,
                    interface_type="maybe_ethernet",
                ),
                NetworkInterfaceInfo(
                    name="en2",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
            ]
        )
    }

    selected = find_ip_prioritised(
        node_a, node_b, topology, node_network, ring=True
    )
    assert selected == thunderbolt_ip
