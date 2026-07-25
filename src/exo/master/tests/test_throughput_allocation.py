import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    allocate_layers_by_measured_speed,
    allocate_layers_by_throughput,
    estimate_memory_bandwidth_gigabytes_per_second,
    find_ip_prioritised,
    live_rebalance_max_layers,
    plan_pipeline_layer_shift_steps,
    validate_live_rebalance_steps,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
    StageTiming,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


def test_allocate_layers_by_throughput_fills_fastest_first() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=12,
        node_throughputs=[819.0, 273.0, 800.0],
        max_layers_per_node=[4, 4, 8],
    )
    assert allocations == [4, 1, 7]
    assert sum(allocations) == 12


def test_allocate_layers_by_throughput_keeps_one_layer_per_node() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=3,
        node_throughputs=[1000.0, 1.0, 1.0],
        max_layers_per_node=[3, 3, 3],
    )
    assert allocations == [1, 1, 1]


def test_allocate_layers_by_throughput_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _ = allocate_layers_by_throughput(
            total_layers=10,
            node_throughputs=[100.0, 100.0],
            max_layers_per_node=[4, 4],
        )


def test_estimate_memory_bandwidth_matches_known_chips() -> None:
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Apple M3 Ultra")
        )
        == 819.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GB10")
        )
        == 273.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GeForce RTX 3090")
        )
        == 936.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Unknown Chip")
        )
        is None
    )


def _fully_connected_three_node_topology() -> tuple[Topology, NodeId, NodeId, NodeId]:
    node_a, node_b, node_c = NodeId(), NodeId(), NodeId()
    topology = Topology()
    for node_id in (node_a, node_b, node_c):
        topology.add_node(node_id)
    pairs = [
        (node_a, node_b),
        (node_b, node_c),
        (node_c, node_a),
        (node_b, node_a),
        (node_c, node_b),
        (node_a, node_c),
    ]
    for index, (source, sink) in enumerate(pairs):
        topology.add_connection(
            Connection(
                source=source, sink=sink, edge=create_socket_connection(index + 1)
            )
        )
    return topology, node_a, node_b, node_c


def test_pipeline_placement_uses_bandwidth_when_identities_known() -> None:
    topology, node_a, node_b, node_c = _fully_connected_three_node_topology()
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(1500),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    command = PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=3,
    )
    node_memory = {
        node_a: create_node_memory(500),
        node_b: create_node_memory(500),
        node_c: create_node_memory(1000),
    }
    node_network = {node_id: create_node_network() for node_id in node_memory}
    node_backends = {node_id: [Backend.MlxMetal] for node_id in node_memory}
    node_identities = {
        node_a: NodeIdentity(chip_id="Apple M3 Ultra"),
        node_b: NodeIdentity(chip_id="NVIDIA GB10"),
        node_c: NodeIdentity(chip_id="Apple M2 Ultra"),
    }

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))

    def layer_count(node_id: NodeId) -> int:
        runner_id = instance.shard_assignments.node_to_runner[node_id]
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        return shard.end_layer - shard.start_layer

    # Fastest nodes are filled to their memory caps first; the slow GB10
    # keeps the single layer every pipeline stage requires.
    assert layer_count(node_a) == 4
    assert layer_count(node_b) == 1
    assert layer_count(node_c) == 7


def _two_node_topology_with_two_links(
    thunderbolt_ip: str, ethernet_ip: str
) -> tuple[Topology, NodeId, NodeId]:
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    for ip in (thunderbolt_ip, ethernet_ip):
        last_octet = int(ip.rsplit(".", 1)[1])
        topology.add_connection(
            Connection(
                source=node_a,
                sink=node_b,
                edge=create_socket_connection(last_octet),
            )
        )
    return topology, node_a, node_b


def test_find_ip_prioritised_prefers_measured_link_speed_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                    link_speed_megabits=200_000,
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # 200 GbE with a measured speed beats thunderbolt's nominal 40 Gb/s.
    assert selected_ip == ethernet_ip


def test_ring_connections_scale_up_only_on_fast_measured_links() -> None:
    from exo.master.placement_utils import (
        FAST_RING_LINK_CONNECTIONS,
        get_ring_connections_per_host,
    )
    from exo.shared.types.topology import Cycle

    fast_ip_a, fast_ip_b = "169.254.0.8", "169.254.0.9"
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(9))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(8))
    )

    def network(speed: int | None) -> dict[NodeId, NodeNetworkInfo]:
        return {
            node_a: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="enp1",
                        ip_address=fast_ip_a,
                        interface_type="ethernet",
                        link_speed_megabits=speed,
                    )
                ]
            ),
            node_b: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="enp1",
                        ip_address=fast_ip_b,
                        interface_type="ethernet",
                        link_speed_megabits=speed,
                    )
                ]
            ),
        }

    cycle = Cycle(node_ids=[node_a, node_b])

    assert (
        get_ring_connections_per_host(cycle, topology, network(200_000))
        == FAST_RING_LINK_CONNECTIONS
    )
    assert get_ring_connections_per_host(cycle, topology, network(10_000)) == 1
    assert get_ring_connections_per_host(cycle, topology, network(None)) == 1


def test_ring_connections_scale_up_when_any_hop_is_fast() -> None:
    from exo.master.placement_utils import (
        FAST_RING_LINK_CONNECTIONS,
        get_ring_connections_per_host,
    )
    from exo.shared.types.topology import Cycle

    topology, node_network, _identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    # sparks adjacent: the 200G spark-spark hop is fast even though every
    # other hop runs over the slow LAN.
    mixed_cycle = Cycle(node_ids=[mac, spark_a, spark_b, pc_3090])
    assert (
        get_ring_connections_per_host(mixed_cycle, topology, node_network)
        == FAST_RING_LINK_CONNECTIONS
    )

    # without the fast spark link every hop is slow -> single connection.
    slow_topology, slow_network, _identities2, mac2, spark_a2, spark_b2, pc2 = (
        _heterogeneous_cluster_topology(spark_link_megabits=None)
    )
    slow_cycle = Cycle(node_ids=[mac2, spark_a2, spark_b2, pc2])
    assert get_ring_connections_per_host(slow_cycle, slow_topology, slow_network) == 1


def test_cycle_reordered_so_fast_link_is_a_ring_hop() -> None:
    from exo.master.placement_utils import order_cycle_for_fastest_links
    from exo.shared.types.topology import Cycle

    node_a, node_b, node_c, node_d = NodeId(), NodeId(), NodeId(), NodeId()
    fast_pair = {node_b, node_d}
    lan_ip_by_node = {
        node_a: "169.254.0.1",
        node_b: "169.254.0.2",
        node_c: "169.254.0.3",
        node_d: "169.254.0.4",
    }
    fast_ip_by_node = {node_b: "169.254.1.2", node_d: "169.254.1.4"}

    topology = Topology()
    for node_id in lan_ip_by_node:
        topology.add_node(node_id)
    for source in lan_ip_by_node:
        for sink in lan_ip_by_node:
            if source == sink:
                continue
            topology.add_connection(
                Connection(
                    source=source,
                    sink=sink,
                    edge=create_socket_connection(
                        int(lan_ip_by_node[sink].rsplit(".", 1)[1])
                    ),
                )
            )
    # Direct fast link between b and d, both directions.
    for source, sink in ((node_b, node_d), (node_d, node_b)):
        topology.add_connection(
            Connection(
                source=source,
                sink=sink,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(
                        address=f"/ip4/{fast_ip_by_node[sink]}/tcp/1234"
                    )
                ),
            )
        )

    def interfaces(node_id: NodeId) -> NodeNetworkInfo:
        node_interfaces = [
            NetworkInterfaceInfo(
                name="eth0",
                ip_address=lan_ip_by_node[node_id],
                interface_type="ethernet",
                link_speed_megabits=10_000,
            )
        ]
        if node_id in fast_pair:
            node_interfaces.append(
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address=fast_ip_by_node[node_id],
                    interface_type="ethernet",
                    link_speed_megabits=200_000,
                )
            )
        return NodeNetworkInfo(interfaces=node_interfaces)

    node_network = {node_id: interfaces(node_id) for node_id in lan_ip_by_node}

    # b and d are NOT adjacent in this order (a-b-c-d wraps d-a).
    unordered = Cycle(node_ids=[node_a, node_b, node_c, node_d])
    ordered = order_cycle_for_fastest_links(unordered, topology, node_network)

    position = {node_id: i for i, node_id in enumerate(ordered.node_ids)}
    distance = abs(position[node_b] - position[node_d])
    assert distance in (1, len(ordered) - 1), (
        f"fast pair should be ring-adjacent, got order {ordered.node_ids}"
    )
    assert set(ordered.node_ids) == set(unordered.node_ids)


def _heterogeneous_cluster_topology(
    lan_megabits: int = 10_000,
    spark_link_megabits: int | None = 200_000,
) -> tuple[
    Topology,
    dict[NodeId, NodeNetworkInfo],
    dict[NodeId, NodeIdentity],
    NodeId,
    NodeId,
    NodeId,
    NodeId,
]:
    """Fully-connected Mac + 2x DGX Spark + RTX 3090 cluster over one LAN.

    The Sparks optionally share a direct fast link (e.g. 200 GbE), mirroring
    the real heterogeneous cluster that motivated hop-aware placement.
    """
    mac, spark_a, spark_b, pc_3090 = NodeId(), NodeId(), NodeId(), NodeId()
    lan_ip_by_node = {
        mac: "169.254.0.1",
        spark_a: "169.254.0.2",
        spark_b: "169.254.0.3",
        pc_3090: "169.254.0.4",
    }
    fast_ip_by_node = {spark_a: "169.254.1.2", spark_b: "169.254.1.3"}

    topology = Topology()
    for node_id in lan_ip_by_node:
        topology.add_node(node_id)
    for source in lan_ip_by_node:
        for sink in lan_ip_by_node:
            if source == sink:
                continue
            topology.add_connection(
                Connection(
                    source=source,
                    sink=sink,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/{lan_ip_by_node[sink]}/tcp/1234"
                        )
                    ),
                )
            )
    if spark_link_megabits is not None:
        for source, sink in ((spark_a, spark_b), (spark_b, spark_a)):
            topology.add_connection(
                Connection(
                    source=source,
                    sink=sink,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/{fast_ip_by_node[sink]}/tcp/1234"
                        )
                    ),
                )
            )

    def node_interfaces(node_id: NodeId) -> NodeNetworkInfo:
        interfaces = [
            NetworkInterfaceInfo(
                name="eth0",
                ip_address=lan_ip_by_node[node_id],
                interface_type="ethernet",
                link_speed_megabits=lan_megabits,
            )
        ]
        if spark_link_megabits is not None and node_id in fast_ip_by_node:
            interfaces.append(
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address=fast_ip_by_node[node_id],
                    interface_type="ethernet",
                    link_speed_megabits=spark_link_megabits,
                )
            )
        return NodeNetworkInfo(interfaces=interfaces)

    node_network = {node_id: node_interfaces(node_id) for node_id in lan_ip_by_node}
    node_identities = {
        mac: NodeIdentity(chip_id="Apple M3 Ultra"),
        spark_a: NodeIdentity(chip_id="NVIDIA GB10"),
        spark_b: NodeIdentity(chip_id="NVIDIA GB10"),
        pc_3090: NodeIdentity(chip_id="NVIDIA GeForce RTX 3090"),
    }
    return topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090


def _pipeline_model_card(storage_bytes: int) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _pipeline_place_command(model_card: ModelCard, min_nodes: int = 1) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=min_nodes,
    )


def _placed_node_ids(
    command: PlaceInstance,
    topology: Topology,
    node_memory: dict[NodeId, MemoryUsage],
    node_network: dict[NodeId, NodeNetworkInfo],
    node_identities: dict[NodeId, NodeIdentity],
) -> set[NodeId]:
    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        {node_id: [Backend.MlxMetal] for node_id in node_memory},
        node_identities=node_identities,
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    return set(instance.shard_assignments.node_to_runner)


def test_placement_excludes_extra_node_when_smaller_cycle_fits() -> None:
    topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    node_memory = {
        mac: create_node_memory(500),
        spark_a: create_node_memory(300),
        spark_b: create_node_memory(300),
        pc_3090: create_node_memory(100),
    }

    placed = _placed_node_ids(
        _pipeline_place_command(_pipeline_model_card(storage_bytes=1000)),
        topology,
        node_memory,
        node_network,
        node_identities,
    )

    # Three nodes fit the model; the 3090 would only add a mandatory hop.
    assert placed == {mac, spark_a, spark_b}


def test_placement_prefers_fast_link_cycle_over_more_total_ram() -> None:
    topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    # The 3090 cycle has more total RAM, which the previous ranking rewarded.
    node_memory = {
        mac: create_node_memory(500),
        spark_a: create_node_memory(300),
        spark_b: create_node_memory(300),
        pc_3090: create_node_memory(400),
    }

    placed = _placed_node_ids(
        _pipeline_place_command(_pipeline_model_card(storage_bytes=1000)),
        topology,
        node_memory,
        node_network,
        node_identities,
    )

    # The Spark pair's 200 GbE ring hop beats the extra RAM of a 3090 cycle
    # whose hops are all on the 10 GbE LAN.
    assert placed == {mac, spark_a, spark_b}


def test_placement_includes_faster_gpu_when_links_are_equal() -> None:
    topology, node_network, node_identities, mac, _spark_a, _spark_b, pc_3090 = (
        _heterogeneous_cluster_topology(spark_link_megabits=None)
    )
    node_memory = {node_id: create_node_memory(400) for node_id in node_identities}

    placed = _placed_node_ids(
        _pipeline_place_command(_pipeline_model_card(storage_bytes=1000)),
        topology,
        node_memory,
        node_network,
        node_identities,
    )

    # With identical links everywhere, hop cost ties and the higher-bandwidth
    # GPUs win: the 936 GB/s 3090 and 819 GB/s Mac are both selected.
    assert len(placed) == 3
    assert {mac, pc_3090} <= placed


def test_placement_uses_all_nodes_when_memory_requires_them() -> None:
    topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    # No three nodes reach 1300 units, so all four are needed.
    node_memory = {node_id: create_node_memory(400) for node_id in node_identities}

    placed = _placed_node_ids(
        _pipeline_place_command(_pipeline_model_card(storage_bytes=1300)),
        topology,
        node_memory,
        node_network,
        node_identities,
    )

    assert placed == {mac, spark_a, spark_b, pc_3090}


def test_placement_respects_min_nodes_over_latency() -> None:
    topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    node_memory = {
        mac: create_node_memory(500),
        spark_a: create_node_memory(300),
        spark_b: create_node_memory(300),
        pc_3090: create_node_memory(100),
    }

    placed = _placed_node_ids(
        _pipeline_place_command(_pipeline_model_card(storage_bytes=1000), min_nodes=4),
        topology,
        node_memory,
        node_network,
        node_identities,
    )

    assert placed == {mac, spark_a, spark_b, pc_3090}


def test_estimate_penalises_extra_hop_more_than_it_rewards_compute() -> None:
    from exo.master.placement_utils import estimate_cycle_decode_seconds_per_token
    from exo.shared.types.topology import Cycle

    topology, node_network, node_identities, mac, spark_a, spark_b, pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    node_memory = {node_id: create_node_memory(500) for node_id in node_identities}
    model_card = _pipeline_model_card(storage_bytes=1000)

    three_node_estimate = estimate_cycle_decode_seconds_per_token(
        cycle=Cycle(node_ids=[mac, spark_a, spark_b]),
        cycle_digraph=topology,
        node_network=node_network,
        node_identities=node_identities,
        node_memory=node_memory,
        model_card=model_card,
    )
    four_node_estimate = estimate_cycle_decode_seconds_per_token(
        cycle=Cycle(node_ids=[mac, spark_a, spark_b, pc_3090]),
        cycle_digraph=topology,
        node_network=node_network,
        node_identities=node_identities,
        node_memory=node_memory,
        model_card=model_card,
    )

    assert three_node_estimate < four_node_estimate


def test_estimate_rewards_faster_ring_links() -> None:
    from exo.master.placement_utils import estimate_cycle_decode_seconds_per_token
    from exo.shared.types.topology import Cycle

    node_memory_value = 500
    estimates: list[float] = []
    for lan_megabits in (200_000, 10_000):
        topology, node_network, node_identities, mac, spark_a, spark_b, _pc_3090 = (
            _heterogeneous_cluster_topology(
                lan_megabits=lan_megabits, spark_link_megabits=None
            )
        )
        estimates.append(
            estimate_cycle_decode_seconds_per_token(
                cycle=Cycle(node_ids=[mac, spark_a, spark_b]),
                cycle_digraph=topology,
                node_network=node_network,
                node_identities=node_identities,
                node_memory={
                    node_id: create_node_memory(node_memory_value)
                    for node_id in node_identities
                },
                model_card=_pipeline_model_card(storage_bytes=1000),
            )
        )

    fast_lan_estimate, slow_lan_estimate = estimates
    assert fast_lan_estimate < slow_lan_estimate


def test_estimate_falls_back_for_unknown_chips() -> None:
    from exo.master.placement_utils import estimate_cycle_decode_seconds_per_token
    from exo.shared.types.topology import Cycle

    topology, node_network, _identities, mac, spark_a, spark_b, _pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    unknown_identities = {
        node_id: NodeIdentity(chip_id="Mystery Chip")
        for node_id in (mac, spark_a, spark_b)
    }

    estimate = estimate_cycle_decode_seconds_per_token(
        cycle=Cycle(node_ids=[mac, spark_a, spark_b]),
        cycle_digraph=topology,
        node_network=node_network,
        node_identities=unknown_identities,
        node_memory={
            node_id: create_node_memory(500) for node_id in unknown_identities
        },
        model_card=_pipeline_model_card(storage_bytes=1000),
    )

    assert 0.0 < estimate < float("inf")


def test_estimate_is_infinite_for_cycles_that_cannot_hold_the_model() -> None:
    from exo.master.placement_utils import estimate_cycle_decode_seconds_per_token
    from exo.shared.types.topology import Cycle

    topology, node_network, node_identities, mac, spark_a, spark_b, _pc_3090 = (
        _heterogeneous_cluster_topology()
    )

    estimate = estimate_cycle_decode_seconds_per_token(
        cycle=Cycle(node_ids=[mac, spark_a, spark_b]),
        cycle_digraph=topology,
        node_network=node_network,
        node_identities=node_identities,
        node_memory={node_id: create_node_memory(10) for node_id in node_identities},
        model_card=_pipeline_model_card(storage_bytes=1000),
    )

    assert estimate == float("inf")


def _mixture_of_experts_card(
    storage_bytes: int, total_experts: int, active_experts: int
) -> ModelCard:
    from exo.shared.models.model_cards import MixtureOfExpertsInfo

    return _pipeline_model_card(storage_bytes=storage_bytes).model_copy(
        update={
            "mixture_of_experts": MixtureOfExpertsInfo(
                routed_experts_total=total_experts,
                routed_experts_active=active_experts,
            )
        }
    )


def test_decode_bytes_read_dense_model_reads_all_weights() -> None:
    from exo.master.placement_utils import estimate_decode_bytes_read_per_token

    assert (
        estimate_decode_bytes_read_per_token(_pipeline_model_card(storage_bytes=1000))
        == 1000.0
    )


def test_decode_bytes_read_moe_model_reads_active_share() -> None:
    from exo.master.placement_utils import estimate_decode_bytes_read_per_token

    card = _mixture_of_experts_card(
        storage_bytes=1000, total_experts=128, active_experts=8
    )
    routed_fraction = 8 / 128
    expected = 1000 * (routed_fraction + 0.05 * (1 - routed_fraction))
    assert abs(estimate_decode_bytes_read_per_token(card) - expected) < 1e-6


def test_estimate_moe_decode_is_faster_than_dense_of_same_size() -> None:
    from exo.master.placement_utils import estimate_cycle_decode_seconds_per_token
    from exo.shared.types.topology import Cycle

    topology, node_network, node_identities, mac, spark_a, spark_b, _pc_3090 = (
        _heterogeneous_cluster_topology()
    )
    node_memory = {node_id: create_node_memory(500) for node_id in node_identities}

    def estimate_for(model_card: ModelCard) -> float:
        return estimate_cycle_decode_seconds_per_token(
            cycle=Cycle(node_ids=[mac, spark_a, spark_b]),
            cycle_digraph=topology,
            node_network=node_network,
            node_identities=node_identities,
            node_memory=node_memory,
            model_card=model_card,
        )

    dense_estimate = estimate_for(_pipeline_model_card(storage_bytes=1000))
    moe_estimate = estimate_for(
        _mixture_of_experts_card(
            storage_bytes=1000, total_experts=128, active_experts=8
        )
    )
    assert moe_estimate < dense_estimate


def test_config_data_parses_routed_expert_aliases() -> None:
    from exo.shared.models.model_cards import ConfigData

    for alias in ("num_experts", "n_routed_experts", "num_local_experts"):
        config = ConfigData.model_validate(
            {"num_hidden_layers": 4, "num_experts_per_tok": 8, alias: 128}
        )
        info = config.mixture_of_experts
        assert info is not None
        assert info.routed_experts_total == 128
        assert info.routed_experts_active == 8


def test_config_data_treats_all_experts_active_as_dense() -> None:
    from exo.shared.models.model_cards import ConfigData

    config = ConfigData.model_validate(
        {"num_hidden_layers": 4, "num_experts_per_tok": 8, "num_experts": 8}
    )
    assert config.mixture_of_experts is None


def _stage_timing(layers_held: int, compute_ms_per_token: float) -> StageTiming:
    return StageTiming(
        layers_held=layers_held,
        compute_ms_per_token=compute_ms_per_token,
        communication_ms_per_token=1.0,
        tokens_measured=64,
    )


def _memory_usage(*, total_bytes: int, available_bytes: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=total_bytes,
        ram_available=available_bytes,
        swap_total=0,
        swap_available=0,
    )


def test_measured_speed_allocation_reserves_live_shift_memory() -> None:
    fast_node, slow_node = NodeId(), NodeId()
    model_card = _pipeline_model_card(storage_bytes=1000)

    allocations = allocate_layers_by_measured_speed(
        model_card=model_card,
        node_ids=[fast_node, slow_node],
        node_memory={
            fast_node: _memory_usage(total_bytes=1000, available_bytes=240),
            slow_node: _memory_usage(total_bytes=1000, available_bytes=500),
        },
        current_layers={fast_node: 7, slow_node: 3},
        stage_timings={
            fast_node: _stage_timing(layers_held=7, compute_ms_per_token=1.0),
            slow_node: _stage_timing(layers_held=3, compute_ms_per_token=4.0),
        },
    )

    assert allocations == {fast_node: 7, slow_node: 3}


def test_live_rebalance_reserves_one_transient_layer() -> None:
    assert (
        live_rebalance_max_layers(
            model_card=_pipeline_model_card(storage_bytes=1000),
            memory_usage=_memory_usage(total_bytes=1000, available_bytes=300),
            current_layer_count=5,
        )
        == 6
    )


def test_live_rebalance_rejects_zero_sized_model() -> None:
    with pytest.raises(ValueError, match="storage size must be positive"):
        _ = live_rebalance_max_layers(
            model_card=_pipeline_model_card(storage_bytes=0),
            memory_usage=_memory_usage(total_bytes=1000, available_bytes=1000),
            current_layer_count=1,
        )


def test_live_rebalance_rejects_unsafe_intermediate_step() -> None:
    node_a, node_b, node_c = NodeId(), NodeId(), NodeId()
    runner_a, runner_b, runner_c = RunnerId(), RunnerId(), RunnerId()
    model_card = _pipeline_model_card(storage_bytes=600)
    current_shards = {
        runner_a: PipelineShardMetadata(
            model_card=model_card,
            device_rank=0,
            world_size=3,
            start_layer=0,
            end_layer=4,
            n_layers=10,
        ),
        runner_b: PipelineShardMetadata(
            model_card=model_card,
            device_rank=1,
            world_size=3,
            start_layer=4,
            end_layer=5,
            n_layers=10,
        ),
        runner_c: PipelineShardMetadata(
            model_card=model_card,
            device_rank=2,
            world_size=3,
            start_layer=5,
            end_layer=10,
            n_layers=10,
        ),
    }
    steps = plan_pipeline_layer_shift_steps(
        current_shards,
        {runner_a: 1, runner_b: 1, runner_c: 8},
    )

    with pytest.raises(ValueError, match="step 3"):
        validate_live_rebalance_steps(
            model_card=model_card,
            node_ids=[node_a, node_b, node_c],
            node_to_runner={
                node_a: runner_a,
                node_b: runner_b,
                node_c: runner_c,
            },
            node_memory={
                node_a: _memory_usage(total_bytes=1000, available_bytes=1000),
                node_b: _memory_usage(total_bytes=320, available_bytes=300),
                node_c: _memory_usage(total_bytes=1000, available_bytes=1000),
            },
            current_layers={node_a: 4, node_b: 1, node_c: 5},
            steps=steps,
        )


def test_measured_speed_allocation_moves_layers_to_faster_node() -> None:
    node_a, node_b = NodeId(), NodeId()
    model_card = _pipeline_model_card(storage_bytes=1000)
    # Node A processed 5 layers in 1 ms (5 layers/ms); node B processed
    # 5 layers in 4 ms (1.25 layers/ms). Decode latency is the sum of stage
    # times, so A is loaded to capacity and B keeps the pipeline minimum.
    allocations = allocate_layers_by_measured_speed(
        model_card=model_card,
        node_ids=[node_a, node_b],
        node_memory={
            node_a: _memory_usage(total_bytes=2000, available_bytes=1000),
            node_b: _memory_usage(total_bytes=2000, available_bytes=1000),
        },
        current_layers={node_a: 5, node_b: 5},
        stage_timings={
            node_a: _stage_timing(layers_held=5, compute_ms_per_token=1.0),
            node_b: _stage_timing(layers_held=5, compute_ms_per_token=4.0),
        },
    )
    assert allocations == {node_a: 9, node_b: 1}


def test_measured_speed_allocation_respects_memory_cap_of_fast_node() -> None:
    node_a, node_b = NodeId(), NodeId()
    model_card = _pipeline_model_card(storage_bytes=1000)
    # A is measured faster but only has memory for 6 layers (100 bytes free
    # + 500 bytes reclaimable = 600 bytes = 6 of the 10 x 100-byte layers).
    allocations = allocate_layers_by_measured_speed(
        model_card=model_card,
        node_ids=[node_a, node_b],
        node_memory={
            node_a: create_node_memory(100),
            node_b: create_node_memory(1000),
        },
        current_layers={node_a: 5, node_b: 5},
        stage_timings={
            node_a: _stage_timing(layers_held=5, compute_ms_per_token=1.0),
            node_b: _stage_timing(layers_held=5, compute_ms_per_token=4.0),
        },
    )
    assert allocations == {node_a: 6, node_b: 4}


def test_measured_speed_allocation_requires_timing_for_every_node() -> None:
    node_a, node_b = NodeId(), NodeId()
    with pytest.raises(ValueError, match="No measured decode timing"):
        _ = allocate_layers_by_measured_speed(
            model_card=_pipeline_model_card(storage_bytes=1000),
            node_ids=[node_a, node_b],
            node_memory={
                node_a: create_node_memory(1000),
                node_b: create_node_memory(1000),
            },
            current_layers={node_a: 5, node_b: 5},
            stage_timings={
                node_a: _stage_timing(layers_held=5, compute_ms_per_token=1.0)
            },
        )


def test_measured_speed_allocation_counts_current_layers_as_reclaimable() -> None:
    node_a, node_b = NodeId(), NodeId()
    model_card = _pipeline_model_card(storage_bytes=1000)
    # Both nodes report 0 bytes free because the running instance holds all
    # the weights; the allocator may preserve each node's current share but
    # must not increase either node during a live shift.
    allocations = allocate_layers_by_measured_speed(
        model_card=model_card,
        node_ids=[node_a, node_b],
        node_memory={
            node_a: create_node_memory(0),
            node_b: create_node_memory(0),
        },
        current_layers={node_a: 5, node_b: 5},
        stage_timings={
            node_a: _stage_timing(layers_held=5, compute_ms_per_token=1.0),
            node_b: _stage_timing(layers_held=5, compute_ms_per_token=1.0),
        },
    )
    # Equal measured speed, but neither node has memory for more than its
    # current 5-layer share, so the split cannot move.
    assert allocations == {node_a: 5, node_b: 5}


def test_measured_speed_allocation_rejects_unusable_timing() -> None:
    node_a, node_b = NodeId(), NodeId()
    with pytest.raises(ValueError, match="unusable"):
        _ = allocate_layers_by_measured_speed(
            model_card=_pipeline_model_card(storage_bytes=1000),
            node_ids=[node_a, node_b],
            node_memory={
                node_a: create_node_memory(1000),
                node_b: create_node_memory(1000),
            },
            current_layers={node_a: 5, node_b: 5},
            stage_timings={
                node_a: _stage_timing(layers_held=5, compute_ms_per_token=0.0),
                node_b: _stage_timing(layers_held=5, compute_ms_per_token=1.0),
            },
        )


def test_find_ip_prioritised_falls_back_to_nominal_speeds_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # Without measured speeds the previous type preference is preserved.
    assert selected_ip == thunderbolt_ip
