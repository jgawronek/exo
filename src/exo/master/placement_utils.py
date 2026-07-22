import itertools
from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


# Approximate GPU memory bandwidth (GB/s) by chip/GPU name substring. Decode
# is memory-bandwidth bound, so pipeline stage time is proportional to the
# bytes of weights a node reads per token divided by its bandwidth. Substring
# order matters: more specific names (e.g. "M3 Ultra") must precede less
# specific ones (e.g. "M3").
_MEMORY_BANDWIDTH_GBPS_BY_CHIP_SUBSTRING: tuple[tuple[str, float], ...] = (
    ("M4 Ultra", 1092.0),
    ("M3 Ultra", 819.0),
    ("M2 Ultra", 800.0),
    ("M1 Ultra", 800.0),
    ("M4 Max", 546.0),
    ("M3 Max", 400.0),
    ("M2 Max", 400.0),
    ("M1 Max", 400.0),
    ("M4 Pro", 273.0),
    ("M3 Pro", 150.0),
    ("M2 Pro", 200.0),
    ("M1 Pro", 200.0),
    ("M4", 120.0),
    ("M3", 100.0),
    ("M2", 100.0),
    ("M1", 68.0),
    ("GB10", 273.0),  # NVIDIA DGX Spark
    ("RTX 5090", 1792.0),
    ("RTX 5080", 960.0),
    ("RTX 4090", 1008.0),
    ("RTX 4080", 717.0),
    ("RTX 3090", 936.0),
    ("RTX 3080", 760.0),
    ("RTX 8000", 672.0),  # Quadro RTX 8000 (Turing)
)


def estimate_memory_bandwidth_gigabytes_per_second(
    node_identity: NodeIdentity,
) -> float | None:
    """Estimated GPU memory bandwidth for a node, or None when unrecognised."""
    chip_name = node_identity.chip_id.lower()
    for chip_substring, bandwidth in _MEMORY_BANDWIDTH_GBPS_BY_CHIP_SUBSTRING:
        if chip_substring.lower() in chip_name:
            return bandwidth
    return None


def allocate_layers_by_throughput(
    total_layers: int,
    node_throughputs: list[float],
    max_layers_per_node: list[int],
) -> list[int]:
    """Split layers to minimise summed per-stage decode time.

    Per-token pipeline decode latency is the sum of every stage's compute
    time, and stage time is (layers on node) / (node throughput), so the sum
    is minimised by loading the fastest nodes to their memory capacity first.
    Every node keeps at least one layer (a pipeline stage cannot be empty).
    Raises ValueError when allocation is impossible; handled by the placement
    caller (``place_instance``) which surfaces it to the API.
    """
    n = len(node_throughputs)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )
    if any(cap < 1 for cap in max_layers_per_node):
        raise ValueError(
            "Every pipeline node must have memory capacity for at least one layer"
        )
    if sum(max_layers_per_node) < total_layers:
        raise ValueError(
            f"Selected nodes only have capacity for {sum(max_layers_per_node)} of "
            f"{total_layers} layers"
        )

    result = [1] * n
    remaining = total_layers - n
    for i in sorted(range(n), key=lambda i: node_throughputs[i], reverse=True):
        take = min(max_layers_per_node[i] - result[i], remaining)
        result[i] += take
        remaining -= take
        if remaining == 0:
            break
    assert remaining == 0
    return result


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
    max_layers_per_node: list[int] | None = None,
) -> list[int]:
    """Split layers across nodes proportionally to their memory fractions.

    ``max_layers_per_node`` caps how many layers each node may receive (how
    many fit in its available memory). Without caps, largest-remainder
    rounding can hand a leftover layer to a node that has no memory slack
    for it. Raises ValueError when allocation is impossible; handled by the
    placement caller (``place_instance``) which surfaces it to the API.
    """
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )
    caps = (
        max_layers_per_node if max_layers_per_node is not None else [total_layers] * n
    )
    assert len(caps) == n
    if any(cap < 1 for cap in caps):
        raise ValueError(
            "A selected pipeline node has insufficient memory to hold even one layer"
        )
    if sum(caps) < total_layers:
        raise ValueError(
            f"Selected nodes only have capacity for {sum(caps)} of "
            f"{total_layers} layers"
        )

    # Largest remainder: floor each (capped), then hand out the remaining
    # layers by fractional part, skipping nodes that are at capacity.
    raw = [fraction * total_layers for fraction in memory_fractions]
    result = [min(int(r), cap) for r, cap in zip(raw, caps, strict=True)]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    remaining = total_layers - sum(result)
    while remaining > 0:
        for i in by_remainder:
            if remaining == 0:
                break
            if result[i] < caps[i]:
                result[i] += 1
                remaining -= 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Memory:
    total_memory = sum(
        (node_memory[node_id].ram_available for node_id in node_ids),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> list[int]:
    max_layers_per_node = [
        (node_memory[node_id].ram_available.in_bytes * model_card.n_layers)
        // model_card.storage_size.in_bytes
        for node_id in node_ids
    ]

    node_bandwidths = [
        estimate_memory_bandwidth_gigabytes_per_second(
            (node_identities or {}).get(node_id, NodeIdentity())
        )
        for node_id in node_ids
    ]

    if all(bandwidth is not None for bandwidth in node_bandwidths):
        # Decode throughput is bounded by the sum of per-stage times, so load
        # the highest-bandwidth nodes first (capped by their memory).
        layer_allocations = allocate_layers_by_throughput(
            total_layers=model_card.n_layers,
            node_throughputs=[
                bandwidth for bandwidth in node_bandwidths if bandwidth is not None
            ],
            max_layers_per_node=max_layers_per_node,
        )
    else:
        # Unknown hardware: fall back to memory-proportional allocation.
        layer_allocations = allocate_layers_proportionally(
            total_layers=model_card.n_layers,
            memory_fractions=[
                node_memory[node_id].ram_available / total_memory
                for node_id in node_ids
            ],
            max_layers_per_node=max_layers_per_node,
        )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )

    return layer_allocations


def _validate_manual_layer_allocations(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    node_layers: Mapping[NodeId, int],
) -> list[int]:
    if set(node_layers) != set(node_ids):
        raise ValueError(
            "Manual layer allocation must specify exactly the selected pipeline nodes"
        )
    if any(layer_count < 1 for layer_count in node_layers.values()):
        raise ValueError(
            "Manual layer allocations must assign at least one layer per node"
        )
    if sum(node_layers.values()) != model_card.n_layers:
        raise ValueError(
            f"Manual layer allocations must sum to {model_card.n_layers} layers"
        )

    allocations = [node_layers[node_id] for node_id in node_ids]
    for index, (node_id, layer_count) in enumerate(
        zip(node_ids, allocations, strict=True)
    ):
        required_memory = (model_card.storage_size * layer_count) // model_card.n_layers
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {index} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {layer_count} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )
    return allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_layers: Mapping[NodeId, int] | None = None,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        if node_layers is not None:
            raise ValueError(
                "Manual layer allocation is not supported for CFG-parallel models"
            )
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(
            model_card, cycle, node_memory, node_layers, node_identities
        )


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_layers: Mapping[NodeId, int] | None = None,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory)

    layer_allocations = (
        _allocate_and_validate_layers(
            cycle.node_ids, node_memory, total_memory, model_card, node_identities
        )
        if node_layers is None
        else _validate_manual_layer_allocations(
            cycle.node_ids, node_memory, model_card, node_layers
        )
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        layer_count = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + layer_count,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_layers: Mapping[NodeId, int] | None = None,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
                node_layers=node_layers,
                node_identities=node_identities,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


# Nominal link speeds (Mb/s) used when the OS does not report a negotiated
# speed. Ordering preserves the previous ring preference:
# thunderbolt > maybe_ethernet > ethernet > wifi > unknown.
_NOMINAL_LINK_SPEED_MEGABITS: Mapping[str, int] = {
    "thunderbolt": 40_000,
    "maybe_ethernet": 10_000,
    "ethernet": 1_000,
    "wifi": 300,
    "unknown": 100,
}

# A single TCP stream saturates well below line rate on fast links, so the
# ring backend opens several parallel connections per neighbour when every
# ring link reports a measured speed at or above this threshold.
FAST_RING_LINK_MIN_MEGABITS = 25_000
FAST_RING_LINK_CONNECTIONS = 4


def _measured_link_speed_megabits(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> int | None:
    """Measured (not nominal) speed of the link ring hosts would pick, or None."""
    selected_ip = find_ip_prioritised(
        source_node_id, sink_node_id, cycle_digraph, node_network, ring=True
    )
    if selected_ip is None:
        return None
    sink_network = node_network.get(sink_node_id, NodeNetworkInfo())
    for interface in sink_network.interfaces:
        if interface.ip_address == selected_ip:
            return interface.link_speed_megabits
    return None


def get_ring_connections_per_host(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> int:
    """Parallel TCP connections per ring neighbour for an instance.

    Conservative: multiple connections are only used when every ring link in
    both directions has a measured (not nominal) speed of at least
    FAST_RING_LINK_MIN_MEGABITS, since extra connections on slow links only
    add ports and sockets without improving throughput.
    """
    world_size = len(selected_cycle)
    if world_size < 2:
        return 1
    for rank, node_id in enumerate(selected_cycle.node_ids):
        right_neighbor = selected_cycle.node_ids[(rank + 1) % world_size]
        for source, sink in ((node_id, right_neighbor), (right_neighbor, node_id)):
            speed = _measured_link_speed_megabits(
                source, sink, cycle_digraph, node_network
            )
            if speed is None or speed < FAST_RING_LINK_MIN_MEGABITS:
                return 1
    return FAST_RING_LINK_CONNECTIONS


def _effective_link_speed_megabits(interface: NetworkInterfaceInfo | None) -> int:
    """Measured link speed when reported, otherwise a nominal per-type speed."""
    if interface is None:
        return _NOMINAL_LINK_SPEED_MEGABITS["unknown"]
    if interface.link_speed_megabits is not None:
        return interface.link_speed_megabits
    return _NOMINAL_LINK_SPEED_MEGABITS.get(
        interface.interface_type, _NOMINAL_LINK_SPEED_MEGABITS["unknown"]
    )


def effective_hop_speed_megabits(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> int | None:
    """Effective speed of the link ring host selection would pick for a hop.

    Returns None when the nodes have no socket connection at all.
    """
    selected_ip = find_ip_prioritised(
        source_node_id, sink_node_id, cycle_digraph, node_network, ring=True
    )
    if selected_ip is None:
        return None
    sink_network = node_network.get(sink_node_id, NodeNetworkInfo())
    for interface in sink_network.interfaces:
        if interface.ip_address == selected_ip:
            return _effective_link_speed_megabits(interface)
    return _effective_link_speed_megabits(None)


def order_cycle_for_fastest_links(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> Cycle:
    """Reorder ring nodes to minimize the total per-byte wire cost of the ring.

    Cycle enumeration returns nodes in arbitrary order, and hop cost is
    order-sensitive: a pair of nodes with a direct 200GbE link only benefits
    when they are ring-adjacent. Every chunk of pipeline traffic crosses
    every hop, so the objective is the sum of reciprocal hop speeds — which
    both rewards putting fast links on hops and penalizes slow bottlenecks.
    Brute-forces all orderings (cycles are small); orderings with a
    disconnected hop are invalid. Keeps the original order unless another is
    strictly better, so placements stay stable.
    """
    node_ids = selected_cycle.node_ids
    world_size = len(node_ids)
    if world_size <= 2 or world_size > 8:
        return selected_cycle

    def hop_speed(a: NodeId, b: NodeId) -> int | None:
        speeds: list[int] = []
        for source, sink in ((a, b), (b, a)):
            speed = effective_hop_speed_megabits(
                source, sink, cycle_digraph, node_network
            )
            if speed is None or speed <= 0:
                return None
            speeds.append(speed)
        return min(speeds)

    def wire_cost(order: tuple[NodeId, ...]) -> float | None:
        total = 0.0
        for rank, node_id in enumerate(order):
            speed = hop_speed(node_id, order[(rank + 1) % len(order)])
            if speed is None:
                return None
            total += 1.0 / speed
        return total

    best_order = tuple(node_ids)
    best_cost = wire_cost(best_order)
    # Fix the first node: ring rotations are equivalent.
    for permutation in itertools.permutations(node_ids[1:]):
        order = (node_ids[0], *permutation)
        cost = wire_cost(order)
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_order, best_cost = order, cost

    return Cycle(node_ids=list(best_order))


def find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Ring links prefer the fastest interface: the negotiated link speed when
    the node reports one (Linux sysfs), otherwise a nominal per-type speed.
    RDMA coordinators prefer ethernet.
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_interface = {iface.ip_address: iface for iface in other_network.interfaces}

    if ring:
        return max(
            ips,
            key=lambda ip: _effective_link_speed_megabits(ip_to_interface.get(ip)),
        )

    # RDMA prefers ethernet coordinator
    priority = {
        "ethernet": 0,
        "wifi": 1,
        "unknown": 2,
        "maybe_ethernet": 3,
        "thunderbolt": 4,
    }

    def interface_type_for_ip(ip: str) -> str:
        interface = ip_to_interface.get(ip)
        return interface.interface_type if interface is not None else "unknown"

    return min(ips, key=lambda ip: priority.get(interface_type_for_ip(ip), 2))


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
