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
    StageTiming,
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


# Bandwidth assumed for chips missing from the table above, so latency
# estimates stay computable on unrecognised hardware. Deliberately modest:
# unknown chips should not look like an upgrade over known ones.
_FALLBACK_MEMORY_BANDWIDTH_GBPS = 100.0

# Share of a mixture-of-experts model's weights that are read every token
# regardless of routing (attention, embeddings, norms, shared experts).
# Config files do not expose per-tensor sizes, so this is a fixed
# approximation; it reproduces published active/total parameter ratios well
# (Qwen3-30B-A3B: true 0.108 vs estimated 0.109; GLM-4.5-355B-A32B: true
# ~0.09 vs estimated ~0.10).
_MOE_DENSE_WEIGHT_SHARE = 0.05


def estimate_decode_bytes_read_per_token(model_card: ModelCard) -> float:
    """Weight bytes one decode step reads across the whole model.

    Dense models read every weight each token. Mixture-of-experts models
    read only the active routed experts plus the always-active dense share,
    which is why their per-token compute is far below their storage size.
    """
    total_bytes = float(model_card.storage_size.in_bytes)
    mixture_of_experts = model_card.mixture_of_experts
    if mixture_of_experts is None:
        return total_bytes
    routed_fraction = (
        mixture_of_experts.routed_experts_active
        / mixture_of_experts.routed_experts_total
    )
    active_fraction = routed_fraction + _MOE_DENSE_WEIGHT_SHARE * (
        1.0 - routed_fraction
    )
    return total_bytes * min(1.0, active_fraction)


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


def allocate_layers_by_measured_speed(
    model_card: ModelCard,
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    current_layers: Mapping[NodeId, int],
    stage_timings: Mapping[NodeId, StageTiming],
) -> dict[NodeId, int]:
    """Re-split pipeline layers using measured per-stage decode timings.

    Pipeline decode latency is the sum of every stage's compute time, so the
    split that minimises it loads the nodes with the fastest *measured*
    per-layer rate (layers per millisecond of decode) to capacity first —
    the same objective as ``allocate_layers_by_throughput``, but driven by
    live telemetry instead of the static bandwidth table. Equalising stage
    times would instead spread layers onto slower nodes and increase the
    per-token sum. Memory caps treat each node's share of the currently
    loaded instance as reclaimable, because rebalancing relaunches the
    instance.

    Raises ValueError when a node has no measured timing yet or the
    allocation is impossible; handled by the API rebalance endpoint, which
    surfaces it as HTTP 400.
    """
    layer_rates: list[float] = []
    for node_id in node_ids:
        timing = stage_timings.get(node_id)
        if timing is None:
            raise ValueError(
                f"No measured decode timing for node {node_id}; "
                "complete at least one generation before rebalancing"
            )
        if timing.layers_held < 1 or timing.compute_ms_per_token <= 0.0:
            raise ValueError(
                f"Measured decode timing for node {node_id} is unusable "
                f"(layers_held={timing.layers_held}, "
                f"compute_ms_per_token={timing.compute_ms_per_token})"
            )
        layer_rates.append(timing.layers_held / timing.compute_ms_per_token)

    max_layers_per_node: list[int] = []
    for node_id in node_ids:
        reclaimable = (
            model_card.storage_size * current_layers.get(node_id, 0)
        ) // model_card.n_layers
        usable = node_memory[node_id].ram_available + reclaimable
        max_layers_per_node.append(
            (usable.in_bytes * model_card.n_layers)
            // model_card.storage_size.in_bytes
        )

    allocations = allocate_layers_by_throughput(
        total_layers=model_card.n_layers,
        node_throughputs=layer_rates,
        max_layers_per_node=max_layers_per_node,
    )
    return dict(zip(node_ids, allocations, strict=True))


def plan_pipeline_layer_shift_steps(
    current_shards: Mapping[RunnerId, PipelineShardMetadata],
    target_layer_counts: Mapping[RunnerId, int],
) -> list[dict[RunnerId, PipelineShardMetadata]]:
    """Decompose a pipeline re-allocation into single-layer boundary shifts.

    Each returned step is a full runner-to-shard map that differs from its
    predecessor by exactly one layer moved between two adjacent ranks, and
    every rank keeps at least one layer at every intermediate step, so a
    running instance can apply the steps one at a time without ever emptying
    a pipeline stage. Returns an empty list when the target equals the
    current allocation.

    Raises ValueError on malformed input (mismatched runner sets, counts not
    summing to the model's layer count, or a rank left without layers);
    handled by the master's ShiftInstanceLayers command handler.
    """
    if set(current_shards) != set(target_layer_counts):
        raise ValueError(
            "Target layer counts must cover exactly the instance's runners"
        )
    if any(count < 1 for count in target_layer_counts.values()):
        raise ValueError("Every pipeline rank must keep at least one layer")

    ranked_runners = sorted(
        current_shards, key=lambda runner_id: current_shards[runner_id].device_rank
    )
    reference_shard = current_shards[ranked_runners[0]]
    total_layers = reference_shard.n_layers
    if sum(target_layer_counts.values()) != total_layers:
        raise ValueError(
            f"Target layer counts sum to {sum(target_layer_counts.values())}, "
            f"expected {total_layers}"
        )

    # Boundaries as cumulative layer counts, with fixed sentinels 0 and
    # total_layers; boundaries[i] is the split between rank i-1 and rank i.
    def cumulative_boundaries(counts: list[int]) -> list[int]:
        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)
        return boundaries

    current_boundaries = cumulative_boundaries(
        [
            current_shards[runner_id].end_layer - current_shards[runner_id].start_layer
            for runner_id in ranked_runners
        ]
    )
    target_boundaries = cumulative_boundaries(
        [target_layer_counts[runner_id] for runner_id in ranked_runners]
    )

    def snapshot(boundaries: list[int]) -> dict[RunnerId, PipelineShardMetadata]:
        return {
            runner_id: current_shards[runner_id].model_copy(
                update={
                    "start_layer": boundaries[rank],
                    "end_layer": boundaries[rank + 1],
                }
            )
            for rank, runner_id in enumerate(ranked_runners)
        }

    steps: list[dict[RunnerId, PipelineShardMetadata]] = []
    total_moves = sum(
        abs(current - target)
        for current, target in zip(current_boundaries, target_boundaries, strict=True)
    )
    for _ in range(total_moves):
        moved = False
        for i in range(1, len(current_boundaries) - 1):
            if current_boundaries[i] < target_boundaries[i] and (
                current_boundaries[i] + 1 < current_boundaries[i + 1]
            ):
                current_boundaries[i] += 1
            elif current_boundaries[i] > target_boundaries[i] and (
                current_boundaries[i] - 1 > current_boundaries[i - 1]
            ):
                current_boundaries[i] -= 1
            else:
                continue
            steps.append(snapshot(current_boundaries))
            moved = True
            break
        assert moved, "Layer shift planning stalled; boundaries are inconsistent"
    assert current_boundaries == target_boundaries
    return steps


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
# ring backend opens several parallel connections per neighbour when any
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

    The MLX ring backend requires every rank to hold the same number of
    connections to its left and right neighbour, which transitively forces
    one count for the whole ring - per-hop counts are not expressible.
    Multiple connections are therefore used when *any* ring hop is fast in
    both directions: fast hops (e.g. a 200G spark-spark link) need several
    streams to approach line rate during prefill, while the extra sockets
    on slow hops are close to free - transfers under 256 KiB (every decode
    payload) use a single socket, and larger prefill transfers split into
    parallel streams on the same interface without losing throughput.
    """
    world_size = len(selected_cycle)
    if world_size < 2:
        return 1
    for rank, node_id in enumerate(selected_cycle.node_ids):
        right_neighbor = selected_cycle.node_ids[(rank + 1) % world_size]
        hop_is_fast = all(
            (
                speed := _measured_link_speed_megabits(
                    source, sink, cycle_digraph, node_network
                )
            )
            is not None
            and speed >= FAST_RING_LINK_MIN_MEGABITS
            for source, sink in ((node_id, right_neighbor), (right_neighbor, node_id))
        )
        if hop_is_fast:
            return FAST_RING_LINK_CONNECTIONS
    return 1


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


# Fixed cost of one ring hop during decode: activation send (~0.4 ms
# measured), plus this rank's share of the per-token token-relay all_gather
# (3.8-7.7 ms per token across the whole ring in hop logs). Calibrated
# live on Qwen3.5-9B-4bit across ring sizes 1-4 (67.5 / 55.0 / 47.0 /
# 43.5 decode TPS), where each added node cost 1.7-3.4 ms per token.
# Decode payloads are a few KiB, so this constant dominates the per-hop
# term on any link faster than ~1 Gb/s.
RING_HOP_BASE_LATENCY_SECONDS = 0.002

# Decode activations are hidden_size elements of (b)float16 per token.
_DECODE_PAYLOAD_BYTES_PER_HIDDEN_ELEMENT = 2


def estimate_cycle_decode_seconds_per_token(
    cycle: Cycle,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_identities: Mapping[NodeId, NodeIdentity],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
) -> float:
    """Estimated per-token pipeline decode latency for a candidate cycle.

    Per-token pipeline decode latency is the sum of every stage's compute
    time plus the cost of every ring hop. Decode is memory-bandwidth bound,
    so stage time is the weight bytes a node reads per token divided by its
    memory bandwidth; hop cost is a fixed per-hop latency plus the decode
    payload over the hop's link speed. Placement uses this to rank candidate
    cycles, rewarding fast ring links and penalising slow hops (e.g. a node
    reachable only over 1 Gb Ethernet).

    Layer allocation mirrors what ``get_shard_assignments`` would produce via
    ``allocate_layers_by_throughput``. Its ValueError (a cycle whose nodes
    cannot hold the model's layers) is handled here by returning
    ``float("inf")``: such a cycle ranks behind every feasible one instead of
    aborting placement while alternatives remain.
    """
    ordered_cycle = order_cycle_for_fastest_links(cycle, cycle_digraph, node_network)
    node_ids = ordered_cycle.node_ids

    node_bandwidths = [
        estimate_memory_bandwidth_gigabytes_per_second(
            node_identities.get(node_id, NodeIdentity())
        )
        or _FALLBACK_MEMORY_BANDWIDTH_GBPS
        for node_id in node_ids
    ]
    max_layers_per_node = [
        (node_memory[node_id].ram_available.in_bytes * model_card.n_layers)
        // model_card.storage_size.in_bytes
        for node_id in node_ids
    ]
    try:
        layer_allocations = allocate_layers_by_throughput(
            total_layers=model_card.n_layers,
            node_throughputs=node_bandwidths,
            max_layers_per_node=max_layers_per_node,
        )
    except ValueError:
        return float("inf")

    bytes_read_per_layer = (
        estimate_decode_bytes_read_per_token(model_card) / model_card.n_layers
    )
    compute_seconds = sum(
        layer_count * bytes_read_per_layer / (bandwidth * 1e9)
        for layer_count, bandwidth in zip(
            layer_allocations, node_bandwidths, strict=True
        )
    )

    world_size = len(node_ids)
    if world_size == 1:
        return compute_seconds

    payload_bits = model_card.hidden_size * _DECODE_PAYLOAD_BYTES_PER_HIDDEN_ELEMENT * 8
    hop_seconds = 0.0
    for rank, node_id in enumerate(node_ids):
        right_neighbor = node_ids[(rank + 1) % world_size]
        directional_speeds = [
            speed
            for source, sink in ((node_id, right_neighbor), (right_neighbor, node_id))
            if (
                speed := effective_hop_speed_megabits(
                    source, sink, cycle_digraph, node_network
                )
            )
            is not None
            and speed > 0
        ]
        hop_speed_megabits = min(
            directional_speeds, default=_NOMINAL_LINK_SPEED_MEGABITS["unknown"]
        )
        hop_seconds += RING_HOP_BASE_LATENCY_SECONDS + payload_bits / (
            hop_speed_megabits * 1e6
        )

    return compute_seconds + hop_seconds


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
