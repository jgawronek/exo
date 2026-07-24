from datetime import UTC, datetime

from exo.shared.apply import apply_node_gathered_info
from exo.shared.types.common import NodeId
from exo.shared.types.events import EventId, NodeGatheredInfo
from exo.shared.types.profiling import SystemPerformanceProfile
from exo.shared.types.state import State
from exo.utils.info_gatherer.nvml_metrics import NvmlMetrics


def test_apply_nvml_metrics_updates_node_system():
    state = State()
    profile = SystemPerformanceProfile(
        gpu_usage=0.42,
        temp=67.0,
        sys_power=120.5,
    )
    event = NodeGatheredInfo(
        event_id=EventId(),
        node_id=NodeId("cuda-node"),
        when=datetime.now(UTC).isoformat(),
        info=NvmlMetrics(system_profile=profile),
    )

    new_state = apply_node_gathered_info(event, state)

    assert new_state.node_system[NodeId("cuda-node")] == profile
