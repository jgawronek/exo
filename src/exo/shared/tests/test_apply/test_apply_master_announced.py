from exo.shared.apply import apply_master_announced
from exo.shared.types.common import NodeId
from exo.shared.types.events import MasterAnnounced
from exo.shared.types.state import State


def test_master_announced_sets_master_node_id() -> None:
    node_id = NodeId()
    new_state = apply_master_announced(MasterAnnounced(node_id=node_id), State())
    assert new_state.master_node_id == node_id


def test_master_announced_replaces_previous_master() -> None:
    old_master, new_master = NodeId(), NodeId()
    state = State(master_node_id=old_master)
    new_state = apply_master_announced(MasterAnnounced(node_id=new_master), state)
    assert new_state.master_node_id == new_master
