from exo.shared.apply import (
    apply_node_shared_directory_status_updated,
    apply_shared_models_directory_set,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    NodeSharedDirectoryStatusUpdated,
    SharedModelsDirectorySet,
)
from exo.shared.types.state import State
from exo.shared.types.storage import SharedDirectoryStatus


def test_setting_path_updates_state() -> None:
    new_state = apply_shared_models_directory_set(
        SharedModelsDirectorySet(path="/mnt/models"), State()
    )
    assert new_state.shared_models_dir == "/mnt/models"
    assert new_state.shared_models_dir_statuses == {}


def test_changing_path_clears_old_statuses() -> None:
    node_id = NodeId()
    state = State(
        shared_models_dir="/mnt/old",
        shared_models_dir_statuses={node_id: SharedDirectoryStatus(valid=True)},
    )
    new_state = apply_shared_models_directory_set(
        SharedModelsDirectorySet(path="/mnt/new"), state
    )
    assert new_state.shared_models_dir == "/mnt/new"
    assert new_state.shared_models_dir_statuses == {}


def test_reannouncing_same_path_keeps_statuses() -> None:
    node_id = NodeId()
    state = State(
        shared_models_dir="/mnt/models",
        shared_models_dir_statuses={node_id: SharedDirectoryStatus(valid=True)},
    )
    new_state = apply_shared_models_directory_set(
        SharedModelsDirectorySet(path="/mnt/models"), state
    )
    assert new_state.shared_models_dir == "/mnt/models"
    assert node_id in new_state.shared_models_dir_statuses


def test_clearing_path_clears_statuses() -> None:
    node_id = NodeId()
    state = State(
        shared_models_dir="/mnt/models",
        shared_models_dir_statuses={node_id: SharedDirectoryStatus(valid=True)},
    )
    new_state = apply_shared_models_directory_set(
        SharedModelsDirectorySet(path=None), state
    )
    assert new_state.shared_models_dir is None
    assert new_state.shared_models_dir_statuses == {}


def test_node_status_recorded() -> None:
    node_id = NodeId()
    state = State(shared_models_dir="/mnt/models")
    status = SharedDirectoryStatus(valid=True, free_bytes=123)
    new_state = apply_node_shared_directory_status_updated(
        NodeSharedDirectoryStatusUpdated(node_id=node_id, status=status), state
    )
    assert new_state.shared_models_dir_statuses == {node_id: status}


def test_stale_node_status_ignored_when_no_path_configured() -> None:
    node_id = NodeId()
    status = SharedDirectoryStatus(valid=False, error="Path does not exist")
    new_state = apply_node_shared_directory_status_updated(
        NodeSharedDirectoryStatusUpdated(node_id=node_id, status=status), State()
    )
    assert new_state.shared_models_dir_statuses == {}
