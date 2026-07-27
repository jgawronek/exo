from exo.shared.apply import apply_node_download_progress
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.worker.tests.constants import MODEL_A_ID, MODEL_B_ID


def test_apply_node_download_progress():
    state = State()
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    event = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event]}


def test_apply_two_node_download_progress():
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=2)
    event1 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )
    event2 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard2,
        total=Memory(),
    )
    state = State(downloads={NodeId("node-1"): [event1]})

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event2), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event1, event2]}


def test_zero_byte_pending_removes_download_record() -> None:
    # A zero-byte DownloadPending means "no download state": it must clear
    # an existing record (e.g. after model deletion) rather than being
    # stored, and must not create a record when none exists.
    node_id = NodeId("node-1")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    completed = DownloadCompleted(
        node_id=node_id,
        shard_metadata=shard,
        total=Memory(),
    )
    state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=completed), State()
    )
    assert len(state.downloads[node_id]) == 1

    reset = DownloadPending(node_id=node_id, shard_metadata=shard)
    state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=reset), state
    )
    assert state.downloads[node_id] == []

    state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=reset), state
    )
    assert state.downloads[node_id] == []
