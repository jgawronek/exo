from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgressData,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.plan import _share_copy_turn  # pyright: ignore[reportPrivateUsage]
from exo.worker.tests.constants import (
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import get_pipeline_shard_metadata

SHARD_A = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
SHARD_B = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
NODE_TO_RUNNER = {NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}


def _share_completed(node_id: NodeId, shard: ShardMetadata) -> DownloadCompleted:
    return DownloadCompleted(
        shard_metadata=shard,
        node_id=node_id,
        total=Memory(),
        model_directory="/mnt/share/model",
        on_share=True,
    )


def _share_copy_ongoing(node_id: NodeId, shard: ShardMetadata) -> DownloadOngoing:
    return DownloadOngoing(
        shard_metadata=shard,
        node_id=node_id,
        model_directory="/local/model",
        from_share=True,
        download_progress=DownloadProgressData(
            total=Memory(),
            downloaded=Memory(),
            downloaded_this_session=Memory(),
            completed_files=0,
            total_files=1,
            speed=0,
            eta_ms=0,
            files={},
        ),
    )


def test_lowest_waiting_node_goes_first() -> None:
    status = {
        NODE_A: [_share_completed(NODE_A, SHARD_A)],
        NODE_B: [_share_completed(NODE_B, SHARD_B)],
    }
    assert _share_copy_turn(NODE_A, NODE_TO_RUNNER, MODEL_A_ID, status)
    assert not _share_copy_turn(NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status)


def test_in_flight_copy_blocks_everyone_else() -> None:
    status = {
        NODE_A: [_share_copy_ongoing(NODE_A, SHARD_A)],
        NODE_B: [_share_completed(NODE_B, SHARD_B)],
    }
    assert not _share_copy_turn(NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status)


def test_next_node_proceeds_after_copy_completes_locally() -> None:
    status = {
        NODE_A: [
            DownloadCompleted(
                shard_metadata=SHARD_A,
                node_id=NODE_A,
                total=Memory(),
                model_directory="/local/model",
                on_share=False,
            )
        ],
        NODE_B: [_share_completed(NODE_B, SHARD_B)],
    }
    assert _share_copy_turn(NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status)


def test_unflagged_own_status_does_not_deadlock() -> None:
    # Own completed event predates the on_share flag: permissive when no
    # flagged peers are waiting, deferential when they are.
    unflagged = DownloadCompleted(
        shard_metadata=SHARD_A,
        node_id=NODE_A,
        total=Memory(),
        model_directory="/mnt/share/model",
    )
    assert _share_copy_turn(
        NODE_A, NODE_TO_RUNNER, MODEL_A_ID, {NODE_A: [unflagged]}
    )
    assert not _share_copy_turn(
        NODE_B,
        NODE_TO_RUNNER,
        MODEL_A_ID,
        {
            NODE_A: [_share_completed(NODE_A, SHARD_A)],
            NODE_B: [unflagged.model_copy(update={"node_id": NODE_B})],
        },
    )
