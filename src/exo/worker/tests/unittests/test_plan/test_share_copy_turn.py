import exo.worker.plan as plan_mod
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import DownloadModel
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerConnected, RunnerLoading
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.plan import _share_copy_turn  # pyright: ignore[reportPrivateUsage]
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

SHARD_A = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
SHARD_B = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
NODE_TO_RUNNER = {NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
PRELOAD_RUNNERS = {RUNNER_1_ID: RunnerConnected(), RUNNER_2_ID: RunnerConnected()}


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
    assert _share_copy_turn(NODE_A, NODE_TO_RUNNER, MODEL_A_ID, status, PRELOAD_RUNNERS)
    assert not _share_copy_turn(
        NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status, PRELOAD_RUNNERS
    )


def test_in_flight_copy_blocks_everyone_else() -> None:
    status = {
        NODE_A: [_share_copy_ongoing(NODE_A, SHARD_A)],
        NODE_B: [_share_completed(NODE_B, SHARD_B)],
    }
    assert not _share_copy_turn(
        NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status, PRELOAD_RUNNERS
    )


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
    assert _share_copy_turn(NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status, PRELOAD_RUNNERS)


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
        NODE_A, NODE_TO_RUNNER, MODEL_A_ID, {NODE_A: [unflagged]}, PRELOAD_RUNNERS
    )
    assert not _share_copy_turn(
        NODE_B,
        NODE_TO_RUNNER,
        MODEL_A_ID,
        {
            NODE_A: [_share_completed(NODE_A, SHARD_A)],
            NODE_B: [unflagged.model_copy(update={"node_id": NODE_B})],
        },
        PRELOAD_RUNNERS,
    )


def test_connected_runner_still_issues_queued_share_copy() -> None:
    """A node whose serialized copy turn arrives after the pipeline connected
    must still be able to issue the copy — Idle-only gating deadlocks it."""
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: SHARD_A, RUNNER_2_ID: SHARD_B},
    )
    bound = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound, status=RunnerConnected())
    own_share_completed = DownloadCompleted(
        shard_metadata=SHARD_A,
        node_id=NODE_A,
        total=Memory(),
        model_directory="/mnt/share/models/model-a",
        on_share=True,
    )
    other_local_completed = DownloadCompleted(
        shard_metadata=SHARD_B,
        node_id=NODE_B,
        total=Memory(),
        model_directory="/local/models/model-a",
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={
            NODE_A: [own_share_completed],
            NODE_B: [other_local_completed],
        },
        instances={INSTANCE_1_ID: instance},
        all_runners={
            RUNNER_1_ID: RunnerConnected(),
            RUNNER_2_ID: RunnerConnected(),
        },
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        download_retry_backoff=KeyedBackoff(),
        shared_copy_source_root="/mnt/share/models",
    )

    assert isinstance(result, DownloadModel)
    assert result.shard_metadata.model_card.model_id == MODEL_A_ID


def test_node_already_loading_does_not_block_the_queue() -> None:
    """Regression: a node that proceeded to load without copying (e.g. its
    share never resolved a copy root) must not be treated as ahead in the
    queue — that livelocked every node behind it."""
    status = {
        NODE_A: [_share_completed(NODE_A, SHARD_A)],
        NODE_B: [_share_completed(NODE_B, SHARD_B)],
    }
    runners = {
        RUNNER_1_ID: RunnerLoading(layers_loaded=0, total_layers=1, source="share"),
        RUNNER_2_ID: RunnerConnected(),
    }
    # NODE_A is the lower id but is already loading from the share; NODE_B
    # must get the copy turn instead of waiting on it forever.
    assert _share_copy_turn(NODE_B, NODE_TO_RUNNER, MODEL_A_ID, status, runners)
