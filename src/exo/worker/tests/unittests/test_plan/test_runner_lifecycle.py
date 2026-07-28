from typing import Any

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import Shutdown
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
)
from exo.utils.keyed_backoff import KeyedBackoff
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


def test_plan_kills_runner_when_instance_missing():
    """
    If a local runner's instance is no longer present in state,
    plan() should return a Shutdown for that runner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances: dict[InstanceId, Instance] = {}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_kills_runner_when_sibling_failed():
    """
    If a sibling runner in the same instance has failed, the local runner
    should be shut down.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerFailed(error_message="boom", diagnostics=[]),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_creates_runner_when_missing_for_node():
    """
    If shard_assignments specify a runner for this node but we don't have
    a local supervisor yet, plan() should emit a CreateRunner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}  # nothing local yet
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    # We patched plan_mod.CreateRunner → CreateRunner
    assert isinstance(result, plan_mod.CreateRunner)
    assert result.instance_id == INSTANCE_1_ID
    assert isinstance(result.bound_instance, BoundInstance)
    assert result.bound_instance.instance is instance
    assert result.bound_instance.bound_runner_id == RUNNER_1_ID


def test_plan_does_not_create_runner_when_supervisor_already_present():
    """
    If we already have a local supervisor for the runner assigned to this node,
    plan() should not emit a CreateRunner again.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_does_not_create_runner_for_unassigned_node():
    """
    If this node does not appear in shard_assignments.node_to_runner,
    plan() should not try to create a runner on this node.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_2_ID: shard},
    )

    runners: dict[RunnerId, FakeRunnerSupervisor] = {}  # no local runners
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_defers_runner_creation_while_predecessor_terminates():
    """
    A replacement runner must wait for its predecessor's process to exit.
    The instance's ring listener port is fixed, so creating the replacement
    while the old process still holds it fails to bind (EADDRINUSE) and
    recycles into an endless loop.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    def make_plan(terminating: frozenset[InstanceId]):
        return plan_mod.plan(
            node_id=NODE_A,
            runners={},
            global_download_status={NODE_A: []},
            instances={INSTANCE_1_ID: instance},
            all_runners={},
            tasks={},
            input_chunk_buffer={},
            image_cache={},
            instance_backoff=KeyedBackoff(),
            download_backoff=KeyedBackoff(),
            terminating_instance_ids=terminating,
        )

    assert make_plan(frozenset({INSTANCE_1_ID})) is None
    # Once the predecessor is reaped, the replacement proceeds.
    assert isinstance(make_plan(frozenset()), plan_mod.CreateRunner)


def test_diverged_runner_statuses_finds_only_stale_entries() -> None:
    """A dropped status event must not strand an instance forever.

    Ring formation is sequenced on cluster-wide runner status, so if a
    transition is lost the last rank waits on a peer state that never
    arrives. The worker restates exactly the statuses state disagrees with.
    """
    from exo.shared.types.worker.runners import RunnerConnecting
    from exo.worker.main import diverged_runner_statuses

    stale, agreed, unknown = RunnerId(), RunnerId(), RunnerId()
    local = {
        stale: RunnerConnecting(),
        agreed: RunnerReady(prefill_server_port=None),
        unknown: RunnerConnecting(),
    }
    replicated = {
        stale: RunnerReady(prefill_server_port=None),
        agreed: RunnerReady(prefill_server_port=None),
    }

    diverged = diverged_runner_statuses(
        local_statuses=local, replicated_statuses=replicated
    )

    # The stale entry and the one state has never seen are republished; the
    # one already agreed on is left alone.
    assert set(diverged) == {stale, unknown}
    assert diverged[stale] == RunnerConnecting()
