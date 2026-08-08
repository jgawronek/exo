from typing import Any

import anyio

import exo.worker.plan as plan_mod
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, TaskStatusUpdated
from exo.shared.types.tasks import Shutdown, TaskId, TaskStatus
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
)
from exo.utils.channels import channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker
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
        download_retry_backoff=KeyedBackoff(),
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
        download_retry_backoff=KeyedBackoff(),
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
        download_retry_backoff=KeyedBackoff(),
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
        download_retry_backoff=KeyedBackoff(),
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
        download_retry_backoff=KeyedBackoff(),
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
            download_retry_backoff=KeyedBackoff(),
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


async def test_wedged_runner_is_reaped_before_a_replacement_is_planned():
    """
    A runner failed via the unresponsive/wedged path must hold its instance
    against replacement until its process has actually exited.

    The instance's ring listener port is fixed for its lifetime, and MLX's ring
    listener sets SO_REUSEPORT, so a replacement created while the predecessor
    still holds that port binds *successfully* alongside the zombie rather than
    failing. The kernel then splits inbound connections across both listeners
    and the ring cannot form. The graceful Shutdown path already defers; this
    covers the timeout path, which reaches teardown through a different branch.
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

    class FakeProcess:
        def __init__(self) -> None:
            self.exited = anyio.Event()

        async def wait(self) -> int:
            await self.exited.wait()
            return 0

    class FakeSupervisor:
        def __init__(self) -> None:
            self.bound_instance = bound_instance
            self.in_progress: dict[TaskId, Any] = {TaskId("connect"): object()}
            self.runner_process = FakeProcess()
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    event_sender, event_receiver = channel[Event]()
    _cmd_sender, _cmd_receiver = channel[Any]()
    _dl_sender, _dl_receiver = channel[Any]()
    _ev_in_sender, ev_in_receiver = channel[Any]()

    worker = Worker(
        NodeId(NODE_A),
        event_receiver=ev_in_receiver,
        event_sender=event_sender,
        command_sender=_cmd_sender,
        download_command_sender=_dl_sender,
        api_port=52415,
    )

    supervisor = FakeSupervisor()
    worker.runners[RUNNER_1_ID] = supervisor  # type: ignore[assignment]

    async with worker._tg:  # pyright: ignore[reportPrivateUsage]
        await worker._fail_unresponsive_runner(RUNNER_1_ID, "wedged")  # pyright: ignore[reportPrivateUsage]

        # Still terminating: no replacement may be planned for this instance yet.
        assert supervisor.shutdown_called
        assert worker._terminating_instance_ids() == frozenset({INSTANCE_1_ID})  # pyright: ignore[reportPrivateUsage]
        assert (
            plan_mod.plan(
                node_id=NODE_A,
                runners={},
                global_download_status={NODE_A: []},
                instances={INSTANCE_1_ID: instance},
                all_runners={
                    RUNNER_1_ID: RunnerFailed(error_message="", diagnostics=[])
                },
                tasks={},
                input_chunk_buffer={},
                image_cache={},
                instance_backoff=KeyedBackoff(),
                download_backoff=KeyedBackoff(),
                download_retry_backoff=KeyedBackoff(),
                terminating_instance_ids=worker._terminating_instance_ids(),  # pyright: ignore[reportPrivateUsage]
            )
            is None
        )

        # The process exits; the reap task releases the instance.
        supervisor.runner_process.exited.set()

    assert worker._terminating_instance_ids() == frozenset()  # pyright: ignore[reportPrivateUsage]

    # In-flight work is driven to a terminal status rather than left hanging.
    events: list[Event] = []
    event_sender.close()
    async for event in event_receiver:
        events.append(event)

    assert any(
        isinstance(event, TaskStatusUpdated)
        and event.task_id == TaskId("connect")
        and event.task_status == TaskStatus.Failed
        for event in events
    ), "the wedged runner's in-flight task was left non-terminal"
