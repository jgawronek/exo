from datetime import datetime, timedelta, timezone

from exo.master.main import find_stalled_generation_tasks
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId

_STALL_TIMEOUT = timedelta(seconds=120)


def _generation_task(task_status: TaskStatus) -> TextGenerationTask:
    return TextGenerationTask(
        task_id=TaskId(),
        command_id=CommandId(),
        instance_id=InstanceId(),
        task_status=task_status,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


def test_running_task_without_recent_progress_is_stalled() -> None:
    now = datetime.now(tz=timezone.utc)
    task = _generation_task(TaskStatus.Running)

    stalled = find_stalled_generation_tasks(
        {task.task_id: task},
        {task.task_id: now - timedelta(seconds=121)},
        now,
        _STALL_TIMEOUT,
    )

    assert stalled == [task]


def test_recent_progress_is_not_stalled() -> None:
    now = datetime.now(tz=timezone.utc)
    task = _generation_task(TaskStatus.Running)

    stalled = find_stalled_generation_tasks(
        {task.task_id: task},
        {task.task_id: now - timedelta(seconds=30)},
        now,
        _STALL_TIMEOUT,
    )

    assert stalled == []


def test_non_running_tasks_are_ignored() -> None:
    now = datetime.now(tz=timezone.utc)
    ancient = now - timedelta(hours=1)
    tasks = {
        status: _generation_task(status)
        for status in (
            TaskStatus.Pending,
            TaskStatus.Complete,
            TaskStatus.Failed,
            TaskStatus.Cancelled,
        )
    }

    stalled = find_stalled_generation_tasks(
        {task.task_id: task for task in tasks.values()},
        {task.task_id: ancient for task in tasks.values()},
        now,
        _STALL_TIMEOUT,
    )

    assert stalled == []


def test_unwatched_task_is_not_stalled() -> None:
    # A task with no recorded progress timestamp (e.g. right after master
    # failover) must be watched for a full window before being judged.
    now = datetime.now(tz=timezone.utc)
    task = _generation_task(TaskStatus.Running)

    stalled = find_stalled_generation_tasks(
        {task.task_id: task}, {}, now, _STALL_TIMEOUT
    )

    assert stalled == []
