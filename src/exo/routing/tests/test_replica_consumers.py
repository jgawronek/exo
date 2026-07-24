from unittest.mock import MagicMock

import pytest

from exo.api.main import API
from exo.routing.event_router import ReplicatedEventDelivery
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import Event, IndexedEvent, TestEvent
from exo.shared.types.state import State
from exo.utils.channels import Sender, channel
from exo.utils.state_replica import StateReplica
from exo.worker.main import Worker


def make_session() -> SessionId:
    return SessionId(master_node_id=NodeId("master"), election_clock=1)


def make_delivery(
    state_replica: StateReplica,
    indexed_event: IndexedEvent,
) -> ReplicatedEventDelivery:
    state_replica.apply(indexed_event)
    return ReplicatedEventDelivery(
        indexed_event=indexed_event,
        state_after_event=state_replica.state,
    )


def make_api(
    *,
    state_replica: StateReplica | None,
) -> tuple[API, Sender[ReplicatedEventDelivery]]:
    event_sender, event_receiver = channel[ReplicatedEventDelivery]()
    api = object.__new__(API)
    api.event_receiver = event_receiver
    api.state_replica = state_replica
    api._state = State()  # pyright: ignore[reportPrivateUsage]
    api._event_log = MagicMock()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    return api, event_sender


def make_worker(
    *,
    state_replica: StateReplica | None,
) -> tuple[Worker, Sender[ReplicatedEventDelivery]]:
    event_sender, event_receiver = channel[ReplicatedEventDelivery]()
    outbound_event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_command_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NodeId("worker"),
        event_receiver=event_receiver,
        event_sender=outbound_event_sender,
        command_sender=command_sender,
        download_command_sender=download_command_sender,
        api_port=52415,
        state_replica=state_replica,
    )
    return worker, event_sender


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_kind", ["api", "worker"])
async def test_consumer_uses_shared_replica_without_reapplying_event(
    consumer_kind: str,
) -> None:
    state_replica = StateReplica(
        session=make_session(),
        initial_state=State(),
        ready=True,
    )
    indexed_event = IndexedEvent(idx=0, event=TestEvent())
    delivery = make_delivery(state_replica, indexed_event)

    if consumer_kind == "api":
        consumer, event_sender = make_api(state_replica=state_replica)
        event_sender.send_nowait(delivery)
        event_sender.close()
        await consumer._apply_state()  # pyright: ignore[reportPrivateUsage]
    else:
        consumer, event_sender = make_worker(state_replica=state_replica)
        event_sender.send_nowait(delivery)
        event_sender.close()
        await consumer._event_applier()  # pyright: ignore[reportPrivateUsage]

    assert consumer.state is state_replica.state
    assert consumer.state.last_event_applied_idx == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_kind", ["api", "worker"])
async def test_standalone_consumer_applies_delivered_event_locally(
    consumer_kind: str,
) -> None:
    source_replica = StateReplica(
        session=make_session(),
        initial_state=State(),
        ready=True,
    )
    indexed_event = IndexedEvent(idx=0, event=TestEvent())
    delivery = make_delivery(source_replica, indexed_event)

    if consumer_kind == "api":
        consumer, event_sender = make_api(state_replica=None)
        event_sender.send_nowait(delivery)
        event_sender.close()
        await consumer._apply_state()  # pyright: ignore[reportPrivateUsage]
    else:
        consumer, event_sender = make_worker(state_replica=None)
        event_sender.send_nowait(delivery)
        event_sender.close()
        await consumer._event_applier()  # pyright: ignore[reportPrivateUsage]

    assert consumer.state.last_event_applied_idx == 0
