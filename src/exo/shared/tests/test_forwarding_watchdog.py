from exo.main import FORWARDING_WEDGE_SECONDS, event_forwarding_wedged
from exo.shared.types.common import NodeId


def test_wedged_when_inbound_alive_but_own_events_stale() -> None:
    node, master = NodeId(), NodeId()
    assert event_forwarding_wedged(
        now=1000.0,
        inbound_advanced_at=995.0,
        own_last_seen_advanced_at=1000.0 - FORWARDING_WEDGE_SECONDS - 1,
        master_node_id=master,
        node_id=node,
    )


def test_not_wedged_while_own_events_fresh() -> None:
    node, master = NodeId(), NodeId()
    assert not event_forwarding_wedged(
        now=1000.0,
        inbound_advanced_at=995.0,
        own_last_seen_advanced_at=990.0,
        master_node_id=master,
        node_id=node,
    )


def test_not_wedged_when_fully_disconnected() -> None:
    # Inbound is dead too: the election layer owns that failure, not the
    # forwarding watchdog.
    node, master = NodeId(), NodeId()
    assert not event_forwarding_wedged(
        now=1000.0,
        inbound_advanced_at=1000.0 - 120.0,
        own_last_seen_advanced_at=1000.0 - 120.0,
        master_node_id=master,
        node_id=node,
    )


def test_not_wedged_without_master_or_as_master() -> None:
    node = NodeId()
    stale = 1000.0 - FORWARDING_WEDGE_SECONDS - 1
    assert not event_forwarding_wedged(
        now=1000.0,
        inbound_advanced_at=995.0,
        own_last_seen_advanced_at=stale,
        master_node_id=None,
        node_id=node,
    )
    assert not event_forwarding_wedged(
        now=1000.0,
        inbound_advanced_at=995.0,
        own_last_seen_advanced_at=stale,
        master_node_id=node,
        node_id=node,
    )
