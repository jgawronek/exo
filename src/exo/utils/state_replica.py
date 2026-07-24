from typing import final

import anyio

from exo.shared.apply import apply
from exo.shared.types.common import SessionId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.state import State


class ReplicaSequenceError(ValueError):
    pass


@final
class StateReplica:
    def __init__(
        self,
        *,
        session: SessionId,
        initial_state: State,
        ready: bool | None = None,
    ) -> None:
        self._session = session
        self._state = initial_state
        self._last_event: IndexedEvent | None = None
        self._ready = (
            initial_state.last_event_applied_idx >= 0 if ready is None else ready
        )
        self._ready_event = anyio.Event()
        if self._ready:
            self._ready_event.set()

    @property
    def session(self) -> SessionId:
        return self._session

    @property
    def state(self) -> State:
        return self._state

    @property
    def ready(self) -> bool:
        return self._ready

    def apply(
        self,
        indexed_event: IndexedEvent,
        *,
        mark_ready: bool = True,
    ) -> State:
        expected_index = self._state.last_event_applied_idx + 1
        if indexed_event.idx == self._state.last_event_applied_idx:
            if self._last_event == indexed_event:
                return self._state
            raise ReplicaSequenceError(
                f"Received conflicting event at index {indexed_event.idx}"
            )
        if indexed_event.idx != expected_index:
            raise ReplicaSequenceError(
                f"Received event at index {indexed_event.idx}; "
                f"expected index {expected_index}"
            )
        self._state = apply(self._state, indexed_event)
        self._last_event = indexed_event
        if mark_ready:
            self.mark_ready()
        return self._state

    def prepare_snapshot(self, session: SessionId, state: State) -> None:
        if session != self._session:
            raise ReplicaSequenceError(
                "Snapshot session does not match replica session"
            )
        was_ready = self._ready
        self._state = state
        self._last_event = None
        self._ready = False
        if was_ready:
            self._ready_event = anyio.Event()

    def install_snapshot(self, session: SessionId, state: State) -> None:
        self.prepare_snapshot(session, state)
        self.mark_ready()

    def mark_ready(self) -> None:
        self._ready = True
        self._ready_event.set()

    def start_unready_session(self, session: SessionId) -> None:
        self._session = session
        self._state = State()
        self._last_event = None
        self._ready = False
        self._ready_event = anyio.Event()

    async def wait_ready(self) -> None:
        await self._ready_event.wait()

    def rebase_session(self, session: SessionId) -> None:
        self._session = session
        self._state = self._state.model_copy(update={"last_event_applied_idx": -1})
        self._last_event = None
