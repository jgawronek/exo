# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.routing.event_router import ReplicatedEventDelivery
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.utils.channels import channel
from exo.utils.state_replica import StateReplica


def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""

    app = FastAPI()

    # Setup exception handler
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    # Add test routes that raise HTTPException
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # Test 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # Test 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_api_rejects_state_requests_while_replica_is_unready() -> None:
    node_id = NodeId("node")
    session = SessionId(master_node_id=NodeId("master"), election_clock=1)
    state_replica = StateReplica(
        session=session,
        initial_state=State(),
        ready=False,
    )
    _, event_receiver = channel[ReplicatedEventDelivery]()
    command_sender, _ = channel[ForwarderCommand]()
    download_command_sender, _ = channel[ForwarderDownloadCommand]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        node_id,
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_command_sender,
        election_receiver=election_receiver,
        state_replica=state_replica,
    )
    client = TestClient(api.app)

    assert client.get("/state").status_code == 503
    assert client.get("/node_id").status_code == 200
