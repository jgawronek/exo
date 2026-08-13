"""Runner diagnostics must survive the event serialization round-trip.

These diagnostics ride inside events that are dumped to JSON/msgpack and
re-validated on every node. JSON has no tuple type, so a tuple `evidence`
field came back as a list and the strict model rejected it — which failed
validation on every peer and poisoned the whole event stream on any runner
fault (a Metal GPU timeout, a ring error). Regression guard.
"""

from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.disk_event_log import (
    _deserialize_event,  # pyright: ignore[reportPrivateUsage]
    _serialize_event,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.runner.diagnostics import (
    KnownRunnerDiagnostic,
    RunnerMetalGpuTimeout,
    RunnerRingSocketReceivingError,
    RunnerRingTransportError,
)

_RUNNER = RunnerId("11111111-1111-4111-8111-111111111111")


def _round_trip_diag(diag: KnownRunnerDiagnostic) -> KnownRunnerDiagnostic:
    status = RunnerFailed(error_message="boom", diagnostics=[diag])
    evt = RunnerStatusUpdated(runner_id=_RUNNER, runner_status=status)
    back = _deserialize_event(_serialize_event(evt))
    assert isinstance(back, RunnerStatusUpdated)
    assert isinstance(back.runner_status, RunnerFailed)
    restored = back.runner_status.diagnostics
    assert len(restored) == 1
    return restored[0]


def test_metal_gpu_timeout_round_trips_with_evidence() -> None:
    diag = RunnerMetalGpuTimeout(
        message="[METAL] GPU Timeout Error (kIOGPUCommandBufferCallbackErrorTimeout)",
        evidence=["line one", "line two"],
    )
    back = _round_trip_diag(diag)
    assert isinstance(back, RunnerMetalGpuTimeout)
    assert back.evidence == ["line one", "line two"]


def test_metal_gpu_timeout_round_trips_with_empty_evidence() -> None:
    back = _round_trip_diag(RunnerMetalGpuTimeout(message="m"))
    assert isinstance(back, RunnerMetalGpuTimeout)
    assert back.evidence == []


def test_ring_transport_error_round_trips() -> None:
    back = _round_trip_diag(
        RunnerRingTransportError(message="ring aborted", evidence=["e"])
    )
    assert isinstance(back, RunnerRingTransportError)


def test_ring_socket_error_round_trips() -> None:
    back = _round_trip_diag(
        RunnerRingSocketReceivingError(
            error_number=0,
            error_name="SUCCESS",
            error_description="",
            message="ring socket receive failed",
            evidence=["a", "b"],
        )
    )
    assert isinstance(back, RunnerRingSocketReceivingError)
    assert back.error_number == 0
