"""Stall-detection logic for the download coordinator.

A download's sockets can stay open with no bytes flowing — the aiohttp
socket-read timeout never fires — so the coordinator tracks forward byte
progress and flags a freeze itself.
"""

from unittest.mock import patch

from exo.download.coordinator import (
    _MAX_CONCURRENT_DOWNLOADS,  # pyright: ignore[reportPrivateUsage]
    DownloadCoordinator,
)
from exo.shared.types.common import ModelId

MODEL = ModelId("org/model")


def _coordinator() -> DownloadCoordinator:
    # Only the stall bookkeeping is exercised; the collaborators are unused.
    return DownloadCoordinator.__new__(DownloadCoordinator)


def _fresh() -> DownloadCoordinator:
    c = _coordinator()
    c._download_progress_marks = {}  # pyright: ignore[reportPrivateUsage]
    return c


class TestStallDetection:
    def test_not_stalled_without_a_mark(self) -> None:
        c = _fresh()
        assert c._download_is_stalled(MODEL) is False  # pyright: ignore[reportPrivateUsage]

    def test_not_stalled_within_timeout(self) -> None:
        c = _fresh()
        with patch("exo.download.coordinator.current_time", return_value=100.0):
            c._mark_download_progress(MODEL, 10)  # pyright: ignore[reportPrivateUsage]
        with patch("exo.download.coordinator.current_time", return_value=100.5):
            assert c._download_is_stalled(MODEL) is False  # pyright: ignore[reportPrivateUsage]

    def test_stalled_after_timeout_with_no_advance(self) -> None:
        c = _fresh()
        with patch("exo.download.coordinator.current_time", return_value=100.0):
            c._mark_download_progress(MODEL, 10)  # pyright: ignore[reportPrivateUsage]
        with patch("exo.download.coordinator.current_time", return_value=1000.0):
            assert c._download_is_stalled(MODEL) is True  # pyright: ignore[reportPrivateUsage]

    def test_advancing_bytes_reset_the_timer(self) -> None:
        c = _fresh()
        with patch("exo.download.coordinator.current_time", return_value=100.0):
            c._mark_download_progress(MODEL, 10)  # pyright: ignore[reportPrivateUsage]
        # More bytes arrive later — the freeze clock restarts.
        with patch("exo.download.coordinator.current_time", return_value=200.0):
            c._mark_download_progress(MODEL, 20)  # pyright: ignore[reportPrivateUsage]
        with patch("exo.download.coordinator.current_time", return_value=250.0):
            assert c._download_is_stalled(MODEL) is False  # pyright: ignore[reportPrivateUsage]

    def test_same_byte_count_does_not_reset_the_timer(self) -> None:
        c = _fresh()
        with patch("exo.download.coordinator.current_time", return_value=100.0):
            c._mark_download_progress(MODEL, 10)  # pyright: ignore[reportPrivateUsage]
        # A repeated progress event with no new bytes must NOT reset the clock.
        with patch("exo.download.coordinator.current_time", return_value=200.0):
            c._mark_download_progress(MODEL, 10)  # pyright: ignore[reportPrivateUsage]
        with patch("exo.download.coordinator.current_time", return_value=1000.0):
            assert c._download_is_stalled(MODEL) is True  # pyright: ignore[reportPrivateUsage]


def test_downloads_serialized_by_default() -> None:
    assert _MAX_CONCURRENT_DOWNLOADS == 1
