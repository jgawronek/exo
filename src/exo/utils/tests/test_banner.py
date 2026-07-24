import os
from unittest.mock import patch

from pytest import CaptureFixture

from exo.utils.banner import print_startup_banner


def test_startup_banner_displays_xeo_name(
    capsys: CaptureFixture[str],
) -> None:
    with patch("exo.utils.banner._is_first_run", return_value=False):
        print_startup_banner(52415)

    standard_error = capsys.readouterr().err
    assert "XEO" in standard_error
    assert "EXO" not in standard_error


def test_xeo_runtime_dir_prevents_first_run_browser_open() -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"XEO_RUNTIME_DIR", "EXO_RUNTIME_DIR"}
    }
    environment["XEO_RUNTIME_DIR"] = "/tmp/xeo-runtime"

    with (
        patch.dict(os.environ, environment, clear=True),
        patch("exo.utils.banner._is_first_run", return_value=True),
        patch("exo.utils.banner._mark_first_run_done"),
        patch("exo.utils.banner.webbrowser.open") as open_browser,
    ):
        print_startup_banner(52415)

    open_browser.assert_not_called()
