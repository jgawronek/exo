import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast


class _ConstantsSnapshot(TypedDict):
    config_home: str
    data_home: str
    cache_home: str
    default_models_dir: str
    models_dirs: list[str]
    models_read_only_dirs: list[str]
    resources_dir: str
    dashboard_dir: str
    enable_image_models: bool
    offline: bool
    tracing_enabled: bool
    max_concurrent_requests: int


_COMPATIBLE_ENVIRONMENT_VARIABLES = (
    "HOME",
    "DEFAULT_MODELS_DIR",
    "MODELS_READ_ONLY_DIRS",
    "MODELS_DIRS",
    "RESOURCES_DIR",
    "DASHBOARD_DIR",
    "ENABLE_IMAGE_MODELS",
    "OFFLINE",
    "TRACING_ENABLED",
    "MAX_CONCURRENT_REQUESTS",
)

_SNAPSHOT_SCRIPT = """
import json
import sys

sys.platform = "darwin"

from exo.shared import constants

print(json.dumps({
    "config_home": str(constants.EXO_CONFIG_HOME),
    "data_home": str(constants.EXO_DATA_HOME),
    "cache_home": str(constants.EXO_CACHE_HOME),
    "default_models_dir": str(constants.EXO_DEFAULT_MODELS_DIR),
    "models_dirs": [str(path) for path in constants.EXO_MODELS_DIRS],
    "models_read_only_dirs": [
        str(path) for path in constants.EXO_MODELS_READ_ONLY_DIRS
    ],
    "resources_dir": str(constants.RESOURCES_DIR),
    "dashboard_dir": str(constants.DASHBOARD_DIR),
    "enable_image_models": constants.EXO_ENABLE_IMAGE_MODELS,
    "offline": constants.EXO_OFFLINE,
    "tracing_enabled": constants.EXO_TRACING_ENABLED,
    "max_concurrent_requests": constants.EXO_MAX_CONCURRENT_REQUESTS,
}))
"""


def _constants_snapshot(
    environment_overrides: dict[str, str],
) -> _ConstantsSnapshot:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.removeprefix("EXO_").removeprefix("XEO_")
        not in _COMPATIBLE_ENVIRONMENT_VARIABLES
    }
    environment.update(environment_overrides)
    source_root = Path(__file__).parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), environment.get("PYTHONPATH", ""))
    )
    completed_process = subprocess.run(
        [sys.executable, "-c", _SNAPSHOT_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return cast(
        _ConstantsSnapshot,
        json.loads(completed_process.stdout),
    )


def test_xeo_environment_variables_take_precedence() -> None:
    snapshot = _constants_snapshot(
        {
            "XEO_HOME": ".xeo-home",
            "EXO_HOME": ".exo-home",
            "XEO_DEFAULT_MODELS_DIR": "/xeo/default-models",
            "EXO_DEFAULT_MODELS_DIR": "/exo/default-models",
            "XEO_MODELS_DIRS": "/xeo/writable-one:/xeo/writable-two",
            "EXO_MODELS_DIRS": "/exo/writable",
            "XEO_MODELS_READ_ONLY_DIRS": "/xeo/read-only",
            "EXO_MODELS_READ_ONLY_DIRS": "/exo/read-only",
            "XEO_RESOURCES_DIR": ".xeo-resources",
            "EXO_RESOURCES_DIR": ".exo-resources",
            "XEO_DASHBOARD_DIR": ".xeo-dashboard",
            "EXO_DASHBOARD_DIR": ".exo-dashboard",
            "XEO_ENABLE_IMAGE_MODELS": "true",
            "EXO_ENABLE_IMAGE_MODELS": "false",
            "XEO_OFFLINE": "true",
            "EXO_OFFLINE": "false",
            "XEO_TRACING_ENABLED": "true",
            "EXO_TRACING_ENABLED": "false",
            "XEO_MAX_CONCURRENT_REQUESTS": "13",
            "EXO_MAX_CONCURRENT_REQUESTS": "7",
        }
    )

    home = Path.home()
    assert snapshot["config_home"] == str(home / ".xeo-home")
    assert snapshot["data_home"] == str(home / ".xeo-home")
    assert snapshot["cache_home"] == str(home / ".xeo-home")
    assert snapshot["default_models_dir"] == "/xeo/default-models"
    assert snapshot["models_dirs"] == [
        "/xeo/default-models",
        "/xeo/writable-one",
        "/xeo/writable-two",
    ]
    assert snapshot["models_read_only_dirs"] == ["/xeo/read-only"]
    assert snapshot["resources_dir"] == str(home / ".xeo-resources")
    assert snapshot["dashboard_dir"] == str(home / ".xeo-dashboard")
    assert snapshot["enable_image_models"] is True
    assert snapshot["offline"] is True
    assert snapshot["tracing_enabled"] is True
    assert snapshot["max_concurrent_requests"] == 13


def test_exo_environment_variables_remain_fallbacks() -> None:
    snapshot = _constants_snapshot(
        {
            "EXO_HOME": ".legacy-home",
            "EXO_DEFAULT_MODELS_DIR": "/legacy/default-models",
            "EXO_MODELS_DIRS": "/legacy/writable",
            "EXO_MODELS_READ_ONLY_DIRS": "/legacy/read-only",
            "EXO_RESOURCES_DIR": ".legacy-resources",
            "EXO_DASHBOARD_DIR": ".legacy-dashboard",
            "EXO_ENABLE_IMAGE_MODELS": "true",
            "EXO_OFFLINE": "true",
            "EXO_TRACING_ENABLED": "true",
            "EXO_MAX_CONCURRENT_REQUESTS": "11",
        }
    )

    home = Path.home()
    assert snapshot["config_home"] == str(home / ".legacy-home")
    assert snapshot["data_home"] == str(home / ".legacy-home")
    assert snapshot["cache_home"] == str(home / ".legacy-home")
    assert snapshot["default_models_dir"] == "/legacy/default-models"
    assert snapshot["models_dirs"] == [
        "/legacy/default-models",
        "/legacy/writable",
    ]
    assert snapshot["models_read_only_dirs"] == ["/legacy/read-only"]
    assert snapshot["resources_dir"] == str(home / ".legacy-resources")
    assert snapshot["dashboard_dir"] == str(home / ".legacy-dashboard")
    assert snapshot["enable_image_models"] is True
    assert snapshot["offline"] is True
    assert snapshot["tracing_enabled"] is True
    assert snapshot["max_concurrent_requests"] == 11


def test_environment_defaults_remain_exo_paths_and_values() -> None:
    snapshot = _constants_snapshot({})

    exo_home = Path.home() / ".exo"
    default_models_dir = exo_home / "models"
    assert snapshot["config_home"] == str(exo_home)
    assert snapshot["data_home"] == str(exo_home)
    assert snapshot["cache_home"] == str(exo_home)
    assert snapshot["default_models_dir"] == str(default_models_dir)
    assert snapshot["models_dirs"] == [str(default_models_dir)]
    assert snapshot["models_read_only_dirs"] == []
    assert snapshot["enable_image_models"] is False
    assert snapshot["offline"] is False
    assert snapshot["tracing_enabled"] is False
    assert snapshot["max_concurrent_requests"] == 8
