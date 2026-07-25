import os
import sys
from collections.abc import Mapping
from pathlib import Path

from exo.shared.environment import get_compatible_environment_value
from exo.utils.dashboard_path import find_dashboard, find_resources


def _get_xdg_dir(
    environment: Mapping[str, str],
    platform: str,
    home: Path,
    xdg_variable_name: str,
    fallback: str,
) -> Path:
    """Get an XDG directory while preserving the established exo path layout."""

    exo_home = get_compatible_environment_value(environment, "EXO_HOME")
    if exo_home is not None:
        return home / exo_home

    if platform != "linux":
        return home / ".exo"

    xdg_value = environment.get(xdg_variable_name)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return home / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir(
    os.environ,
    sys.platform,
    Path.home(),
    "XDG_CONFIG_HOME",
    ".config",
)
EXO_DATA_HOME = _get_xdg_dir(
    os.environ,
    sys.platform,
    Path.home(),
    "XDG_DATA_HOME",
    ".local/share",
)
EXO_CACHE_HOME = _get_xdg_dir(
    os.environ,
    sys.platform,
    Path.home(),
    "XDG_CACHE_HOME",
    ".cache",
)

# Default models directory (always included as first entry in writable dirs)
_EXO_DEFAULT_MODELS_DIR_ENV = get_compatible_environment_value(
    os.environ,
    "EXO_DEFAULT_MODELS_DIR",
)
EXO_DEFAULT_MODELS_DIR = (
    Path(_EXO_DEFAULT_MODELS_DIR_ENV).expanduser()
    if _EXO_DEFAULT_MODELS_DIR_ENV is not None
    else EXO_DATA_HOME / "models"
)


def _parse_colon_dirs(environment_value: str | None) -> tuple[Path, ...]:
    if environment_value is None:
        return ()
    return tuple(
        Path(path).expanduser() for path in environment_value.split(":") if path
    )


# Read-only model directories (colon-separated). Never written to or deleted from.
_EXO_MODELS_READ_ONLY_DIRS_ENV = _parse_colon_dirs(
    get_compatible_environment_value(
        os.environ,
        "EXO_MODELS_READ_ONLY_DIRS",
    )
)
# Writable model directories (colon-separated). Default dir is always prepended.
_EXO_MODELS_DIRS_ENV = _parse_colon_dirs(
    get_compatible_environment_value(os.environ, "EXO_MODELS_DIRS")
)

# If a directory appears in both lists, treat it as read-only.
_read_only_set = frozenset(_EXO_MODELS_READ_ONLY_DIRS_ENV)
EXO_MODELS_DIRS: tuple[Path, ...] = tuple(
    d
    for d in (EXO_DEFAULT_MODELS_DIR, *_EXO_MODELS_DIRS_ENV)
    if d not in _read_only_set
)
EXO_MODELS_READ_ONLY_DIRS: tuple[Path, ...] = _EXO_MODELS_READ_ONLY_DIRS_ENV

_RESOURCES_DIR_ENV = get_compatible_environment_value(
    os.environ,
    "EXO_RESOURCES_DIR",
)
RESOURCES_DIR = (
    find_resources() if _RESOURCES_DIR_ENV is None else Path.home() / _RESOURCES_DIR_ENV
)
_DASHBOARD_DIR_ENV = get_compatible_environment_value(
    os.environ,
    "EXO_DASHBOARD_DIR",
)
DASHBOARD_DIR = (
    find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
)

# Log files (data/logs or cache)
EXO_LOG_DIR = EXO_CACHE_HOME / "exo_log"
EXO_LOG = EXO_LOG_DIR / "exo.log"
EXO_RUNNER_LOG_DIR = EXO_LOG_DIR / "runner_log"
EXO_RUNNER_STDOUT_LOG = EXO_RUNNER_LOG_DIR / "stdout.log"
EXO_RUNNER_STDERR_LOG = EXO_RUNNER_LOG_DIR / "stderr.log"

EXO_TEST_LOG = EXO_CACHE_HOME / "exo_test.log"
EXO_PID_FILE = EXO_CACHE_HOME / "exo.pid"

# Identity (config)
EXO_NODE_ZID = EXO_CACHE_HOME / "node_zid"
EXO_CONFIG_FILE = EXO_CONFIG_HOME / "config.toml"

# Persisted cluster-wide shared models directory (set from the dashboard).
EXO_SHARED_MODELS_DIR_FILE = EXO_CONFIG_HOME / "shared_models_dir"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

EXO_MAX_CHUNK_SIZE = 512 * 1024

EXO_CUSTOM_MODEL_CARDS_DIR = EXO_DATA_HOME / "custom_model_cards"

EXO_EVENT_LOG_DIR = EXO_DATA_HOME / "event_log"
EXO_IMAGE_CACHE_DIR = EXO_CACHE_HOME / "images"
EXO_TRACING_CACHE_DIR = EXO_CACHE_HOME / "traces"

EXO_ENABLE_IMAGE_MODELS = (
    get_compatible_environment_value(
        os.environ,
        "EXO_ENABLE_IMAGE_MODELS",
        "false",
    ).lower()
    == "true"
)

EXO_OFFLINE = (
    get_compatible_environment_value(os.environ, "EXO_OFFLINE", "false").lower()
    == "true"
)

EXO_TRACING_ENABLED = (
    get_compatible_environment_value(
        os.environ,
        "EXO_TRACING_ENABLED",
        "false",
    ).lower()
    == "true"
)

ENABLE_DISAGGREGATION = os.getenv("ENABLE_DISAGGREGATION", "false").lower() == "true"

EXO_MAX_CONCURRENT_REQUESTS = int(
    get_compatible_environment_value(
        os.environ,
        "EXO_MAX_CONCURRENT_REQUESTS",
        "8",
    )
)

EXO_MAX_INSTANCE_RETRIES = 5
