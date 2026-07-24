from collections.abc import Mapping
from typing import overload


@overload
def get_compatible_environment_value(
    environment: Mapping[str, str],
    exo_variable_name: str,
) -> str | None: ...


@overload
def get_compatible_environment_value(
    environment: Mapping[str, str],
    exo_variable_name: str,
    default: str,
) -> str: ...


def get_compatible_environment_value(
    environment: Mapping[str, str],
    exo_variable_name: str,
    default: str | None = None,
) -> str | None:
    """Read XEO first while retaining the corresponding EXO fallback."""

    xeo_variable_name = f"XEO_{exo_variable_name.removeprefix('EXO_')}"
    return environment.get(
        xeo_variable_name,
        environment.get(exo_variable_name, default),
    )
