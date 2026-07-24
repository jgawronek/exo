import pytest

from exo.shared.environment import get_compatible_environment_value


def test_xeo_value_takes_precedence_over_exo_value() -> None:
    environment = {
        "XEO_MEMORY_THRESHOLD": "0.75",
        "EXO_MEMORY_THRESHOLD": "0.50",
    }

    assert (
        get_compatible_environment_value(
            environment,
            "EXO_MEMORY_THRESHOLD",
        )
        == "0.75"
    )


def test_exo_value_remains_the_fallback() -> None:
    environment = {"EXO_DRAFT_TOKENS": "4"}

    assert (
        get_compatible_environment_value(
            environment,
            "EXO_DRAFT_TOKENS",
        )
        == "4"
    )


def test_explicit_default_is_returned_when_both_names_are_unset() -> None:
    assert (
        get_compatible_environment_value(
            {},
            "EXO_STALL_TIMEOUT_SECONDS",
            "120",
        )
        == "120"
    )


def test_empty_xeo_value_still_takes_precedence() -> None:
    environment = {
        "XEO_BOOTSTRAP_PEERS": "",
        "EXO_BOOTSTRAP_PEERS": "/ip4/legacy-peer",
    }

    assert (
        get_compatible_environment_value(
            environment,
            "EXO_BOOTSTRAP_PEERS",
        )
        == ""
    )


@pytest.mark.parametrize(
    "exo_variable_name",
    (
        "EXO_MACMON_PATH",
        "EXO_DRAFT_MODEL",
        "EXO_NO_BATCH",
        "EXO_MEMORY_THRESHOLD",
        "EXO_PREFILL_MEMORY_THRESHOLD",
        "EXO_STALL_TIMEOUT_SECONDS",
        "EXO_DRAFT_TOKENS",
        "EXO_LIBP2P_NAMESPACE",
        "EXO_ZENOH_NAMESPACE",
        "EXO_BOOTSTRAP_PEERS",
        "EXO_FAST_SYNCH",
        "EXO_RUNTIME_DIR",
        "EXO_OFFLINE",
    ),
)
def test_runtime_environment_names_have_xeo_aliases(
    exo_variable_name: str,
) -> None:
    xeo_variable_name = f"XEO_{exo_variable_name.removeprefix('EXO_')}"
    environment = {
        xeo_variable_name: "preferred",
        exo_variable_name: "fallback",
    }

    assert (
        get_compatible_environment_value(
            environment,
            exo_variable_name,
        )
        == "preferred"
    )
