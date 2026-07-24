import sys

from pytest import MonkeyPatch


def test_main_arguments_prefer_xeo_environment_values(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("XEO_OFFLINE", "true")
    monkeypatch.setenv("EXO_OFFLINE", "false")
    monkeypatch.setenv(
        "XEO_BOOTSTRAP_PEERS",
        "/ip4/xeo-one,/ip4/xeo-two",
    )
    monkeypatch.setenv("EXO_BOOTSTRAP_PEERS", "/ip4/exo")
    monkeypatch.setattr(sys, "argv", ["xeo"])

    from exo.main import Args

    arguments = Args.parse()

    assert arguments.offline is True
    assert arguments.bootstrap_peers == [
        "/ip4/xeo-one",
        "/ip4/xeo-two",
    ]


def test_runner_flags_override_both_environment_names() -> None:
    from exo.main import Args, apply_runner_environment_overrides

    environment = {
        "XEO_NO_BATCH": "",
        "EXO_NO_BATCH": "",
        "XEO_FAST_SYNCH": "false",
        "EXO_FAST_SYNCH": "false",
    }
    arguments = Args(
        no_batch=True,
        fast_synch=True,
        namespace="test",
        zenoh_port=52414,
        discovery_port=52413,
    )

    apply_runner_environment_overrides(arguments, environment)

    assert environment["XEO_NO_BATCH"] == "1"
    assert environment["EXO_NO_BATCH"] == "1"
    assert environment["XEO_FAST_SYNCH"] == "true"
    assert environment["EXO_FAST_SYNCH"] == "true"
