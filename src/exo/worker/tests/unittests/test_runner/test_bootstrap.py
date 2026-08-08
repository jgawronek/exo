import pytest

from exo.worker.runner.bootstrap import resolve_metal_fast_synchronization


@pytest.mark.parametrize(
    ("override", "expected_value"),
    [
        # Off by default: MLX's fast fence has no timeout and deadlocks, and
        # defaulting it on for jaccl made every RDMA instance hang.
        (None, "0"),
        ("false", "0"),
        ("true", "1"),
        # Anything unrecognised must not silently enable the broken path.
        ("", "0"),
        ("1", "0"),
        ("TRUE", "0"),
    ],
)
def test_metal_fast_synchronization_is_opt_in(
    *,
    override: str | None,
    expected_value: str,
) -> None:
    assert resolve_metal_fast_synchronization(override) == expected_value
