from exo.shared.instance_launch_limits import (
    MAX_PREFILL_STEP_SIZE,
    MIN_CONTEXT_LENGTH,
    MIN_PREFILL_STEP_SIZE,
    clamp_context_length,
    clamp_prefill_step_size,
)


class TestClampContextLength:
    def test_none_when_card_unknown(self) -> None:
        assert clamp_context_length(8192, 0) is None

    def test_none_when_requested_none(self) -> None:
        assert clamp_context_length(None, 32768) is None

    def test_clamps_below_minimum(self) -> None:
        assert clamp_context_length(100, 32768) == MIN_CONTEXT_LENGTH

    def test_clamps_above_card(self) -> None:
        assert clamp_context_length(512_000, 32_768) == 32_768

    def test_passes_through_in_range(self) -> None:
        assert clamp_context_length(8192, 32768) == 8192

    def test_card_smaller_than_minimum(self) -> None:
        assert clamp_context_length(100, 512) == 512


class TestClampPrefillStepSize:
    def test_default_when_none(self) -> None:
        assert clamp_prefill_step_size(None, 4096) == 4096

    def test_clamps_below_minimum(self) -> None:
        assert clamp_prefill_step_size(64, 4096) == MIN_PREFILL_STEP_SIZE

    def test_clamps_above_maximum(self) -> None:
        assert clamp_prefill_step_size(16_384, 4096) == MAX_PREFILL_STEP_SIZE

    def test_passes_through_in_range(self) -> None:
        assert clamp_prefill_step_size(512, 4096) == 512
