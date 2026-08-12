"""Clamp ranges for per-instance launch options (context + prefill chunk)."""

from typing import Final

# Context window: never above the model card; floor keeps the control useful.
MIN_CONTEXT_LENGTH: Final[int] = 1024

# Prefill chunk: floor matches pipeline wavefront minimum; ceiling is the
# Metal-safe default for a single command buffer on Apple Silicon.
MIN_PREFILL_STEP_SIZE: Final[int] = 256
MAX_PREFILL_STEP_SIZE: Final[int] = 4096


def clamp_context_length(
    requested: int | None, card_context_length: int
) -> int | None:
    """Clamp a requested context length to ``[1024, card]``.

    Returns ``None`` when the card has no context length (control disabled)
    or when ``requested`` is ``None`` (leave unbounded).
    """
    if card_context_length <= 0:
        return None
    if requested is None:
        return None
    upper = card_context_length
    lower = min(MIN_CONTEXT_LENGTH, upper)
    return max(lower, min(requested, upper))


def clamp_prefill_step_size(requested: int | None, default: int) -> int:
    """Clamp prefill chunk size to ``[256, 4096]``, falling back to ``default``."""
    value = default if requested is None else requested
    return max(MIN_PREFILL_STEP_SIZE, min(value, MAX_PREFILL_STEP_SIZE))
