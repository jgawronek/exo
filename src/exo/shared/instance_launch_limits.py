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


# Thinking budget: floor keeps room for at least a short reasoning pass —
# closing the phase after a handful of tokens degrades answers more than
# disabling thinking outright.
MIN_THINKING_BUDGET: Final[int] = 256

# Sampling temperature bounds mirror the OpenAI-compatible request range.
MIN_TEMPERATURE: Final[float] = 0.0
MAX_TEMPERATURE: Final[float] = 2.0


def clamp_thinking_budget(requested: int | None) -> int | None:
    """Clamp a thinking budget to ``[256, ∞)``; ``None`` leaves it unbounded."""
    if requested is None:
        return None
    return max(MIN_THINKING_BUDGET, requested)


def clamp_temperature(requested: float | None) -> float | None:
    """Clamp a default temperature to ``[0.0, 2.0]``; ``None`` keeps the engine default."""
    if requested is None:
        return None
    return max(MIN_TEMPERATURE, min(requested, MAX_TEMPERATURE))
