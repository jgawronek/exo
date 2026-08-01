"""Attribute the fixed per-decode-step cost that single-stream generation pays.

Decode step time behaves as ``step = fixed + batch_size * per_token``. On a
three-node 397B ring the fixed term measured ~22ms against a ~64ms per-token
term, so at batch 1 roughly a quarter of every token is spent outside the
model. This profiler exists to say *where*, since the aggregate number alone
does not name a thing to fix.

Off unless ``XEO_DECODE_PROFILE`` (or ``EXO_DECODE_PROFILE``) is set, and the
timing calls compile down to a monotonic clock read per bucket so that leaving
the import in place costs nothing on the hot path.
"""

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import final

from loguru import logger

from exo.shared.environment import get_compatible_environment_value

DECODE_PROFILE_ENABLED: bool = get_compatible_environment_value(
    os.environ, "EXO_DECODE_PROFILE", ""
).lower() in ("1", "true", "yes")

# Steps between reports. Long enough that a single slow step does not dominate
# the average, short enough to see a change while a generation is still running.
DECODE_PROFILE_INTERVAL_STEPS: int = int(
    get_compatible_environment_value(os.environ, "EXO_DECODE_PROFILE_INTERVAL", "100")
)


@final
class StepProfiler:
    """Accumulates per-bucket wall time across decode steps.

    Buckets nest freely: a caller may time ``generator.step()`` as a whole and
    separately time phases inside it, so long as the names differ. Reported
    totals are per-step averages, which is the unit the fixed/per-token model
    is expressed in.
    """

    def __init__(self, label: str, interval_steps: int | None = None) -> None:
        self._label: str = label
        self._interval: int = interval_steps or DECODE_PROFILE_INTERVAL_STEPS
        self._totals: dict[str, float] = {}
        self._steps: int = 0
        self._tokens: int = 0

    @contextmanager
    def bucket(self, name: str) -> Iterator[None]:
        if not DECODE_PROFILE_ENABLED:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] = self._totals.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def add(self, name: str, seconds: float) -> None:
        """Record a bucket timed by the caller (e.g. across a yield boundary)."""
        if not DECODE_PROFILE_ENABLED:
            return
        self._totals[name] = self._totals.get(name, 0.0) + seconds

    def end_step(self, tokens_emitted: int = 1) -> None:
        """Close a decode step, reporting once the interval has elapsed."""
        if not DECODE_PROFILE_ENABLED:
            return
        self._steps += 1
        self._tokens += tokens_emitted
        if self._steps < self._interval:
            return
        self.report()

    @property
    def totals(self) -> Mapping[str, float]:
        return dict(self._totals)

    def report(self) -> None:
        if not DECODE_PROFILE_ENABLED or self._steps == 0:
            return
        steps = self._steps
        tokens = self._tokens
        measured = sum(self._totals.values())
        parts = " ".join(
            f"{name}={value / steps * 1000:.2f}ms"
            for name, value in sorted(self._totals.items(), key=lambda item: -item[1])
        )
        logger.info(
            f"[decode-profile {self._label}] steps={steps} tokens={tokens} "
            f"avg_step={measured / steps * 1000:.2f}ms "
            f"tokens_per_step={tokens / steps:.2f} {parts}"
        )
        self._totals = {}
        self._steps = 0
        self._tokens = 0
