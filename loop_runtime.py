"""Runtime helpers for unattended gemini_uncached_loop runs.

* :func:`retry_delay` - exponential backoff with jitter for consecutive
  request failures.
* :class:`RotatingJsonlWriter` - append one JSON object per line and
  rotate to ``<path>.1`` when the file grows past a size limit.
* :class:`DailyCounter` - persistent per-day request counter so a
  scheduled job can enforce a daily cap across restarts.

Only the standard library is used, so these helpers work on any Python
3.11+ interpreter, with or without gemini-webapi installed.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path


def retry_delay(
    base_interval: float,
    consecutive_failures: int,
    *,
    max_delay: float = 300.0,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retrying after *consecutive_failures*.

    The delay doubles per consecutive failure, is capped at *max_delay*,
    and is jittered by a factor of 0.5 to 1.5 so retries do not line up
    in a fixed rhythm.
    """
    if consecutive_failures < 0:
        raise ValueError("consecutive_failures must be >= 0")
    if base_interval < 0:
        raise ValueError("base_interval must be >= 0")
    if max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    chooser = rng if rng is not None else random
    expected = min(base_interval * (2**consecutive_failures), max_delay)
    return expected * chooser.uniform(0.5, 1.5)


class RotatingJsonlWriter:
    """Append JSON records to a file, rotating once it exceeds max_bytes."""

    def __init__(self, path: str | Path, *, max_bytes: int = 5_000_000) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=True) + "\n"
        data = line.encode("utf-8")

        if self.path.exists():
            size = self.path.stat().st_size
            if size > 0 and size + len(data) > self.max_bytes:
                backup = Path(str(self.path) + ".1")
                if backup.exists():
                    backup.unlink()
                self.path.replace(backup)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class DailyCounter:
    """Persistent request counter that resets when the date changes.

    State is a small JSON file: ``{"date": "YYYY-MM-DD", "count": N}``.
    A missing or corrupt file is treated as a fresh counter, so a bad
    write can never wedge a scheduled run.
    """

    def __init__(self, path: str | Path, limit: int, *, today: str | None = None) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.path = Path(path)
        self.limit = limit
        self._today = today or date.today().isoformat()
        self._count = 0
        self._load()

    def _load(self) -> None:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("date") == self._today:
                self._count = int(state.get("count", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            self._count = 0

    def allow(self) -> bool:
        """True while fewer than ``limit`` requests have been recorded today."""
        return self._count < self.limit

    def record(self) -> None:
        """Count one request and persist the counter."""
        self._count += 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"date": self._today, "count": self._count}),
                encoding="utf-8",
            )
        except OSError:
            pass
