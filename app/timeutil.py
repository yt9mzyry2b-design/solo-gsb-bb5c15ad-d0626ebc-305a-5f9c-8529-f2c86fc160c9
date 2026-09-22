"""Timestamp parsing / formatting helpers.

Two representations are accepted on input:

* numbers -- epoch timestamps, units auto-detected among
  seconds / milliseconds / microseconds / nanoseconds by order of magnitude;
* strings -- ISO-8601 (``2026-09-22T08:00:00Z`` ...) or numeric strings.

All internal arithmetic works on epoch **seconds** (float).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def parse_timestamp(value: Any) -> float:
    """Parse a JSON timestamp into epoch seconds.

    Raises ``ValueError`` for values that cannot be interpreted.
    """
    if isinstance(value, bool):
        raise ValueError(f"unsupported timestamp value: {value!r}")
    if isinstance(value, (int, float)):
        return _normalise_units(float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty timestamp string")
        # Numeric string -> epoch, units auto-detected.
        try:
            return _normalise_units(float(text))
        except ValueError:
            pass
        # ISO-8601; tolerate a trailing Z.
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"unparseable timestamp: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise ValueError(f"unsupported timestamp type: {type(value).__name__}")


def _normalise_units(ts: float) -> float:
    if not math.isfinite(ts):
        raise ValueError("non-finite timestamp")
    a = abs(ts)
    # Order-of-magnitude bands, independent of the calendar year:
    #   seconds      |t| < 1e11   (covers 1970..year 5138, negatives too)
    #   milliseconds 1e11..1e14
    #   microseconds 1e14..1e17
    #   nanoseconds  >= 1e17
    if a >= 1.0e17:
        return ts / 1_000_000_000.0
    if a >= 1.0e14:
        return ts / 1_000_000.0
    if a >= 1.0e11:
        return ts / 1_000.0
    return ts


def to_iso(epoch_seconds: float) -> str:
    """Format epoch seconds as an ISO-8601 UTC string with millisecond precision."""
    if not math.isfinite(epoch_seconds):
        return None  # type: ignore[return-value]
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
