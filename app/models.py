"""Internal data structures for the alignment engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    """One event record pulled from a JSONL stream."""

    seq: int                 # 0-based position within its source stream
    time: float              # parsed timestamp, epoch seconds
    label: str               # matching tag
    record: dict[str, Any]   # original JSON object (kept for echo)


@dataclass(frozen=True)
class Anchor:
    """A matched ``(reference, target)`` event pair used to fit the clock map."""

    ref: Event
    tgt: Event
    confidence: float
