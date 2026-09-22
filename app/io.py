"""JSONL ingestion and alignment-result assembly."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .matching import ClockMapping, PiecewiseLinearMapping, align_sequences
from .models import Event
from .timeutil import parse_timestamp, to_iso

MAX_RECORDS = 50_000

DEFAULT_TIME_FIELDS = ("timestamp", "time", "ts", "datetime", "date")
DEFAULT_LABEL_FIELDS = ("label", "tag", "event", "event_label", "type", "name")


class InputError(ValueError):
    """Raised for malformed client input (maps to HTTP 400)."""


@dataclass
class AlignConfig:
    time_field: str | None = None
    label_field: str | None = None
    iterations: int = 3
    time_weight: float = 0.5


# --- parsing ----------------------------------------------------------------


def _pick_field(record: dict[str, Any], explicit: str | None,
                candidates: Sequence[str]) -> str:
    if explicit is not None:
        if explicit not in record:
            raise InputError(f"field {explicit!r} not present in record")
        return explicit
    for name in candidates:
        if name in record:
            return name
    raise InputError(
        "could not find a "
        f"{'/' .join(candidates)} field in record: {json.dumps(record)[:200]}"
    )


def parse_events(raw: Any, cfg: AlignConfig) -> list[Event]:
    """Parse a JSON body (list of objects) or a JSONL string/bytes into Events."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        records: list[Any] = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InputError(f"invalid JSON on line {lineno}: {exc}") from exc
    elif isinstance(raw, list):
        records = raw
    else:
        raise InputError(
            "payload must be a JSONL text stream or a JSON array of objects"
        )

    if len(records) > MAX_RECORDS:
        raise InputError(
            f"too many records: {len(records)} (limit {MAX_RECORDS})"
        )

    events: list[Event] = []
    time_key: str | None = cfg.time_field
    label_key: str | None = cfg.label_field
    for seq, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise InputError(f"record #{seq} is not a JSON object")
        if time_key is None:
            time_key = _pick_field(rec, None, DEFAULT_TIME_FIELDS)
        elif time_key not in rec:
            raise InputError(f"record #{seq} missing field {time_key!r}")
        if label_key is None:
            label_key = _pick_field(rec, None, DEFAULT_LABEL_FIELDS)
        elif label_key not in rec:
            raise InputError(f"record #{seq} missing field {label_key!r}")
        try:
            t = parse_timestamp(rec[time_key])
        except (ValueError, TypeError, OverflowError, OSError) as exc:
            raise InputError(f"record #{seq}: {exc}") from exc
        label = rec[label_key]
        events.append(Event(seq=seq, time=t, label=str(label), record=rec))

    # Monotonic DP assumes chronological order; sort defensively while the
    # records themselves keep their original positions (seq / index).
    events.sort(key=lambda e: (e.time, e.seq))
    return events


# --- result assembly --------------------------------------------------------


def _segments(mapping: ClockMapping) -> list[dict[str, Any]]:
    if not isinstance(mapping, PiecewiseLinearMapping) or len(mapping.xs) < 2:
        return []
    out: list[dict[str, Any]] = []
    for k in range(len(mapping.xs) - 1):
        x0, x1 = mapping.xs[k], mapping.xs[k + 1]
        y0, y1 = mapping.ys[k], mapping.ys[k + 1]
        if x1 == x0:
            continue
        slope = (y1 - y0) / (x1 - x0)
        out.append({
            "index": k,
            "start_target_time": x0,
            "start_target_time_iso": to_iso(x0),
            "end_target_time": x1,
            "end_target_time_iso": to_iso(x1),
            "start_reference_time": y0,
            "end_reference_time": y1,
            "offset_at_start": y0 - x0,
            "slope": slope,
            "drift_ppm": (slope - 1.0) * 1_000_000.0,
        })
    return out


def run_alignment(refs_raw: Any, tgts_raw: Any,
                  cfg: AlignConfig) -> dict[str, Any]:
    refs = parse_events(refs_raw, cfg)
    tgts = parse_events(tgts_raw, cfg)

    if not refs and not tgts:
        raise InputError("both reference and target streams are empty")

    anchors, mapping, residual_scale = align_sequences(
        refs, tgts, iterations=cfg.iterations, time_weight=cfg.time_weight
    )

    ref_matched = {a.ref.seq: a for a in anchors}
    tgt_matched = {a.tgt.seq: a for a in anchors}

    aligned: list[dict[str, Any]] = []
    for ev in tgts:
        t_cal = mapping.predict(ev.time)
        extrap = mapping.is_extrapolated(ev.time)
        anchor = tgt_matched.get(ev.seq)
        if anchor is not None:
            aligned.append({
                "index": ev.seq,
                "status": "matched",
                "label": ev.label,
                "original_time": ev.time,
                "original_time_iso": to_iso(ev.time),
                "calibrated_time": t_cal,
                "calibrated_time_iso": to_iso(t_cal),
                "confidence": anchor.confidence,
                "matched_reference_index": anchor.ref.seq,
                "residual": t_cal - anchor.ref.time,
                "extrapolated": extrap,
            })
        else:
            aligned.append({
                "index": ev.seq,
                "status": "inserted",
                "label": ev.label,
                "original_time": ev.time,
                "original_time_iso": to_iso(ev.time),
                "calibrated_time": t_cal,
                "calibrated_time_iso": to_iso(t_cal),
                "confidence": 0.0,
                "matched_reference_index": None,
                "residual": None,
                "extrapolated": extrap,
            })

    missing: list[dict[str, Any]] = []
    for ev in refs:
        anchor = ref_matched.get(ev.seq)
        if anchor is None:
            missing.append({
                "index": ev.seq,
                "status": "missing",
                "label": ev.label,
                "reference_time": ev.time,
                "reference_time_iso": to_iso(ev.time),
                "record": ev.record,
            })

    anchor_out = [
        {
            "reference_index": a.ref.seq,
            "target_index": a.tgt.seq,
            "label": a.ref.label,
            "reference_time": a.ref.time,
            "target_time": a.tgt.time,
            "mapped_time": mapping.predict(a.tgt.time),
            "residual": mapping.predict(a.tgt.time) - a.ref.time,
            "confidence": a.confidence,
        }
        for a in sorted(anchors, key=lambda x: x.ref.time)
    ]

    matched_conf = [a.confidence for a in anchors]
    aligned.sort(key=lambda r: r["index"])
    missing.sort(key=lambda r: r["index"])

    return {
        "summary": {
            "reference_count": len(refs),
            "target_count": len(tgts),
            "matched_count": len(anchors),
            "inserted_count": len(tgts) - len(anchors),
            "missing_count": len(refs) - len(anchors),
            "mean_confidence": (sum(matched_conf) / len(matched_conf)
                                if matched_conf else 0.0),
            "min_confidence": min(matched_conf) if matched_conf else 0.0,
            "residual_rms": (
                math.sqrt(sum(
                    (mapping.predict(a.tgt.time) - a.ref.time) ** 2
                    for a in anchors
                ) / len(anchors))
                if anchors else None
            ),
            "residual_scale": residual_scale,
            "mapping_type": type(mapping).__name__,
        },
        "clock_mapping": {
            "type": type(mapping).__name__,
            "global_slope": mapping.slope,
            "segments": _segments(mapping),
        },
        "anchors": anchor_out,
        "aligned_events": aligned,
        "missing_reference_events": missing,
    }
