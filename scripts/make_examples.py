#!/usr/bin/env python3
"""Generate reference/target example JSONL logs.

The target log has:
* a piecewise clock drift (offset + slow rate drift, then a clock jump),
* two inserted (spurious) events that do not exist in the reference,
* one reference event dropped (missing on the target side),
* timestamp jitter.

Run:  python scripts/make_examples.py
"""

from __future__ import annotations

import json
import math
import os
import random

LABELS = [
    "LINE_START", "HEATER_ON", "TEMP_RAMP", "VALVE_OPEN",
    "PRESSURE_OK", "WELD_BEGIN", "WELD_END", "COOLING",
    "QC_SNAPSHOT", "LINE_STOP",
]

T0 = 1_790_000_000.0          # fixed epoch base for reproducible examples
JUMP_AT = T0 + 900.0          # clock jump happens at t=900s


def target_time(ref_time: float) -> float:
    """True (hidden from the API) target-clock reading for a reference time."""
    if ref_time < JUMP_AT:
        return ref_time + 5.0 + 0.02 * ref_time_span(ref_time)
    return ref_time + 47.0 + 0.015 * ref_time_span(ref_time)


def ref_time_span(t: float) -> float:
    return t - T0


def main() -> None:
    random.seed(42)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "examples")
    os.makedirs(out_dir, exist_ok=True)

    # 60 reference events, one every 30 s, repeating the label cycle.
    refs = []
    for k in range(60):
        rt = T0 + 30 * k
        refs.append({
            "timestamp": round(rt, 3),
            "label": LABELS[k % len(LABELS)],
            "line": 1,
            "seq": k,
        })

    # Target: drop reference #20 (missing), insert two spurious events.
    targets = []
    for rec in refs:
        if rec["seq"] == 20:
            continue
        tt = target_time(rec["timestamp"]) + random.uniform(-0.05, 0.05)
        targets.append({
            "timestamp": round(tt, 3),
            "label": rec["label"],
            "line": 1,
            "seq": rec["seq"],
        })

    targets.insert(25, {
        "timestamp": round(target_time(T0 + 760.0) + random.uniform(-0.05, 0.05), 3),
        "label": "SPURIOUS_ALARM",
        "line": 1,
        "seq": -1,
    })
    targets.insert(45, {
        "timestamp": round(target_time(T0 + 1330.0) + random.uniform(-0.05, 0.05), 3),
        "label": "GHOST_RETRY",
        "line": 1,
        "seq": -2,
    })

    with open(os.path.join(out_dir, "reference.jsonl"), "w") as f:
        for rec in refs:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(out_dir, "target.jsonl"), "w") as f:
        for rec in targets:
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {len(refs)} reference / {len(targets)} target records "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
