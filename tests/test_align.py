"""Tests for the alignment engine and HTTP layer."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.io import AlignConfig, InputError, parse_events, run_alignment
from app.main import app
from app.matching import (
    ConstantMapping,
    PiecewiseLinearMapping,
    align_sequences,
    robust_sigma,
)
from app.timeutil import parse_timestamp, to_iso

client = TestClient(app)


# --- timestamp helpers ------------------------------------------------------


def test_parse_units():
    assert parse_timestamp(1_790_000_000) == 1_790_000_000.0
    # Small absolute epoch (1970s) stays seconds.
    assert parse_timestamp(1_000_000) == 1_000_000.0
    assert parse_timestamp(1_790_000_000_000) == pytest.approx(1_790_000_000.0)
    assert parse_timestamp(1_790_000_000_000_000) == pytest.approx(
        1_790_000_000.0
    )
    assert parse_timestamp(1_790_000_000_000_000_000) == pytest.approx(
        1_790_000_000.0
    )
    assert parse_timestamp("2026-09-22T00:00:00Z") == pytest.approx(
        parse_timestamp("2026-09-22T00:00:00+00:00")
    )
    with pytest.raises(ValueError):
        parse_timestamp("not-a-time")
    iso = to_iso(1_790_000_000.0)
    assert iso.endswith("Z") and "T" in iso


# --- synthetic alignment scenarios ------------------------------------------


def make_ref(n=40, step=10.0, base=1_000_000.0):
    labels = ["A", "B", "C", "D", "E"]
    return [
        {"timestamp": base + i * step, "label": labels[i % 5], "id": i}
        for i in range(n)
    ]


def drift(t):
    """Piecewise true map: offset+rate drift, then a 60 s clock jump."""
    if t < 1_000_200.0:
        return t + 5.0 + 0.02 * (t - 1_000_000.0)
    return t + 65.0 + 0.01 * (t - 1_000_000.0)


def make_drifting_target(refs, drop_ids=(), insert=()):
    out = []
    for r in refs:
        if r["id"] in drop_ids:
            continue
        out.append({
            "timestamp": drift(r["timestamp"]) + 0.001 * (r["id"] % 3),
            "label": r["label"],
            "id": r["id"],
        })
    for at, lab in insert:
        out.append({"timestamp": at, "label": lab, "id": -1})
    out.sort(key=lambda r: r["timestamp"])
    return out


def test_piecewise_recovers_known_drift():
    refs = make_ref()
    tgts = make_drifting_target(
        refs, drop_ids={17}, insert=[(1_000_155.0, "GHOST"),
                                     (1_000_301.0, "ALARM_X")]
    )
    res = run_alignment(refs, tgts, AlignConfig())
    s = res["summary"]

    assert s["matched_count"] == 39
    assert s["inserted_count"] == 2
    assert s["missing_count"] == 1
    assert s["mapping_type"] == "PiecewiseLinearMapping"
    assert s["residual_rms"] == pytest.approx(0.0, abs=1e-3)
    assert 0.0 <= s["mean_confidence"] <= 1.0

    missing = {m["index"] for m in res["missing_reference_events"]}
    assert missing == {17}
    inserted = {e["label"] for e in res["aligned_events"]
                if e["status"] == "inserted"}
    assert inserted == {"GHOST", "ALARM_X"}

    # Matched calibration must reproduce reference times.
    for a in res["anchors"]:
        assert a["mapped_time"] == pytest.approx(a["reference_time"], abs=2e-3)


def test_global_rate_drift_is_fit_when_no_breakpoint():
    refs = make_ref(n=30)
    tgts = [
        {"timestamp": 1.001 * r["timestamp"] - 3.2, "label": r["label"],
         "id": r["id"]}
        for r in refs
    ]
    res = run_alignment(refs, tgts, AlignConfig())
    assert res["summary"]["matched_count"] == 30
    for a in res["anchors"]:
        assert a["mapped_time"] == pytest.approx(a["reference_time"], abs=1e-6)
    assert res["clock_mapping"]["global_slope"] == pytest.approx(
        1.0 / 1.001, rel=1e-4
    )


def test_repeated_same_labels_match_in_order():
    refs = [{"timestamp": 100 + 10 * i, "label": "X"} for i in range(20)]
    tgts = [{"timestamp": 112 + 10 * i, "label": "X"} for i in range(19)]
    res = run_alignment(refs, tgts, AlignConfig())
    assert res["summary"]["matched_count"] == 19
    assert res["summary"]["missing_count"] == 1
    # Monotone, non-crossing alignment.
    pairs = [(a["reference_index"], a["target_index"]) for a in res["anchors"]]
    assert pairs == sorted(pairs)
    assert all(j2 > j1 and i2 > i1
               for (j1, i1), (j2, i2) in zip(pairs, pairs[1:]))


def test_no_common_labels():
    refs = [{"timestamp": 100.0 + i, "label": f"R{i}"} for i in range(5)]
    tgts = [{"timestamp": 200.0 + i, "label": f"T{i}"} for i in range(5)]
    res = run_alignment(refs, tgts, AlignConfig())
    assert res["summary"]["matched_count"] == 0
    assert res["summary"]["missing_count"] == 5
    assert res["summary"]["inserted_count"] == 5
    assert res["clock_mapping"]["type"] == "ConstantMapping"
    # Unmatched target events still get calibrated timestamps (identity-ish).
    ev = res["aligned_events"][0]
    assert ev["calibrated_time"] == pytest.approx(ev["original_time"], abs=1e-9)
    assert ev["confidence"] == 0.0


def test_extrapolation_flags():
    refs = make_ref(n=20)
    tgts = make_drifting_target(refs)
    tgts.insert(0, {"timestamp": drift(1_000_000.0) - 500, "label": "EARLY",
                    "id": -9})
    tgts.append({"timestamp": drift(1_000_000.0 + 19 * 10) + 500,
                 "label": "LATE", "id": -8})
    res = run_alignment(refs, tgts, AlignConfig())
    status = {(e["label"], e["status"], e["extrapolated"])
              for e in res["aligned_events"]
              if e["label"] in ("EARLY", "LATE")}
    assert status == {("EARLY", "inserted", True), ("LATE", "inserted", True)}
    assert not any(e["extrapolated"] for e in res["aligned_events"]
                   if e["status"] == "matched")


def test_single_anchor_uses_constant_offset():
    refs = [{"timestamp": 100.0, "label": "A"},
            {"timestamp": 110.0, "label": "B"}]
    tgts = [{"timestamp": 150.0, "label": "A"},
            {"timestamp": 160.0, "label": "C"}]
    res = run_alignment(refs, tgts, AlignConfig())
    assert res["summary"]["matched_count"] == 1
    a = res["anchors"][0]
    assert a["mapped_time"] == pytest.approx(100.0)
    assert res["clock_mapping"]["type"] == "ConstantMapping"


def test_empty_streams_error():
    with pytest.raises(InputError):
        run_alignment([], [], AlignConfig())


def test_missing_fields_error():
    with pytest.raises(InputError):
        parse_events([{"nope": 1}], AlignConfig())
    with pytest.raises(InputError):
        parse_events('{"label": "A", "when": 1}', AlignConfig())


def test_jsonl_string_input_and_iso_echo():
    refs = '{"timestamp": 1000.0, "label": "A"}\n{"timestamp": 1010.0, "label": "B"}\n'
    tgts = '{"timestamp": 1005.0, "label": "A"}\n{"timestamp": 1015.0, "label": "B"}\n'
    res = run_alignment(refs, tgts, AlignConfig())
    assert res["summary"]["matched_count"] == 2
    for ev in res["aligned_events"]:
        assert ev["calibrated_time_iso"].endswith("Z")


def test_robust_sigma():
    # Pure constant residuals -> degenerate MAD, floored scale.
    assert robust_sigma([1.0, 1.0, 1.0, 1.0]) == 1.0
    # MAD tolerates an extreme outlier.
    assert robust_sigma([1.0, 1.0, 1.0, 1.0, 100.0]) < 50.0
    assert robust_sigma([]) == 1.0
    assert robust_sigma([1.0]) == 1.0


# --- mapping objects --------------------------------------------------------


def test_piecewise_extrapolation_uses_edge_slope():
    cfg = AlignConfig()
    refs = parse_events(
        [{"timestamp": 0.0, "label": "A"}, {"timestamp": 100.0, "label": "B"},
         {"timestamp": 200.0, "label": "C"}], cfg)
    tgts = parse_events(
        [{"timestamp": 10.0, "label": "A"}, {"timestamp": 120.0, "label": "B"},
         {"timestamp": 230.0, "label": "C"}], cfg)
    anchors, mapping, _scale = align_sequences(refs, tgts, iterations=1)
    assert isinstance(mapping, PiecewiseLinearMapping)
    # left edge slope = (100-0)/(120-10) -> predict t=0
    assert mapping.predict(0.0) == pytest.approx(
        0.0 - 10.0 * (100.0 / 110.0)
    )
    assert mapping.is_extrapolated(0.0)
    assert not mapping.is_extrapolated(120.0)


def test_constant_mapping():
    m = ConstantMapping(7.0)
    assert m.predict(3.0) == 10.0
    assert m.is_extrapolated(3.0)


# --- HTTP layer -------------------------------------------------------------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "version" in body


def test_align_json_body():
    refs = make_ref(n=20)
    tgts = make_drifting_target(refs, drop_ids={5})
    r = client.post("/align", json={"reference": refs, "target": tgts})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["matched_count"] == 19
    assert body["summary"]["missing_count"] == 1
    assert "elapsed_seconds" in body["summary"]
    assert body["clock_mapping"]["segments"]


def test_align_multipart_files():
    import json as _json
    refs = make_ref(n=10)
    tgts = make_drifting_target(refs)
    rf = _json.dumps
    r = client.post(
        "/align",
        files={
            "reference": ("reference.jsonl",
                          "\n".join(rf(x) for x in refs), "application/x-ndjson"),
            "target": ("target.jsonl",
                       "\n".join(rf(x) for x in tgts), "application/x-ndjson"),
        },
        data={"iterations": "2"},
    )
    assert r.status_code == 200
    assert r.json()["summary"]["matched_count"] == 10


def test_align_bad_input_400():
    r = client.post("/align", json={"reference": [{"x": 1}], "target": []})
    assert r.status_code == 400
    r = client.post("/align", json={"reference": []})
    assert r.status_code == 400


def test_align_unsupported_media_type():
    r = client.post("/align", content="abc",
                    headers={"content-type": "text/plain"})
    assert r.status_code == 415


# --- manual anchors ---------------------------------------------------------


ANCHORS = [
    {"reference_index": 0, "target_index": 0},
    {"reference_index": 25, "target_index": 23},
    {"reference_index": 39, "target_index": 36},
]


def test_manual_anchors_locked_and_segmented():
    refs = make_ref()
    tgts = make_drifting_target(refs, drop_ids={5, 17, 30})
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=ANCHORS))
    s = res["summary"]

    assert s["manual_anchor_count"] == 3
    assert s["auto_anchor_count"] == s["matched_count"] - 3
    assert s["matched_count"] == 37            # 40 refs - 3 drops
    assert s["inserted_count"] == 0           # no spurious events

    by_pair = {(a["reference_index"], a["target_index"]): a
               for a in res["anchors"]}
    for ri, ti in ANCHORS:
        a = by_pair[(ri, ti)]
        assert a["source"] == "manual"
        assert a["confidence"] == 1.0

    # Automatic matches never reuse a locked event or cross a lock.
    auto = [(a["reference_index"], a["target_index"])
            for a in res["anchors"] if a["source"] == "auto"]
    locked_r = {ri for ri, _ in ANCHORS}
    locked_t = {ti for _, ti in ANCHORS}
    assert all(ri not in locked_r and ti not in locked_t for ri, ti in auto)
    ordered_locks = sorted(ANCHORS, key=lambda x: x["reference_index"])
    for ri, ti in auto:
        for left, right in zip(ordered_locks, ordered_locks[1:]):
            if left["reference_index"] < ri < right["reference_index"]:
                assert left["target_index"] < ti < right["target_index"]
        if ri > ordered_locks[-1]["reference_index"]:
            assert ti > ordered_locks[-1]["target_index"]
        if ri < ordered_locks[0]["reference_index"]:
            assert ti < ordered_locks[0]["target_index"]

    # The merged piecewise map passes through every lock.
    for a in res["anchors"]:
        assert a["mapped_time"] == pytest.approx(a["reference_time"], abs=2e-3)


def test_manual_anchor_single_lock():
    refs, tgts = make_ref(n=20), make_drifting_target(make_ref(n=20))
    res = run_alignment(
        refs, tgts,
        AlignConfig(manual_anchors=[{"reference_index": 10,
                                      "target_index": 10}]),
    )
    assert res["summary"]["manual_anchor_count"] == 1
    lock = [a for a in res["anchors"] if a["source"] == "manual"]
    assert len(lock) == 1
    assert lock[0]["reference_index"] == 10
    assert lock[0]["confidence"] == 1.0


def test_manual_anchors_only():
    refs, tgts = make_ref(n=5), make_drifting_target(make_ref(n=5))
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=[
        {"reference_index": i, "target_index": i} for i in range(5)
    ]))
    s = res["summary"]
    assert s["matched_count"] == 5
    assert s["auto_anchor_count"] == 0
    assert s["inserted_count"] == 0
    assert s["missing_count"] == 0
    assert res["clock_mapping"]["type"] == "PiecewiseLinearMapping"
    assert all(a["source"] == "manual" for a in res["anchors"])


def test_manual_anchors_correct_natural_match():
    # One target occurrence dropped: truth shifts after the gap.  Locking
    # (ref 3, tgt 2) forces the matcher into the correct gaps.
    refs = [{"timestamp": 100.0 + 10 * i, "label": "A"} for i in range(6)]
    tgts = [{"timestamp": 103.0 + 10 * i, "label": "A"}
            for i in (0, 1, 3, 4, 5)]
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=[
        {"reference_index": 3, "target_index": 2}]))
    pairs = {(a["reference_index"], a["target_index"])
             for a in res["anchors"]}
    assert (3, 2) in pairs
    assert (2, 2) not in pairs and (3, 3) not in pairs
    # Strictly ordered, no crossing on either side.
    rs = sorted(r for r, _ in pairs)
    ts = [t for r, t in sorted(pairs)]
    assert rs == sorted(set(rs))
    assert ts == sorted(ts)


def test_manual_anchors_json_body():
    refs, tgts = make_ref(n=20), make_drifting_target(make_ref(n=20))
    r = client.post("/align", json={
        "reference": refs, "target": tgts,
        "anchors": [{"reference_index": 0, "target_index": 0},
                    {"reference_index": 19, "target_index": 19}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["manual_anchor_count"] == 2
    locks = [a for a in body["anchors"] if a["source"] == "manual"]
    assert {(a["reference_index"], a["target_index"]) for a in locks} == {
        (0, 0), (19, 19)}


def test_manual_anchors_multipart_form():
    refs = make_ref(n=10)
    tgts = make_drifting_target(refs)
    r = client.post(
        "/align",
        files={
            "reference": ("reference.jsonl",
                          "\n".join(json.dumps(x) for x in refs),
                          "application/x-ndjson"),
            "target": ("target.jsonl",
                       "\n".join(json.dumps(x) for x in tgts),
                       "application/x-ndjson"),
        },
        data={"anchors": json.dumps([
            {"reference_index": 2, "target_index": 2},
            {"reference_index": 8, "target_index": 8},
        ])},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["manual_anchor_count"] == 2
    assert all(a["source"] in ("manual", "auto") for a in body["anchors"])


@pytest.mark.parametrize("anchors, fragment", [
    ([{"reference_index": 99, "target_index": 0}], "reference_index 99 out of range"),
    ([{"reference_index": 0, "target_index": 99}], "target_index 99 out of range"),
    ([{"reference_index": -1, "target_index": 0}], "reference_index"),
    ([{"reference_index": 1, "target_index": 1},
      {"reference_index": 2, "target_index": 1}], "already locked"),
    ([{"reference_index": 1, "target_index": 1},
      {"reference_index": 1, "target_index": 2}], "already locked"),
    ([{"reference_index": 2, "target_index": 5},
      {"reference_index": 6, "target_index": 3}], "cross"),
    ([{"reference_index": 0}], "target_index"),
])
def test_manual_anchors_validation_400(anchors, fragment):
    refs = [{"timestamp": 100.0 + 10 * i, "label": "X"} for i in range(10)]
    tgts = [{"timestamp": 150.0 + 10 * i, "label": "X"} for i in range(10)]
    with pytest.raises(InputError) as exc:
        run_alignment(refs, tgts, AlignConfig(manual_anchors=anchors))
    assert fragment in str(exc.value)
    # Same condition through HTTP -> 400 with a locatable message.
    r = client.post("/align", json={"reference": refs, "target": tgts,
                                    "anchors": anchors})
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


def test_manual_anchor_label_mismatch_400():
    refs = make_ref(n=10)
    tgts = make_drifting_target(refs)
    tgts[3] = {"timestamp": tgts[3]["timestamp"], "label": "OTHER"}
    r = client.post("/align", json={
        "reference": refs, "target": tgts,
        "anchors": [{"reference_index": 3, "target_index": 3}],
    })
    assert r.status_code == 400
    assert "label mismatch" in r.json()["detail"]
    assert "anchors[0]" in r.json()["detail"]


def test_manual_anchors_bad_json_400():
    refs = make_ref(n=5)
    tgts = make_drifting_target(refs)
    r = client.post(
        "/align",
        files={
            "reference": ("r.jsonl",
                          "\n".join(json.dumps(x) for x in refs),
                          "application/x-ndjson"),
            "target": ("t.jsonl",
                       "\n".join(json.dumps(x) for x in tgts),
                       "application/x-ndjson"),
        },
        data={"anchors": "not-json"},
    )
    assert r.status_code == 400
    assert "anchors" in r.json()["detail"]
