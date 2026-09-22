"""Tests for the alignment engine and HTTP layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.io import (
    AlignConfig,
    InputError,
    ManualAnchorSpec,
    parse_events,
    run_alignment,
)
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


def make_x(n, t0=100.0, step=10.0, offset=15.0):
    refs = [{"timestamp": t0 + i * step, "label": "X"} for i in range(n)]
    tgts = [{"timestamp": t0 + offset + i * step, "label": "X"}
            for i in range(n)]
    return refs, tgts


def test_manual_anchor_forces_nonobvious_match():
    # Repeated "X" events with a constant +15 s offset.  Two manual anchors
    # deliberately force a shifted alignment in the middle.  The locked
    # pairs must be present verbatim, and the automatic DP must stay inside
    # the intervals the anchors carve out -- no locked event reused and no
    # auto pair crossing an anchor boundary.
    refs, tgts = make_x(10)
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=[
        ManualAnchorSpec(reference_index=5, target_index=3),
        ManualAnchorSpec(reference_index=7, target_index=8),
    ]))
    s = res["summary"]
    assert s["manual_anchor_count"] == 2
    assert s["auto_anchor_count"] == s["matched_count"] - 2
    manual = {(a["reference_index"], a["target_index"])
              for a in res["anchors"] if a["source"] == "manual"}
    assert manual == {(5, 3), (7, 8)}

    locked_r = {5, 7}
    locked_t = {3, 8}
    pairs = [(a["reference_index"], a["target_index"])
             for a in res["anchors"]]
    assert pairs == sorted(pairs)
    for a in res["anchors"]:
        rj, tj = a["reference_index"], a["target_index"]
        if a["source"] == "auto":
            assert rj not in locked_r and tj not in locked_t
            if rj < 5:
                assert tj < 3
            elif rj < 7:
                assert 3 < tj < 8
            else:
                assert tj > 8
    # The piecewise map interpolates exactly through every anchor.
    for a in res["anchors"]:
        assert a["residual"] == pytest.approx(0.0, abs=1e-9)


def test_manual_anchors_split_intervals_without_crossing():
    refs, tgts = make_x(20)
    specs = [
        ManualAnchorSpec(reference_index=4, target_index=4),
        ManualAnchorSpec(reference_index=15, target_index=15),
    ]
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=specs))
    pairs = [(a["reference_index"], a["target_index"])
             for a in res["anchors"]]
    assert (4, 4) in pairs and (15, 15) in pairs
    assert pairs == sorted(pairs)
    assert all(j2 > j1 and i2 > i1
               for (j1, i1), (j2, i2) in zip(pairs, pairs[1:]))
    # Auto matches stay strictly inside the locked intervals.
    for a in res["anchors"]:
        if a["source"] == "auto":
            rj, tj = a["reference_index"], a["target_index"]
            if rj < 4:
                assert tj < 4
            elif 4 < rj < 15:
                assert 4 < tj < 15
            else:
                assert tj > 15
            assert rj not in (4, 15) and tj not in (4, 15)
    assert res["summary"]["manual_anchor_count"] == 2


def test_manual_anchor_single_seeds_constant_mapping():
    refs, tgts = make_x(6)
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=[
        ManualAnchorSpec(reference_index=2, target_index=2),
    ]))
    assert res["summary"]["manual_anchor_count"] == 1
    assert res["summary"]["matched_count"] == 6
    for a in res["anchors"]:
        assert a["mapped_time"] == pytest.approx(a["reference_time"], abs=1e-6)


def test_manual_anchors_all_events_locked():
    refs, tgts = make_x(4)
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=[
        ManualAnchorSpec(reference_index=i, target_index=i)
        for i in range(4)
    ]))
    assert res["summary"]["matched_count"] == 4
    assert res["summary"]["manual_anchor_count"] == 4
    assert res["summary"]["auto_anchor_count"] == 0
    assert all(a["source"] == "manual" for a in res["anchors"])


def test_manual_anchor_with_drifting_fixture():
    refs = make_ref(n=30)
    tgts = make_drifting_target(refs, drop_ids={17})
    specs = [ManualAnchorSpec(reference_index=5, target_index=5),
             ManualAnchorSpec(reference_index=22, target_index=21)]
    res = run_alignment(refs, tgts, AlignConfig(manual_anchors=specs))
    assert res["summary"]["manual_anchor_count"] == 2
    pairs = {(a["reference_index"], a["target_index"])
             for a in res["anchors"]}
    assert (5, 5) in pairs and (22, 21) in pairs
    # The dropped reference 17 stays missing.
    assert 17 in {m["index"] for m in res["missing_reference_events"]}


def test_manual_anchor_index_out_of_bounds():
    refs, tgts = make_x(5)
    with pytest.raises(InputError, match="anchors\\[0\\].reference_index"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(reference_index=99, target_index=0)]))
    with pytest.raises(InputError, match="anchors\\[1\\].target_index"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(reference_index=0, target_index=0),
            ManualAnchorSpec(reference_index=1, target_index=99)]))


def test_manual_anchor_duplicate_and_label_errors():
    refs = [{"timestamp": 100.0 + 10 * i, "label": "X"} for i in range(4)]
    tgts = [{"timestamp": 115.0 + 10 * i, "label": "X"} for i in range(4)]
    with pytest.raises(InputError, match="already locked by anchors\\[0\\]"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(0, 0), ManualAnchorSpec(0, 1)]))
    with pytest.raises(InputError, match="already locked by anchors\\[0\\]"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(0, 0), ManualAnchorSpec(1, 0)]))

    refs2 = [{"timestamp": 100.0, "label": "A"},
             {"timestamp": 110.0, "label": "B"}]
    tgts2 = [{"timestamp": 105.0, "label": "A"},
             {"timestamp": 115.0, "label": "C"}]
    # Chronological input: index 0 on both sides is 'A' -- consistent; the
    # mismatch is between reference[1]='B' and target[1]='C'.
    with pytest.raises(InputError, match="anchors\\[0\\]: label mismatch"):
        run_alignment(refs2, tgts2, AlignConfig(manual_anchors=[
            ManualAnchorSpec(1, 1)]))
    with pytest.raises(InputError, match="anchors\\[0\\]: label mismatch"):
        run_alignment(refs2, tgts2, AlignConfig(manual_anchors=[
            ManualAnchorSpec(0, 1)]))


def test_manual_anchor_order_crossing():
    refs, tgts = make_x(5)
    with pytest.raises(InputError, match="order crossing"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(0, 3), ManualAnchorSpec(3, 0)]))
    # Crossing declared in reverse input order is still a crossing.
    with pytest.raises(InputError, match="order crossing"):
        run_alignment(refs, tgts, AlignConfig(manual_anchors=[
            ManualAnchorSpec(3, 0), ManualAnchorSpec(0, 3)]))


def test_manual_anchor_http_json_and_400s():
    refs, tgts = make_x(10)
    r = client.post("/align", json={
        "reference": refs, "target": tgts,
        "anchors": [{"reference_index": 2, "target_index": 2}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["manual_anchor_count"] == 1
    assert body["anchors"][2]["source"] in ("manual", "auto")
    assert any(a["source"] == "manual" for a in body["anchors"])

    bad_bodies = [
        [{"reference_index": 50, "target_index": 0}],
        [{"reference_index": 0, "target_index": -1}],
        [{"target_index": 0}],
        [{"reference_index": "0", "target_index": 0}],
        [{"reference_index": 0, "target_index": 0},
         {"reference_index": 1, "target_index": 0}],
        42,
    ]
    for anchors in bad_bodies:
        r = client.post("/align", json={
            "reference": refs, "target": tgts, "anchors": anchors})
        assert r.status_code == 400, anchors
        assert r.json()["detail"]


def test_manual_anchor_http_multipart():
    import json as _json
    refs, tgts = make_x(8)
    r = client.post(
        "/align",
        files={
            "reference": ("reference.jsonl",
                          "\n".join(_json.dumps(x) for x in refs),
                          "application/x-ndjson"),
            "target": ("target.jsonl",
                       "\n".join(_json.dumps(x) for x in tgts),
                       "application/x-ndjson"),
            "anchors": (None, _json.dumps(
                [{"reference_index": 1, "target_index": 1},
                 {"reference_index": 6, "target_index": 6}])),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["manual_anchor_count"] == 2
    assert {(a["reference_index"], a["target_index"])
            for a in body["anchors"] if a["source"] == "manual"} == {(1, 1),
                                                                      (6, 6)}

    # One JSON object per repeatable form part.
    r = client.post(
        "/align",
        files=[
            ("reference", ("reference.jsonl",
                           "\n".join(_json.dumps(x) for x in refs))),
            ("target", ("target.jsonl",
                        "\n".join(_json.dumps(x) for x in tgts))),
            ("anchors", (None, _json.dumps(
                {"reference_index": 1, "target_index": 1}))),
            ("anchors", (None, _json.dumps(
                {"reference_index": 6, "target_index": 6}))),
        ],
    )
    assert r.status_code == 200
    assert r.json()["summary"]["manual_anchor_count"] == 2

    # Malformed anchor JSON -> 400 with a locatable message.
    r = client.post(
        "/align",
        files=[
            ("reference", ("reference.jsonl",
                           "\n".join(_json.dumps(x) for x in refs))),
            ("target", ("target.jsonl",
                        "\n".join(_json.dumps(x) for x in tgts))),
            ("anchors", (None, "{not-json")),
        ],
    )
    assert r.status_code == 400
    assert "anchors form part #0" in r.json()["detail"]
