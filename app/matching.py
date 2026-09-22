"""Monotonic label/DP event matching and piecewise-linear clock mapping.

Algorithm overview
------------------
1. Same-label event pairs are the only admissible matches.  Each candidate
   pair gets a weight

       w = label_bonus + time_weight * exp(-(r / sigma)^2 / 2)

   where ``r = mapping(target_time) - ref_time`` is the timestamp residual
   under the clock map estimated so far.  ``label_bonus`` is fixed at 1.0
   and the Gaussian temporal term reaches 0.5, so a consistent pair adds
   ~1.4-1.5 while skipping an event adds 0: inserted/missing events are
   skipped naturally, no gap penalty needed.

2. Order must be preserved on both sides (industrial logs are monotonic).
   Finding the maximum-weight chain of pairs is a weighted LIS solved with
   a Fenwick (BIT) maximum keyed on reference position in O(C log m) for C
   admissible same-label pairs.

3. The matched anchors are used to fit the piecewise-linear clock map
   (linear interpolation between anchors, edge slope extrapolation outside
   the anchor span).  Steps 1-2 are re-run with the refined map; the soft
   temporal Gaussian lets anchors migrate and the solution stabilises
   within a few iterations.

4. *Manual anchors* supplied by the client are non-skippable locks.  The
   sequences are cut at those locks into half-open gaps (before the first,
   between consecutive pairs, after the last) and the matcher above runs
   independently inside each gap: locked events are excluded from the
   candidate windows, so automatic matches can neither reuse a locked
   event nor cross a manual anchor.  All anchors are then merged and the
   piecewise map is fit once over the union.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Sequence

from .models import Anchor, Event

# --- tunables ---------------------------------------------------------------

LABEL_BONUS = 1.0
DEFAULT_TIME_WEIGHT = 0.5
DEFAULT_ITERATIONS = 3
WINDOW_SIGMA = 3.0               # candidate pruning radius (refinement passes)
CANDIDATE_CAP = 32               # max candidates per target event


# --- small numerical helpers ------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in [0, 1].  Values must be sorted."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def seed_pairs(
    refs: Sequence[Event], tgts: Sequence[Event]
) -> list[tuple[Event, Event]]:
    """In-order pairing of same-label events used to bootstrap the clock map."""
    by_ref: dict[str, list[Event]] = {}
    for ev in refs:
        by_ref.setdefault(ev.label, []).append(ev)
    buckets: dict[str, list[Event]] = {}
    for ev in tgts:
        buckets.setdefault(ev.label, []).append(ev)
    pairs: list[tuple[Event, Event]] = []
    for label, rlist in by_ref.items():
        pairs.extend(zip(rlist, buckets.get(label, [])))
    return pairs


def _nearest_neighbour_pairs(
    refs: Sequence[Event], tgts: Sequence[Event]
) -> list[tuple[Event, Event]]:
    """Pair each target with its nearest same-label reference on raw times.

    No positional alignment is assumed, so inserted/missing events cannot
    corrupt the pairing.  Sound when the true clock offset is below half the
    inter-event gap of a repeated label.
    """
    ref_by_label: dict[str, list[Event]] = {}
    for ev in refs:
        ref_by_label.setdefault(ev.label, []).append(ev)
    pairs: list[tuple[Event, Event]] = []
    for tgt in tgts:
        bucket = ref_by_label.get(tgt.label)
        if not bucket:
            continue
        times = [e.time for e in bucket]
        k = bisect_left(times, tgt.time)
        best_k, best_d = -1, float("inf")
        if k < len(bucket):
            best_k, best_d = k, abs(times[k] - tgt.time)
        if k > 0 and abs(times[k - 1] - tgt.time) < best_d:
            best_k = k - 1
        if best_k >= 0:
            pairs.append((bucket[best_k], tgt))
    return pairs


def bootstrap_mapping(
    refs: Sequence[Event], tgts: Sequence[Event]
) -> ClockMapping:
    """Initial clock map, chosen between two complementary bootstraps:

    * *temporal* -- nearest same-label reference on raw timestamps; robust
      to inserted/missing events, but only when the drift within the span is
      smaller than the repeated-label period;
    * *positional* -- in-order ``zip`` pairing of same-label events; robust
      to arbitrarily large rate drift, but assumes streams are positionally
      aligned (few insertions/deletions).

    Each yields offsets; the pairing whose offsets are most self-consistent
    (small MAD, measured relative to the span so it works in any units)
    wins.  A least-squares line is fit through the winning pairs, covering
    both offset and rate drift.  Clock jumps just appear as line outliers
    repaired in later piecewise passes.
    """
    temporal = _nearest_neighbour_pairs(refs, tgts)
    positional = seed_pairs(refs, tgts)

    span = max(
        (max(e.time for e in refs) - min(e.time for e in refs)) if refs else 0.0,
        (max(e.time for e in tgts) - min(e.time for e in tgts)) if tgts else 0.0,
        1.0,
    )

    def score(pairs: list[tuple[Event, Event]]) -> float:
        if len(pairs) < 2:
            return float("inf")
        offsets = [t.time - r.time for r, t in pairs]
        # Fraction of the span unexplained by a single common offset.
        return robust_sigma(offsets) / span

    pairs = temporal if temporal and (
        not positional or score(temporal) <= score(positional)
    ) else positional

    if not pairs:
        return ConstantMapping(0.0)
    if len(pairs) < 3:
        return ConstantMapping(
            median(t.time - r.time for r, t in pairs)
        )
    ts = [t.time for _, t in pairs]
    ys = [r.time for r, _ in pairs]
    n = len(pairs)
    mt, my = sum(ts) / n, sum(ys) / n
    var = sum((t - mt) ** 2 for t in ts)
    if var <= 1e-6:
        return ConstantMapping(
            median(t.time - r.time for r, t in pairs)
        )
    slope = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / var
    intercept = my - slope * mt
    return LinearMapping(slope, intercept, min(ts), max(ts))


def robust_sigma(residuals: Sequence[float]) -> float:
    """MAD-based Gaussian scale estimate; never collapses below 1.0.

    The floor matters before the first fit: the matching candidate window is
    proportional to this sigma, and a degenerate MAD (all residuals equal,
    e.g. a constant offset) must not shrink the window to zero.
    """
    if not residuals:
        return 1.0
    values = sorted(residuals)
    med = _percentile(values, 0.5)
    mad = _percentile(sorted(abs(v - med) for v in values), 0.5)
    sigma = 1.4826 * mad
    if sigma > 1e-9:
        return sigma
    spread = 0.5 * (
        _percentile(values, 0.84) - _percentile(values, 0.16)
    )
    return max(spread, 1.0)


# --- clock mappings (map *target* time into reference time) -----------------


class ClockMapping:
    def predict(self, t: float) -> float:
        raise NotImplementedError

    def is_extrapolated(self, t: float) -> bool:
        raise NotImplementedError

    @property
    def slope(self) -> float:
        raise NotImplementedError


class ConstantMapping(ClockMapping):
    """Pure offset ``ref = t + b`` (used when there is a single anchor)."""

    def __init__(self, b: float):
        self.b = b

    def predict(self, t: float) -> float:
        return t + self.b

    def is_extrapolated(self, t: float) -> bool:
        return True

    @property
    def slope(self) -> float:
        return 0.0


class LinearMapping(ClockMapping):
    """Global least-squares line ``ref = slope * t + intercept``."""

    def __init__(self, slope: float, intercept: float,
                 t_min: float, t_max: float):
        self._slope = slope
        self.intercept = intercept
        self.t_min = t_min
        self.t_max = t_max

    @classmethod
    def fit(cls, anchors: Sequence[Anchor]) -> "LinearMapping":
        ts = [a.tgt.time for a in anchors]
        ys = [a.ref.time for a in anchors]
        n = len(ts)
        mt = sum(ts) / n
        my = sum(ys) / n
        var = sum((t - mt) ** 2 for t in ts)
        if var <= 1e-12:
            slope = 1.0
        else:
            slope = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / var
        return cls(slope, my - slope * mt, min(ts), max(ts))

    def predict(self, t: float) -> float:
        return self._slope * t + self.intercept

    def is_extrapolated(self, t: float) -> bool:
        return t < self.t_min or t > self.t_max

    @property
    def slope(self) -> float:
        return self._slope


class PiecewiseLinearMapping(ClockMapping):
    """Linear interpolation between consecutive anchors.

    Outside the anchor span the slope of the nearest edge segment is used.
    """

    def __init__(self, anchors: Sequence[Anchor]):
        pts = sorted(anchors, key=lambda a: a.ref.time)
        self.points: list[tuple[float, float]] = [
            (a.tgt.time, a.ref.time) for a in pts
        ]
        self.xs = [p[0] for p in self.points]
        self.ys = [p[1] for p in self.points]

    def predict(self, t: float) -> float:
        xs, ys = self.xs, self.ys
        n = len(xs)
        if n == 0:
            return t
        if t <= xs[0]:
            if n >= 2 and xs[1] != xs[0]:
                return ys[0] + (t - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] + (t - xs[0])
        if t >= xs[-1]:
            if n >= 2 and xs[-1] != xs[-2]:
                return ys[-1] + (t - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + (t - xs[-1])
        k = bisect_right(xs, t) - 1
        x0, x1 = xs[k], xs[k + 1]
        if x1 == x0:
            return ys[k]
        return ys[k] + (t - x0) * (ys[k + 1] - ys[k]) / (x1 - x0)

    def is_extrapolated(self, t: float) -> bool:
        return t < self.xs[0] or t > self.xs[-1]

    @property
    def slope(self) -> float:
        """Global average slope over the anchor span."""
        if len(self.xs) < 2 or self.xs[-1] == self.xs[0]:
            return 1.0
        return (self.ys[-1] - self.ys[0]) / (self.xs[-1] - self.xs[0])


# --- Fenwick-tree maximum weighted LIS --------------------------------------


@dataclass
class _Pair:
    j: int          # reference event global position (seq) -- BIT key
    i: int          # target event global position (seq)
    w: float        # pair weight
    dp: float = 0.0
    parent: int = -1


def _build_label_indexes(
    refs: Sequence[Event],
) -> tuple[dict[str, list[Event]], dict[str, list[float]]]:
    """Time-sorted same-label buckets, reused across gaps and iterations."""
    ref_by_label: dict[str, list[Event]] = {}
    for ev in refs:
        ref_by_label.setdefault(ev.label, []).append(ev)
    for bucket in ref_by_label.values():
        bucket.sort(key=lambda e: e.time)
    ref_times_by_label = {
        label: [e.time for e in bucket]
        for label, bucket in ref_by_label.items()
    }
    return ref_by_label, ref_times_by_label


def _candidate_pairs(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    mapping: Callable[[float], float],
    sigma: float,
    time_weight: float,
    ref_lo: int = 0,
    ref_hi: int | None = None,
    tgt_lo: int = 0,
    tgt_hi: int | None = None,
    label_indexes: tuple[dict[str, list[Event]],
                         dict[str, list[float]]] | None = None,
) -> list[_Pair]:
    """Build admissible same-label (ref, target) candidate pairs.

    Same-label reference events inside the ±3σ temporal window around the
    mapped target time are admitted; if the window is empty (large initial
    offset, or the bootstrap line straddles a clock jump), the two nearest
    same-label events are used instead.

    ``ref_lo``/``ref_hi``/``tgt_lo``/``tgt_hi`` restrict both sides to the
    half-open ``[lo, hi)`` *sequence-index* windows.  They are used to run
    the matcher independently inside each gap between client-locked manual
    anchors so that automatic matches can neither reuse a locked event nor
    cross a manual anchor.

    ``label_indexes`` may supply pre-built same-label reference buckets to
    avoid rebuilding them per gap / iteration.
    """
    if ref_hi is None:
        ref_hi = len(refs)
    if tgt_hi is None:
        tgt_hi = len(tgts)
    if label_indexes is None:
        ref_by_label, ref_times_by_label = _build_label_indexes(refs)
    else:
        ref_by_label, ref_times_by_label = label_indexes

    radius = WINDOW_SIGMA * sigma
    inv_two_sigma2 = 1.0 / (2.0 * max(sigma, 1e-9) ** 2)
    pairs: list[_Pair] = []

    def emit(ev: Event, centre: float, tgt: Event) -> None:
        residual = centre - ev.time
        w = LABEL_BONUS + time_weight * math.exp(
            -(residual * residual) * inv_two_sigma2
        )
        pairs.append(_Pair(j=ev.seq, i=tgt.seq, w=w))

    for tgt in tgts:
        if not (tgt_lo <= tgt.seq < tgt_hi):
            continue
        bucket = ref_by_label.get(tgt.label)
        if not bucket:
            continue
        ref_times = ref_times_by_label[tgt.label]
        centre = mapping(tgt.time)

        lo = bisect_left(ref_times, centre - radius)
        hi = bisect_right(ref_times, centre + radius)

        # Keep only the candidates whose sequence position lies inside this
        # gap (locked boundary events and other gaps are excluded).
        kept = [
            k for k in range(lo, hi)
            if ref_lo <= bucket[k].seq < ref_hi
        ]
        if kept:
            candidates: Any = kept
            if len(kept) > CANDIDATE_CAP:
                candidates = sorted(
                    kept, key=lambda k: abs(bucket[k].time - centre)
                )[:CANDIDATE_CAP]
            for k in candidates:
                emit(bucket[k], centre, tgt)
        else:
            # Nothing in the temporal window belongs to this gap (e.g. the
            # seed line straddles a clock jump, or the window overlaps a
            # neighbouring gap).  Take the two nearest same-label events
            # *inside the gap*: keeping both sides of a jump lets the DP
            # choose instead of permanently locking the wrong event.
            ins = bisect_left(ref_times, centre)
            found_hi = None
            for k in range(ins, len(bucket)):
                if bucket[k].seq >= ref_hi:
                    break
                if bucket[k].seq >= ref_lo:
                    found_hi = k
                    break
            found_lo = None
            for k in range(ins - 1, -1, -1):
                if bucket[k].seq < ref_lo:
                    break
                if bucket[k].seq < ref_hi:
                    found_lo = k
                    break
            if found_hi is not None:
                emit(bucket[found_hi], centre, tgt)
            if found_lo is not None:
                emit(bucket[found_lo], centre, tgt)
    return pairs


def monotonic_match(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    mapping: ClockMapping,
    sigma: float,
    time_weight: float,
    ref_lo: int = 0,
    ref_hi: int | None = None,
    tgt_lo: int = 0,
    tgt_hi: int | None = None,
    label_indexes: tuple[dict[str, list[Event]],
                         dict[str, list[float]]] | None = None,
) -> list[Anchor]:
    """Maximum-weight monotonic matching via a Fenwick-tree weighted LIS.

    A single Fenwick tree keyed on the *global* reference position enforces
    strict ordering across all labels.  Pairs are processed in batches of
    equal target position (queries first, commits after) so each target
    event is used at most once; the prefix query enforces the same on the
    reference side.  No per-event closures are allocated in the hot loop.

    The half-open ``[lo, hi)`` sequence-index windows restrict matching to a
    single gap between locked manual anchors (see ``_candidate_pairs``).
    """
    pairs = _candidate_pairs(
        refs, tgts, mapping.predict, sigma, time_weight,
        ref_lo=ref_lo, ref_hi=ref_hi, tgt_lo=tgt_lo, tgt_hi=tgt_hi,
        label_indexes=label_indexes,
    )
    if not pairs:
        return []

    pairs.sort(key=lambda p: (p.i, p.j))

    size = len(refs) + 1
    bit_val = [float("-inf")] * size
    bit_id = [-1] * size

    best_overall, best_pid = 0.0, -1
    n_pairs = len(pairs)
    pos = 0

    while pos < n_pairs:
        i = pairs[pos].i
        end = pos + 1
        while end < n_pairs and pairs[end].i == i:
            end += 1

        # Phase 1: query only, so same-batch pairs never see each other.
        for pid in range(pos, end):
            p = pairs[pid]
            k = p.j                     # slots 1..j hold reference pos < j
            prev_val, prev_id = 0.0, -1
            while k:
                v = bit_val[k]
                if v > prev_val:
                    prev_val, prev_id = v, bit_id[k]
                k -= k & -k
            score = prev_val + p.w
            if score > 0.0:
                p.dp, p.parent = score, prev_id

        # Phase 2: at most one commit per reference position for this target,
        # then push into the Fenwick tree.
        best_by_j: dict[int, tuple[float, int]] = {}
        for pid in range(pos, end):
            p = pairs[pid]
            score = p.dp
            if score <= 0.0:
                continue
            cur = best_by_j.get(p.j)
            if cur is None or score > cur[0]:
                best_by_j[p.j] = (score, pid)
        for j, (score, pid) in best_by_j.items():
            if score > best_overall:
                best_overall, best_pid = score, pid
            k = j + 1
            while k < size:
                if score > bit_val[k]:
                    bit_val[k], bit_id[k] = score, pid
                k += k & -k

        pos = end

    # Backtrack.
    chain: list[_Pair] = []
    cur = best_pid
    while cur != -1:
        chain.append(pairs[cur])
        cur = pairs[cur].parent
    chain.reverse()

    ref_by_seq = {e.seq: e for e in refs}
    tgt_by_seq = {e.seq: e for e in tgts}
    return [Anchor(ref=ref_by_seq[p.j], tgt=tgt_by_seq[p.i], confidence=0.0)
            for p in chain]


# --- leave-one-out residuals (anchor confidence) ----------------------------


def _loo_residual(anchors_sorted: Sequence[Anchor], k: int) -> float:
    """Residual at anchor k under a local line fit that ignores anchor k.

    Interior anchors are predicted from the two adjacent anchors.  Edge
    anchors are predicted by extrapolating the line through their two
    nearest neighbours on the interior side (a pure offset would be a
    meaningless baseline when the clock has a rate drift).
    """
    n = len(anchors_sorted)
    a = anchors_sorted
    tk, yk = a[k].tgt.time, a[k].ref.time

    def line_pred(p: int, q: int, t: float) -> float:
        dt = a[q].tgt.time - a[p].tgt.time
        if dt == 0.0:
            return a[p].ref.time
        slope = (a[q].ref.time - a[p].ref.time) / dt
        return a[p].ref.time + slope * (t - a[p].tgt.time)

    if n == 1:
        return 0.0
    if 0 < k < n - 1:
        pred = line_pred(k - 1, k + 1, tk)
    elif k == 0:
        pred = line_pred(1, 2, tk) if n >= 3 else line_pred(1, 1, tk)
    else:
        pred = (line_pred(n - 3, n - 2, tk) if n >= 3
                else line_pred(0, 0, tk))
    return pred - yk


# --- top-level driver -------------------------------------------------------


def _fit_anchor_mapping(anchors: Sequence[Anchor]) -> ClockMapping:
    """Piecewise-linear map through ``anchors`` (or constant/single anchor)."""
    if len(anchors) >= 2:
        return PiecewiseLinearMapping(anchors)
    if len(anchors) == 1:
        return ConstantMapping(
            anchors[0].ref.time - anchors[0].tgt.time
        )
    return ConstantMapping(0.0)


def align_sequences(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    iterations: int = DEFAULT_ITERATIONS,
    time_weight: float = DEFAULT_TIME_WEIGHT,
    manual_pairs: Sequence[tuple[Event, Event]] | None = None,
) -> tuple[list[Anchor], ClockMapping, float]:
    """Run the iterative match/fit loop.

    Returns ``(anchors, clock_mapping, confidence_scale_seconds)``.
    The scale is the leave-one-out residual spread of the final anchors,
    useful as a temporal unit when scoring unmatched events.

    ``manual_pairs`` are client-locked ``(reference, target)`` pairs.  They
    are validated upstream (in-range, no double occupation, same label, no
    crossing) and are treated as mandatory anchors: the matcher is run
    separately inside each gap between consecutive locks and can neither
    consume a locked event nor jump over a lock.
    """
    if not manual_pairs:
        return _align_unconstrained(refs, tgts, iterations, time_weight)
    return _align_constrained(
        refs, tgts, iterations, time_weight,
        [Anchor(ref=r, tgt=t, confidence=1.0, source="manual")
         for r, t in manual_pairs],
    )


def _align_unconstrained(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    iterations: int,
    time_weight: float,
) -> tuple[list[Anchor], ClockMapping, float]:
    """Original behaviour with no client locks."""
    mapping = bootstrap_mapping(refs, tgts)
    total_iters = max(1, iterations)
    # Bootstrap pass: only nearest same-label candidates are admitted (the
    # temporal window is tight around the median offset), so it cannot
    # confuse two occurrences of a repeated label; the piecewise map from
    # pass 1 then spans clock jumps and rate drift with the full ±3σ window.
    sigma = 1.0

    anchors: list[Anchor] = []

    for it in range(total_iters):
        anchors = monotonic_match(
            refs, tgts, mapping, sigma, time_weight,
        )
        if len(anchors) >= 2:
            mapping = PiecewiseLinearMapping(anchors)
        elif len(anchors) == 1:
            mapping = ConstantMapping(
                anchors[0].ref.time - anchors[0].tgt.time
            )
        # zero anchors: keep the bootstrap mapping

        # Refine the temporal Gaussian from leave-one-out residuals of the
        # piecewise anchor set (in-sample residuals are exactly zero).
        if it < total_iters - 1 and len(anchors) >= 2:
            ordered = sorted(anchors, key=lambda a: a.ref.time)
            loo = [_loo_residual(ordered, k) for k in range(len(ordered))]
            sigma = max(robust_sigma(loo), 1e-3)

    # Anchor confidence from leave-one-out residuals of the local line
    # neighbourhood.  The scale is a high percentile so that a handful of
    # genuine piecewise breakpoints don't flatten every confidence to zero.
    ordered = sorted(anchors, key=lambda a: a.ref.time)
    if len(anchors) >= 2:
        loo = [abs(_loo_residual(ordered, k)) for k in range(len(ordered))]
        conf_scale = max(_percentile(sorted(loo), 0.9), 1e-3)
        scored: list[Anchor] = []
        for k, a in enumerate(ordered):
            c = math.exp(-0.5 * (loo[k] / conf_scale) ** 2)
            scored.append(Anchor(ref=a.ref, tgt=a.tgt, confidence=c))
        anchors = scored
    elif len(anchors) == 1:
        conf_scale = 1.0
        anchors = [Anchor(ref=ordered[0].ref, tgt=ordered[0].tgt,
                          confidence=0.5)]
    else:
        conf_scale = 1.0

    return anchors, mapping, conf_scale


def _align_constrained(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    iterations: int,
    time_weight: float,
    manual_anchors: Sequence[Anchor],
) -> tuple[list[Anchor], ClockMapping, float]:
    """Match within the gaps cut by client-locked anchors, then merge.

    The manual anchors have already been validated; they are ordered by
    increasing ``reference`` index here (their target indices then increase
    strictly as well -- the crossing check rejected any other case).
    """
    total_iters = max(1, iterations)

    # A global line through the locks is a far better seed than a blind
    # bootstrap whenever at least two locks exist; one lock at least fixes
    # the offset.  The per-gap matchers still refine this with a
    # piecewise map including the boundary anchors.
    global_seed: ClockMapping | None = None
    if len(manual_anchors) >= 2:
        global_seed = LinearMapping.fit(manual_anchors)
    elif manual_anchors:
        m0 = manual_anchors[0]
        global_seed = ConstantMapping(m0.ref.time - m0.tgt.time)

    # seq -> Event lookup, so gap slicing stays O(events) in total.
    ref_by_seq = sorted(refs, key=lambda e: e.seq)
    tgt_by_seq = sorted(tgts, key=lambda e: e.seq)
    ref_seqs = [e.seq for e in ref_by_seq]
    tgt_seqs = [e.seq for e in tgt_by_seq]
    # Same-label reference buckets are shared by every gap and iteration.
    label_indexes = _build_label_indexes(refs)

    cuts: list[tuple[int | None, int | None, int | None, int | None,
                      Anchor | None, Anchor | None]] = []
    # Each cut = (ref_lo, ref_hi, tgt_lo, tgt_hi, left_anchor, right_anchor)
    # over half-open sequence-index windows.  Adjacent gaps share the
    # boundary anchor (used only to fit the mapping, never re-matched).
    prev: Anchor | None = None
    for cur in manual_anchors:
        r_hi = cur.ref.seq
        i_hi = cur.tgt.seq
        r_lo = prev.ref.seq + 1 if prev is not None else 0
        i_lo = prev.tgt.seq + 1 if prev is not None else 0
        cuts.append((r_lo, r_hi, i_lo, i_hi, prev, cur))
        prev = cur
    last = manual_anchors[-1]
    cuts.append((
        last.ref.seq + 1, len(refs),
        last.tgt.seq + 1, len(tgts),
        last, None,
    ))

    auto_anchors: list[Anchor] = []

    for r_lo, r_hi, i_lo, i_hi, left, right in cuts:
        gap_refs = ref_by_seq[
            bisect_left(ref_seqs, r_lo):bisect_left(ref_seqs, r_hi)
        ]
        gap_tgts = tgt_by_seq[
            bisect_left(tgt_seqs, i_lo):bisect_left(tgt_seqs, i_hi)
        ]
        if not gap_refs or not gap_tgts:
            continue

        boundary = [a for a in (left, right) if a is not None]

        # Seed mapping for this gap: the global line through the locks, or
        # a bootstrap built from the gap events plus its boundary anchors.
        mapping: ClockMapping
        if global_seed is not None:
            mapping = global_seed
        else:
            seed_refs = [a.ref for a in boundary] + gap_refs
            seed_tgts = [a.tgt for a in boundary] + gap_tgts
            mapping = bootstrap_mapping(seed_refs, seed_tgts)

        sigma = 1.0
        for it in range(total_iters):
            matched = monotonic_match(
                refs, tgts, mapping, sigma, time_weight,
                ref_lo=r_lo, ref_hi=r_hi, tgt_lo=i_lo, tgt_hi=i_hi,
                label_indexes=label_indexes,
            )
            fit_set = boundary + matched
            mapping = _fit_anchor_mapping(fit_set)

            # Refine the temporal Gaussian (boundary locks included).
            if it < total_iters - 1 and len(fit_set) >= 2:
                ordered = sorted(fit_set, key=lambda a: a.ref.time)
                loo = [_loo_residual(ordered, k)
                       for k in range(len(ordered))]
                sigma = max(robust_sigma(loo), 1e-3)

        auto_anchors.extend(matched)

    # Merge: locked anchors (confidence fixed at 1.0) plus automatic ones.
    all_anchors: list[Anchor] = list(manual_anchors) + auto_anchors
    ordered = sorted(all_anchors, key=lambda a: a.ref.time)
    mapping = _fit_anchor_mapping(ordered)

    # Confidence for automatic anchors from leave-one-out residuals of the
    # merged anchor chain.  Manual locks are fixed to 1.0 -- they are
    # ground truth by construction and must not be down-weighted by a jump.
    auto_ordered = [a for a in ordered if a.source == "auto"]
    if auto_ordered and len(ordered) >= 2:
        loo_map = {
            id(a): abs(_loo_residual(ordered, k))
            for k, a in enumerate(ordered)
        }
        auto_loo = [loo_map[id(a)] for a in auto_ordered]
        conf_scale = max(_percentile(sorted(auto_loo), 0.9), 1e-3)
        scored: list[Anchor] = []
        for a in ordered:
            if a.source == "manual":
                scored.append(a)
            else:
                c = math.exp(-0.5 * (loo_map[id(a)] / conf_scale) ** 2)
                scored.append(Anchor(ref=a.ref, tgt=a.tgt, confidence=c))
        anchors = scored
    else:
        # No automatic matches at all: only locks remain.
        conf_scale = 1.0
        anchors = ordered

    return anchors, mapping, conf_scale
