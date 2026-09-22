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

Manual anchors
--------------
Client-supplied anchors are mandatory matches.  They seed the first clock
map directly and partition both streams into intervals; the DP above runs
per interval (see :func:`segmented_match`), so automatic matches can never
claim a locked event or cross an anchor.  Manual anchors survive into the
final anchor set tagged ``source="manual"``.
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


def _candidate_pairs(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    mapping: Callable[[float], float],
    sigma: float,
    time_weight: float,
) -> list[_Pair]:
    """Build admissible same-label (ref, target) candidate pairs.

    Same-label reference events inside the ±3σ temporal window around the
    mapped target time are admitted; if the window is empty (large initial
    offset, or the bootstrap line straddles a clock jump), the two nearest
    same-label events are used instead.
    """
    ref_by_label: dict[str, list[Event]] = {}
    for ev in refs:
        ref_by_label.setdefault(ev.label, []).append(ev)
    ref_times_by_label = {
        label: [e.time for e in bucket]
        for label, bucket in ref_by_label.items()
    }

    radius = WINDOW_SIGMA * sigma
    inv_two_sigma2 = 1.0 / (2.0 * max(sigma, 1e-9) ** 2)
    pairs: list[_Pair] = []

    def emit(bucket: list[Event], k: int, centre: float,
             tgt: Event) -> None:
        residual = centre - bucket[k].time
        w = LABEL_BONUS + time_weight * math.exp(
            -(residual * residual) * inv_two_sigma2
        )
        pairs.append(_Pair(j=bucket[k].seq, i=tgt.seq, w=w))

    for tgt in tgts:
        bucket = ref_by_label.get(tgt.label)
        if not bucket:
            continue
        ref_times = ref_times_by_label[tgt.label]
        centre = mapping(tgt.time)

        lo = bisect_left(ref_times, centre - radius)
        hi = bisect_right(ref_times, centre + radius)
        if hi > lo:
            candidates: Any = range(lo, hi)
            if hi - lo > CANDIDATE_CAP:
                candidates = sorted(
                    range(lo, hi),
                    key=lambda k: abs(bucket[k].time - centre),
                )[:CANDIDATE_CAP]
            for k in candidates:
                emit(bucket, k, centre, tgt)
        else:
            # Outside the temporal window (e.g. the bootstrap global line
            # straddles a clock jump).  Take the two nearest same-label
            # events instead of the single nearest one: keeping both sides
            # of the jump lets the DP choose, instead of permanently
            # locking a jump-region target to the wrong reference event.
            ins = bisect_left(ref_times, centre)
            if ins < len(bucket):
                emit(bucket, ins, centre, tgt)
            if ins > 0:
                emit(bucket, ins - 1, centre, tgt)
    return pairs


def monotonic_match(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    mapping: ClockMapping,
    sigma: float,
    time_weight: float,
) -> list[Anchor]:
    """Maximum-weight monotonic matching via a Fenwick-tree weighted LIS.

    A single Fenwick tree keyed on *reference rank* (position within the
    supplied subsequence) enforces strict ordering across all labels.
    Pairs are processed in batches of equal target rank (queries first,
    commits after) so each target event is used at most once; the prefix
    query enforces the same on the reference side.  No per-event closures
    are allocated in the hot loop.

    ``refs`` / ``tgts`` may be contiguous sub-slices of the global streams
    (manual anchors split the sequences into intervals); ranks are positions
    inside the slice while global ``seq`` values are kept on the emitted
    anchors.
    """
    pairs = _candidate_pairs(
        refs, tgts, mapping.predict, sigma, time_weight
    )
    if not pairs:
        return []

    # Global seq -> rank within the supplied subsequence.  Ordering within a
    # stream is identical under time/seq ordering (both sides are sorted), so
    # rank == strictly increasing seq order.
    ref_rank = {e.seq: k for k, e in enumerate(refs)}
    tgt_rank = {e.seq: k for k, e in enumerate(tgts)}
    for p in pairs:
        p.j = ref_rank[p.j]
        p.i = tgt_rank[p.i]

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

    # Backtrack.  After the rank remap above, p.j/p.i are positions inside
    # the supplied refs/tgts subsequences, so they index the lists directly.
    chain: list[_Pair] = []
    cur = best_pid
    while cur != -1:
        chain.append(pairs[cur])
        cur = pairs[cur].parent
    chain.reverse()

    return [Anchor(ref=refs[p.j], tgt=tgts[p.i], confidence=0.0)
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


def _mapping_from_anchors(anchors: Sequence[Anchor]) -> ClockMapping:
    """Piecewise map for >=2 anchors, constant offset for one, identity for none."""
    if len(anchors) >= 2:
        return PiecewiseLinearMapping(anchors)
    if len(anchors) == 1:
        return ConstantMapping(
            anchors[0].ref.time - anchors[0].tgt.time
        )
    return ConstantMapping(0.0)


def segmented_match(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    manual: Sequence[Anchor],
    mapping: ClockMapping,
    sigma: float,
    time_weight: float,
) -> list[Anchor]:
    """Order-preserving DP match inside each interval between manual anchors.

    Manual anchors are mandatory matches that partition both streams into
    intervals (before the first anchor, between consecutive anchors, after
    the last anchor).  The original :func:`monotonic_match` runs independently
    on each interval, so automatic matches can neither occupy locked events
    nor cross an anchor; chaining interval results in boundary order yields a
    globally monotonic match.
    """
    locked_refs = {a.ref.seq for a in manual}
    locked_tgts = {a.tgt.seq for a in manual}
    ordered = sorted(manual, key=lambda a: a.ref.seq)

    matched: list[Anchor] = []
    prev_r = prev_t = -1
    for a in ordered:
        r_hi, t_hi = a.ref.seq, a.tgt.seq
        ref_seg = [
            e for e in refs if prev_r < e.seq < r_hi
            and e.seq not in locked_refs
        ]
        tgt_seg = [
            e for e in tgts if prev_t < e.seq < t_hi
            and e.seq not in locked_tgts
        ]
        if ref_seg and tgt_seg:
            matched.extend(
                monotonic_match(ref_seg, tgt_seg, mapping, sigma, time_weight)
            )
        matched.append(a)
        prev_r, prev_t = r_hi, t_hi

    ref_seg = [e for e in refs if e.seq > prev_r and e.seq not in locked_refs]
    tgt_seg = [e for e in tgts if e.seq > prev_t and e.seq not in locked_tgts]
    if ref_seg and tgt_seg:
        matched.extend(
            monotonic_match(ref_seg, tgt_seg, mapping, sigma, time_weight)
        )

    matched.sort(key=lambda a: a.ref.seq)
    return matched


def align_sequences(
    refs: Sequence[Event],
    tgts: Sequence[Event],
    iterations: int = DEFAULT_ITERATIONS,
    time_weight: float = DEFAULT_TIME_WEIGHT,
    manual_anchors: Sequence[Anchor] | None = None,
) -> tuple[list[Anchor], ClockMapping, float]:
    """Run the iterative match/fit loop.

    ``manual_anchors`` (if given) are non-skippable matches supplied by the
    caller.  They seed the initial clock map and split both streams into
    intervals; automatic DP matching is confined to each interval and cannot
    reuse a locked event or cross an anchor.  Manual anchors are always
    present in the returned anchor set (with ``source="manual"``).

    Returns ``(anchors, clock_mapping, confidence_scale_seconds)``.
    The scale is the leave-one-out residual spread of the final anchors,
    useful as a temporal unit when scoring unmatched events.
    """
    manual = sorted(manual_anchors or [], key=lambda a: a.ref.seq)
    if manual:
        # The mandatory matches alone determine the first clock map; no
        # global bootstrap over the noisy full streams is needed.
        mapping: ClockMapping = _mapping_from_anchors(manual)
    else:
        mapping = bootstrap_mapping(refs, tgts)
    total_iters = max(1, iterations)
    # Bootstrap pass: only nearest same-label candidates are admitted (the
    # temporal window is tight around the median offset), so it cannot
    # confuse two occurrences of a repeated label; the piecewise map from
    # pass 1 then spans clock jumps and rate drift with the full ±3σ window.
    sigma = 1.0

    anchors: list[Anchor] = list(manual)

    for it in range(total_iters):
        if manual:
            anchors = segmented_match(
                refs, tgts, manual, mapping, sigma, time_weight
            )
        else:
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
            scored.append(Anchor(ref=a.ref, tgt=a.tgt, confidence=c,
                                 source=a.source))
        anchors = scored
    elif len(anchors) == 1:
        conf_scale = 1.0
        anchors = [Anchor(ref=ordered[0].ref, tgt=ordered[0].tgt,
                          confidence=0.5, source=ordered[0].source)]
    else:
        conf_scale = 1.0

    return anchors, mapping, conf_scale
