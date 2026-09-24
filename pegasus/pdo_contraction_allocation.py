"""Pure geometry/query-allocation helpers for contraction-aware population allocation.

This module intentionally contains no model/oracle code.  It operates on already-realized
candidate sequences and scalar predicted utilities.  The core rule is lexicographic:
protect each lineage's predicted opportunity within a small slack, then minimize
population-level reachable contraction among those value-preserving alternatives.

No candidate is deleted from PDO.  The selected candidate is only queried *earlier*;
deferred candidates remain available to later PDO rounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AllocationOption:
    lineage_id: int
    preference_id: int
    sequence: str
    incumbent_sequence: str
    scale_k: int
    predicted_utility: float
    predicted_upper: float
    pool_key: int
    local_index: int
    initial_rank: int

    @property
    def realized_k(self) -> int:
        return hamming(self.incumbent_sequence, self.sequence)


def hamming(a: str, b: str) -> int:
    aa = str(a)
    bb = str(b)
    if len(aa) != len(bb):
        raise ValueError("sequences must have equal length")
    return int(sum(x != y for x, y in zip(aa, bb)))


def normalized_reachable_contraction(
    incumbent_i: str,
    incumbent_j: str,
    endpoint_i: str,
    endpoint_j: str,
) -> float:
    """Scale-normalized positive pairwise contraction in [0, 1]."""
    d0 = hamming(incumbent_i, incumbent_j)
    if d0 == 0:
        return 0.0
    ki = hamming(incumbent_i, endpoint_i)
    kj = hamming(incumbent_j, endpoint_j)
    denom = min(d0, ki + kj)
    if denom <= 0:
        return 0.0
    d1 = hamming(endpoint_i, endpoint_j)
    delta = max(d0 - d1, 0)
    chi = float(delta) / float(denom)
    # Triangle inequality implies <=1; tolerate floating noise only.
    if chi < -1e-12 or chi > 1.0 + 1e-12:
        raise RuntimeError("normalized contraction invariant failed")
    return float(min(1.0, max(0.0, chi)))


def value_preserving_options(
    options: Sequence[AllocationOption],
    *,
    slack: float,
    cap: int,
) -> tuple[AllocationOption, list[AllocationOption]]:
    """Return predicted primary plus top value-preserving alternatives.

    Alternatives may come from any realized scale.  The primary is never removed.
    """
    rows = list(options)
    if not rows:
        raise ValueError("options cannot be empty")
    eps = float(slack)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("slack must be finite and nonnegative")
    cc = int(cap)
    if cc <= 0:
        raise ValueError("cap must be positive")
    for r in rows:
        if not math.isfinite(float(r.predicted_utility)):
            raise ValueError("predicted utilities must be finite")
        if not math.isfinite(float(r.predicted_upper)):
            raise ValueError("predicted upper utilities must be finite")
    rows.sort(
        key=lambda r: (
            -float(r.predicted_utility),
            -float(r.predicted_upper),
            int(r.initial_rank),
            int(r.scale_k),
            str(r.sequence),
        )
    )
    primary = rows[0]
    threshold = float(primary.predicted_utility) - eps - 1e-15
    eligible = [r for r in rows if float(r.predicted_utility) >= threshold]
    eligible = eligible[:cc]
    if all((r.pool_key, r.local_index) != (primary.pool_key, primary.local_index) for r in eligible):
        eligible = [primary] + eligible[: max(0, cc - 1)]
    return primary, eligible


def _selection_cost(
    selected: Mapping[int, AllocationOption],
    fixed: Mapping[int, AllocationOption] | None = None,
) -> float:
    merged: dict[int, AllocationOption] = {}
    if fixed:
        merged.update({int(k): v for k, v in fixed.items()})
    for k, v in selected.items():
        kk = int(k)
        if kk in merged:
            raise ValueError("a lineage cannot be both fixed and active")
        merged[kk] = v
    ids = sorted(merged)
    total = 0.0
    for a in range(len(ids)):
        oi = merged[ids[a]]
        for b in range(a + 1, len(ids)):
            oj = merged[ids[b]]
            if int(oi.preference_id) != int(oj.preference_id):
                continue
            total += normalized_reachable_contraction(
                oi.incumbent_sequence,
                oj.incumbent_sequence,
                oi.sequence,
                oj.sequence,
            )
    return float(total)


def population_contraction_cost(
    selected: Mapping[int, AllocationOption],
    fixed: Mapping[int, AllocationOption] | None = None,
) -> float:
    return _selection_cost(selected, fixed)


def coordinate_minimize_contraction(
    alternatives: Mapping[int, Sequence[AllocationOption]],
    *,
    fixed: Mapping[int, AllocationOption] | None = None,
    passes: int = 3,
    use_contraction: bool = True,
) -> tuple[dict[int, AllocationOption], float, float]:
    """Choose one option per active lineage.

    Initializes at each lineage's first (predicted-best) option.  When contraction is
    enabled, deterministic coordinate descent selects among value-preserving alternatives.
    The common objective never increases.  ``use_contraction=False`` is the matched
    value-only scheduler control.
    """
    if int(passes) < 0:
        raise ValueError("passes cannot be negative")
    alts = {int(k): list(v) for k, v in alternatives.items()}
    if any(len(v) == 0 for v in alts.values()):
        raise ValueError("every active lineage must have at least one alternative")
    if not alts:
        return {}, 0.0, 0.0
    selected = {lid: rows[0] for lid, rows in alts.items()}
    initial_cost = _selection_cost(selected, fixed)
    if not use_contraction or int(passes) == 0:
        return selected, initial_cost, initial_cost

    current = float(initial_cost)
    for _ in range(int(passes)):
        changed = False
        # Forward and reverse sweeps reduce fixed-order artifacts while remaining deterministic.
        for order in (sorted(alts), list(reversed(sorted(alts)))):
            for lid in order:
                incumbent_choice = selected[lid]
                best = incumbent_choice
                best_cost = current
                for cand in alts[lid]:
                    if cand == incumbent_choice:
                        continue
                    trial = dict(selected)
                    trial[lid] = cand
                    cost = _selection_cost(trial, fixed)
                    # Primary criterion = contraction; exact ties prefer predicted value.
                    if cost < best_cost - 1e-12:
                        best, best_cost = cand, cost
                    elif abs(cost - best_cost) <= 1e-12:
                        if (
                            float(cand.predicted_utility),
                            float(cand.predicted_upper),
                            -int(cand.initial_rank),
                            -int(cand.scale_k),
                            str(cand.sequence),
                        ) > (
                            float(best.predicted_utility),
                            float(best.predicted_upper),
                            -int(best.initial_rank),
                            -int(best.scale_k),
                            str(best.sequence),
                        ):
                            best = cand
                if best != incumbent_choice:
                    selected[lid] = best
                    current = float(best_cost)
                    changed = True
            current = _selection_cost(selected, fixed)
        if not changed:
            break
    final_cost = _selection_cost(selected, fixed)
    if final_cost > initial_cost + 1e-10:
        raise RuntimeError("coordinate allocation increased contraction objective")
    return selected, float(initial_cost), float(final_cost)


__all__ = [
    "AllocationOption",
    "hamming",
    "normalized_reachable_contraction",
    "value_preserving_options",
    "population_contraction_cost",
    "coordinate_minimize_contraction",
]
