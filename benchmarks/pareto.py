"""Pareto archive, hypervolume, and final-100 reduction for E3."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import FINAL_LIBRARY_SIZE, HV_REFERENCE, OPT_NAMES


def nondominated_mask_maximize(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((x.shape[0],), dtype=bool)
    ge = np.all(x[None, :, :] >= x[:, None, :], axis=2)
    gt = np.any(x[None, :, :] > x[:, None, :], axis=2)
    dominates = ge & gt
    np.fill_diagonal(dominates, False)
    return ~dominates.any(axis=1)


def nondominated_sort_maximize(scores: np.ndarray) -> list[list[int]]:
    """Return fronts as lists of indices (rank 0 = nondominated)."""
    x = np.asarray(scores, dtype=np.float64)
    n = x.shape[0]
    remaining = set(range(n))
    fronts: list[list[int]] = []
    while remaining:
        idx = sorted(remaining)
        sub = x[idx]
        mask = nondominated_mask_maximize(sub)
        front = [idx[i] for i, keep in enumerate(mask) if keep]
        fronts.append(front)
        remaining -= set(front)
    return fronts


def hypervolume_maximize(
    points: np.ndarray,
    ref: Sequence[float] = HV_REFERENCE,
) -> float:
    """Maximize-oriented HV; prefer pymoo compiled kernel when available."""
    x = np.asarray(points, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("points must be [N,D]")
    if x.shape[1] != r.size:
        raise ValueError("dimension mismatch")
    if x.shape[0] == 0:
        return 0.0
    mask = np.all(x >= r, axis=1) & np.any(x > r, axis=1)
    x = x[mask]
    if x.shape[0] == 0:
        return 0.0
    x = x[nondominated_mask_maximize(x)]
    try:
        from pymoo.util.function_loader import load_function

        hv_fn = load_function("hv")
        return float(hv_fn(np.ascontiguousarray(-r), np.ascontiguousarray(-x)))
    except Exception:
        pass
    try:
        from pymoo.indicators.hv import HV

        return float(HV(ref_point=-r)(-x))
    except Exception:
        return float(_hv_recursive(x, r))


def _hv_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    n, d = points.shape
    if n == 0:
        return 0.0
    if d == 1:
        return float(max(0.0, np.max(points[:, 0] - ref[0])))
    levels = np.sort(np.unique(points[:, -1]))
    hv = 0.0
    previous = float(ref[-1])
    for level in levels:
        level = float(level)
        if level <= previous:
            continue
        projected = points[points[:, -1] >= level, :-1]
        if projected.size:
            projected = projected[nondominated_mask_maximize(projected)]
            hv += (level - previous) * _hv_recursive(projected, ref[:-1])
        previous = level
    return hv


def hypervolume_contribution(points: np.ndarray, ref: Sequence[float] = HV_REFERENCE) -> np.ndarray:
    """Per-point exclusive HV contribution (full HV − HV without point)."""
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    total = hypervolume_maximize(x, ref=ref)
    contrib = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rest = np.delete(x, i, axis=0)
        contrib[i] = total - hypervolume_maximize(rest, ref=ref)
    return contrib


# Exact leave-one-out HV contribution is O(n) full 6D HVs. On E3 archives the
# first nondominated front is often 300–600 points, which stalls finalize for
# many minutes (hours under multi-job CPU contention). Spec §7 prefers HV
# contribution when stable and otherwise allows crowding-distance.
EXACT_HV_CONTRIB_MAX_FRONT = 80


def crowding_distance_maximize(points: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distances for maximize-oriented objective vectors."""
    x = np.asarray(points, dtype=np.float64)
    n, m = x.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    dist = np.zeros(n, dtype=np.float64)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for j in range(m):
        order = np.argsort(x[:, j], kind="mergesort")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        lo = float(x[order[0], j])
        hi = float(x[order[-1], j])
        span = hi - lo
        if span <= 0.0:
            continue
        for t in range(1, n - 1):
            prev_f = float(x[order[t - 1], j])
            next_f = float(x[order[t + 1], j])
            dist[order[t]] += (next_f - prev_f) / span
    return dist


def select_final_100(
    sequences: Sequence[str],
    scores: np.ndarray,
    *,
    k: int = FINAL_LIBRARY_SIZE,
    ref: Sequence[float] = HV_REFERENCE,
    exact_hv_contrib_max_front: int = EXACT_HV_CONTRIB_MAX_FRONT,
) -> list[int]:
    """Method-independent archive reduction (spec §7).

    1. Nondominated fronts (maximize normalized objectives).
    2. Add complete fronts until the next front would exceed k.
    3. Fill the remainder from the splitting front by:
       - exact exclusive HV contribution when the front is small enough to be
         computationally stable in 6D, else
       - crowding distance (spec §7 fallback).
    """
    seqs = [str(s) for s in sequences]
    x = np.asarray(scores, dtype=np.float64)
    if len(seqs) != x.shape[0]:
        raise ValueError("sequences/scores length mismatch")
    # Deduplicate exact sequences, keep first occurrence.
    seen: set[str] = set()
    uniq: list[int] = []
    for i, s in enumerate(seqs):
        if s in seen:
            continue
        seen.add(s)
        uniq.append(i)
    if len(uniq) <= k:
        return uniq
    ux = x[uniq]
    fronts = nondominated_sort_maximize(ux)
    chosen_local: list[int] = []
    for front in fronts:
        if len(chosen_local) + len(front) <= k:
            chosen_local.extend(front)
            continue
        need = k - len(chosen_local)
        fx = ux[front]
        util = fx.mean(axis=1)
        if len(front) <= int(exact_hv_contrib_max_front):
            score = hypervolume_contribution(fx, ref=ref)
        else:
            score = crowding_distance_maximize(fx)
        # Highest diversity score first; tie-break by utility then sequence.
        order = sorted(
            range(len(front)),
            key=lambda j: (-float(score[j]), -float(util[j]), seqs[uniq[front[j]]]),
        )
        chosen_local.extend(front[j] for j in order[:need])
        break
    return [uniq[i] for i in chosen_local]


def pairwise_hamming_stats(sequences: Sequence[str]) -> dict[str, float]:
    seqs = [str(s) for s in sequences]
    n = len(seqs)
    if n < 2:
        return {
            "pairwise_hamming_mean": float("nan"),
            "pairwise_hamming_std": float("nan"),
            "nn_hamming_mean": float("nan"),
            "nn_hamming_std": float("nan"),
        }
    L = len(seqs[0])
    if not all(len(s) == L for s in seqs):
        raise ValueError("all sequences must share length for Hamming stats")
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(n, L)
    diffs = (arr[:, None, :] != arr[None, :, :]).sum(axis=2).astype(np.float64)
    # Length-normalized (project convention in PEGASUS diversity reporting).
    diffs /= float(L)
    iu = np.triu_indices(n, k=1)
    pair = diffs[iu]
    np.fill_diagonal(diffs, np.inf)
    nn = diffs.min(axis=1)
    return {
        "pairwise_hamming_mean": float(pair.mean()),
        "pairwise_hamming_std": float(pair.std(ddof=1)) if pair.size > 1 else 0.0,
        "nn_hamming_mean": float(nn.mean()),
        "nn_hamming_std": float(nn.std(ddof=1)) if nn.size > 1 else 0.0,
    }


__all__ = [
    "OPT_NAMES",
    "EXACT_HV_CONTRIB_MAX_FRONT",
    "nondominated_mask_maximize",
    "nondominated_sort_maximize",
    "hypervolume_maximize",
    "hypervolume_contribution",
    "crowding_distance_maximize",
    "select_final_100",
    "pairwise_hamming_stats",
]
