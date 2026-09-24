"""Distribution-free statistics, Pareto and diversity utilities for K-FOCUS."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta as beta_dist


def anytime_dkw_radius(n: int, delta: float, arm_index: int = 1) -> float:
    """Anytime-valid DKW radius by summable arm/time alpha spending.

    We allocate delta * 6/(pi^2 j^2) to dynamically created arm j and then
    6/(pi^2 n^2) across sample counts inside the arm.  A union bound therefore
    protects every arm at every adaptively reached sample count.
    """
    n = int(n)
    j = int(arm_index)
    if n <= 0:
        return 1.0
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must lie in (0,1)")
    if j <= 0:
        raise ValueError("arm_index must be positive")
    c = 6.0 / (math.pi * math.pi)
    arm_delta = float(delta) * c / float(j * j)
    time_delta = arm_delta * c / float(n * n)
    time_delta = min(max(time_delta, 1e-300), 0.999999)
    return float(min(1.0, math.sqrt(math.log(2.0 / time_delta) / (2.0 * n))))


def empirical_survival(gains: Sequence[float], thresholds: np.ndarray) -> np.ndarray:
    g = np.asarray(gains, dtype=np.float64).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if g.size == 0:
        return np.zeros_like(t)
    return np.mean(g[:, None] >= t[None, :], axis=0)


def tail_index(gains: Sequence[float], gain_cap: float) -> tuple[float, float]:
    """Return empirical Psi=max_Delta Delta*P(G>=Delta) and argmax."""
    cap = max(0.0, float(gain_cap))
    g = np.clip(np.asarray(gains, dtype=np.float64).reshape(-1), 0.0, cap)
    if cap <= 0.0:
        return 0.0, 0.0
    thresholds = np.unique(np.concatenate(([0.0], g[g > 0.0], [cap])))
    s = empirical_survival(g, thresholds)
    values = thresholds * s
    k = int(np.argmax(values))
    return float(values[k]), float(thresholds[k])


@dataclass(frozen=True)
class TailBounds:
    estimate: float
    lower: float
    upper: float
    estimate_delta: float
    lower_delta: float
    upper_delta: float
    cdf_radius: float


def tail_index_bounds(
    gains: Sequence[float],
    gain_cap: float,
    *,
    delta: float,
    arm_index: int,
) -> TailBounds:
    cap = max(0.0, float(gain_cap))
    g = np.clip(np.asarray(gains, dtype=np.float64).reshape(-1), 0.0, cap)
    if cap <= 0.0:
        return TailBounds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if g.size == 0:
        return TailBounds(0.0, 0.0, cap, 0.0, 0.0, cap, 1.0)
    eps = anytime_dkw_radius(g.size, delta, arm_index)
    thresholds = np.unique(np.concatenate(([0.0], g[g > 0.0], [cap])))
    s = empirical_survival(g, thresholds)
    est_v = thresholds * s
    lo_v = thresholds * np.clip(s - eps, 0.0, 1.0)
    hi_v = thresholds * np.clip(s + eps, 0.0, 1.0)
    ie = int(np.argmax(est_v)); il = int(np.argmax(lo_v)); iu = int(np.argmax(hi_v))
    return TailBounds(
        float(est_v[ie]), float(lo_v[il]), float(hi_v[iu]),
        float(thresholds[ie]), float(thresholds[il]), float(thresholds[iu]), float(eps),
    )


def practical_priority(gains: Sequence[float], gain_cap: float, total_samples: int) -> float:
    """Finite-sample acquisition score; stopping still uses TailBounds.upper."""
    est, _ = tail_index(gains, gain_cap)
    n = len(tuple(gains))
    bonus = float(gain_cap) * math.sqrt(math.log(float(total_samples) + 2.0) / float(n + 1))
    return float(est + bonus)



def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided binomial upper confidence bound.

    ``alpha`` is the one-sided error probability for this particular check.
    For zero successes this reduces to ``1 - alpha**(1/n)``, giving the
    desired O(1/n) rare-event certification rate.
    """
    k = int(successes); n = int(trials)
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError("successes must lie in [0,trials]")
    a = float(min(max(alpha, 1e-300), 0.999999999999))
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - a ** (1.0 / float(n)))
    return float(beta_dist.ppf(1.0 - a, k + 1, n - k))


def checkpoint_sample_count(n: int, minimum: int) -> tuple[int, int]:
    """Largest geometric confidence checkpoint <= n and its zero-based level."""
    n = int(n); minimum = max(1, int(minimum))
    if n < minimum:
        return 0, -1
    level = int(math.floor(math.log2(float(n) / float(minimum))))
    return int(minimum * (2 ** level)), level


def strong_tail_single_threshold_bound(
    gains: Sequence[float],
    gain_cap: float,
    epsilon: float,
    *,
    alpha_arm: float,
    min_samples: int,
) -> dict[str, float]:
    """Query-efficient valid upper bound for the full tail index Psi.

    Let G in [0, gain_cap] and tau=min(epsilon,gain_cap).  For Delta<=tau,
    Delta P(G>=Delta)<=tau.  For Delta>tau,
    Delta P(G>=Delta)<=gain_cap P(G>=tau).  Hence

        Psi <= max(tau, gain_cap * P(G>=tau)).

    We estimate only the Bernoulli event G>=tau.  Confidence is evaluated on
    geometric checkpoints and spends ``alpha_arm * 6/(pi^2 (k+1)^2)`` at
    checkpoint k, so repeated adaptive checks remain valid for that arm.
    """
    cap = max(0.0, float(gain_cap)); eps = max(0.0, float(epsilon))
    if cap <= eps:
        return {
            "psi_ucb": cap, "threshold": min(eps, cap), "p_ucb": 0.0,
            "n_used": 0.0, "successes": 0.0, "checkpoint_level": -1.0,
            "ready": 1.0,
        }
    n_cp, level = checkpoint_sample_count(len(gains), min_samples)
    if n_cp <= 0:
        return {
            "psi_ucb": float("inf"), "threshold": eps, "p_ucb": 1.0,
            "n_used": 0.0, "successes": 0.0, "checkpoint_level": -1.0,
            "ready": 0.0,
        }
    g = np.asarray(gains[:n_cp], dtype=np.float64)
    tau = min(eps, cap)
    k = int(np.sum(g >= tau))
    c = 6.0 / (math.pi * math.pi)
    alpha_check = float(alpha_arm) * c / float((level + 1) ** 2)
    p_up = clopper_pearson_upper(k, n_cp, alpha_check)
    psi_up = max(tau, cap * p_up)
    return {
        "psi_ucb": float(psi_up), "threshold": float(tau),
        "p_ucb": float(p_up), "n_used": float(n_cp),
        "successes": float(k), "checkpoint_level": float(level),
        "ready": 1.0,
    }


def zero_success_required_checkpoint(
    gain_cap: float, epsilon: float, *, alpha_arm: float, min_samples: int, max_n: int = 1_048_576
) -> int:
    """Smallest geometric checkpoint that would certify Psi<=epsilon with zero hits."""
    cap = max(0.0, float(gain_cap)); eps = max(0.0, float(epsilon))
    if cap <= eps:
        return 0
    n = max(1, int(min_samples)); level = 0
    c = 6.0 / (math.pi * math.pi)
    while n <= int(max_n):
        alpha_check = float(alpha_arm) * c / float((level + 1) ** 2)
        p_up = clopper_pearson_upper(0, n, alpha_check)
        if max(eps, cap * p_up) <= eps * (1.0 + 1e-12):
            return int(n)
        n *= 2; level += 1
    return int(max_n)



def zero_hit_strong_reachability_samples(q_min: float, alpha: float) -> int:
    """Exact samples needed to rule out a q_min-probability strong hit after zero hits.

    If each independent draw has P(hit)>=q_min, observing zero hits in n draws
    has probability at most (1-q_min)^n.  Therefore n satisfying
    (1-q_min)^n <= alpha is a fixed-confidence certificate.
    """
    q = float(q_min); a = float(alpha)
    if not (0.0 < q < 1.0):
        raise ValueError("q_min must lie in (0,1)")
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    return int(math.ceil(math.log(a) / math.log1p(-q)))


def strong_reachability_status(
    gains: Sequence[float], strong_gain: float, q_min: float, alpha: float
) -> dict[str, float]:
    """Fixed-confidence status for an actionable strong-improvement event.

    A hit is G >= strong_gain.  K-FOCUS commits immediately on such a hit, so
    the terminal/plateau episode normally contains zero hits.  With n_required
    zero-hit draws, we may rule out P(hit)>=q_min at one-sided error alpha.
    """
    g = np.asarray(gains, dtype=np.float64).reshape(-1)
    d = max(0.0, float(strong_gain))
    hits = int(np.sum(g >= d))
    need = zero_hit_strong_reachability_samples(q_min, alpha)
    # Exact CP bound is useful for reporting even before the fixed threshold is met.
    p_ucb = clopper_pearson_upper(hits, int(g.size), alpha) if g.size else 1.0
    certified = bool(hits == 0 and int(g.size) >= need)
    return {
        "strong_gain": d, "q_min": float(q_min), "alpha": float(alpha),
        "n": float(g.size), "hits": float(hits), "n_required": float(need),
        "p_ucb": float(p_ucb), "certified_no_actionable_strong_arm": float(certified),
    }


def parse_probability_levels(value) -> tuple[float, ...]:
    """Parse a descending multiscale probability schedule.

    The schedule is an *anytime certificate ladder*, not a reward cutoff.
    Reaching level q certifies absence of a ``strong_gain`` event with
    probability at least q on every root arm (at the requested family-wise
    confidence).  Rarer opportunities remain explicitly unresolved unless a
    deeper level is requested.
    """
    if isinstance(value, str):
        vals = [float(x.strip()) for x in value.split(',') if x.strip()]
    else:
        vals = [float(x) for x in value]
    if not vals:
        raise ValueError("tail probability levels must be non-empty")
    if any((x <= 0.0 or x >= 1.0) for x in vals):
        raise ValueError("tail probability levels must lie in (0,1)")
    # Strictly descending: each level searches a rarer tail than the previous.
    if any(vals[i+1] >= vals[i] for i in range(len(vals)-1)):
        raise ValueError("tail probability levels must be strictly descending")
    return tuple(vals)


def multiscale_zero_hit_requirements(levels, alpha_arm: float) -> tuple[int, ...]:
    """Deterministic cumulative zero-hit sample targets for each probability level."""
    lev = parse_probability_levels(levels)
    return tuple(zero_hit_strong_reachability_samples(q, alpha_arm) for q in lev)


def multiscale_strong_tail_status(
    gains_by_arm: Sequence[Sequence[float]],
    strong_gain: float,
    levels,
    family_alpha: float,
) -> dict:
    """State of the multiscale strong-tail ladder for fixed root arms.

    Only the final requested level is used as the formal stopping certificate.
    Intermediate levels are deterministic search milestones that prioritize
    common strong opportunities first.  Since a meaningful hit restarts the
    episode, a plateau episode should contain zero hits.
    """
    lev = parse_probability_levels(levels)
    k = max(1, len(gains_by_arm))
    alpha_arm = float(family_alpha) / float(k)
    req = multiscale_zero_hit_requirements(lev, alpha_arm)
    ns=[]; hits=[]
    for g in gains_by_arm:
        a=np.asarray(g,dtype=np.float64).reshape(-1)
        ns.append(int(a.size)); hits.append(int(np.sum(a >= float(strong_gain))))
    any_hit=any(h>0 for h in hits)
    passed=[]
    for nreq in req:
        passed.append((not any_hit) and all(n >= nreq for n in ns))
    current_idx=0
    while current_idx < len(lev)-1 and passed[current_idx]:
        current_idx += 1
    final_certified=bool(passed[-1])
    q_current=float(lev[current_idx])
    n_current=int(req[current_idx])
    q_final=float(lev[-1]); n_final=int(req[-1])
    p_ucbs=[]
    for n,h in zip(ns,hits):
        p_ucbs.append(clopper_pearson_upper(h,n,alpha_arm) if n>0 else 1.0)
    return {
        'levels': lev, 'requirements': req, 'alpha_arm': alpha_arm,
        'arm_samples': tuple(ns), 'arm_hits': tuple(hits),
        'passed_levels': tuple(bool(x) for x in passed),
        'current_level_index': int(current_idx),
        'current_q': q_current, 'current_required_samples': n_current,
        'final_q': q_final, 'final_required_samples': n_final,
        'final_certified': final_certified, 'any_strong_hit': bool(any_hit),
        'max_root_hit_probability_ucb': float(max(p_ucbs) if p_ucbs else 0.0),
    }

def nondominated_mask_max(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("scores must be [N,D]")
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominates = np.all(x >= x[i], axis=1) & np.any(x > x[i], axis=1)
        if np.any(dominates):
            keep[i] = False
    return keep


def normalized_hamming_tokens(a: Sequence[int], b: Sequence[int]) -> float:
    x = np.asarray(a, dtype=np.int64).reshape(-1)
    y = np.asarray(b, dtype=np.int64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Hamming inputs must have equal length")
    if x.size > 2:
        x = x[1:-1]; y = y[1:-1]
    return float(np.mean(x != y))


def diversity_summary(tokens: np.ndarray) -> dict[str, float]:
    x = np.asarray(tokens, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] == 0:
        return {"count": 0, "unique_fraction": float("nan"), "pairwise_hamming_mean": float("nan"), "nearest_neighbor_hamming_mean": float("nan")}
    uniq = len({tuple(row.tolist()) for row in x}) / float(x.shape[0])
    if x.shape[0] < 2:
        return {"count": int(x.shape[0]), "unique_fraction": float(uniq), "pairwise_hamming_mean": float("nan"), "nearest_neighbor_hamming_mean": float("nan")}
    z = x[:, 1:-1] if x.shape[1] > 2 else x
    d = np.mean(z[:, None, :] != z[None, :, :], axis=-1).astype(np.float64)
    tri = d[np.triu_indices(x.shape[0], k=1)]
    dd = d.copy(); np.fill_diagonal(dd, np.inf)
    return {
        "count": int(x.shape[0]),
        "unique_fraction": float(uniq),
        "pairwise_hamming_mean": float(np.mean(tri)),
        "nearest_neighbor_hamming_mean": float(np.mean(np.min(dd, axis=1))),
    }


def simplex_grid(dim: int, level: int, *, min_weight: float = 1e-3) -> np.ndarray:
    """Positive simplex lattice used only for adaptive preference-cover diagnostics."""
    d = int(dim); m = int(level)
    if d < 2 or m < 1:
        raise ValueError("dim>=2 and level>=1 required")
    rows: list[list[int]] = []
    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + [remaining]); return
        for k in range(remaining + 1):
            rec(prefix + [k], remaining-k, slots-1)
    rec([], m, d)
    x = np.asarray(rows, dtype=np.float64) / float(m)
    x = x + float(min_weight)
    return x / x.sum(axis=1, keepdims=True)


def adaptive_cover_order(points: np.ndarray, *, seed_index: int | None = None) -> list[int]:
    """Farthest-first L1 cover ordering on a finite preference lattice."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return []
    first = int(seed_index) if seed_index is not None else int(np.argmin(np.sum((p - 1.0/p.shape[1])**2, axis=1)))
    chosen = [first]
    dist = np.sum(np.abs(p - p[first]), axis=1)
    dist[first] = -1.0
    while len(chosen) < p.shape[0]:
        i = int(np.argmax(dist)); chosen.append(i)
        dist = np.minimum(dist, np.sum(np.abs(p - p[i]), axis=1)); dist[chosen] = -1.0
    return chosen


__all__ = [
    "TailBounds", "anytime_dkw_radius", "empirical_survival", "tail_index",
    "clopper_pearson_upper", "checkpoint_sample_count", "strong_tail_single_threshold_bound",
    "zero_success_required_checkpoint", "zero_hit_strong_reachability_samples", "strong_reachability_status",
    "tail_index_bounds", "practical_priority", "nondominated_mask_max",
    "normalized_hamming_tokens", "diversity_summary", "simplex_grid",
    "adaptive_cover_order",
]
