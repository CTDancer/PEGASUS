"""Pure joint-population PDO geometry helpers.

The helpers in this file deliberately contain no model/oracle code.  They implement the
three pieces needed by joint Population PDO:

1. empirical PDO sets from a coherent committee of complete vector readouts;
2. numerical decision-span / shared-rank accounting in a common feature coordinate;
3. exact one-step population-diversity coordination via signed reachable contraction.

The formal exact-linear theorem uses exact subspace membership.  The implementation uses
an SVD rank tolerance only for numerical linear algebra.  No scale label enters the
geometry: H1/H2/H4/H8/... candidates are simply realized action directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.linalg

from .pdo_contraction_allocation import AllocationOption, hamming


@dataclass(frozen=True)
class EmpiricalPDOSets:
    """No-op-inclusive empirical PDO sets for one lineage.

    ``plausible_indices`` is the union of epsilon-good actions across committee members.
    ``robust_indices`` is the intersection of epsilon-good actions across members.
    """

    plausible_indices: np.ndarray
    robust_indices: np.ndarray
    epsilon: float

    @property
    def resolved(self) -> bool:
        return bool(len(self.robust_indices) > 0)


def empirical_pdo_sets(member_utilities: np.ndarray, epsilon: float) -> EmpiricalPDOSets:
    """Return union/intersection epsilon-good action sets across coherent hypotheses."""
    u = np.asarray(member_utilities, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] <= 0 or u.shape[1] <= 0 or np.any(~np.isfinite(u)):
        raise ValueError("member_utilities must be a finite [committee, actions] matrix")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    best = np.max(u, axis=1, keepdims=True)
    good = u >= (best - eps - 1e-12)
    plausible = np.flatnonzero(np.any(good, axis=0)).astype(np.int64)
    robust = np.flatnonzero(np.all(good, axis=0)).astype(np.int64)
    return EmpiricalPDOSets(plausible, robust, eps)


def _as_rows(x: np.ndarray | Sequence[Sequence[float]], dim: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        if dim is None:
            return np.zeros((0, 0), dtype=np.float64)
        return np.zeros((0, int(dim)), dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or np.any(~np.isfinite(a)):
        raise ValueError("feature directions must be a finite 2D matrix")
    if dim is not None and a.shape[1] != int(dim):
        raise ValueError("feature dimension mismatch")
    return a


def _robust_svd(
    x: np.ndarray,
    *,
    full_matrices: bool = False,
    compute_uv: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """SVD with a scipy fallback only when numpy's LAPACK call fails to converge."""
    try:
        return np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        # Keep numpy as the primary path so successful runs stay bit-identical.
        # gesvd is more conservative than gesdd and resolves rare non-convergence cases.
        return scipy.linalg.svd(
            x,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesvd",
        )


def feature_scale(rows: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Deterministic diagonal conditioning; invertible scaling preserves exact rank."""
    x = _as_rows(rows)
    if x.shape[1] == 0:
        return np.ones(0, dtype=np.float64)
    s = np.std(x, axis=0)
    return np.where(s > float(floor), s, 1.0).astype(np.float64)


def numerical_rank(rows: np.ndarray, *, rtol: float = 1e-7, scale: np.ndarray | None = None) -> int:
    """SVD row-rank with an explicit relative tolerance.

    ``rtol`` is a numerical tolerance, not a statistical confidence parameter.
    """
    x = _as_rows(rows)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return 0
    rr = float(rtol)
    if not np.isfinite(rr) or rr <= 0:
        raise ValueError("rtol must be finite and positive")
    if scale is not None:
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        if len(sc) != x.shape[1] or np.any(~np.isfinite(sc)) or np.any(sc <= 0):
            raise ValueError("invalid feature scale")
        x = x / sc.reshape(1, -1)
    s = _robust_svd(x, full_matrices=False, compute_uv=False)
    if len(s) == 0 or float(s[0]) <= 0:
        return 0
    tol = max(np.finfo(np.float64).eps * max(x.shape) * float(s[0]), rr * float(s[0]))
    return int(np.sum(s > tol))


def residual_dimension(
    observed: np.ndarray,
    decision_rows: np.ndarray,
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> int:
    """dim(H + S) - dim(H) for row spans H and S."""
    s = _as_rows(decision_rows)
    d = int(s.shape[1]) if s.shape[1] else (int(np.asarray(observed).shape[-1]) if np.asarray(observed).ndim == 2 and np.asarray(observed).shape[1] else 0)
    h = _as_rows(observed, dim=d) if d else _as_rows(observed)
    if s.shape[0] == 0:
        return 0
    if h.shape[0] == 0:
        return numerical_rank(s, rtol=rtol, scale=scale)
    return int(
        numerical_rank(np.concatenate([h, s], axis=0), rtol=rtol, scale=scale)
        - numerical_rank(h, rtol=rtol, scale=scale)
    )


def joint_span_statistics(
    observed: np.ndarray,
    decision_by_lineage: Mapping[int, np.ndarray],
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute per-lineage and joint unresolved decision dimensions."""
    mats = {int(k): _as_rows(v) for k, v in decision_by_lineage.items()}
    dim = next((v.shape[1] for v in mats.values() if v.shape[1] > 0), None)
    h = _as_rows(observed, dim=dim) if dim is not None else _as_rows(observed)
    per = {
        lid: residual_dimension(h, mat, rtol=rtol, scale=scale)
        for lid, mat in mats.items()
    }
    nonempty = [m for m in mats.values() if m.shape[0] > 0]
    if nonempty:
        union = np.concatenate(nonempty, axis=0)
        joint = residual_dimension(h, union, rtol=rtol, scale=scale)
    else:
        joint = 0
    summed = int(sum(per.values()))
    gamma = float(summed / joint) if joint > 0 else (1.0 if summed == 0 else float("inf"))
    return {
        "per_lineage_dimension": per,
        "sum_lineage_dimension": summed,
        "joint_dimension": int(joint),
        "sharing_factor": gamma,
    }


def shared_rank_gain(
    observed: np.ndarray,
    decision_by_lineage: Mapping[int, np.ndarray],
    query_direction: np.ndarray,
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> tuple[int, dict[int, int]]:
    """How many lineage residual dimensions are removed by observing one direction.

    Under exact subspace arithmetic each lineage gain is either zero or one.  Numerical
    SVD tolerance implements the same rank-difference definition directly.
    """
    q = np.asarray(query_direction, dtype=np.float64).reshape(1, -1)
    h = _as_rows(observed, dim=q.shape[1])
    hq = np.concatenate([h, q], axis=0) if h.shape[0] else q
    gains: dict[int, int] = {}
    for lid, rows in decision_by_lineage.items():
        s = _as_rows(rows, dim=q.shape[1])
        before = residual_dimension(h, s, rtol=rtol, scale=scale)
        after = residual_dimension(hq, s, rtol=rtol, scale=scale)
        gain = int(before - after)
        if gain < 0 or gain > 1:
            raise RuntimeError("one query direction changed residual dimension by more than one")
        gains[int(lid)] = gain
    return int(sum(gains.values())), gains


def scale_unique_rank_contributions(
    observed: np.ndarray,
    decision_by_scale: Mapping[int, np.ndarray],
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> dict[int, int]:
    """Order-independent unique residual rank contributed by each scale.

    For scale r this is d(all scales) - d(all scales except r).  Shared directions are
    intentionally not assigned to either scale.
    """
    mats = {int(k): _as_rows(v) for k, v in decision_by_scale.items()}
    nonempty = [m for m in mats.values() if m.shape[0] > 0]
    if not nonempty:
        return {k: 0 for k in mats}
    dim = nonempty[0].shape[1]
    h = _as_rows(observed, dim=dim)
    all_rows = np.concatenate(nonempty, axis=0)
    d_all = residual_dimension(h, all_rows, rtol=rtol, scale=scale)
    out: dict[int, int] = {}
    for k in sorted(mats):
        others = [m for kk, m in mats.items() if kk != k and m.shape[0] > 0]
        d_without = residual_dimension(
            h,
            np.concatenate(others, axis=0) if others else np.zeros((0, dim), dtype=np.float64),
            rtol=rtol,
            scale=scale,
        )
        out[k] = int(max(0, d_all - d_without))
    return out



@dataclass(frozen=True)
class FastSharedRankContext:
    """Precomputed row-space bases for O(KD^2)-free candidate scoring."""

    scale: np.ndarray
    observed_basis: np.ndarray
    lineage_residual_basis: dict[int, np.ndarray]
    lineage_dimensions: dict[int, int]
    rtol: float


def orthonormal_row_basis(
    rows: np.ndarray,
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return orthonormal row-space basis vectors in scaled feature coordinates."""
    x = _as_rows(rows)
    if x.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if scale is not None:
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        if len(sc) != x.shape[1] or np.any(~np.isfinite(sc)) or np.any(sc <= 0):
            raise ValueError("invalid feature scale")
        x = x / sc.reshape(1, -1)
    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    _u, sv, vt = _robust_svd(x, full_matrices=False, compute_uv=True)
    if len(sv) == 0 or float(sv[0]) <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    rr = float(rtol)
    tol = max(np.finfo(np.float64).eps * max(x.shape) * float(sv[0]), rr * float(sv[0]))
    rank = int(np.sum(sv > tol))
    return np.asarray(vt[:rank], dtype=np.float64)


def _residualize_scaled(rows_scaled: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(rows_scaled, dtype=np.float64)
    if basis.shape[0] == 0:
        return x.copy()
    return x - (x @ basis.T) @ basis


def build_fast_shared_rank_context(
    observed: np.ndarray,
    decision_by_lineage: Mapping[int, np.ndarray],
    *,
    rtol: float = 1e-7,
    scale: np.ndarray | None = None,
) -> FastSharedRankContext:
    mats = {int(k): _as_rows(v) for k, v in decision_by_lineage.items()}
    dim = next((m.shape[1] for m in mats.values() if m.shape[1] > 0), None)
    if dim is None:
        oo = np.asarray(observed)
        dim = int(oo.shape[1]) if oo.ndim == 2 else 0
    h = _as_rows(observed, dim=dim) if dim else _as_rows(observed)
    if scale is None:
        all_rows = [r for r in (mats.values()) if r.shape[0] > 0]
        conditioning = np.concatenate(([h] if h.shape[0] else []) + all_rows, axis=0) if (h.shape[0] or all_rows) else np.zeros((0, dim), dtype=np.float64)
        sc = feature_scale(conditioning) if dim else np.ones(0, dtype=np.float64)
    else:
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        if len(sc) != dim:
            raise ValueError("feature scale dimension mismatch")
    hb = orthonormal_row_basis(h, rtol=rtol, scale=sc if len(sc) else None)
    per_basis: dict[int, np.ndarray] = {}
    per_dim: dict[int, int] = {}
    for lid, mat in mats.items():
        if mat.shape[0] == 0:
            qb = np.zeros((0, dim), dtype=np.float64)
        else:
            ms = mat / sc.reshape(1, -1) if len(sc) else mat.copy()
            residual = _residualize_scaled(ms, hb)
            qb = orthonormal_row_basis(residual, rtol=rtol, scale=None)
        per_basis[int(lid)] = qb
        per_dim[int(lid)] = int(qb.shape[0])
    return FastSharedRankContext(
        scale=sc,
        observed_basis=hb,
        lineage_residual_basis=per_basis,
        lineage_dimensions=per_dim,
        rtol=float(rtol),
    )


def fast_shared_rank_gain(
    context: FastSharedRankContext,
    query_direction: np.ndarray,
) -> tuple[int, dict[int, int]]:
    """Fast exact-subspace-style shared gain using precomputed residual row spaces."""
    q = np.asarray(query_direction, dtype=np.float64).reshape(-1)
    if len(context.scale) and len(q) != len(context.scale):
        raise ValueError("query feature dimension mismatch")
    qs = q / context.scale if len(context.scale) else q.copy()
    if context.observed_basis.shape[0]:
        qr = qs - (qs @ context.observed_basis.T) @ context.observed_basis
    else:
        qr = qs
    norm = float(np.linalg.norm(qr))
    # With diagonally conditioned features, this threshold only removes directions already
    # numerically contained in H.  It is tied to the same SVD rtol used to build H/S bases.
    if norm <= max(1e-12, float(context.rtol)):
        return 0, {lid: 0 for lid in context.lineage_residual_basis}
    gains: dict[int, int] = {}
    membership_tol = max(1e-10, 10.0 * float(context.rtol))
    for lid, basis in context.lineage_residual_basis.items():
        if basis.shape[0] == 0:
            gains[int(lid)] = 0
            continue
        outside = qr - (qr @ basis.T) @ basis
        rel = float(np.linalg.norm(outside) / norm)
        gains[int(lid)] = int(rel <= membership_tol)
    return int(sum(gains.values())), gains

def raw_reachable_contraction(
    incumbent_i: str,
    incumbent_j: str,
    endpoint_i: str,
    endpoint_j: str,
) -> int:
    """Signed Hamming contraction d_before-d_after; negative means expansion."""
    return int(hamming(incumbent_i, incumbent_j) - hamming(endpoint_i, endpoint_j))


def population_raw_contraction_cost(selected: Mapping[int, AllocationOption]) -> float:
    ids = sorted(int(k) for k in selected)
    total = 0.0
    for ii in range(len(ids)):
        a = selected[ids[ii]]
        for jj in range(ii + 1, len(ids)):
            b = selected[ids[jj]]
            if int(a.preference_id) != int(b.preference_id):
                continue
            total += raw_reachable_contraction(
                a.incumbent_sequence,
                b.incumbent_sequence,
                a.sequence,
                b.sequence,
            )
    return float(total)


def coordinate_minimize_raw_contraction(
    alternatives: Mapping[int, Sequence[AllocationOption]],
    *,
    passes: int = 3,
    use_diversity: bool = True,
) -> tuple[dict[int, AllocationOption], float, float]:
    """Coordinate-minimize exact signed contraction over admissible action sets.

    The first option for each lineage defines the primary tuple (exact-best in PEGASUS v2).  Every accepted
    coordinate update weakly decreases signed contraction, which is exactly equivalent to
    weakly increasing one-step mean pairwise Hamming diversity because the incumbent
    pairwise distances are fixed.  This is a maintenance guarantee relative to the
    protected primary tuple, not a claim of global combinatorial optimality.
    """
    pp = int(passes)
    if pp < 0:
        raise ValueError("passes cannot be negative")
    alts = {int(k): list(v) for k, v in alternatives.items()}
    if not alts or any(len(v) == 0 for v in alts.values()):
        if not alts:
            return {}, 0.0, 0.0
        raise ValueError("every lineage must have at least one alternative")
    selected = {lid: rows[0] for lid, rows in alts.items()}
    initial = population_raw_contraction_cost(selected)
    if not use_diversity or pp == 0:
        return selected, initial, initial
    current = float(initial)
    for _ in range(pp):
        changed = False
        for order in (sorted(alts), list(reversed(sorted(alts)))):
            for lid in order:
                incumbent = selected[lid]
                best = incumbent
                best_cost = current
                for cand in alts[lid]:
                    if cand == incumbent:
                        continue
                    trial = dict(selected)
                    trial[lid] = cand
                    cost = population_raw_contraction_cost(trial)
                    if cost < best_cost - 1e-12:
                        best, best_cost = cand, cost
                    elif abs(cost - best_cost) <= 1e-12:
                        # Exact diversity ties: prefer higher predicted value, then a stable key.
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
                if best != incumbent:
                    selected[lid] = best
                    current = float(best_cost)
                    changed = True
            current = population_raw_contraction_cost(selected)
        if not changed:
            break
    final = population_raw_contraction_cost(selected)
    if final > initial + 1e-10:
        raise RuntimeError("diversity coordination increased raw contraction")
    return selected, float(initial), float(final)


__all__ = [
    "EmpiricalPDOSets",
    "empirical_pdo_sets",
    "feature_scale",
    "numerical_rank",
    "residual_dimension",
    "joint_span_statistics",
    "shared_rank_gain",
    "scale_unique_rank_contributions",
    "FastSharedRankContext",
    "orthonormal_row_basis",
    "build_fast_shared_rank_context",
    "fast_shared_rank_gain",
    "raw_reachable_contraction",
    "population_raw_contraction_cost",
    "coordinate_minimize_raw_contraction",
]
