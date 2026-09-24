"""Statistics for a shared-residual distributional Koopman model.

The scientific hypothesis is

    z_T = mu_C(x_s) + eps_C,

where z is the fixed information-normalized anchor, mu_C is the already
validated direct finite-time Koopman conditional mean, and eps_C has a
chain/horizon-specific law that is approximately shared across source states.

This module contains only deterministic/statistical helpers.  It makes no
assumption that eps_C is Gaussian and never uses the failed innovation-control
matrices.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / den) if den > 1e-12 else float("nan")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 4 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    try:
        return float(spearmanr(x, y).statistic)
    except Exception:
        return float("nan")


def relative_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(p - y) / max(float(np.linalg.norm(y)), 1e-12))


def fixed_orthonormal_directions(dim: int, count: int, seed: int) -> np.ndarray:
    d = int(dim)
    k = min(int(count), d)
    if d < 1 or k < 1:
        raise ValueError("dim/count must be positive")
    rng = np.random.default_rng(int(seed))
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return np.asarray(q[:, :k], dtype=np.float64)


def covariance(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("samples must be [N,D] with N>=2")
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / float(x.shape[0] - 1)


def state_covariances(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2:
        raise ValueError("samples must be [states, continuations, dim]")
    xc = x - x.mean(axis=1, keepdims=True)
    return np.einsum("smd,sme->sde", xc, xc) / float(x.shape[1] - 1)


def stable_risk_value(values: np.ndarray, beta: float, axis: int = -1) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    b = float(beta)
    if abs(b) < 1e-12:
        return np.mean(x, axis=axis)
    m = np.max(b * x, axis=axis, keepdims=True)
    z = np.mean(np.exp(b * x - m), axis=axis)
    return (np.squeeze(m, axis=axis) + np.log(np.maximum(z, 1e-300))) / b


def projected(samples: np.ndarray, directions: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    u = np.asarray(directions, dtype=np.float64)
    return np.einsum("...d,dk->...k", x, u)


def directional_profiles(
    residuals: np.ndarray,
    directions: np.ndarray,
    *,
    quantiles: Sequence[float],
    betas: Sequence[float],
) -> dict[str, np.ndarray]:
    """Return direction-wise stochastic shape summaries.

    residuals may be [N,D] or [S,M,D].  In the latter case summaries are
    computed separately per state along M.
    """
    p = projected(residuals, directions)
    axis = -2  # sample / continuation axis in either [N,K] or [S,M,K]
    mean = np.mean(p, axis=axis)
    var = np.var(p, axis=axis, ddof=1)
    out: dict[str, np.ndarray] = {"mean": mean, "variance": var}
    for q in quantiles:
        out[f"quantile_{float(q):g}"] = np.quantile(p, float(q), axis=axis)
    for beta in betas:
        out[f"risk_{float(beta):g}"] = stable_risk_value(p, float(beta), axis=axis)
    return out


def normalized_sliced_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    directions: np.ndarray,
    *,
    reference_std: np.ndarray | None = None,
) -> float:
    """Average 1-D Wasserstein distance normalized by reference scale."""
    pa = projected(np.asarray(a, dtype=np.float64), directions)
    pb = projected(np.asarray(b, dtype=np.float64), directions)
    if pa.ndim != 2 or pb.ndim != 2:
        raise ValueError("a/b must be [samples,D]")
    if reference_std is None:
        reference_std = np.std(pb, axis=0, ddof=1)
    scale = np.maximum(np.asarray(reference_std, dtype=np.float64), 1e-8)
    vals = [wasserstein_distance(pa[:, j], pb[:, j]) / scale[j] for j in range(pa.shape[1])]
    return float(np.mean(vals))


def bootstrap_split_floor(
    bank: np.ndarray,
    directions: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> float:
    x = np.asarray(bank, dtype=np.float64)
    n = int(sample_size)
    if x.shape[0] < n:
        raise ValueError("bank smaller than sample_size")
    rng = np.random.default_rng(int(seed))
    std = np.std(projected(x, directions), axis=0, ddof=1)
    vals = []
    for _ in range(int(repeats)):
        ia = rng.choice(x.shape[0], size=n, replace=False)
        ib = rng.choice(x.shape[0], size=n, replace=False)
        vals.append(normalized_sliced_wasserstein(x[ia], x[ib], directions, reference_std=std))
    return float(np.mean(vals)) if vals else float("nan")


def profile_cosine_to_global(state_profile: np.ndarray, global_profile: np.ndarray) -> float:
    a = np.asarray(state_profile, dtype=np.float64)
    g = np.asarray(global_profile, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError("state_profile must have state dimension")
    vals = [safe_cosine(row, g) for row in a]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def expected_group_max(values: np.ndarray, budget: int) -> float:
    """Mean max over non-overlapping groups; deterministic and no bootstrap leakage."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    b = int(budget)
    groups = x.size // b
    if groups < 1:
        return float("nan")
    y = x[: groups * b].reshape(groups, b)
    return float(np.mean(np.max(y, axis=1)))


def expected_max_bonus_from_bank(
    bank: np.ndarray,
    directions: np.ndarray,
    budgets: Sequence[int],
    *,
    repeats: int = 256,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Monte-Carlo E[max_{1..B} u^T eps] for every direction."""
    p = projected(np.asarray(bank, dtype=np.float64), directions)
    rng = np.random.default_rng(int(seed))
    out: dict[int, np.ndarray] = {}
    for budget in budgets:
        b = int(budget)
        idx = rng.integers(0, p.shape[0], size=(int(repeats), b))
        # [repeat,B,K] -> [K]
        out[b] = p[idx].max(axis=1).mean(axis=0)
    return out


def fit_multioutput_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    sx = StandardScaler().fit(np.asarray(train_x, dtype=np.float64))
    xtr = sx.transform(train_x)
    xte = sx.transform(test_x)
    sy = StandardScaler().fit(np.asarray(train_y, dtype=np.float64))
    ytr = sy.transform(train_y)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(xtr, ytr)
    return sy.inverse_transform(model.predict(xte))


def conditional_gain_over_global(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    pred = fit_multioutput_ridge(train_x, train_y, test_x, alpha=alpha)
    base = np.broadcast_to(np.mean(train_y, axis=0, keepdims=True), np.asarray(test_y).shape)
    pred_cos = safe_cosine(pred, test_y)
    base_cos = safe_cosine(base, test_y)
    pred_rmse = relative_rmse(pred, test_y)
    base_rmse = relative_rmse(base, test_y)
    return {
        "conditional_cosine": pred_cos,
        "global_cosine": base_cos,
        "cosine_gain": pred_cos - base_cos if math.isfinite(pred_cos) and math.isfinite(base_cos) else float("nan"),
        "conditional_relative_rmse": pred_rmse,
        "global_relative_rmse": base_rmse,
        "relative_rmse_gain": base_rmse - pred_rmse,
    }


def horizon_accuracy_and_regret(records: Sequence[Mapping[str, float | str | int]]) -> tuple[float, float]:
    """Evaluate predicted-vs-actual best chain per (state, scalar, context, B)."""
    groups: dict[tuple, list[Mapping[str, float | str | int]]] = {}
    for r in records:
        key = (r["state_id"], r["scalar"], r["context_queries"], r["B"])
        groups.setdefault(key, []).append(r)
    correct, regrets = [], []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        pred = np.asarray([float(r["predicted_best"]) for r in rows])
        true = np.asarray([float(r["actual_best"]) for r in rows])
        ip = int(np.nanargmax(pred))
        it = int(np.nanargmax(true))
        correct.append(float(ip == it))
        regrets.append(float(true[it] - true[ip]))
    return (float(np.mean(correct)) if correct else float("nan"),
            float(np.mean(regrets)) if regrets else float("nan"))


__all__ = [
    "safe_cosine", "safe_spearman", "relative_rmse", "fixed_orthonormal_directions",
    "covariance", "state_covariances", "stable_risk_value", "projected",
    "directional_profiles", "normalized_sliced_wasserstein", "bootstrap_split_floor",
    "profile_cosine_to_global", "expected_group_max", "expected_max_bonus_from_bank",
    "fit_multioutput_ridge", "conditional_gain_over_global", "horizon_accuracy_and_regret",
]
