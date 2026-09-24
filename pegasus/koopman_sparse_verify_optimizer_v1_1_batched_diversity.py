"""Final simplified Koopman sparse-verification optimizer.

Frozen architecture
-------------------
    PROPOSE -> KOOPMAN READ -> RANK -> SPARSE VERIFY -> ACCEPT -> UPDATE

This speed-optimized production implementation preserves the architecture supported by the focused
bridge + freeze diagnostics.  It is intentionally a clean implementation, not a
subclass of K-FOCUS v1-v5.

Core responsibilities
---------------------
1. Physical stochastic proposal
   For each lineage/current incumbent, generate a large slate of ACTUAL discrete
   peptides from every validated finite-time flow horizon.  Every hypothetical
   candidate independently reconstructs the stochastic chain start before
   physical sampling, exactly matching the validated native root proposal law.

2. Information-preserving Koopman readout
   Maintain one shared multi-output ridge readout
       f_hat(x) = W^T phi(x)
   from all genuinely paid six-objective labels.  The frozen Koopman/LKF model is
   never modified.

   Production default uses the validated zero-parameter exact-parent calibration
       f_tilde(y) = f(x) + [f_hat(y) - f_hat(x)].
   Absolute readout is retained only as a clean ablation.

3. Sparse exact verification
   Rank the entire cheap physical slate once.  Query predicted ranks 1..K in that
   STATIC order and accept the FIRST exact gain > accept_epsilon.  Every queried
   label (success or failure) enters the shared readout dataset for the NEXT
   slate.  If all K fail, regenerate a fresh stochastic slate on the lineage's
   next turn instead of querying deeper.

4. Monotone exact acceptance
   Ordinary optimization accepts any exact improvement larger than a small
   numerical epsilon.  ``strong_gain`` (default 0.02) is diagnostic only and is
   never an optimization gate.

5. Diversity
   Multiple independent lineages are maintained.  Candidate proposals must
   satisfy the requested Hamming floor against the other live lineages for the
   same preference.  Final reported sequences are selected only from exact
   queried archive entries with the requested output Hamming floor.

Deliberately absent
-------------------
* HCLR;
* residual witness optimization;
* generic residual transport;
* contextual residual-response regression;
* contextual tail-lift acquisition;
* a separate Residual-Echo scheduler;
* preference-specific reward models;
* direct latent Koopman control;
* a root-tail stopping certificate inside optimization.

A formal strong-tail certificate, if desired, should be run as a separate audit
and must not control monotone optimization acceptance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .probes import decode_esm_tokens
from .residual_distribution_koopman_model import load_residual_distribution_checkpoint
from .terminal_controlled_koopman_control import (
    augmented_tchebycheff_utility,
    parse_preferences,
)
from .terminal_controlled_koopman_geometry import chain_name, parse_chains, prepare_chain_start
from .utils import seed_everything, write_csv, write_json
from .kfocus.objectives import (
    CachedSixPropertyOracle,
    OPT_NAMES,
    RAW_NAMES,
    objective_summary,
)
from .kfocus.regions import terminal_z
from .kfocus.stats import (
    diversity_summary,
    nondominated_mask_max,
    normalized_hamming_tokens,
)

IMPLEMENTATION_VERSION = "koopman-sparse-verify-final-v1.1-batched"
DIVERSITY_REPORTING_VERSION = "sequence-diversity-v1.0"
DIVERSITY_BASIN_RADII = (0.15, 0.20, 0.25)
METHOD_NAME = "Koopman Sparse-Verify (batched)"
ROOT_START_SEED_XOR = 0x6A09E667
PROPOSAL_LAW = (
    "batched_independent_recorruption_per_candidate_then_physical_flow"
)
DEFAULT_AMHR2_TARGET = (
    "GPHMPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLP"
)
DEFAULT_CHAINS = "0,1;0.25,1;0.5,1;0.75,1;0,0.5,1;0,0.25,0.5,1"


@dataclass
class RidgeReadout:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        return ((xx - self.x_mean) / self.x_scale) @ self.coef + self.y_mean


@dataclass
class ArchiveEntry:
    sequence: str
    tokens: torch.Tensor
    z: np.ndarray
    raw: np.ndarray
    scores: np.ndarray
    utility_at_query: float
    first_query_index: int
    source: str
    preference_id: int | None
    lineage_id: int | None
    slate_id: int | None
    verification_rank: int | None


@dataclass
class LineageState:
    preference_id: int
    lineage_id: int
    tokens: torch.Tensor
    raw: np.ndarray
    scores: np.ndarray
    utility: float
    initial_utility: float
    accepted_moves: int = 0
    slates_attempted: int = 0
    verification_queries: int = 0


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _token_key(tokens: torch.Tensor | np.ndarray) -> tuple[int, ...]:
    return tuple(
        int(v)
        for v in torch.as_tensor(tokens).detach().cpu().reshape(-1).tolist()
    )


def _hamming(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    aa = torch.as_tensor(a).detach().cpu().reshape(-1).numpy()
    bb = torch.as_tensor(b).detach().cpu().reshape(-1).numpy()
    if aa.shape != bb.shape:
        raise ValueError("token shapes do not match")
    if aa.size > 2:
        aa, bb = aa[1:-1], bb[1:-1]
    return float(np.mean(aa != bb))


def _sequence_hamming(a: str, b: str) -> float:
    """Normalized Hamming distance for equal-length decoded peptide strings."""
    aa = str(a)
    bb = str(b)
    if len(aa) != len(bb):
        raise ValueError(
            f"sequence lengths do not match for Hamming distance: {len(aa)} vs {len(bb)}"
        )
    if not aa:
        return 0.0
    return float(sum(x != y for x, y in zip(aa, bb)) / len(aa))


def _interior_token_matrix(tokens: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return [N,L] biological-token matrix, excluding BOS/EOS when present."""
    arr = torch.as_tensor(tokens).detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected token matrix [N,L], got shape={arr.shape}")
    if arr.shape[1] > 2:
        arr = arr[:, 1:-1]
    return np.asarray(arr)


def _pairwise_hamming_matrix(tokens: torch.Tensor | np.ndarray) -> np.ndarray:
    """Exact normalized pairwise Hamming matrix in biological sequence space."""
    arr = _interior_token_matrix(tokens)
    n = int(arr.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.mean(arr[:, None, :] != arr[None, :, :], axis=2, dtype=np.float64)


def _connected_components_from_distance(
    distance: np.ndarray,
    radius: float,
) -> list[list[int]]:
    """Connected components of the Hamming graph with edges d_H <= radius."""
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance matrix must be square")
    n = int(d.shape[0])
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= float(radius) + 1e-12:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0] if g else -1))


def _extended_diversity_summary(
    tokens: torch.Tensor | np.ndarray,
    *,
    basin_radii: Sequence[float] = DIVERSITY_BASIN_RADII,
) -> dict[str, Any]:
    """Augment the existing diversity summary with local-collapse/basin metrics.

    Existing ``diversity_summary`` fields are preserved verbatim for backwards
    compatibility.  Added metrics are computed directly in normalized Hamming
    sequence space and therefore do not alter optimization or selection.
    """
    base = dict(diversity_summary(np.asarray(tokens)))
    d = _pairwise_hamming_matrix(tokens)
    n = int(d.shape[0])

    base["sequence_count"] = n
    if n == 0:
        base.update(
            {
                "unique_count": 0,
                "unique_fraction": float("nan"),
                "pairwise_hamming_median": float("nan"),
                "pairwise_hamming_p10": float("nan"),
                "nearest_neighbor_hamming_mean": float("nan"),
                "nearest_neighbor_hamming_median": float("nan"),
                "nearest_neighbor_hamming_p10": float("nan"),
                "nearest_neighbor_hamming_min": float("nan"),
            }
        )
        for radius in basin_radii:
            tag = f"{float(radius):.2f}"
            base[f"basin_count_r{tag}"] = 0
            base[f"largest_basin_size_r{tag}"] = 0
            base[f"largest_basin_fraction_r{tag}"] = float("nan")
            base[f"basin_sizes_r{tag}"] = []
        return base

    arr = _interior_token_matrix(tokens)
    unique_count = len({tuple(int(x) for x in row.tolist()) for row in arr})
    base["unique_count"] = int(unique_count)
    base["unique_fraction"] = float(unique_count / n)

    if n >= 2:
        tri = d[np.triu_indices(n, k=1)]
        nn_d = d.copy()
        np.fill_diagonal(nn_d, np.inf)
        nn = np.min(nn_d, axis=1)

        # Keep pre-existing mean pairwise value when provided by stats.py; the
        # direct calculation is used only as a fallback.
        base.setdefault("pairwise_hamming_mean", float(np.mean(tri)))
        base["pairwise_hamming_median"] = float(np.median(tri))
        base["pairwise_hamming_p10"] = float(np.percentile(tri, 10.0))
        base["nearest_neighbor_hamming_mean"] = float(np.mean(nn))
        base["nearest_neighbor_hamming_median"] = float(np.median(nn))
        base["nearest_neighbor_hamming_p10"] = float(np.percentile(nn, 10.0))
        base["nearest_neighbor_hamming_min"] = float(np.min(nn))
    else:
        base.setdefault("pairwise_hamming_mean", float("nan"))
        base["pairwise_hamming_median"] = float("nan")
        base["pairwise_hamming_p10"] = float("nan")
        base["nearest_neighbor_hamming_mean"] = float("nan")
        base["nearest_neighbor_hamming_median"] = float("nan")
        base["nearest_neighbor_hamming_p10"] = float("nan")
        base["nearest_neighbor_hamming_min"] = float("nan")

    for radius in basin_radii:
        tag = f"{float(radius):.2f}"
        comps = _connected_components_from_distance(d, float(radius))
        sizes = [int(len(c)) for c in comps]
        largest = max(sizes) if sizes else 0
        base[f"basin_count_r{tag}"] = int(len(comps))
        base[f"largest_basin_size_r{tag}"] = int(largest)
        base[f"largest_basin_fraction_r{tag}"] = float(largest / n) if n else float("nan")
        base[f"basin_sizes_r{tag}"] = sizes

    return base


def _lineage_contribution_summary(
    selected: Sequence[ArchiveEntry],
) -> dict[str, Any]:
    """How broadly the final exact library is sourced across optimization lineages."""
    n = len(selected)
    source_counts: dict[str, int] = {}
    lineage_counts: dict[str, int] = {}
    for e in selected:
        source_counts[e.source] = source_counts.get(e.source, 0) + 1
        if e.lineage_id is not None:
            key = str(int(e.lineage_id))
            lineage_counts[key] = lineage_counts.get(key, 0) + 1

    lineage_total = int(sum(lineage_counts.values()))
    lineage_probs = (
        np.asarray(list(lineage_counts.values()), dtype=np.float64) / lineage_total
        if lineage_total > 0
        else np.zeros(0, dtype=np.float64)
    )
    if len(lineage_probs) > 1:
        entropy = float(
            -np.sum(lineage_probs * np.log(lineage_probs))
            / math.log(len(lineage_probs))
        )
    elif len(lineage_probs) == 1:
        entropy = 0.0
    else:
        entropy = float("nan")

    return {
        "contributing_optimization_lineages": int(len(lineage_counts)),
        "lineage_counts": lineage_counts,
        "lineage_max_output_fraction": (
            float(max(lineage_counts.values()) / n)
            if n > 0 and lineage_counts
            else 0.0
        ),
        "lineage_entropy_normalized": entropy,
        "source_counts": source_counts,
        "source_fractions": {
            k: float(v / n) if n else float("nan")
            for k, v in source_counts.items()
        },
        "discovery_output_fraction": (
            float(source_counts.get("discovery", 0) / n) if n else float("nan")
        ),
        "sparse_verification_output_fraction": (
            float(source_counts.get("sparse_verification", 0) / n)
            if n
            else float("nan")
        ),
    }


def _trajectory_diversity_summary(
    query_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Hamming diversity of paid proposals and accepted optimization moves."""
    queried_dist: list[float] = []
    accepted_dist: list[float] = []
    for row in query_rows:
        parent = str(row.get("incumbent_sequence", ""))
        child = str(row.get("candidate_sequence", ""))
        if not parent or not child:
            continue
        dist = _sequence_hamming(parent, child)
        queried_dist.append(dist)
        if int(row.get("accepted", 0)) == 1:
            accepted_dist.append(dist)

    def stats(values: Sequence[float], prefix: str) -> dict[str, Any]:
        if not values:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_hamming_mean": float("nan"),
                f"{prefix}_hamming_median": float("nan"),
                f"{prefix}_hamming_p10": float("nan"),
                f"{prefix}_hamming_min": float("nan"),
                f"{prefix}_hamming_max": float("nan"),
            }
        x = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_count": int(len(x)),
            f"{prefix}_hamming_mean": float(np.mean(x)),
            f"{prefix}_hamming_median": float(np.median(x)),
            f"{prefix}_hamming_p10": float(np.percentile(x, 10.0)),
            f"{prefix}_hamming_min": float(np.min(x)),
            f"{prefix}_hamming_max": float(np.max(x)),
        }

    out = {}
    out.update(stats(queried_dist, "queried_parent_child"))
    out.update(stats(accepted_dist, "accepted_parent_child"))
    return out


def _final_lineage_diversity_summary(
    lineages: Sequence[LineageState],
) -> dict[str, Any]:
    """Pairwise/nearest-neighbor Hamming diversity among live final lineages."""
    if not lineages:
        return {
            "lineage_count": 0,
            "pairwise_hamming_mean": float("nan"),
            "nearest_neighbor_hamming_mean": float("nan"),
            "nearest_neighbor_hamming_min": float("nan"),
        }
    toks = np.stack([lin.tokens.detach().cpu().numpy() for lin in lineages], axis=0)
    div = _extended_diversity_summary(toks)
    return {
        "lineage_count": int(len(lineages)),
        "pairwise_hamming_mean": div.get("pairwise_hamming_mean", float("nan")),
        "pairwise_hamming_median": div.get("pairwise_hamming_median", float("nan")),
        "nearest_neighbor_hamming_mean": div.get(
            "nearest_neighbor_hamming_mean", float("nan")
        ),
        "nearest_neighbor_hamming_p10": div.get(
            "nearest_neighbor_hamming_p10", float("nan")
        ),
        "nearest_neighbor_hamming_min": div.get(
            "nearest_neighbor_hamming_min", float("nan")
        ),
    }


def _utility(
    scores: np.ndarray,
    preference: np.ndarray,
    rho: float,
) -> np.ndarray | float:
    x = np.asarray(scores, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x.reshape(1, -1)
    out = np.asarray(
        augmented_tchebycheff_utility(
            x,
            np.asarray(preference, dtype=np.float64),
            reference=np.ones(6, dtype=np.float64),
            rho=float(rho),
        ),
        dtype=np.float64,
    ).reshape(-1)
    return float(out[0]) if scalar else out


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeReadout:
    """Fit the exact standardized multi-output ridge used in the freeze diagnostic."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.ndim != 2:
        raise ValueError("ridge features must be [N,D]")
    if yy.ndim == 1:
        yy = yy[:, None]
    if yy.ndim != 2 or yy.shape[0] != xx.shape[0]:
        raise ValueError("ridge targets must be [N,M] and align with features")
    xm = xx.mean(axis=0, keepdims=True)
    xs = xx.std(axis=0, keepdims=True)
    xs = np.where(xs > 1e-8, xs, 1.0)
    ym = yy.mean(axis=0, keepdims=True)
    zx = (xx - xm) / xs
    zy = yy - ym
    gram = zx.T @ zx + float(alpha) * np.eye(zx.shape[1], dtype=np.float64)
    rhs = zx.T @ zy
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ rhs
    return RidgeReadout(xm, xs, ym, coef, float(alpha))


def _choose_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    *,
    seed: int,
    folds: int,
) -> tuple[float, float]:
    """Choose ridge alpha ONCE from discovery labels; keep it fixed thereafter."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim == 1:
        yy = yy[:, None]
    n = len(xx)
    if n < 8:
        raise ValueError("at least 8 discovery labels are required for ridge CV")
    k = max(2, min(int(folds), max(2, n // 4)))
    parts = np.array_split(np.random.default_rng(int(seed)).permutation(n), k)
    best_alpha = float(alphas[0])
    best_loss = float("inf")
    for alpha in alphas:
        losses: list[float] = []
        for test in parts:
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            train = np.flatnonzero(mask)
            if len(train) < 2 or len(test) == 0:
                continue
            pred = _fit_ridge(xx[train], yy[train], float(alpha)).predict(xx[test])
            losses.append(float(np.mean((pred - yy[test]) ** 2)))
        loss = float(np.mean(losses)) if losses else float("inf")
        if loss < best_loss - 1e-15:
            best_alpha = float(alpha)
            best_loss = loss
    return best_alpha, best_loss


@torch.no_grad()
def _native_pool(
    model,
    count: int,
    token_length: int,
    nfe: int,
    seed: int,
    batch_size: int,
    *,
    show_progress: bool,
) -> torch.Tensor:
    """Cheap unconditional physical pool used only for information-rich discovery."""
    process = model.base_model.process
    seen: dict[tuple[int, ...], torch.Tensor] = {}
    round_id = 0
    bar = None
    if tqdm is not None and show_progress:
        bar = tqdm(
            total=int(count),
            desc="Cheap information-rich discovery pool",
            unit="seq",
            dynamic_ncols=True,
        )
    try:
        while len(seen) < int(count):
            before = len(seen)
            b = min(max(2 * (int(count) - len(seen)), 16), int(batch_size))
            g = torch.Generator(device=process.device)
            g.manual_seed(int(seed) + 1009 * round_id)
            x = process.sample_terminal(
                batch_size=b,
                seq_len=int(token_length),
                nfe=int(nfe),
                generator=g,
            )
            for row in x:
                cpu = row.detach().cpu().clone()
                seen.setdefault(_token_key(cpu), cpu)
                if len(seen) >= int(count):
                    break
            if bar is not None:
                bar.update(len(seen) - before)
            round_id += 1
            if round_id > 4000:
                raise RuntimeError("could not generate enough unique discovery sequences")
    finally:
        if bar is not None:
            bar.close()
    return torch.stack(list(seen.values())[: int(count)])


def _diverse_maximin_indices(
    z: np.ndarray,
    tokens: torch.Tensor,
    count: int,
    min_hamming: float,
    seed: int,
) -> np.ndarray:
    """Oracle-free information-space maximin packing with exact Hamming floor."""
    x = np.asarray(z, dtype=np.float64)
    tok = torch.as_tensor(tokens).detach().cpu().numpy()
    if x.shape[0] != tok.shape[0]:
        raise ValueError("z/tokens size mismatch")
    if int(count) > len(x):
        return np.zeros(0, dtype=int)
    centered = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    xx = centered / np.where(scale > 1e-8, scale, 1.0)
    interior = tok[:, 1:-1]
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(len(x)))
    feasible = np.ones(len(x), dtype=bool)
    min_dist = np.full(len(x), np.inf, dtype=np.float64)
    chosen: list[int] = []
    while len(chosen) < int(count):
        candidates = np.flatnonzero(feasible)
        if candidates.size == 0:
            break
        if not chosen:
            idx = first if feasible[first] else int(candidates[0])
        else:
            idx = int(candidates[np.argmax(min_dist[candidates])])
        chosen.append(idx)
        d = np.sum((xx - xx[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        ham = np.mean(interior != interior[idx][None, :], axis=1)
        feasible &= ham >= float(min_hamming) - 1e-12
        feasible[np.asarray(chosen, dtype=int)] = False
    return np.asarray(chosen, dtype=int)


@torch.no_grad()
def _sample_independent_root_pool(
    model,
    incumbent_tokens: torch.Tensor,
    chain: Sequence[float],
    *,
    count: int,
    seed: int,
    forbidden_keys: set[tuple[int, ...]],
    diversity_anchors: Sequence[torch.Tensor],
    min_hamming: float,
    batch_size: int,
    max_cheap_draws_per_candidate: int,
    max_root_attempts: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Batched implementation of the validated independent-root proposal law.

    Scientific law is unchanged from v1.0:

        incumbent x
          -> independent stochastic x_start^(b) for each slate slot b
          -> physical endpoint y_b from the requested finite-time chain
          -> oracle-free Hamming / exact-duplicate rejection.

    The v1.0 implementation reproduced this law by calling the old
    ``sample_candidate`` helper once per slate slot.  That helper is designed for
    residual-region rejection and, even for a plain root proposal, generates a
    full ``batch_size`` continuation batch and computes terminal Koopman features
    before returning ONE candidate.  Repeating that serially for a 384-candidate
    slate is computationally pathological.

    Here we keep one persistent independently sampled ``x_start`` per requested
    candidate slot, propagate all currently unresolved starts together, and
    resample only unresolved slots.  This is distributionally the same
    first-eligible rejection construction but vectorized across independent
    candidate slots.  No residual mean or terminal Koopman embedding is computed
    during proposal generation; terminal features are computed once later for
    the completed slate, where they are actually needed for ranking.

    ``batch_size`` now controls physical transition chunking, not 64 redundant
    continuations per accepted candidate.
    """
    del max_root_attempts  # retained in signature for CLI/backward compatibility

    n = int(count)
    if n <= 0:
        raise ValueError("count must be positive")
    max_draws = int(max_cheap_draws_per_candidate)
    if max_draws <= 0:
        raise ValueError("max_cheap_draws_per_candidate must be positive")

    c = tuple(model.match_chain(chain))
    base = model.base_model
    device = model.device
    incumbent = torch.as_tensor(incumbent_tokens).reshape(1, -1).to(device)

    # One independent stochastic chain start per conceptual slate candidate.
    # prepare_chain_start already supports a batch of parent states (used by the
    # frozen Koopman diagnostics); a single torch.Generator supplies independent
    # random variates across rows.
    start_gen = torch.Generator(device=device)
    start_gen.manual_seed(_stable_seed(seed, "batched-starts"))
    parents = incumbent.expand(n, -1).contiguous()
    x_starts = prepare_chain_start(
        base,
        parents,
        c,
        generator=start_gen,
    )

    accepted: list[torch.Tensor | None] = [None] * n
    accepted_keys: set[tuple[int, ...]] = set()
    draws_per_slot = np.zeros(n, dtype=np.int64)
    rejection_diversity = np.zeros(n, dtype=np.int64)
    rejection_duplicate = np.zeros(n, dtype=np.int64)
    unresolved = np.arange(n, dtype=np.int64)

    # One generator per chain/pool.  Sampling in chunks changes only RNG
    # bookkeeping, not the product proposal law across independent starts.
    terminal_gen = torch.Generator(device=device)
    terminal_gen.manual_seed(_stable_seed(seed, "batched-terminals"))
    chunk = max(1, int(batch_size))

    # Cache anchor interiors once.
    anchor_interiors = []
    for other in diversity_anchors:
        oi = torch.as_tensor(other).detach().cpu().reshape(-1)
        if oi.numel() > 2:
            oi = oi[1:-1]
        anchor_interiors.append(oi.numpy())

    round_id = 0
    while unresolved.size:
        if np.any(draws_per_slot[unresolved] >= max_draws):
            bad = unresolved[draws_per_slot[unresolved] >= max_draws]
            raise RuntimeError(
                f"{chain_name(c)} diversity/dedup rejection exhausted for "
                f"{len(bad)} candidate slots after {max_draws} draws/slot"
            )

        next_unresolved: list[int] = []
        for lo in range(0, len(unresolved), chunk):
            ids_np = unresolved[lo:lo + chunk]
            ids = torch.as_tensor(ids_np, device=device, dtype=torch.long)
            terminal = base.sample_chain(
                x_starts.index_select(0, ids),
                c,
                generator=terminal_gen,
            ).detach().cpu()

            for local_j, slot_v in enumerate(ids_np.tolist()):
                slot = int(slot_v)
                draws_per_slot[slot] += 1
                row = terminal[local_j].clone()
                key = _token_key(row)

                if key in forbidden_keys or key in accepted_keys:
                    rejection_duplicate[slot] += 1
                    next_unresolved.append(slot)
                    continue

                interior = row.reshape(-1)
                if interior.numel() > 2:
                    interior = interior[1:-1]
                arr = interior.numpy()
                diverse = True
                for oi in anchor_interiors:
                    if arr.size != oi.size:
                        raise ValueError("diversity token lengths differ")
                    if float(np.mean(arr != oi)) < float(min_hamming) - 1e-12:
                        diverse = False
                        break
                if not diverse:
                    rejection_diversity[slot] += 1
                    next_unresolved.append(slot)
                    continue

                accepted[slot] = row
                accepted_keys.add(key)

        unresolved = np.asarray(next_unresolved, dtype=np.int64)
        round_id += 1

    if any(x is None for x in accepted):
        raise RuntimeError("internal error: unresolved batched root candidate")

    meta = [
        {
            "candidate_index_within_chain": int(i),
            "root_attempt_index": int(i),
            "draw_seed": int(_stable_seed(seed, "batched-slot", i)),
            "start_seed": int(_stable_seed(seed, "batched-starts")),
            "cheap_draws": int(draws_per_slot[i]),
            "diversity_rejections": int(rejection_diversity[i]),
            "duplicate_rejections": int(rejection_duplicate[i]),
            "batched_root_generation": True,
            "residual_features_computed_during_proposal": False,
        }
        for i in range(n)
    ]
    return torch.stack([x for x in accepted if x is not None]), meta


def _predict_scores(
    readout: RidgeReadout,
    z_candidates: np.ndarray,
    *,
    mode: str,
    incumbent_scores: np.ndarray,
    incumbent_z: np.ndarray,
) -> np.ndarray:
    pred = readout.predict(np.asarray(z_candidates, dtype=np.float64))
    if mode == "anchor_absolute":
        return np.clip(pred, 0.0, 1.0)
    if mode == "anchor_parent_delta":
        parent_pred = readout.predict(
            np.asarray(incumbent_z, dtype=np.float64).reshape(1, -1)
        )[0]
        return np.clip(
            np.asarray(incumbent_scores, dtype=np.float64).reshape(1, -1)
            + (pred - parent_pred.reshape(1, -1)),
            0.0,
            1.0,
        )
    raise ValueError(
        f"unknown readout mode {mode!r}; expected anchor_absolute or anchor_parent_delta"
    )


class KoopmanSparseVerifyRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        seed_everything(int(args.seed))
        self.model, self.checkpoint_payload = load_residual_distribution_checkpoint(
            args.residual_checkpoint,
            terminal_checkpoint=args.terminal_checkpoint or None,
            base_lkf_checkpoint=args.lkf_checkpoint or None,
            device=args.device,
            strict_sha=not bool(args.allow_checkpoint_sha_mismatch),
        )
        requested_chains = parse_chains(args.chains)
        self.chains = tuple(tuple(self.model.match_chain(c)) for c in requested_chains)
        # Optional named / explicit weight override for the Pareto preference sweep.
        # Default path (--preferences only) is unchanged for production equal-weight runs.
        prefs_text = str(args.preferences)
        pref_name = str(getattr(args, "preference_name", "") or "").strip()
        weights_text = str(getattr(args, "weights", "") or "").strip()
        if pref_name or weights_text:
            from .pareto_weight_sweep_preferences import resolve_preferences_cli

            prefs_text, resolved_name = resolve_preferences_cli(
                preference_name=pref_name,
                weights=weights_text,
                preferences=str(args.preferences),
            )
            self.preference_name = resolved_name
        else:
            self.preference_name = None
        self.preferences = tuple(
            np.asarray(p, dtype=np.float64) for p in parse_preferences(prefs_text, 6)
        )
        self.oracle = CachedSixPropertyOracle.from_peptiverse(
            args.peptiverse_root,
            target=args.target,
            manifest_path=args.peptiverse_manifest or None,
            device=args.peptiverse_device,
            clip_optimization_scores=not bool(args.no_clip_optimization_scores),
        )
        self.out = Path(args.output_dir).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)

        self.archive: list[ArchiveEntry] = []
        self.archive_by_sequence: dict[str, ArchiveEntry] = {}
        self.query_history: list[dict[str, Any]] = []
        self.slate_history: list[dict[str, Any]] = []
        self.top_prediction_history: list[dict[str, Any]] = []
        self.readout_history: list[dict[str, Any]] = []
        self.lineages: dict[int, list[LineageState]] = {}
        self.readout_alpha: float | None = None
        self.readout_alpha_cv_mse: float | None = None
        self.discovery_unique_queries = 0
        self.slate_counter = 0
        self.stop_requested = False

    def _budget_remaining(self) -> int | None:
        limit = int(self.args.max_unique_oracle_queries)
        if limit <= 0:
            return None
        return max(0, limit - int(self.oracle.unique_oracle_queries))

    def _record_exact(
        self,
        tokens: torch.Tensor,
        z: np.ndarray,
        rec,
        *,
        source: str,
        preference_id: int | None,
        lineage_id: int | None,
        slate_id: int | None,
        verification_rank: int | None,
        utility_at_query: float,
    ) -> ArchiveEntry:
        seq = str(rec.sequence)
        if seq in self.archive_by_sequence:
            return self.archive_by_sequence[seq]
        entry = ArchiveEntry(
            sequence=seq,
            tokens=tokens.detach().cpu().clone(),
            z=np.asarray(z, dtype=np.float64).reshape(-1).copy(),
            raw=np.asarray(rec.raw, dtype=np.float64).reshape(6).copy(),
            scores=np.asarray(rec.scores, dtype=np.float64).reshape(6).copy(),
            utility_at_query=float(utility_at_query),
            first_query_index=int(self.oracle.unique_oracle_queries),
            source=str(source),
            preference_id=preference_id,
            lineage_id=lineage_id,
            slate_id=slate_id,
            verification_rank=verification_rank,
        )
        self.archive.append(entry)
        self.archive_by_sequence[seq] = entry
        return entry

    def _archive_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.archive:
            raise RuntimeError("readout requested before any exact labels")
        return (
            np.stack([e.z for e in self.archive], axis=0),
            np.stack([e.scores for e in self.archive], axis=0),
        )

    def _fit_current_readout(self) -> RidgeReadout:
        if self.readout_alpha is None:
            raise RuntimeError("ridge alpha has not been initialized")
        z, scores = self._archive_arrays()
        return _fit_ridge(z, scores, float(self.readout_alpha))

    def _save_progress(self) -> None:
        if self.archive:
            write_csv(
                self.out / "evaluated_archive.csv",
                [self._archive_row(e) for e in self.archive],
            )
        if self.query_history:
            write_csv(self.out / "query_history.csv", self.query_history)
        if self.slate_history:
            write_csv(self.out / "slate_history.csv", self.slate_history)
        if self.top_prediction_history:
            write_csv(
                self.out / "top_prediction_history.csv",
                self.top_prediction_history,
            )
        if self.readout_history:
            write_csv(self.out / "readout_history.csv", self.readout_history)

    @staticmethod
    def _archive_row(e: ArchiveEntry) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sequence": e.sequence,
            "first_query_index": int(e.first_query_index),
            "source": e.source,
            "preference_id": "" if e.preference_id is None else int(e.preference_id),
            "lineage_id": "" if e.lineage_id is None else int(e.lineage_id),
            "slate_id": "" if e.slate_id is None else int(e.slate_id),
            "verification_rank": (
                "" if e.verification_rank is None else int(e.verification_rank)
            ),
            "utility_at_query": float(e.utility_at_query),
        }
        for j, name in enumerate(RAW_NAMES):
            row[f"raw_{name}"] = float(e.raw[j])
        for j, name in enumerate(OPT_NAMES):
            row[f"score_{name}"] = float(e.scores[j])
        return row

    def discovery(self) -> None:
        a = self.args
        output_hamming = (
            float(a.output_min_hamming)
            if a.output_min_hamming is not None
            else float(a.min_lineage_hamming)
        )
        target = max(
            int(a.initial_readout_queries),
            int(a.lineages),
            int(a.num_output_sequences),
        )
        hamming_floor = max(float(a.min_lineage_hamming), output_hamming)
        pool_size = max(
            int(target),
            int(target) * max(1, int(a.discovery_pool_multiplier)),
        )
        selected = None
        selected_z = None

        for expansion in range(int(a.discovery_max_pool_expansions) + 1):
            pool = _native_pool(
                self.model,
                pool_size,
                int(a.peptide_length) + 2,
                int(a.native_nfe),
                _stable_seed(a.seed, "discovery-pool", expansion),
                int(a.discovery_generation_batch),
                show_progress=not bool(a.no_progress),
            )
            z = terminal_z(
                self.model,
                pool,
                batch_size=int(a.feature_batch_size),
            )
            idx = _diverse_maximin_indices(
                z,
                pool,
                target,
                hamming_floor,
                _stable_seed(a.seed, "discovery-maximin", expansion),
            )
            if len(idx) >= target:
                take = torch.as_tensor(idx[:target], dtype=torch.long)
                selected = pool[take]
                selected_z = z[idx[:target]]
                break
            if expansion >= int(a.discovery_max_pool_expansions):
                raise RuntimeError(
                    f"discovery could pack only {len(idx)}/{target} exact-query candidates "
                    f"at Hamming floor {hamming_floor:.3f}"
                )
            print(
                f"[{METHOD_NAME}] cheap discovery packed {len(idx)}/{target}; "
                f"expanding pool {pool_size} -> {2 * pool_size}",
                flush=True,
            )
            pool_size *= 2

        assert selected is not None and selected_z is not None
        seqs = decode_esm_tokens(selected)
        bar = None
        if tqdm is not None and not bool(a.no_progress):
            bar = tqdm(
                total=len(seqs),
                desc="PeptiVerse information-rich discovery",
                unit="query",
                dynamic_ncols=True,
            )
        try:
            for i, (tok, seq) in enumerate(zip(selected, seqs)):
                if self._budget_remaining() == 0:
                    raise RuntimeError(
                        "oracle budget exhausted during mandatory initial discovery"
                    )
                rec = self.oracle.evaluate_one(seq)
                # Discovery is preference-neutral; utility_at_query is reported
                # under preference 0 only as a convenience and never trains W.
                u0 = float(_utility(rec.scores, self.preferences[0], a.rho))
                self._record_exact(
                    tok,
                    selected_z[i],
                    rec,
                    source="discovery",
                    preference_id=None,
                    lineage_id=None,
                    slate_id=None,
                    verification_rank=None,
                    utility_at_query=u0,
                )
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(unique_q=self.oracle.unique_oracle_queries)
        finally:
            if bar is not None:
                bar.close()

        self.discovery_unique_queries = int(self.oracle.unique_oracle_queries)
        z, scores = self._archive_arrays()
        alphas = tuple(
            float(x.strip())
            for x in str(a.ridge_alphas).split(",")
            if x.strip()
        )
        if not alphas:
            raise ValueError("--ridge-alphas cannot be empty")
        self.readout_alpha, self.readout_alpha_cv_mse = _choose_alpha(
            z,
            scores,
            alphas,
            seed=_stable_seed(a.seed, "ridge-alpha"),
            folds=int(a.cv_folds),
        )
        self.readout_history.append(
            {
                "event": "initial_alpha_selection",
                "slate_id": "",
                "preference_id": "",
                "lineage_id": "",
                "cycle": "",
                "training_labels": int(len(self.archive)),
                "alpha": float(self.readout_alpha),
                "cv_mse": float(self.readout_alpha_cv_mse),
                "readout_mode": str(a.readout_mode),
                "alpha_fixed_for_all_future_slates": True,
            }
        )
        self._initialize_lineages()
        self._save_progress()

    def _initialize_lineages(self) -> None:
        a = self.args
        discovery = [e for e in self.archive if e.source == "discovery"]
        if len(discovery) < int(a.lineages):
            raise RuntimeError("not enough discovery labels to initialize lineages")
        for pref_id, pref in enumerate(self.preferences):
            ranked = sorted(
                discovery,
                key=lambda e: float(_utility(e.scores, pref, a.rho)),
                reverse=True,
            )
            chosen: list[LineageState] = []
            for e in ranked:
                if all(
                    normalized_hamming_tokens(
                        e.tokens.numpy(),
                        old.tokens.numpy(),
                    )
                    >= float(a.min_lineage_hamming)
                    for old in chosen
                ):
                    u = float(_utility(e.scores, pref, a.rho))
                    chosen.append(
                        LineageState(
                            preference_id=int(pref_id),
                            lineage_id=int(len(chosen)),
                            tokens=e.tokens.clone(),
                            raw=e.raw.copy(),
                            scores=e.scores.copy(),
                            utility=u,
                            initial_utility=u,
                        )
                    )
                    if len(chosen) >= int(a.lineages):
                        break
            if len(chosen) < int(a.lineages):
                raise RuntimeError(
                    f"discovery archive packs only {len(chosen)}/{a.lineages} "
                    f"lineages for preference {pref_id}"
                )
            self.lineages[pref_id] = chosen

    def _diversity_anchors(
        self,
        preference_id: int,
        lineage_id: int,
    ) -> list[torch.Tensor]:
        return [
            lin.tokens
            for lin in self.lineages[int(preference_id)]
            if int(lin.lineage_id) != int(lineage_id)
        ]

    def _generate_unified_slate(
        self,
        lineage: LineageState,
        *,
        slate_id: int,
    ) -> tuple[torch.Tensor, list[str], list[dict[str, Any]]]:
        a = self.args
        forbidden = set(
            _token_key(e.tokens) for e in self.archive
        )
        forbidden.add(_token_key(lineage.tokens))
        anchors = self._diversity_anchors(
            lineage.preference_id,
            lineage.lineage_id,
        )

        pools: list[torch.Tensor] = []
        chain_labels: list[str] = []
        metas: list[dict[str, Any]] = []
        slate_keys: set[tuple[int, ...]] = set()

        for ci, chain in enumerate(self.chains):
            pool, per_candidate = _sample_independent_root_pool(
                self.model,
                lineage.tokens,
                chain,
                count=int(a.slate_per_chain),
                seed=_stable_seed(
                    a.seed,
                    "slate",
                    lineage.preference_id,
                    lineage.lineage_id,
                    slate_id,
                    ci,
                ),
                forbidden_keys=forbidden | slate_keys,
                diversity_anchors=anchors,
                min_hamming=float(a.min_lineage_hamming),
                batch_size=int(a.cheap_generation_batch),
                max_cheap_draws_per_candidate=int(
                    a.max_cheap_draws_per_candidate
                ),
                max_root_attempts=int(a.max_root_attempts_per_chain),
            )
            pools.append(pool)
            label = chain_name(chain)
            chain_labels.extend([label] * int(pool.shape[0]))
            for row in pool:
                slate_keys.add(_token_key(row))
            for m in per_candidate:
                metas.append({"chain": label, **m})

        return torch.cat(pools, dim=0), chain_labels, metas

    def _run_one_slate(
        self,
        preference_id: int,
        lineage: LineageState,
        cycle: int,
    ) -> None:
        a = self.args
        if self.stop_requested:
            return
        if self._budget_remaining() == 0:
            self.stop_requested = True
            return

        self.slate_counter += 1
        slate_id = int(self.slate_counter)
        readout = self._fit_current_readout()
        readout_train_size = int(len(self.archive))
        self.readout_history.append(
            {
                "event": "fit_before_slate",
                "slate_id": slate_id,
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "cycle": int(cycle),
                "training_labels": readout_train_size,
                "alpha": float(self.readout_alpha),
                "cv_mse": "",
                "readout_mode": str(a.readout_mode),
                "alpha_fixed_for_all_future_slates": True,
            }
        )

        t_slate0 = time.perf_counter()
        slate, chain_labels, proposal_meta = self._generate_unified_slate(
            lineage,
            slate_id=slate_id,
        )
        proposal_seconds = float(time.perf_counter() - t_slate0)

        t_rank0 = time.perf_counter()
        seqs = decode_esm_tokens(slate)
        z = terminal_z(
            self.model,
            slate,
            batch_size=int(a.feature_batch_size),
        )
        incumbent_z = terminal_z(
            self.model,
            lineage.tokens.reshape(1, -1),
            batch_size=int(a.feature_batch_size),
        )[0]
        pred_scores = _predict_scores(
            readout,
            z,
            mode=str(a.readout_mode),
            incumbent_scores=lineage.scores,
            incumbent_z=incumbent_z,
        )
        pref = self.preferences[int(preference_id)]
        pred_u = np.asarray(
            _utility(pred_scores, pref, a.rho),
            dtype=np.float64,
        ).reshape(-1)
        order = np.argsort(-pred_u, kind="mergesort")
        rank_of = np.empty(len(order), dtype=int)
        rank_of[order] = np.arange(1, len(order) + 1)
        ranking_seconds = float(time.perf_counter() - t_rank0)

        log_top = min(int(a.log_top_predictions), len(order))
        top_records: dict[int, dict[str, Any]] = {}
        for rank, idx in enumerate(order[:log_top], start=1):
            idx = int(idx)
            row: dict[str, Any] = {
                "slate_id": slate_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "candidate_index": idx,
                "predicted_rank": int(rank),
                "sequence": seqs[idx],
                "chain": chain_labels[idx],
                "predicted_utility": float(pred_u[idx]),
                "predicted_gain_from_exact_incumbent": float(
                    pred_u[idx] - lineage.utility
                ),
                "queried": 0,
                "accepted": 0,
                "exact_utility": "",
                "exact_gain": "",
            }
            for j, name in enumerate(OPT_NAMES):
                row[f"pred_score_{name}"] = float(pred_scores[idx, j])
            top_records[idx] = row

        old_utility = float(lineage.utility)
        old_sequence = decode_esm_tokens(
            lineage.tokens.reshape(1, -1)
        )[0]
        queried = 0
        accepted_idx = -1
        accepted_gain = 0.0
        strong_hits = 0

        # IMPORTANT: order is intentionally static inside the slate, exactly as
        # validated.  Paid failures update the NEXT slate's readout only.
        t_oracle0 = time.perf_counter()
        for rank, idx in enumerate(
            order[: int(a.verification_k)],
            start=1,
        ):
            if self._budget_remaining() == 0:
                self.stop_requested = True
                break
            idx = int(idx)
            seq = seqs[idx]
            rec = self.oracle.evaluate_one(seq)
            exact_u = float(_utility(rec.scores, pref, a.rho))
            gain = float(exact_u - old_utility)
            queried += 1
            lineage.verification_queries += 1
            is_strong = gain >= float(a.strong_gain)
            strong_hits += int(is_strong)
            accepted = gain > float(a.accept_epsilon)

            self._record_exact(
                slate[idx],
                z[idx],
                rec,
                source="sparse_verification",
                preference_id=int(preference_id),
                lineage_id=int(lineage.lineage_id),
                slate_id=slate_id,
                verification_rank=int(rank),
                utility_at_query=exact_u,
            )

            qrow: dict[str, Any] = {
                "query_index": int(self.oracle.unique_oracle_queries),
                "slate_id": slate_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence": old_sequence,
                "incumbent_utility": old_utility,
                "verification_rank": int(rank),
                "candidate_index": idx,
                "candidate_sequence": seq,
                "chain": chain_labels[idx],
                "readout_training_labels": readout_train_size,
                "readout_mode": str(a.readout_mode),
                "predicted_utility": float(pred_u[idx]),
                "predicted_gain": float(pred_u[idx] - old_utility),
                "exact_utility": exact_u,
                "exact_gain": gain,
                "accepted": int(accepted),
                "strong_gain_hit": int(is_strong),
                "strong_gain_threshold_is_diagnostic_only": True,
            }
            for j, name in enumerate(OPT_NAMES):
                qrow[f"pred_score_{name}"] = float(pred_scores[idx, j])
                qrow[f"exact_score_{name}"] = float(rec.scores[j])
            for j, name in enumerate(RAW_NAMES):
                qrow[f"raw_{name}"] = float(rec.raw[j])
            self.query_history.append(qrow)

            if idx in top_records:
                top_records[idx]["queried"] = 1
                top_records[idx]["accepted"] = int(accepted)
                top_records[idx]["exact_utility"] = exact_u
                top_records[idx]["exact_gain"] = gain

            if accepted:
                accepted_idx = idx
                accepted_gain = gain
                lineage.tokens = slate[idx].detach().cpu().clone()
                lineage.raw = np.asarray(rec.raw, dtype=np.float64).copy()
                lineage.scores = np.asarray(rec.scores, dtype=np.float64).copy()
                lineage.utility = exact_u
                lineage.accepted_moves += 1
                break

        oracle_seconds = float(time.perf_counter() - t_oracle0)
        lineage.slates_attempted += 1
        self.top_prediction_history.extend(top_records.values())
        self.slate_history.append(
            {
                "slate_id": slate_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence_before": old_sequence,
                "incumbent_utility_before": old_utility,
                "slate_size": int(len(seqs)),
                "slate_per_chain": int(a.slate_per_chain),
                "verification_k": int(a.verification_k),
                "verification_queries": int(queried),
                "accepted": int(accepted_idx >= 0),
                "accepted_rank": (
                    int(rank_of[accepted_idx]) if accepted_idx >= 0 else -1
                ),
                "accepted_sequence": (
                    seqs[accepted_idx] if accepted_idx >= 0 else ""
                ),
                "accepted_chain": (
                    chain_labels[accepted_idx] if accepted_idx >= 0 else ""
                ),
                "accepted_gain": float(accepted_gain),
                "incumbent_utility_after": float(lineage.utility),
                "cumulative_gain_from_lineage_start": float(
                    lineage.utility - lineage.initial_utility
                ),
                "strong_hits_among_verified": int(strong_hits),
                "strong_gain_threshold": float(a.strong_gain),
                "strong_gain_used_as_acceptance_gate": False,
                "readout_training_labels": readout_train_size,
                "readout_mode": str(a.readout_mode),
                "best_predicted_utility": float(pred_u[int(order[0])]),
                "best_predicted_gain": float(
                    pred_u[int(order[0])] - old_utility
                ),
                "proposal_seconds": proposal_seconds,
                "ranking_seconds": ranking_seconds,
                "oracle_seconds": oracle_seconds,
                "slate_wall_seconds": float(
                    proposal_seconds + ranking_seconds + oracle_seconds
                ),
                "proposal_physical_draws": int(
                    sum(int(m.get("cheap_draws", 0)) for m in proposal_meta)
                ),
                "proposal_candidates": int(len(seqs)),
                "proposal_draws_per_candidate": float(
                    sum(int(m.get("cheap_draws", 0)) for m in proposal_meta)
                    / max(len(seqs), 1)
                ),
                "batched_root_generation": True,
            }
        )

        if (
            int(a.save_every_slates) > 0
            and slate_id % int(a.save_every_slates) == 0
        ):
            self._save_progress()

    def optimize(self) -> None:
        a = self.args
        total = (
            len(self.preferences)
            * int(a.lineages)
            * int(a.slates_per_lineage)
        )
        bar = None
        if tqdm is not None and not bool(a.no_progress):
            bar = tqdm(
                total=total,
                desc="Koopman sparse-verify optimization",
                unit="slate",
                dynamic_ncols=True,
            )
        try:
            for cycle in range(int(a.slates_per_lineage)):
                for pref_id in range(len(self.preferences)):
                    for lineage in self.lineages[pref_id]:
                        if self.stop_requested:
                            return
                        before_q = int(self.oracle.unique_oracle_queries)
                        before_u = float(lineage.utility)
                        self._run_one_slate(pref_id, lineage, cycle)
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix(
                                unique_q=self.oracle.unique_oracle_queries,
                                last_gain=f"{lineage.utility - before_u:+.4f}",
                                dq=self.oracle.unique_oracle_queries - before_q,
                            )
        finally:
            if bar is not None:
                bar.close()

    def _select_outputs_for_preference(
        self,
        pref_id: int,
    ) -> list[ArchiveEntry]:
        a = self.args
        pref = self.preferences[int(pref_id)]
        ranked = sorted(
            self.archive,
            key=lambda e: float(_utility(e.scores, pref, a.rho)),
            reverse=True,
        )
        floor = (
            float(a.output_min_hamming)
            if a.output_min_hamming is not None
            else float(a.min_lineage_hamming)
        )
        chosen: list[ArchiveEntry] = []
        for e in ranked:
            if all(
                normalized_hamming_tokens(
                    e.tokens.numpy(),
                    old.tokens.numpy(),
                )
                >= floor
                for old in chosen
            ):
                chosen.append(e)
                if len(chosen) >= int(a.num_output_sequences):
                    break
        if len(chosen) < int(a.num_output_sequences):
            raise RuntimeError(
                f"exact archive packs only {len(chosen)}/{a.num_output_sequences} "
                f"final outputs at Hamming floor {floor}; discovery should have "
                "guaranteed this floor"
            )
        return chosen

    def finalize(self) -> dict[str, Any]:
        self._save_progress()
        a = self.args

        lineage_rows: list[dict[str, Any]] = []
        for pref_id, lines in self.lineages.items():
            pref = self.preferences[int(pref_id)]
            for lin in lines:
                seq = decode_esm_tokens(lin.tokens.reshape(1, -1))[0]
                row: dict[str, Any] = {
                    "preference_id": int(pref_id),
                    "weights": ",".join(f"{float(x):.8g}" for x in pref),
                    "lineage_id": int(lin.lineage_id),
                    "sequence": seq,
                    "utility": float(lin.utility),
                    "initial_utility": float(lin.initial_utility),
                    "cumulative_gain": float(lin.utility - lin.initial_utility),
                    "accepted_moves": int(lin.accepted_moves),
                    "slates_attempted": int(lin.slates_attempted),
                    "verification_queries": int(lin.verification_queries),
                }
                for j, name in enumerate(RAW_NAMES):
                    row[f"raw_{name}"] = float(lin.raw[j])
                for j, name in enumerate(OPT_NAMES):
                    row[f"score_{name}"] = float(lin.scores[j])
                lineage_rows.append(row)
        write_csv(self.out / "lineage_final.csv", lineage_rows)

        generated_rows: list[dict[str, Any]] = []
        per_pref_summary: dict[str, Any] = {}
        diversity_rows: list[dict[str, Any]] = []
        for pref_id, pref in enumerate(self.preferences):
            selected = self._select_outputs_for_preference(pref_id)
            for rank, e in enumerate(selected):
                row: dict[str, Any] = {
                    "preference_id": int(pref_id),
                    "weights": ",".join(f"{float(x):.8g}" for x in pref),
                    "output_rank": int(rank),
                    "sequence": e.sequence,
                    "utility": float(_utility(e.scores, pref, a.rho)),
                    "source": e.source,
                    "first_query_index": int(e.first_query_index),
                }
                for j, name in enumerate(RAW_NAMES):
                    row[f"raw_{name}"] = float(e.raw[j])
                for j, name in enumerate(OPT_NAMES):
                    row[f"score_{name}"] = float(e.scores[j])
                generated_rows.append(row)

            raw = np.stack([e.raw for e in selected], axis=0)
            scores = np.stack([e.scores for e in selected], axis=0)
            toks = np.stack([e.tokens.numpy() for e in selected], axis=0)
            utilities = np.asarray(
                [_utility(e.scores, pref, a.rho) for e in selected],
                dtype=np.float64,
            )
            qrows = [
                q for q in self.query_history
                if int(q["preference_id"]) == int(pref_id)
            ]
            srows = [
                s for s in self.slate_history
                if int(s["preference_id"]) == int(pref_id)
            ]
            accepted_q = [q for q in qrows if int(q["accepted"]) == 1]
            final_diversity = _extended_diversity_summary(toks)
            lineage_contribution = _lineage_contribution_summary(selected)
            trajectory_diversity = _trajectory_diversity_summary(qrows)
            live_lineage_diversity = _final_lineage_diversity_summary(
                self.lineages[int(pref_id)]
            )

            diversity_row: dict[str, Any] = {
                "preference_id": int(pref_id),
                "weights": ",".join(f"{float(x):.8g}" for x in pref),
            }
            for k, v in final_diversity.items():
                diversity_row[f"final_{k}"] = (
                    json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                )
            for k, v in lineage_contribution.items():
                diversity_row[f"lineage_{k}"] = (
                    json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                )
            for k, v in trajectory_diversity.items():
                diversity_row[f"trajectory_{k}"] = v
            for k, v in live_lineage_diversity.items():
                diversity_row[f"live_{k}"] = v
            diversity_rows.append(diversity_row)

            per_pref_summary[str(pref_id)] = {
                "weights": pref.tolist(),
                "generated_objectives": objective_summary(raw, scores),
                "generated_utility_mean": float(np.mean(utilities)),
                "generated_utility_best": float(np.max(utilities)),
                "generated_diversity": final_diversity,
                "lineage_contribution": lineage_contribution,
                "trajectory_diversity": trajectory_diversity,
                "live_lineage_diversity": live_lineage_diversity,
                "verification_queries": int(len(qrows)),
                "verification_positive_rate": (
                    float(np.mean([int(q["accepted"]) for q in qrows]))
                    if qrows else 0.0
                ),
                "accepted_moves": int(len(accepted_q)),
                "mean_accepted_rank": (
                    float(np.mean([int(q["verification_rank"]) for q in accepted_q]))
                    if accepted_q else float("nan")
                ),
                "mean_accepted_gain": (
                    float(np.mean([float(q["exact_gain"]) for q in accepted_q]))
                    if accepted_q else 0.0
                ),
                "slates": int(len(srows)),
                "slate_success_rate": (
                    float(np.mean([int(s["accepted"]) for s in srows]))
                    if srows else 0.0
                ),
            }

        write_csv(self.out / "generated.csv", generated_rows)
        write_csv(self.out / "diversity_metrics.csv", diversity_rows)

        all_scores = np.stack([e.scores for e in self.archive], axis=0)
        nd = nondominated_mask_max(all_scores)
        pareto = [
            self._archive_row(e)
            for e, keep in zip(self.archive, nd)
            if bool(keep)
        ]
        write_csv(self.out / "pareto_archive.csv", pareto)

        raw_generated = np.asarray(
            [
                [float(r[f"raw_{name}"]) for name in RAW_NAMES]
                for r in generated_rows
            ],
            dtype=np.float64,
        )
        scores_generated = np.asarray(
            [
                [float(r[f"score_{name}"]) for name in OPT_NAMES]
                for r in generated_rows
            ],
            dtype=np.float64,
        )
        tokens_generated = np.stack(
            [
                self.archive_by_sequence[str(r["sequence"])].tokens.numpy()
                for r in generated_rows
            ],
            axis=0,
        )

        accepted_queries = [
            q for q in self.query_history if int(q["accepted"]) == 1
        ]
        global_generated_diversity = _extended_diversity_summary(tokens_generated)
        global_trajectory_diversity = _trajectory_diversity_summary(self.query_history)
        summary = {
            "schema_version": "koopman-sparse-verify-final-v1",
            "implementation_version": IMPLEMENTATION_VERSION,
            "diversity_reporting_version": DIVERSITY_REPORTING_VERSION,
            "diversity_basin_radii": [float(x) for x in DIVERSITY_BASIN_RADII],
            "method": METHOD_NAME,
            "target": str(a.target),
            "raw_objective_names": list(RAW_NAMES),
            "optimization_objective_names": list(OPT_NAMES),
            "preferences": [p.tolist() for p in self.preferences],
            "preference_name": getattr(self, "preference_name", None),
            "rho": float(a.rho),
            "chains": [list(c) for c in self.chains],
            "proposal_law": PROPOSAL_LAW,
            "architecture_frozen": True,
            "core_algorithm": [
                "independent physical multi-horizon proposal",
                "shared multi-output Koopman ridge readout",
                str(a.readout_mode),
                "static top-K sparse exact verification",
                "accept first exact positive gain",
                "update readout dataset with every paid verification",
                "regenerate fresh slate after acceptance or top-K failure",
            ],
            "readout": {
                "mode": str(a.readout_mode),
                "initial_labels": int(self.discovery_unique_queries),
                "fixed_ridge_alpha": float(self.readout_alpha),
                "initial_alpha_cv_mse": float(self.readout_alpha_cv_mse),
                "current_training_labels": int(len(self.archive)),
                "alpha_reselected_online": False,
                "readout_refit_frequency": "once at start of each slate",
                "within_slate_reranking_after_query": False,
            },
            "search": {
                "lineages_per_preference": int(a.lineages),
                "slates_per_lineage": int(a.slates_per_lineage),
                "slate_per_chain": int(a.slate_per_chain),
                "slate_size": int(a.slate_per_chain) * len(self.chains),
                "verification_k": int(a.verification_k),
                "accept_epsilon": float(a.accept_epsilon),
                "strong_gain_diagnostic_threshold": float(a.strong_gain),
                "strong_gain_is_acceptance_gate": False,
                "min_lineage_hamming": float(a.min_lineage_hamming),
                "output_min_hamming": (
                    float(a.output_min_hamming)
                    if a.output_min_hamming is not None
                    else float(a.min_lineage_hamming)
                ),
            },
            "oracle_accounting": self.oracle.accounting(),
            "discovery_unique_queries": int(self.discovery_unique_queries),
            "optimization_unique_queries": int(
                max(
                    0,
                    self.oracle.unique_oracle_queries
                    - self.discovery_unique_queries,
                )
            ),
            "exact_archive_size": int(len(self.archive)),
            "pareto_archive_size": int(len(pareto)),
            "generated_count": int(len(generated_rows)),
            "accepted_moves_total": int(len(accepted_queries)),
            "verification_positive_rate": (
                float(np.mean([int(q["accepted"]) for q in self.query_history]))
                if self.query_history else 0.0
            ),
            "strong_hits_total": int(
                sum(int(q["strong_gain_hit"]) for q in self.query_history)
            ),
            "generated_objectives": objective_summary(
                raw_generated,
                scores_generated,
            ),
            "generated_diversity": global_generated_diversity,
            "trajectory_diversity": global_trajectory_diversity,
            "per_preference": per_pref_summary,
            "deliberately_not_in_core": [
                "HCLR",
                "residual transport",
                "contextual residual response",
                "contextual tail lift",
                "Residual Echo scheduler",
                "preference-specific reward model",
                "direct latent control",
                "optimization-time root-tail certificate",
            ],
        }
        write_json(self.out / "summary.json", summary)

        print(f"\n{METHOD_NAME} final generated objective summary", flush=True)
        for name, vals in summary["generated_objectives"].items():
            direction = "min" if name == "Hemolysis" else "max"
            print(
                f"  {name:12s} raw mean={vals['raw_mean']:.6g} "
                f"raw best({direction})={vals['raw_best']:.6g} "
                f"opt mean={vals['optimization_mean']:.6g} "
                f"opt best={vals['optimization_best']:.6g}",
                flush=True,
            )
        print(
            f"  oracle unique queries={self.oracle.unique_oracle_queries} "
            f"(discovery={self.discovery_unique_queries}, "
            f"optimization={summary['optimization_unique_queries']})",
            flush=True,
        )
        print(
            f"  accepted moves={summary['accepted_moves_total']} "
            f"verification positive rate={summary['verification_positive_rate']:.4f}",
            flush=True,
        )
        div = summary["generated_diversity"]
        print(
            "  diversity: "
            f"unique={div.get('unique_fraction', float('nan')):.4f} "
            f"pairwise_mean={div.get('pairwise_hamming_mean', float('nan')):.4f} "
            f"NN_mean={div.get('nearest_neighbor_hamming_mean', float('nan')):.4f} "
            f"NN_p10={div.get('nearest_neighbor_hamming_p10', float('nan')):.4f} "
            f"NN_min={div.get('nearest_neighbor_hamming_min', float('nan')):.4f}",
            flush=True,
        )
        print(
            "  Hamming basins: "
            + ", ".join(
                f"r={r:.2f}: {div.get(f'basin_count_r{r:.2f}', 0)} "
                f"(largest={div.get(f'largest_basin_fraction_r{r:.2f}', float('nan')):.3f})"
                for r in DIVERSITY_BASIN_RADII
            ),
            flush=True,
        )
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] proposal_law={PROPOSAL_LAW}\n"
            f"[{METHOD_NAME}] readout={self.args.readout_mode}, "
            f"slate={int(self.args.slate_per_chain) * len(self.chains)}, "
            f"K={self.args.verification_k}, "
            f"accept_epsilon={self.args.accept_epsilon}, "
            f"strong_gain={self.args.strong_gain} (diagnostic only)",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residual-checkpoint", required=True)
    p.add_argument("--terminal-checkpoint", default="")
    p.add_argument("--lkf-checkpoint", default="")
    p.add_argument("--allow-checkpoint-sha-mismatch", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--peptiverse-root", default="${PEPTIVERSE_ROOT}")
    p.add_argument("--peptiverse-manifest", default="")
    p.add_argument("--peptiverse-device", default="cuda")
    p.add_argument("--target", default=DEFAULT_AMHR2_TARGET)
    p.add_argument("--preferences", default="1,1,1,1,1,1")
    p.add_argument(
        "--preference-name",
        default="",
        help=(
            "Named preference from pegasus.pareto_weight_sweep_preferences "
            "(equal/nonhem/nonfouling/solubility/permeability/halflife/affinity). "
            "When set, overrides --preferences with the canonical vector."
        ),
    )
    p.add_argument(
        "--weights",
        default="",
        help="Optional alias for a single length-6 preference vector (overrides --preferences).",
    )
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--no-clip-optimization-scores", action="store_true")

    p.add_argument("--peptide-length", type=int, default=12)
    p.add_argument("--native-nfe", type=int, default=1)
    p.add_argument("--chains", default=DEFAULT_CHAINS)

    # Information-rich discovery / initial readout.
    p.add_argument("--initial-readout-queries", type=int, default=128)
    p.add_argument("--discovery-pool-multiplier", type=int, default=8)
    p.add_argument("--discovery-max-pool-expansions", type=int, default=3)
    p.add_argument("--discovery-generation-batch", type=int, default=256)
    p.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100")
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--feature-batch-size", type=int, default=256)

    # Frozen production search architecture.
    p.add_argument("--lineages", type=int, default=8)
    p.add_argument("--slates-per-lineage", type=int, default=25)
    p.add_argument("--slate-per-chain", type=int, default=64)
    p.add_argument("--verification-k", type=int, default=8)
    p.add_argument(
        "--readout-mode",
        choices=("anchor_parent_delta", "anchor_absolute"),
        default="anchor_parent_delta",
    )
    p.add_argument("--accept-epsilon", type=float, default=1e-6)
    p.add_argument(
        "--strong-gain",
        type=float,
        default=0.02,
        help="Diagnostic threshold only; NEVER blocks a smaller exact improvement.",
    )

    # Physical sampling / diversity.
    p.add_argument("--min-lineage-hamming", type=float, default=0.20)
    p.add_argument("--output-min-hamming", type=float, default=None)
    p.add_argument("--cheap-generation-batch", type=int, default=256)
    p.add_argument("--max-cheap-draws-per-candidate", type=int, default=4096)
    p.add_argument("--max-root-attempts-per-chain", type=int, default=8192)

    # Reporting / safety.
    p.add_argument("--num-output-sequences", type=int, default=100)
    p.add_argument("--log-top-predictions", type=int, default=16)
    p.add_argument("--save-every-slates", type=int, default=10)
    p.add_argument(
        "--max-unique-oracle-queries",
        type=int,
        default=0,
        help="0 means no additional hard cap beyond the finite slate schedule.",
    )
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.initial_readout_queries) < 8:
        raise ValueError("--initial-readout-queries must be at least 8")
    if int(args.lineages) <= 0:
        raise ValueError("--lineages must be positive")
    if int(args.slates_per_lineage) < 0:
        raise ValueError("--slates-per-lineage cannot be negative")
    if int(args.slate_per_chain) <= 0:
        raise ValueError("--slate-per-chain must be positive")
    if int(args.verification_k) <= 0:
        raise ValueError("--verification-k must be positive")
    if int(args.verification_k) > int(args.slate_per_chain) * len(parse_chains(args.chains)):
        raise ValueError("--verification-k exceeds total slate size")
    if float(args.accept_epsilon) < 0.0:
        raise ValueError("--accept-epsilon cannot be negative")
    if float(args.strong_gain) <= 0.0:
        raise ValueError("--strong-gain must be positive")
    if not (0.0 <= float(args.min_lineage_hamming) <= 1.0):
        raise ValueError("--min-lineage-hamming must lie in [0,1]")
    KoopmanSparseVerifyRunner(args).run()


if __name__ == "__main__":
    main()
