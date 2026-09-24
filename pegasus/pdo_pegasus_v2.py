"""Unified PEGASUS v2: physical branching -> Koopman look-ahead -> coarse PDO -> fine PDO.

This module operationalizes the evidence-backed division of labor:

* LKF is the *only* physical generator.  It instantiates stochastic intermediate
  states and all terminal sequences.
* Koopman A_C is a conditional finite-time observability operator.  It is used only
  to forecast the expected terminal KFM observable of an already-realized physical
  intermediate state, before spending the remaining generation compute on that branch.
* Coarse PDO converts a generated terminal slate into an adaptive exact-query decision
  instead of relying on a fixed top-K surrogate ranking.
* Fine PDO is the recovery-safe Hamming-1 mechanism from PEGASUS v1.3.
* Exact six-objective oracle vectors are the sole acceptance authority at both scales.

Deliberately absent: residual-bank sampling/control, latent actuation, witness decoding,
information-seeking bonuses for coarse branch allocation, and hard Koopman pruning.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_v1 as v1
from .kfocus.objectives import OPT_NAMES, RAW_NAMES
from .kfocus.regions import terminal_z
from .pdo_math import (
    EmpiricalConfidenceRecovery,
    best_exact_improving_index,
    decision_relevant_probe_choice,
    empirical_confidence_verification_index,
    version_set_pdo_decision,
)
from .pdo_uncertainty import build_local_response_state
from .terminal_controlled_koopman_geometry import chain_name, prepare_chain_start
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-v2-unified-koopman-coarse-pdo"
METHOD_NAME = "Unified PEGASUS v2"


@dataclass(frozen=True)
class CoarseBranch:
    global_index: int
    chain_index: int
    chain: tuple[float, ...]
    chain_label: str
    start_index_within_chain: int
    start_tokens: torch.Tensor
    predicted_terminal_z: np.ndarray
    predicted_scores: np.ndarray
    predicted_utility: float
    allocated_candidates: int = 0


def _uniform_integer_allocation(n: int, budget: int) -> np.ndarray:
    """Deterministic nearly-uniform integer allocation summing exactly to budget."""
    n = int(n); budget = int(budget)
    if n <= 0 or budget < 0:
        raise ValueError("invalid allocation dimensions")
    q, r = divmod(budget, n)
    out = np.full(n, q, dtype=np.int64)
    if r:
        out[:r] += 1
    return out


def _rank_weights(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(s) == 0 or np.any(~np.isfinite(s)):
        raise ValueError("branch scores must be a nonempty finite vector")
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(len(s), dtype=np.int64)
    return 1.0 / (ranks.astype(np.float64) + 1.0)


def _largest_remainder_unbounded(weights: np.ndarray, total: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = int(total)
    if len(w) == 0 or total < 0 or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("invalid largest-remainder arguments")
    if total == 0:
        return np.zeros(len(w), dtype=np.int64)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return _uniform_integer_allocation(len(w), total)
    ideal = total * w / sw
    out = np.floor(ideal).astype(np.int64)
    rem = total - int(np.sum(out))
    if rem:
        frac = ideal - out
        order = np.argsort(-frac, kind="mergesort")
        out[order[:rem]] += 1
    if int(np.sum(out)) != total:
        raise RuntimeError("largest-remainder allocation failed to conserve budget")
    return out


def safe_koopman_branch_allocation(
    scores: np.ndarray,
    *,
    budget: int,
    adaptive_fraction: float,
    minimum_per_branch: int = 1,
) -> np.ndarray:
    """Protected uniform reserve plus rank-weighted Koopman allocation.

    ``adaptive_fraction`` controls only the model-side continuation budget.  A uniform
    reserve is allocated first; A_C can redistribute only the remaining fraction.  No
    branch is hard-pruned when ``minimum_per_branch >= 1``.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(s); b = int(budget); f = float(adaptive_fraction); m = int(minimum_per_branch)
    if n <= 0 or b <= 0 or not (0.0 <= f <= 1.0) or m < 0:
        raise ValueError("invalid Koopman branch allocation arguments")
    if b < n * m:
        raise ValueError("generation budget is too small for the protected branch floor")
    protected_budget = int(round((1.0 - f) * b))
    protected_budget = max(protected_budget, n * m)
    protected_budget = min(protected_budget, b)
    base_counts = np.full(n, m, dtype=np.int64)
    base_counts += _uniform_integer_allocation(n, protected_budget - n * m)
    remainder = b - int(np.sum(base_counts))
    if remainder > 0:
        base_counts += _largest_remainder_unbounded(_rank_weights(s), remainder)
    if int(np.sum(base_counts)) != b:
        raise RuntimeError("Koopman allocation does not conserve generation budget")
    if np.any(base_counts < m):
        raise RuntimeError("Koopman allocation violated the protected branch floor")
    return base_counts


def coarse_decision_coordinates(
    candidate_z: np.ndarray,
    incumbent_z: np.ndarray,
    *,
    svd_rtol: float = 1e-8,
) -> tuple[np.ndarray, float, int]:
    """Return no-op-inclusive transductive KFM coordinates for coarse PDO.

    Candidate endpoints are already physically realizable sequences, so KFM is used here
    only as a decision representation.  The exact incumbent/no-op is row 0 with coordinate
    zero.  Coordinates are normalized and projected onto the realized candidate span.
    """
    zz = np.asarray(candidate_z, dtype=np.float64)
    z0 = np.asarray(incumbent_z, dtype=np.float64).reshape(-1)
    if zz.ndim != 2 or zz.shape[1] != len(z0) or np.any(~np.isfinite(zz)) or np.any(~np.isfinite(z0)):
        raise ValueError("candidate/incumbent KFM feature shape mismatch")
    rtol = float(svd_rtol)
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("svd_rtol must be finite and positive")
    d = zz - z0.reshape(1, -1)
    norms = np.linalg.norm(d, axis=1)
    positive = norms[norms > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    x = d / scale
    if x.size == 0 or np.linalg.norm(x) <= 1e-14:
        q = np.zeros((len(zz) + 1, 0), dtype=np.float64)
        return q, scale, 0
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    thresh = max(float(s[0]) * rtol, 1e-12)
    rank = int(np.sum(s > thresh))
    basis = vt[:rank].T if rank else np.zeros((x.shape[1], 0), dtype=np.float64)
    qc = x @ basis
    q = np.concatenate([np.zeros((1, rank), dtype=np.float64), qc], axis=0)
    return q, scale, rank


@torch.no_grad()
def _sample_allocated_fixed_start_pool(
    model: Any,
    starts: torch.Tensor,
    chain: Sequence[float],
    counts: np.ndarray,
    *,
    seed: int,
    forbidden_keys: set[tuple[int, ...]],
    accepted_keys: set[tuple[int, ...]],
    diversity_anchors: Sequence[torch.Tensor],
    min_hamming: float,
    batch_size: int,
    max_cheap_draws_per_candidate: int,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]]]:
    """Generate terminal continuations from fixed realized stochastic starts.

    Every slot is permanently assigned to one physical intermediate start.  Rejected
    duplicate/diversity draws are retried from that same start, preserving the intended
    branch-conditioned proposal law.
    """
    cnt = np.asarray(counts, dtype=np.int64).reshape(-1)
    if torch.as_tensor(starts).ndim != 2 or len(cnt) != len(starts):
        raise ValueError("starts/counts shape mismatch")
    if np.any(cnt < 0) or int(np.sum(cnt)) <= 0:
        raise ValueError("counts must be nonnegative with positive total")
    max_draws = int(max_cheap_draws_per_candidate)
    if max_draws <= 0:
        raise ValueError("max_cheap_draws_per_candidate must be positive")

    tm = model.base_model
    device = model.device
    starts_d = torch.as_tensor(starts, dtype=torch.long, device=device)
    owners = np.repeat(np.arange(len(cnt), dtype=np.int64), cnt)
    n = len(owners)
    slot_starts = starts_d.index_select(0, torch.as_tensor(owners, device=device, dtype=torch.long))
    accepted: list[torch.Tensor | None] = [None] * n
    draws = np.zeros(n, dtype=np.int64)
    rej_dup = np.zeros(n, dtype=np.int64)
    rej_div = np.zeros(n, dtype=np.int64)
    unresolved = np.arange(n, dtype=np.int64)

    anchor_interiors: list[np.ndarray] = []
    for other in diversity_anchors:
        oi = torch.as_tensor(other).detach().cpu().reshape(-1)
        if oi.numel() > 2:
            oi = oi[1:-1]
        anchor_interiors.append(oi.numpy())

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    chunk = max(1, int(batch_size))
    c = tuple(model.match_chain(chain))

    while len(unresolved):
        if np.any(draws[unresolved] >= max_draws):
            bad = unresolved[draws[unresolved] >= max_draws]
            raise RuntimeError(
                f"{chain_name(c)} branch-conditioned rejection exhausted for {len(bad)} slots "
                f"after {max_draws} draws/slot"
            )
        next_unresolved: list[int] = []
        for lo in range(0, len(unresolved), chunk):
            ids_np = unresolved[lo:lo + chunk]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            terminal = tm.sample_chain(
                slot_starts.index_select(0, ids),
                c,
                generator=gen,
            ).detach().cpu()
            for j, slot_v in enumerate(ids_np.tolist()):
                slot = int(slot_v)
                draws[slot] += 1
                row = terminal[j].clone()
                key = base._token_key(row)
                if key in forbidden_keys or key in accepted_keys:
                    rej_dup[slot] += 1
                    next_unresolved.append(slot)
                    continue
                interior = row.reshape(-1)
                if interior.numel() > 2:
                    interior = interior[1:-1]
                arr = interior.numpy()
                ok = True
                for oi in anchor_interiors:
                    if arr.size != oi.size:
                        raise ValueError("diversity token lengths differ")
                    if float(np.mean(arr != oi)) < float(min_hamming) - 1e-12:
                        ok = False
                        break
                if not ok:
                    rej_div[slot] += 1
                    next_unresolved.append(slot)
                    continue
                accepted[slot] = row
                accepted_keys.add(key)
        unresolved = np.asarray(next_unresolved, dtype=np.int64)

    if any(x is None for x in accepted):
        raise RuntimeError("internal error: unresolved branch-conditioned candidate")
    meta = [
        {
            "branch_start_index": int(owners[i]),
            "cheap_draws": int(draws[i]),
            "duplicate_rejections": int(rej_dup[i]),
            "diversity_rejections": int(rej_div[i]),
            "branch_conditioned_generation": True,
            "residual_bank_used": False,
        }
        for i in range(n)
    ]
    return torch.stack([x for x in accepted if x is not None]), owners, meta


class UnifiedPDOPegasusRunner(v1.PDOPegasusRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.koopman_branch_history: list[dict[str, Any]] = []
        self.coarse_pdo_decision_history: list[dict[str, Any]] = []
        self.coarse_pdo_query_history: list[dict[str, Any]] = []
        self._coarse_pdo_fresh_queries = 0
        self._coarse_pdo_accepted_moves = 0
        self._coarse_pdo_failed_confidence_verifications = 0
        self._coarse_pdo_recovery_probes = 0

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.koopman_branch_history:
            write_csv(self.out / "koopman_branch_history.csv", self.koopman_branch_history)
        if self.coarse_pdo_decision_history:
            write_csv(self.out / "coarse_pdo_decision_history.csv", self.coarse_pdo_decision_history)
        if self.coarse_pdo_query_history:
            write_csv(self.out / "coarse_pdo_query_history.csv", self.coarse_pdo_query_history)

    def _make_realized_branches(
        self,
        lineage: base.LineageState,
        *,
        slate_id: int,
        readout: base.RidgeReadout,
    ) -> tuple[list[CoarseBranch], np.ndarray, np.ndarray]:
        a = self.args
        tm = self.model.base_model
        nstart = int(a.koopman_pilot_starts_per_chain)
        if nstart <= 0:
            raise ValueError("koopman_pilot_starts_per_chain must be positive")
        incumbent_z = terminal_z(
            self.model,
            lineage.tokens.reshape(1, -1),
            batch_size=int(a.feature_batch_size),
        )[0]
        branches: list[CoarseBranch] = []
        gidx = 0
        for ci, chain in enumerate(self.chains):
            c = tuple(self.model.match_chain(chain))
            parents = lineage.tokens.reshape(1, -1).expand(nstart, -1).contiguous().to(self.model.device)
            gen = torch.Generator(device=self.model.device)
            gen.manual_seed(base._stable_seed(
                a.seed, "pdo2-branch-start", lineage.preference_id, lineage.lineage_id, slate_id, ci
            ))
            starts = prepare_chain_start(tm, parents, c, generator=gen).detach().cpu()
            if str(a.koopman_branch_allocation_mode) == "ac_opportunity":
                # The only A_C call in production: conditional mean terminal observable
                # from an already-realized physical stochastic start.
                pred_z_t = self.model.predict_mean_z(starts.to(self.model.device), c)
                pred_z = pred_z_t.detach().cpu().numpy().astype(np.float64)
                pred_scores = base._predict_scores(
                    readout,
                    pred_z,
                    mode=str(a.readout_mode),
                    incumbent_scores=lineage.scores,
                    incumbent_z=incumbent_z,
                )
                pref = self.preferences[int(lineage.preference_id)]
                pred_u = np.asarray(base._utility(pred_scores, pref, a.rho), dtype=np.float64).reshape(-1)
            elif str(a.koopman_branch_allocation_mode) == "uniform":
                # Clean A_C ablation: do not invoke the Koopman finite-time operator at all.
                pred_z = np.full((nstart, int(self.model.anchor_dim)), np.nan, dtype=np.float64)
                pred_scores = np.full((nstart, len(OPT_NAMES)), np.nan, dtype=np.float64)
                pred_u = np.zeros(nstart, dtype=np.float64)
            else:
                raise ValueError(f"unknown Koopman branch allocation mode {a.koopman_branch_allocation_mode!r}")
            for sj in range(nstart):
                branches.append(CoarseBranch(
                    global_index=gidx,
                    chain_index=ci,
                    chain=c,
                    chain_label=chain_name(c),
                    start_index_within_chain=sj,
                    start_tokens=starts[sj].clone(),
                    predicted_terminal_z=pred_z[sj].copy(),
                    predicted_scores=np.asarray(pred_scores[sj], dtype=np.float64).copy(),
                    predicted_utility=float(pred_u[sj]),
                ))
                gidx += 1
        scores = np.asarray([b.predicted_utility for b in branches], dtype=np.float64)
        total_budget = int(a.slate_per_chain) * len(self.chains)
        if str(a.koopman_branch_allocation_mode) == "uniform":
            counts = _uniform_integer_allocation(len(branches), total_budget)
        elif str(a.koopman_branch_allocation_mode) == "ac_opportunity":
            counts = safe_koopman_branch_allocation(
                scores,
                budget=total_budget,
                adaptive_fraction=float(a.koopman_adaptive_fraction),
                minimum_per_branch=int(a.koopman_min_candidates_per_branch),
            )
        else:
            raise ValueError(f"unknown Koopman branch allocation mode {a.koopman_branch_allocation_mode!r}")
        branches = [
            CoarseBranch(**{**b.__dict__, "allocated_candidates": int(counts[i])})
            for i, b in enumerate(branches)
        ]
        return branches, counts, incumbent_z

    def _generate_koopman_conditioned_slate(
        self,
        lineage: base.LineageState,
        *,
        slate_id: int,
        readout: base.RidgeReadout,
    ) -> tuple[torch.Tensor, list[str], list[dict[str, Any]], list[CoarseBranch], np.ndarray]:
        """Generate the full fixed-compute slate via realized starts + A_C allocation."""
        a = self.args
        branches, counts, incumbent_z = self._make_realized_branches(
            lineage, slate_id=slate_id, readout=readout
        )
        forbidden = {base._token_key(e.tokens) for e in self.archive}
        forbidden.add(base._token_key(lineage.tokens))
        accepted_keys: set[tuple[int, ...]] = set()
        anchors = self._diversity_anchors(lineage.preference_id, lineage.lineage_id)

        pools: list[torch.Tensor] = []
        labels: list[str] = []
        metas: list[dict[str, Any]] = []
        for ci, chain in enumerate(self.chains):
            inds = [i for i, b in enumerate(branches) if b.chain_index == ci]
            starts = torch.stack([branches[i].start_tokens for i in inds], dim=0)
            local_counts = np.asarray([counts[i] for i in inds], dtype=np.int64)
            if int(np.sum(local_counts)) == 0:
                continue
            pool, owners, local_meta = _sample_allocated_fixed_start_pool(
                self.model,
                starts,
                chain,
                local_counts,
                seed=base._stable_seed(
                    a.seed, "pdo2-continuations", lineage.preference_id, lineage.lineage_id, slate_id, ci
                ),
                forbidden_keys=forbidden,
                accepted_keys=accepted_keys,
                diversity_anchors=anchors,
                min_hamming=float(a.min_lineage_hamming),
                batch_size=int(a.cheap_generation_batch),
                max_cheap_draws_per_candidate=int(a.max_cheap_draws_per_candidate),
            )
            pools.append(pool)
            cname = chain_name(chain)
            labels.extend([cname] * int(len(pool)))
            for m, owner in zip(local_meta, owners.tolist()):
                gi = inds[int(owner)]
                b = branches[gi]
                metas.append({
                    "chain": cname,
                    "branch_global_index": int(gi),
                    "branch_start_index": int(b.start_index_within_chain),
                    "branch_predicted_utility": float(b.predicted_utility),
                    "branch_allocated_candidates": int(b.allocated_candidates),
                    **m,
                })

        if not pools:
            raise RuntimeError("Koopman branch allocation generated an empty slate")
        slate = torch.cat(pools, dim=0)
        expected = int(a.slate_per_chain) * len(self.chains)
        if len(slate) != expected or len(labels) != expected or len(metas) != expected:
            raise RuntimeError(
                f"branch-conditioned slate size mismatch: got {len(slate)}, expected {expected}"
            )
        for b in branches:
            self.koopman_branch_history.append({
                "slate_id": int(slate_id),
                "cycle": "",
                "preference_id": int(lineage.preference_id),
                "lineage_id": int(lineage.lineage_id),
                "chain": b.chain_label,
                "branch_global_index": int(b.global_index),
                "branch_start_index": int(b.start_index_within_chain),
                "predicted_terminal_utility": float(b.predicted_utility),
                "allocated_candidates": int(b.allocated_candidates),
                "adaptive_fraction": float(a.koopman_adaptive_fraction),
                "allocation_mode": str(a.koopman_branch_allocation_mode),
                "protected_uniform_fraction": float(1.0 - float(a.koopman_adaptive_fraction))
                    if str(a.koopman_branch_allocation_mode) == "ac_opportunity" else 1.0,
                "residual_bank_used": False,
                "hard_pruning_used": False,
            })
        return slate, labels, metas, branches, incumbent_z

    def _coarse_exact_map(
        self,
        seqs: Sequence[str],
        lineage: base.LineageState,
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {0: np.asarray(lineage.scores, dtype=np.float64).copy()}
        for j, seq in enumerate(seqs, start=1):
            e = self.archive_by_sequence.get(str(seq))
            if e is not None:
                out[int(j)] = np.asarray(e.scores, dtype=np.float64).copy()
        return out

    def _record_coarse_exact(
        self,
        *,
        candidate_index: int,
        tokens: torch.Tensor,
        z: np.ndarray,
        seq: str,
        chain: str,
        preference_id: int,
        lineage: base.LineageState,
        slate_id: int,
        cycle: int,
        query_order: int,
        source: str,
        mode: str,
        predicted_utility: float,
        gain_lower: float,
        gain_upper: float,
        regret_upper: float,
        info_score: float | None = None,
    ) -> tuple[base.ArchiveEntry | None, bool]:
        existing = self.archive_by_sequence.get(str(seq))
        before_q = int(self.oracle.unique_oracle_queries)
        if existing is not None:
            return existing, False
        if self._budget_remaining() == 0:
            self.stop_requested = True
            return None, False
        rec = self.oracle.evaluate_one(str(seq))
        fresh = int(self.oracle.unique_oracle_queries) > before_q
        exact_u = float(base._utility(rec.scores, self.preferences[int(preference_id)], self.args.rho))
        entry = self._record_exact(
            tokens,
            z,
            rec,
            source=source,
            preference_id=int(preference_id),
            lineage_id=int(lineage.lineage_id),
            slate_id=int(slate_id),
            verification_rank=int(query_order),
            utility_at_query=exact_u,
        )
        if fresh:
            self._coarse_pdo_fresh_queries += 1
            lineage.verification_queries += 1
        row: dict[str, Any] = {
            "query_index": int(self.oracle.unique_oracle_queries),
            "slate_id": int(slate_id),
            "cycle": int(cycle),
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "incumbent_sequence": base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0],
            "incumbent_utility": float(lineage.utility),
            "verification_rank": int(query_order),
            "candidate_index": int(candidate_index - 1),
            "candidate_sequence": str(seq),
            "chain": str(chain),
            "predicted_utility": float(predicted_utility),
            "exact_utility": exact_u,
            "exact_gain": float(exact_u - float(lineage.utility)),
            "accepted": 0,
            "strong_gain_hit": int(exact_u - float(lineage.utility) >= float(self.args.strong_gain)),
            "strong_gain_threshold_is_diagnostic_only": True,
            "coarse_pdo": True,
            "coarse_pdo_mode": str(mode),
            "predicted_gain_lower": float(gain_lower),
            "predicted_gain_upper": float(gain_upper),
            "predicted_regret_upper": float(regret_upper),
            "information_score": "" if info_score is None else float(info_score),
            "fresh_oracle_query": int(fresh),
        }
        for j, name in enumerate(OPT_NAMES):
            row[f"exact_score_{name}"] = float(rec.scores[j])
        for j, name in enumerate(RAW_NAMES):
            row[f"raw_{name}"] = float(rec.raw[j])
        self.query_history.append(row)
        self.coarse_pdo_query_history.append(dict(row))
        self._record_preference_query_use(preference_id, seq)
        return entry, fresh

    def _accept_coarse_entry(
        self,
        lineage: base.LineageState,
        *,
        candidate_tokens: torch.Tensor,
        entry: base.ArchiveEntry,
        slate_id: int,
        candidate_sequence: str,
    ) -> bool:
        exact_u = float(base._utility(entry.scores, self.preferences[int(lineage.preference_id)], self.args.rho))
        gain = float(exact_u - float(lineage.utility))
        if gain <= float(self.args.accept_epsilon):
            return False
        lineage.tokens = candidate_tokens.detach().cpu().clone()
        lineage.raw = np.asarray(entry.raw, dtype=np.float64).copy()
        lineage.scores = np.asarray(entry.scores, dtype=np.float64).copy()
        lineage.utility = exact_u
        lineage.accepted_moves += 1
        self._coarse_pdo_accepted_moves += 1
        for rows in (self.query_history, self.coarse_pdo_query_history):
            for row in reversed(rows):
                if int(row.get("slate_id", -1)) != int(slate_id):
                    continue
                if str(row.get("candidate_sequence", "")) == str(candidate_sequence):
                    row["accepted"] = 1
                    break
        return True

    def _run_coarse_pdo_slate(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        *,
        query_cap_override: int | None = None,
    ) -> dict[str, Any]:
        a = self.args
        if self.stop_requested or self._budget_remaining() == 0:
            if self._budget_remaining() == 0:
                self.stop_requested = True
            return {"paid_queries": 0, "accepted": 0, "gain": 0.0, "reason": "budget_exhausted"}

        self.slate_counter += 1
        slate_id = int(self.slate_counter)
        readout = self._fit_current_readout()
        readout_train_size = int(len(self.archive))
        self.readout_history.append({
            "event": "fit_before_unified_coarse_slate",
            "slate_id": slate_id,
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "cycle": int(cycle),
            "training_labels": readout_train_size,
            "alpha": float(self.readout_alpha),
            "cv_mse": "",
            "readout_mode": str(a.readout_mode),
            "alpha_fixed_for_all_future_slates": True,
        })

        old_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        old_scores = np.asarray(lineage.scores, dtype=np.float64).copy()
        old_utility = float(lineage.utility)
        t0 = time.perf_counter()
        slate, chain_labels, proposal_meta, branches, incumbent_z = self._generate_koopman_conditioned_slate(
            lineage, slate_id=slate_id, readout=readout
        )
        proposal_seconds = float(time.perf_counter() - t0)

        t1 = time.perf_counter()
        seqs = base.decode_esm_tokens(slate)
        z = terminal_z(self.model, slate, batch_size=int(a.feature_batch_size))
        pred_scores = base._predict_scores(
            readout, z, mode=str(a.readout_mode), incumbent_scores=old_scores, incumbent_z=incumbent_z
        )
        pref = self.preferences[int(preference_id)]
        pred_u = np.asarray(base._utility(pred_scores, pref, a.rho), dtype=np.float64).reshape(-1)
        order = np.argsort(-pred_u, kind="mergesort")
        rank_of = np.empty(len(order), dtype=np.int64); rank_of[order] = np.arange(1, len(order)+1)
        ranking_seconds = float(time.perf_counter() - t1)

        log_top = min(int(a.log_top_predictions), len(order))
        top_records: dict[int, dict[str, Any]] = {}
        for rank, idx_v in enumerate(order[:log_top], start=1):
            idx = int(idx_v)
            row: dict[str, Any] = {
                "slate_id": slate_id, "cycle": int(cycle), "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id), "candidate_index": idx,
                "predicted_rank": int(rank), "sequence": seqs[idx], "chain": chain_labels[idx],
                "predicted_utility": float(pred_u[idx]),
                "predicted_gain_from_exact_incumbent": float(pred_u[idx] - old_utility),
                "queried": 0, "accepted": 0, "exact_utility": "", "exact_gain": "",
                "coarse_pdo": True,
            }
            for j, name in enumerate(OPT_NAMES): row[f"pred_score_{name}"] = float(pred_scores[idx,j])
            top_records[idx] = row

        qcoord, qscale, qrank = coarse_decision_coordinates(
            z, incumbent_z, svd_rtol=float(a.pdo_coarse_decision_span_svd_rtol)
        )
        # Reference/no-op first; candidate prior rows follow in slate order.
        incumbent_prior = base._predict_scores(
            readout, incumbent_z.reshape(1,-1), mode=str(a.readout_mode),
            incumbent_scores=old_scores, incumbent_z=incumbent_z,
        )[0]
        prior_all = np.concatenate([incumbent_prior.reshape(1,-1), pred_scores], axis=0)
        exact_map = self._coarse_exact_map(seqs, lineage)
        recovery = EmpiricalConfidenceRecovery()
        queried_fresh = 0
        query_attempts = 0
        accepted_idx: int | None = None  # no-op-inclusive index
        reason = "coarse_pdo_no_query"
        cap = int(a.verification_k) if int(a.pdo_coarse_query_cap_per_slate) < 0 else int(a.pdo_coarse_query_cap_per_slate)
        if query_cap_override is not None:
            cap = min(cap, int(query_cap_override))
        cap = max(cap, 0)
        t2 = time.perf_counter()

        while not self.stop_requested:
            exact_map = self._coarse_exact_map(seqs, lineage)
            response = build_local_response_state(
                prior_all,
                qcoord,
                exact_map,
                self.pdo_calibration,
                reference_action_index=0,
                local_ridge_alphas=v1._parse_float_tuple(a.pdo_coarse_local_ridge_alphas),
                information_lambda=float(a.pdo_coarse_information_lambda),
                clip_scores=not bool(a.no_clip_optimization_scores),
            )
            decision = version_set_pdo_decision(
                response.member_scores,
                incumbent_scores=old_scores,
                preference=pref,
                rho=float(a.rho),
                epsilon_dec=float(a.pdo_coarse_epsilon_dec),
                enable_certification=(
                    recovery.confidence_allowed(True)
                    and int(response.local_nonreference_observations) >= int(a.pdo_coarse_min_probes_before_confidence)
                ),
            )
            best_idx = int(decision.best_mean_index)
            self.coarse_pdo_decision_history.append({
                "slate_id": slate_id, "cycle": int(cycle), "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id), "incumbent_sequence": old_sequence,
                "candidate_count_including_noop": int(len(seqs)+1), "decision_rank": int(qrank),
                "representation_scale": float(qscale), "version_set_members": int(response.ensemble_size),
                "exact_nonnoop_candidates": int(response.local_nonreference_observations),
                "distinct_member_winners": int(decision.distinct_member_winners),
                "common_epsilon_action_count": int(len(decision.common_epsilon_indices)),
                "plausible_winner_count": int(len(decision.plausible_indices)),
                "best_mean_index_noop_inclusive": best_idx,
                "best_mean_sequence": "NOOP" if best_idx == 0 else seqs[best_idx-1],
                "best_mean_gain_lower": float(decision.gain_lower[best_idx]),
                "best_mean_gain_upper": float(decision.gain_upper[best_idx]),
                "best_mean_regret_upper": float(decision.regret_upper[best_idx]),
                "empirical_confidence_enabled": int(decision.certification_enabled),
                "recovery_pending": int(recovery.pending),
                "fresh_queries_used": int(queried_fresh), "fresh_query_cap": int(cap),
                "epsilon_dec": float(a.pdo_coarse_epsilon_dec), "claim_level": "empirical_confidence",
            })

            verify_idx = empirical_confidence_verification_index(
                decision,
                no_op_index=0,
                accept_epsilon=float(a.accept_epsilon),
                min_confidence_gain=float(a.pdo_coarse_min_confidence_gain),
            )
            if verify_idx is not None:
                idx = int(verify_idx)
                if idx == 0:
                    reason = "coarse_pdo_noop_confident"
                    break
                seq = seqs[idx-1]
                entry = self.archive_by_sequence.get(seq)
                if entry is None:
                    if queried_fresh >= cap:
                        reason = "coarse_pdo_query_cap_before_confidence_verification"
                        break
                    entry, fresh = self._record_coarse_exact(
                        candidate_index=idx, tokens=slate[idx-1], z=z[idx-1], seq=seq,
                        chain=chain_labels[idx-1], preference_id=preference_id, lineage=lineage,
                        slate_id=slate_id, cycle=cycle, query_order=query_attempts+1,
                        source="pdo_coarse_confidence_verification", mode="verification",
                        predicted_utility=float(pred_u[idx-1]), gain_lower=float(decision.gain_lower[idx]),
                        gain_upper=float(decision.gain_upper[idx]), regret_upper=float(decision.regret_upper[idx]),
                    )
                    if entry is None:
                        reason = "global_budget_exhausted_before_coarse_confidence_verification"; break
                    query_attempts += 1; queried_fresh += int(fresh)
                    if idx-1 in top_records:
                        eu = float(base._utility(entry.scores,pref,a.rho)); top_records[idx-1].update({
                            "queried":1,"exact_utility":eu,"exact_gain":eu-old_utility
                        })
                accepted = self._accept_coarse_entry(
                    lineage, candidate_tokens=slate[idx-1], entry=entry, slate_id=slate_id, candidate_sequence=seq
                )
                if accepted:
                    accepted_idx = idx; reason = "coarse_pdo_empirical_confidence_accepted"; break
                recovery.observe_verification(
                    float(base._utility(entry.scores,pref,a.rho)) - old_utility,
                    float(a.accept_epsilon),
                )
                self._coarse_pdo_failed_confidence_verifications += 1
                reason = "coarse_pdo_confidence_failed_recovery_required"
                continue

            if decision.best_certified_index is not None:
                reason = "coarse_pdo_epsilon_stop_without_robust_positive_gain"
                break
            if queried_fresh >= cap:
                reason = "coarse_pdo_query_cap_reached"
                break
            if cap <= 0:
                reason = "coarse_pdo_zero_query_cap"
                break

            observed = set(int(i) for i in exact_map)
            idx, info_score, acquisition_mode, target_size = decision_relevant_probe_choice(
                qcoord,
                decision,
                observed_indices=observed,
                no_op_index=0,
                ridge_lambda=float(a.pdo_coarse_information_lambda),
                epsilon_dec=float(a.pdo_coarse_epsilon_dec),
                plausible_cap=int(a.pdo_coarse_acquisition_plausible_cap),
                min_information_score=float(a.pdo_coarse_min_information_score),
            )
            if idx is None or int(idx) == 0:
                reason = "coarse_pdo_no_unobserved_decision_relevant_candidate"
                break
            idx = int(idx); seq = seqs[idx-1]
            was_recovery = bool(recovery.pending)
            entry, fresh = self._record_coarse_exact(
                candidate_index=idx, tokens=slate[idx-1], z=z[idx-1], seq=seq,
                chain=chain_labels[idx-1], preference_id=preference_id, lineage=lineage,
                slate_id=slate_id, cycle=cycle, query_order=query_attempts+1,
                source="pdo_coarse_acquisition", mode="acquisition",
                predicted_utility=float(pred_u[idx-1]), gain_lower=float(decision.gain_lower[idx]),
                gain_upper=float(decision.gain_upper[idx]), regret_upper=float(decision.regret_upper[idx]),
                info_score=float(info_score),
            )
            if entry is None:
                reason = "global_budget_exhausted_before_coarse_acquisition"; break
            query_attempts += 1; queried_fresh += int(fresh)
            if self.coarse_pdo_query_history:
                self.coarse_pdo_query_history[-1]["acquisition_mode"] = str(acquisition_mode)
                self.coarse_pdo_query_history[-1]["information_target_size"] = int(target_size)
                self.coarse_pdo_query_history[-1]["is_forced_recovery_probe"] = int(was_recovery)
            if idx-1 in top_records:
                eu = float(base._utility(entry.scores,pref,a.rho)); top_records[idx-1].update({
                    "queried":1,"exact_utility":eu,"exact_gain":eu-old_utility
                })
            if was_recovery:
                if not fresh:
                    raise RuntimeError("coarse PDO recovery probe unexpectedly reused a cached label")
                recovery.observe_recovery_probe(fresh=True)
                self._coarse_pdo_recovery_probes += 1
            reason = "coarse_pdo_acquired_decision_information"

        if accepted_idx is None:
            final_map = self._coarse_exact_map(seqs, lineage)
            fallback_idx, _ = best_exact_improving_index(
                final_map,
                incumbent_scores=old_scores,
                preference=pref,
                rho=float(a.rho),
                accept_epsilon=float(a.accept_epsilon),
            )
            if fallback_idx is not None and int(fallback_idx) != 0:
                idx = int(fallback_idx); seq = seqs[idx-1]
                entry = self.archive_by_sequence.get(seq)
                if entry is None:
                    raise RuntimeError("coarse PDO fallback selected candidate without exact archive entry")
                if self._accept_coarse_entry(
                    lineage, candidate_tokens=slate[idx-1], entry=entry, slate_id=slate_id, candidate_sequence=seq
                ):
                    accepted_idx = idx
                    reason = f"coarse_pdo_best_exact_fallback_after_{reason}"

        oracle_seconds = float(time.perf_counter() - t2)
        lineage.slates_attempted += 1
        if accepted_idx is not None and accepted_idx-1 in top_records:
            top_records[accepted_idx-1]["accepted"] = 1
        self.top_prediction_history.extend(top_records.values())
        accepted_candidate_idx = -1 if accepted_idx is None else int(accepted_idx-1)
        accepted_gain = float(lineage.utility - old_utility)
        branch_counts = np.asarray([b.allocated_candidates for b in branches], dtype=np.int64)
        self.slate_history.append({
            "slate_id": slate_id, "cycle": int(cycle), "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id), "incumbent_sequence_before": old_sequence,
            "incumbent_utility_before": old_utility, "slate_size": int(len(seqs)),
            "slate_per_chain": int(a.slate_per_chain), "verification_k": int(cap),
            "verification_queries": int(queried_fresh), "accepted": int(accepted_idx is not None),
            "accepted_rank": int(rank_of[accepted_candidate_idx]) if accepted_candidate_idx >= 0 else -1,
            "accepted_sequence": seqs[accepted_candidate_idx] if accepted_candidate_idx >= 0 else "",
            "accepted_chain": chain_labels[accepted_candidate_idx] if accepted_candidate_idx >= 0 else "",
            "accepted_gain": accepted_gain, "incumbent_utility_after": float(lineage.utility),
            "cumulative_gain_from_lineage_start": float(lineage.utility - lineage.initial_utility),
            "readout_training_labels": readout_train_size, "readout_mode": str(a.readout_mode),
            "best_predicted_utility": float(pred_u[int(order[0])]),
            "best_predicted_gain": float(pred_u[int(order[0])] - old_utility),
            "proposal_seconds": proposal_seconds, "ranking_seconds": ranking_seconds,
            "oracle_seconds": oracle_seconds,
            "slate_wall_seconds": float(proposal_seconds + ranking_seconds + oracle_seconds),
            "proposal_physical_draws": int(sum(int(m.get("cheap_draws",0)) for m in proposal_meta)),
            "proposal_candidates": int(len(seqs)),
            "proposal_draws_per_candidate": float(sum(int(m.get("cheap_draws",0)) for m in proposal_meta)/max(len(seqs),1)),
            "batched_root_generation": False,
            "branch_conditioned_generation": True,
            "koopman_A_C_used_for_branch_allocation": int(str(a.koopman_branch_allocation_mode)=="ac_opportunity"),
            "koopman_adaptive_fraction": float(a.koopman_adaptive_fraction),
            "branch_count": int(len(branches)),
            "min_branch_candidates": int(np.min(branch_counts)),
            "max_branch_candidates": int(np.max(branch_counts)),
            "coarse_pdo_used": True,
            "coarse_pdo_decision_rank": int(qrank),
            "coarse_pdo_stop_reason": reason,
            "residual_bank_used": False,
        })
        self._coarse_status[(int(preference_id), int(lineage.lineage_id))] = {
            "accepted": bool(accepted_idx is not None), "accepted_gain": accepted_gain,
            "best_predicted_gain": float(pred_u[int(order[0])] - old_utility), "slate_id": slate_id,
        }
        if int(a.save_every_slates) > 0 and slate_id % int(a.save_every_slates) == 0:
            self._save_progress()
        return {
            "slate_id": slate_id, "paid_queries": int(queried_fresh),
            "accepted": int(accepted_idx is not None), "gain": accepted_gain, "reason": reason,
        }

    def _run_coarse_static_slate(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        *,
        query_cap_override: int | None = None,
    ) -> dict[str, Any]:
        """Component ablation: same hierarchical proposal, legacy static top-K verification."""
        a = self.args
        if self.stop_requested or self._budget_remaining() == 0:
            if self._budget_remaining() == 0:
                self.stop_requested = True
            return {"paid_queries":0,"accepted":0,"gain":0.0,"reason":"budget_exhausted"}
        self.slate_counter += 1
        slate_id = int(self.slate_counter)
        readout = self._fit_current_readout()
        readout_train_size = int(len(self.archive))
        old_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1,-1))[0]
        old_scores = np.asarray(lineage.scores,dtype=np.float64).copy()
        old_utility = float(lineage.utility)
        t0=time.perf_counter()
        slate, chain_labels, proposal_meta, branches, incumbent_z = self._generate_koopman_conditioned_slate(
            lineage, slate_id=slate_id, readout=readout
        )
        proposal_seconds=float(time.perf_counter()-t0)
        t1=time.perf_counter()
        seqs=base.decode_esm_tokens(slate)
        z=terminal_z(self.model,slate,batch_size=int(a.feature_batch_size))
        pred_scores=base._predict_scores(readout,z,mode=str(a.readout_mode),incumbent_scores=old_scores,incumbent_z=incumbent_z)
        pref=self.preferences[int(preference_id)]
        pred_u=np.asarray(base._utility(pred_scores,pref,a.rho),dtype=np.float64).reshape(-1)
        order=np.argsort(-pred_u,kind="mergesort")
        rank_of=np.empty(len(order),dtype=np.int64); rank_of[order]=np.arange(1,len(order)+1)
        ranking_seconds=float(time.perf_counter()-t1)
        cap=int(a.verification_k)
        if query_cap_override is not None: cap=min(cap,int(query_cap_override))
        if int(a.pdo_coarse_query_cap_per_slate)>0: cap=min(cap,int(a.pdo_coarse_query_cap_per_slate))
        accepted_idx=-1; paid=0; t2=time.perf_counter()
        for qrank, idx_v in enumerate(order[:cap],start=1):
            if self._budget_remaining()==0:
                self.stop_requested=True; break
            idx=int(idx_v); seq=seqs[idx]
            before=int(self.oracle.unique_oracle_queries)
            rec=self.oracle.evaluate_one(seq)
            fresh=int(self.oracle.unique_oracle_queries)>before
            exact_u=float(base._utility(rec.scores,pref,a.rho)); gain=exact_u-old_utility
            entry=self._record_exact(
                slate[idx],z[idx],rec,source="coarse_static_ablation",preference_id=int(preference_id),
                lineage_id=int(lineage.lineage_id),slate_id=slate_id,verification_rank=qrank,utility_at_query=exact_u
            )
            if fresh:
                paid+=1; lineage.verification_queries+=1
            row={
                "query_index":int(self.oracle.unique_oracle_queries),"slate_id":slate_id,"cycle":int(cycle),
                "preference_id":int(preference_id),"lineage_id":int(lineage.lineage_id),
                "incumbent_sequence":old_sequence,"incumbent_utility":old_utility,
                "verification_rank":qrank,"candidate_index":idx,"candidate_sequence":seq,"chain":chain_labels[idx],
                "readout_training_labels":readout_train_size,"readout_mode":str(a.readout_mode),
                "predicted_utility":float(pred_u[idx]),"predicted_gain":float(pred_u[idx]-old_utility),
                "exact_utility":exact_u,"exact_gain":gain,"accepted":0,
                "strong_gain_hit":int(gain>=float(a.strong_gain)),"strong_gain_threshold_is_diagnostic_only":True,
                "coarse_pdo":False,"coarse_static_topk_ablation":True,
            }
            for j,n in enumerate(OPT_NAMES): row[f"pred_score_{n}"]=float(pred_scores[idx,j]); row[f"exact_score_{n}"]=float(rec.scores[j])
            for j,n in enumerate(RAW_NAMES): row[f"raw_{n}"]=float(rec.raw[j])
            self.query_history.append(row); self._record_preference_query_use(preference_id,seq)
            if gain>float(a.accept_epsilon):
                lineage.tokens=slate[idx].detach().cpu().clone(); lineage.raw=np.asarray(entry.raw,dtype=np.float64).copy()
                lineage.scores=np.asarray(entry.scores,dtype=np.float64).copy(); lineage.utility=exact_u
                lineage.accepted_moves+=1; accepted_idx=idx; self.query_history[-1]["accepted"]=1; break
        oracle_seconds=float(time.perf_counter()-t2)
        lineage.slates_attempted+=1
        accepted_gain=float(lineage.utility-old_utility)
        bcounts=np.asarray([b.allocated_candidates for b in branches],dtype=np.int64)
        self.slate_history.append({
            "slate_id":slate_id,"cycle":int(cycle),"preference_id":int(preference_id),"lineage_id":int(lineage.lineage_id),
            "incumbent_sequence_before":old_sequence,"incumbent_utility_before":old_utility,
            "slate_size":len(seqs),"slate_per_chain":int(a.slate_per_chain),"verification_k":cap,
            "verification_queries":paid,"accepted":int(accepted_idx>=0),
            "accepted_rank":int(rank_of[accepted_idx]) if accepted_idx>=0 else -1,
            "accepted_sequence":seqs[accepted_idx] if accepted_idx>=0 else "",
            "accepted_chain":chain_labels[accepted_idx] if accepted_idx>=0 else "",
            "accepted_gain":accepted_gain,"incumbent_utility_after":float(lineage.utility),
            "cumulative_gain_from_lineage_start":float(lineage.utility-lineage.initial_utility),
            "readout_training_labels":readout_train_size,"readout_mode":str(a.readout_mode),
            "best_predicted_utility":float(pred_u[int(order[0])]),"best_predicted_gain":float(pred_u[int(order[0])]-old_utility),
            "proposal_seconds":proposal_seconds,"ranking_seconds":ranking_seconds,"oracle_seconds":oracle_seconds,
            "slate_wall_seconds":proposal_seconds+ranking_seconds+oracle_seconds,
            "proposal_physical_draws":int(sum(int(m.get("cheap_draws",0)) for m in proposal_meta)),
            "proposal_candidates":len(seqs),"proposal_draws_per_candidate":float(sum(int(m.get("cheap_draws",0)) for m in proposal_meta)/max(len(seqs),1)),
            "batched_root_generation":False,"branch_conditioned_generation":True,
            "koopman_A_C_used_for_branch_allocation":int(str(a.koopman_branch_allocation_mode)=="ac_opportunity"),
            "koopman_adaptive_fraction":float(a.koopman_adaptive_fraction),"branch_count":len(branches),
            "min_branch_candidates":int(np.min(bcounts)),"max_branch_candidates":int(np.max(bcounts)),
            "coarse_pdo_used":False,"coarse_pdo_stop_reason":"static_topk_ablation","residual_bank_used":False,
        })
        self._coarse_status[(int(preference_id),int(lineage.lineage_id))]={
            "accepted":bool(accepted_idx>=0),"accepted_gain":accepted_gain,
            "best_predicted_gain":float(pred_u[int(order[0])]-old_utility),"slate_id":slate_id,
        }
        return {"paid_queries":paid,"accepted":int(accepted_idx>=0),"gain":accepted_gain,"reason":"static_topk_ablation"}

    def _run_unified_coarse(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        *,
        query_cap_override: int | None = None,
    ) -> dict[str, Any]:
        if str(self.args.pdo_coarse_verification_mode) == "pdo":
            return self._run_coarse_pdo_slate(
                preference_id, lineage, cycle, query_cap_override=query_cap_override
            )
        if str(self.args.pdo_coarse_verification_mode) == "static_topk":
            return self._run_coarse_static_slate(
                preference_id, lineage, cycle, query_cap_override=query_cap_override
            )
        raise ValueError(f"unknown coarse verification mode {self.args.pdo_coarse_verification_mode!r}")

    def optimize(self) -> None:
        """Coarse first, then fine: same-turn exact coarse labels warm-start fine PDO."""
        a = self.args
        total = len(self.preferences) * int(a.lineages) * int(a.slates_per_lineage)
        bar = None
        if tqdm is not None and not bool(a.no_progress):
            bar = tqdm(total=total, desc="Unified PEGASUS v2", unit="turn", dynamic_ncols=True)
        try:
            for cycle in range(int(a.slates_per_lineage)):
                for pref_id in range(len(self.preferences)):
                    for lineage in self.lineages[pref_id]:
                        if self.stop_requested:
                            return
                        before_q = int(self.oracle.unique_oracle_queries); before_u = float(lineage.utility)
                        mode = str(a.pdo_budget_mode)
                        if mode == "protected_baseline":
                            self._run_unified_coarse(pref_id, lineage, cycle)
                            if self.stop_requested:
                                return
                            # Fine labels are extra in protected mode and can immediately reuse
                            # all exact coarse vectors paid earlier in this same turn.
                            self._run_pdo_turn(pref_id, lineage, cycle)
                        elif mode == "matched_budget":
                            total_cap = int(a.verification_k)
                            coarse = self._run_unified_coarse(
                                pref_id, lineage, cycle, query_cap_override=total_cap
                            )
                            if self.stop_requested:
                                return
                            remaining = max(0, total_cap - int(coarse.get("paid_queries",0)))
                            if remaining > 0:
                                old_k = int(a.verification_k)
                                try:
                                    a.verification_k = int(remaining)
                                    self._run_pdo_turn(pref_id, lineage, cycle)
                                finally:
                                    a.verification_k = old_k
                        else:
                            raise ValueError(f"unknown PDO budget mode {mode!r}")
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix(
                                unique_q=self.oracle.unique_oracle_queries,
                                last_gain=f"{lineage.utility-before_u:+.4f}",
                                dq=self.oracle.unique_oracle_queries-before_q,
                            )
        finally:
            if bar is not None:
                bar.close()

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        unified = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "schedule": "coarse_then_fine_same_turn_cross_scale_reuse",
            "roles": {
                "LKF": "physical stochastic branching, terminal generation, fine H1 reachability",
                "Koopman_A_C": "conditional future observability for branch continuation allocation only",
                "coarse_PDO": "adaptive terminal-slate decision observability and exact-query selection",
                "fine_PDO": "Hamming-1 decision observability",
                "exact_oracle": "sole acceptance authority",
            },
            "koopman_branch_allocation": {
                "mode": str(self.args.koopman_branch_allocation_mode),
                "pilot_starts_per_chain": int(self.args.koopman_pilot_starts_per_chain),
                "adaptive_fraction": float(self.args.koopman_adaptive_fraction),
                "protected_uniform_fraction": float(1.0-float(self.args.koopman_adaptive_fraction))
                    if str(self.args.koopman_branch_allocation_mode)=="ac_opportunity" else 1.0,
                "minimum_candidates_per_realized_branch": int(self.args.koopman_min_candidates_per_branch),
                "hard_pruning": False,
                "residual_bank_or_residual_control": False,
            },
            "coarse_pdo": {
                "mode": str(self.args.pdo_coarse_verification_mode),
                "epsilon_dec": float(self.args.pdo_coarse_epsilon_dec),
                "fresh_query_cap_per_slate": int(self.args.pdo_coarse_query_cap_per_slate),
                "fresh_queries": int(self._coarse_pdo_fresh_queries),
                "accepted_moves": int(self._coarse_pdo_accepted_moves),
                "failed_empirical_confidence_verifications": int(self._coarse_pdo_failed_confidence_verifications),
                "forced_recovery_probes": int(self._coarse_pdo_recovery_probes),
                "candidate_geometry": "realized terminal KFM displacement span + exact no-op",
                "prior": "causal global KFM readout frozen within each slate",
                "same_turn_labels_reused_by_fine_PDO": True,
                "formal_certificate_claimed": False,
            },
            "query_accounting": {
                "actual_global_unique_queries": int(self.oracle.unique_oracle_queries),
                "coarse_pdo_fresh_queries": int(self._coarse_pdo_fresh_queries),
                "fine_pdo_fresh_queries": int(self._pdo_fresh_queries),
                "protected_mode_fine_queries_are_extra": bool(str(self.args.pdo_budget_mode)=="protected_baseline"),
            },
        }
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["proposal_law"] = (
            "realized_stochastic_chain_starts_then_A_C_safe_breadth_allocation_then_"
            "physical_LKF_continuations"
        )
        summary["core_algorithm"] = [
            "physical stochastic intermediate starts from frozen LKF",
            "frozen Koopman A_C conditional terminal-observable look-ahead",
            "75%-protected / 25%-adaptive model-side continuation allocation by default",
            "physical terminal continuation generation from realized starts",
            "coarse PDO over exact no-op + realized terminal candidate slate",
            "adaptive sparse exact vector verification with recovery-safe fallback",
            "same-turn reuse of every coarse exact vector by fine PDO",
            "fine Hamming-1 PDO in frozen LKF intervention geometry",
            "exact monotone acceptance at both scales",
        ]
        summary.setdefault("readout", {})["within_slate_global_readout_refit_after_query"] = False
        summary["readout"]["coarse_pdo_local_exact_corrections"] = True
        if isinstance(summary.get("pdo"), dict) and isinstance(summary["pdo"].get("budget_accounting"), dict):
            summary["pdo"]["budget_accounting"]["matched_budget_semantics"] = (
                "coarse PDO runs first and consumes at most verification_k fresh slots; "
                "fine PDO receives only the unused slots. Protected mode keeps the ordinary "
                "coarse quota and treats fine PDO queries as sparse extras."
            )
        summary.setdefault("search", {})["coarse_verification_mode"] = str(self.args.pdo_coarse_verification_mode)
        summary["search"]["koopman_pilot_starts_per_chain"] = int(self.args.koopman_pilot_starts_per_chain)
        summary["search"]["koopman_adaptive_fraction"] = float(self.args.koopman_adaptive_fraction)
        summary["unified_pdo_pegasus_v2"] = unified
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] coarse->fine schedule; exact coarse labels are reusable by fine PDO immediately\n"
            f"[{METHOD_NAME}] LKF=physical reachability; A_C=future observability; PDO=decision observability; oracle=truth\n"
            f"[{METHOD_NAME}] residual banks/control disabled by design; no hard Koopman pruning",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = v1.build_parser()
    p.description = __doc__
    p.set_defaults(
        pdo_budget_mode="protected_baseline",
        pdo_mode_policy="always_pdo",
    )
    kg = p.add_argument_group("Unified Koopman branch-conditioned coarse generation")
    kg.add_argument(
        "--koopman-branch-allocation-mode",
        choices=("ac_opportunity", "uniform"),
        default="ac_opportunity",
        help="ac_opportunity uses frozen A_C opportunity to reallocate only the adaptive continuation fraction.",
    )
    kg.add_argument("--koopman-pilot-starts-per-chain", type=int, default=6)
    kg.add_argument("--koopman-adaptive-fraction", type=float, default=0.25)
    kg.add_argument("--koopman-min-candidates-per-branch", type=int, default=1)

    cg = p.add_argument_group("Coarse Pareto Decision Observability")
    cg.add_argument(
        "--pdo-coarse-verification-mode", choices=("pdo", "static_topk"), default="pdo"
    )
    cg.add_argument("--pdo-coarse-epsilon-dec", type=float, default=0.002)
    cg.add_argument("--pdo-coarse-min-confidence-gain", type=float, default=0.0)
    cg.add_argument(
        "--pdo-coarse-query-cap-per-slate", type=int, default=-1,
        help="Negative means inherit --verification-k (recommended/default).",
    )
    cg.add_argument("--pdo-coarse-min-probes-before-confidence", type=int, default=2)
    cg.add_argument("--pdo-coarse-acquisition-plausible-cap", type=int, default=32)
    cg.add_argument("--pdo-coarse-min-information-score", type=float, default=1e-12)
    cg.add_argument("--pdo-coarse-information-lambda", type=float, default=1.0)
    cg.add_argument("--pdo-coarse-local-ridge-alphas", default="0.03,0.1,0.3,1,3")
    cg.add_argument("--pdo-coarse-decision-span-svd-rtol", type=float, default=1e-8)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    v1._validate_args(args)
    if int(args.koopman_pilot_starts_per_chain) <= 0:
        raise ValueError("--koopman-pilot-starts-per-chain must be positive")
    if not (0.0 <= float(args.koopman_adaptive_fraction) <= 1.0):
        raise ValueError("--koopman-adaptive-fraction must lie in [0,1]")
    if int(args.koopman_min_candidates_per_branch) < 0:
        raise ValueError("--koopman-min-candidates-per-branch cannot be negative")
    nbranches = int(args.koopman_pilot_starts_per_chain) * len(base.parse_chains(args.chains))
    budget = int(args.slate_per_chain) * len(base.parse_chains(args.chains))
    if budget < nbranches * int(args.koopman_min_candidates_per_branch):
        raise ValueError("slate budget is too small for the protected realized-branch floor")
    for name in (
        "pdo_coarse_epsilon_dec", "pdo_coarse_min_confidence_gain",
        "pdo_coarse_min_information_score", "pdo_coarse_information_lambda",
        "pdo_coarse_decision_span_svd_rtol",
    ):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_','-')} must be finite")
    if float(args.pdo_coarse_epsilon_dec) < 0 or float(args.pdo_coarse_min_information_score) < 0:
        raise ValueError("coarse PDO epsilon/information floor must be nonnegative")
    if float(args.pdo_coarse_information_lambda) <= 0 or float(args.pdo_coarse_decision_span_svd_rtol) <= 0:
        raise ValueError("coarse PDO information lambda/SVD rtol must be positive")
    if int(args.pdo_coarse_query_cap_per_slate) == 0:
        raise ValueError("--pdo-coarse-query-cap-per-slate cannot be zero; use -1 to inherit verification-k")
    if int(args.pdo_coarse_query_cap_per_slate) > int(args.verification_k):
        raise ValueError("--pdo-coarse-query-cap-per-slate cannot exceed --verification-k")
    if int(args.pdo_coarse_min_probes_before_confidence) < 1:
        raise ValueError("--pdo-coarse-min-probes-before-confidence must be at least 1")
    if int(args.pdo_coarse_acquisition_plausible_cap) == 0:
        raise ValueError("--pdo-coarse-acquisition-plausible-cap cannot be zero")
    alphas = v1._parse_float_tuple(args.pdo_coarse_local_ridge_alphas)
    if any((not math.isfinite(float(x))) or float(x) <= 0 for x in alphas):
        raise ValueError("--pdo-coarse-local-ridge-alphas must all be finite and positive")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    UnifiedPDOPegasusRunner(args).run()


if __name__ == "__main__":
    main()
