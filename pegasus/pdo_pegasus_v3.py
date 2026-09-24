"""Unified PEGASUS v3: Koopman-guided branching + exploit-first coarse PDO + fine PDO.

v3 changes ONLY the coarse terminal-slate verification layer relative to v2.

Frozen v2 components
--------------------
* LKF remains the sole physical generator and fine H1 reachability model.
* Frozen Koopman A_C is used only for conditional future-observability and safe
  75%-protected / 25%-adaptive continuation breadth allocation by default.
* Exact six-objective oracle vectors remain the only acceptance authority.
* Fine PDO remains the recovery-safe Hamming-1 mechanism from PEGASUS v1.3.

v3 coarse decision principle
----------------------------
The coarse slate is a heterogeneous collection of terminal sequences, not a local H1
intervention family.  Therefore v3 does NOT use fine-style local action-response
interpolation or information-first acquisition at coarse scale.

Instead:
1. Freeze a small contender pool from the pre-query global-readout ranking.
2. Query the top predicted endpoint first (pure exploitation).
3. Add every exact vector to the shared global task archive immediately.
4. Refit a coherent committee of complete multi-objective *global* KFM readouts.
5. Continue only while an unqueried contender can still plausibly beat the best exact
   decision by epsilon_dec; among plausible challengers, query the one with the highest
   updated median utility (not the most informative one).
6. At early-stop or the query cap, accept only the best exactly evaluated improving
   candidate; otherwise no-op.

This makes static top-K the limiting behavior when exact labels do not materially change
ranking, while preserving adaptive reranking, early stopping, exact-vector reuse, and the
coarse->fine cross-scale information pathway.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_v1 as v1
from . import pdo_pegasus_v2 as v2
from .kfocus.objectives import OPT_NAMES
from .kfocus.regions import terminal_z
from .pdo_math import augmented_tchebycheff_utility
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-v3-exploit-first-global-coarse-pdo"
METHOD_NAME = "Unified PEGASUS v3"


@dataclass(frozen=True)
class GlobalCoarseCommittee:
    """Joint committee of complete multi-objective global task readouts."""

    member_scores: np.ndarray          # [C, A, m], no-op inclusive
    member_utilities: np.ndarray       # [C, A]
    member_labels: tuple[str, ...]
    median_scores: np.ndarray          # [A, m]
    median_utilities: np.ndarray       # [A]
    utility_lower: np.ndarray          # [A]
    utility_upper: np.ndarray          # [A]


def _validate_matrix(name: str, x: np.ndarray, *, rows: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D matrix")
    if rows is not None and arr.shape[0] != int(rows):
        raise ValueError(f"{name} row count mismatch")
    return arr


def _global_readout_members(
    train_z: np.ndarray,
    train_scores: np.ndarray,
    *,
    alpha: float,
    jackknife_folds: int,
    min_train_labels: int,
) -> tuple[list[base.RidgeReadout], tuple[str, ...]]:
    """Fit a deterministic global-readout committee with one fixed regularization.

    The full-data model is always present.  Additional members are leave-one-block-out
    fits using deterministic interleaved blocks.  This is an empirical version set, not
    a formal posterior/confidence set.  Keeping alpha fixed avoids conflating task-model
    uncertainty with hyperparameter retuning inside a slate.
    """
    z = _validate_matrix("train_z", train_z)
    y = _validate_matrix("train_scores", train_scores, rows=len(z))
    aa = float(alpha)
    if not math.isfinite(aa) or aa <= 0:
        raise ValueError("alpha must be finite and positive")
    folds = int(jackknife_folds)
    minimum = int(min_train_labels)
    if folds < 1:
        raise ValueError("jackknife_folds must be at least 1")
    if minimum < 2:
        raise ValueError("min_train_labels must be at least 2")
    if len(z) < minimum:
        raise ValueError("not enough labels to fit the global coarse readout committee")

    models: list[base.RidgeReadout] = [base._fit_ridge(z, y, aa)]
    labels: list[str] = ["global_full"]
    if folds >= 2 and len(z) >= minimum + folds:
        ids = np.arange(len(z), dtype=np.int64)
        for f in range(folds):
            keep = (ids % folds) != f
            if int(np.sum(keep)) < minimum:
                continue
            models.append(base._fit_ridge(z[keep], y[keep], aa))
            labels.append(f"global_leave_block_{f}_of_{folds}")
    return models, tuple(labels)


def build_global_coarse_committee(
    train_z: np.ndarray,
    train_scores: np.ndarray,
    candidate_z: np.ndarray,
    *,
    incumbent_z: np.ndarray,
    incumbent_scores: np.ndarray,
    exact_scores_by_action: Mapping[int, np.ndarray],
    preference: np.ndarray,
    rho: float,
    readout_mode: str,
    alpha: float,
    jackknife_folds: int = 4,
    min_train_labels: int = 8,
) -> GlobalCoarseCommittee:
    """Predict a no-op-inclusive slate with a joint global task-readout committee.

    ``exact_scores_by_action`` uses no-op-inclusive indices: index 0 is the incumbent,
    and candidate j occupies index j+1.  Every exact action is clamped to its exact full
    objective vector in *every* committee member, preserving joint vector consistency.
    """
    ztr = _validate_matrix("train_z", train_z)
    ytr = _validate_matrix("train_scores", train_scores, rows=len(ztr))
    zc = _validate_matrix("candidate_z", candidate_z)
    z0 = np.asarray(incumbent_z, dtype=np.float64).reshape(-1)
    s0 = np.asarray(incumbent_scores, dtype=np.float64).reshape(-1)
    if zc.shape[1] != len(z0) or ytr.shape[1] != len(s0):
        raise ValueError("global committee feature/objective dimension mismatch")
    if 0 not in exact_scores_by_action:
        raise ValueError("exact_scores_by_action must contain the no-op/incumbent at index 0")

    models, labels = _global_readout_members(
        ztr,
        ytr,
        alpha=float(alpha),
        jackknife_folds=int(jackknife_folds),
        min_train_labels=int(min_train_labels),
    )
    members: list[np.ndarray] = []
    for model in models:
        pred_c = base._predict_scores(
            model,
            zc,
            mode=str(readout_mode),
            incumbent_scores=s0,
            incumbent_z=z0,
        )
        member = np.concatenate([s0.reshape(1, -1), pred_c], axis=0)
        for idx_raw, score_raw in exact_scores_by_action.items():
            idx = int(idx_raw)
            if idx < 0 or idx >= len(member):
                raise ValueError("exact action index out of range")
            score = np.asarray(score_raw, dtype=np.float64).reshape(-1)
            if len(score) != member.shape[1] or np.any(~np.isfinite(score)):
                raise ValueError("invalid exact score vector")
            member[idx] = score
        members.append(member)
    member_scores = np.stack(members, axis=0)
    flat_u = augmented_tchebycheff_utility(
        member_scores.reshape(-1, member_scores.shape[-1]),
        np.asarray(preference, dtype=np.float64),
        rho=float(rho),
        reference=np.ones(member_scores.shape[-1], dtype=np.float64),
    )
    member_u = np.asarray(flat_u, dtype=np.float64).reshape(member_scores.shape[0], member_scores.shape[1])
    return GlobalCoarseCommittee(
        member_scores=member_scores,
        member_utilities=member_u,
        member_labels=labels,
        median_scores=np.median(member_scores, axis=0),
        median_utilities=np.median(member_u, axis=0),
        utility_lower=np.min(member_u, axis=0),
        utility_upper=np.max(member_u, axis=0),
    )


def freeze_initial_contender_indices(predicted_utilities: np.ndarray, cap: int) -> np.ndarray:
    """Freeze the pre-query decision-relevant coarse contender set.

    Returned indices are no-op-inclusive candidate indices (1..N).  No-op is not part of
    the returned array because it is always exactly known separately.
    """
    u = np.asarray(predicted_utilities, dtype=np.float64).reshape(-1)
    if len(u) == 0 or np.any(~np.isfinite(u)):
        raise ValueError("predicted_utilities must be a nonempty finite vector")
    cc = int(cap)
    if cc <= 0:
        raise ValueError("contender cap must be positive")
    order = np.argsort(-u, kind="mergesort")[: min(cc, len(u))]
    return order.astype(np.int64) + 1


def best_exact_decision_index(
    exact_scores_by_action: Mapping[int, np.ndarray],
    *,
    preference: np.ndarray,
    rho: float,
) -> tuple[int, float]:
    if not exact_scores_by_action:
        raise ValueError("exact_scores_by_action cannot be empty")
    inds = sorted(int(i) for i in exact_scores_by_action)
    scores = np.stack([np.asarray(exact_scores_by_action[i], dtype=np.float64) for i in inds], axis=0)
    u = np.asarray(base._utility(scores, np.asarray(preference, dtype=np.float64), float(rho)), dtype=np.float64)
    j = int(np.argmax(u))
    return int(inds[j]), float(u[j])


def plausible_challenger_indices(
    committee: GlobalCoarseCommittee,
    *,
    contender_indices: Sequence[int],
    observed_indices: Sequence[int],
    best_exact_utility: float,
    epsilon_dec: float,
) -> np.ndarray:
    """Unqueried contenders that some coherent global readout says can change decision."""
    eps = float(epsilon_dec)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("epsilon_dec must be finite and nonnegative")
    obs = set(int(i) for i in observed_indices)
    cand = sorted(set(int(i) for i in contender_indices))
    n = committee.member_scores.shape[1]
    if any(i <= 0 or i >= n for i in cand):
        raise ValueError("contender index out of range")
    out = [
        i for i in cand
        if i not in obs and float(committee.utility_upper[i]) > float(best_exact_utility) + eps + 1e-12
    ]
    return np.asarray(out, dtype=np.int64)


def choose_exploit_first_challenger(
    committee: GlobalCoarseCommittee,
    plausible_indices: Sequence[int],
    *,
    initial_rank: np.ndarray | None = None,
) -> int | None:
    """Choose highest updated median-value plausible challenger; never by information."""
    inds = np.asarray(list(plausible_indices), dtype=np.int64)
    if len(inds) == 0:
        return None
    med = committee.median_utilities[inds]
    upper = committee.utility_upper[inds]
    if initial_rank is None:
        rank = inds.astype(np.float64)
    else:
        rr = np.asarray(initial_rank)
        rank = rr[inds - 1].astype(np.float64)
    # np.lexsort uses the last key as primary: maximize median, then upper; minimize rank.
    order = np.lexsort((rank, -upper, -med))
    return int(inds[int(order[0])])


class UnifiedPDOPegasusV3Runner(v2.UnifiedPDOPegasusRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.coarse_v3_decision_history: list[dict[str, Any]] = []
        self.coarse_v3_query_history: list[dict[str, Any]] = []
        self._coarse_v3_early_stops = 0
        self._coarse_v3_reranked_second_queries = 0
        self._coarse_v3_slates = 0

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.coarse_v3_decision_history:
            write_csv(self.out / "coarse_v3_decision_history.csv", self.coarse_v3_decision_history)
        if self.coarse_v3_query_history:
            write_csv(self.out / "coarse_v3_query_history.csv", self.coarse_v3_query_history)

    def _archive_arrays_for_global_committee(self) -> tuple[np.ndarray, np.ndarray]:
        return self._archive_arrays()

    def _build_v3_committee(
        self,
        *,
        candidate_z: np.ndarray,
        incumbent_z: np.ndarray,
        incumbent_scores: np.ndarray,
        exact_map: Mapping[int, np.ndarray],
        preference: np.ndarray,
    ) -> GlobalCoarseCommittee:
        ztr, ytr = self._archive_arrays_for_global_committee()
        return build_global_coarse_committee(
            ztr,
            ytr,
            candidate_z,
            incumbent_z=incumbent_z,
            incumbent_scores=incumbent_scores,
            exact_scores_by_action=exact_map,
            preference=preference,
            rho=float(self.args.rho),
            readout_mode=str(self.args.readout_mode),
            alpha=float(self.readout_alpha),
            jackknife_folds=int(self.args.pdo_coarse_v3_global_jackknife_folds),
            min_train_labels=int(self.args.pdo_coarse_v3_min_global_train_labels),
        )

    def _run_coarse_exploit_first_slate(
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
        self._coarse_v3_slates += 1
        slate_id = int(self.slate_counter)
        readout = self._fit_current_readout()
        readout_train_size = int(len(self.archive))
        self.readout_history.append({
            "event": "fit_before_v3_coarse_slate",
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
            readout,
            z,
            mode=str(a.readout_mode),
            incumbent_scores=old_scores,
            incumbent_z=incumbent_z,
        )
        pref = self.preferences[int(preference_id)]
        pred_u = np.asarray(base._utility(pred_scores, pref, a.rho), dtype=np.float64).reshape(-1)
        order = np.argsort(-pred_u, kind="mergesort")
        rank_of = np.empty(len(order), dtype=np.int64)
        rank_of[order] = np.arange(1, len(order) + 1)
        contender_indices = freeze_initial_contender_indices(
            pred_u, int(a.pdo_coarse_v3_contender_cap)
        )
        contender_set = set(int(i) for i in contender_indices.tolist())
        ranking_seconds = float(time.perf_counter() - t1)

        log_top = min(int(a.log_top_predictions), len(order))
        top_records: dict[int, dict[str, Any]] = {}
        for rank, idx_v in enumerate(order[:log_top], start=1):
            idx = int(idx_v)
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
                "predicted_gain_from_exact_incumbent": float(pred_u[idx] - old_utility),
                "queried": 0,
                "accepted": 0,
                "exact_utility": "",
                "exact_gain": "",
                "coarse_pdo": True,
                "coarse_pdo_v3_exploit_first": True,
                "in_frozen_contender_pool": int((idx + 1) in contender_set),
            }
            for j, name in enumerate(OPT_NAMES):
                row[f"pred_score_{name}"] = float(pred_scores[idx, j])
            top_records[idx] = row

        cap = int(a.verification_k) if int(a.pdo_coarse_query_cap_per_slate) < 0 else int(a.pdo_coarse_query_cap_per_slate)
        if query_cap_override is not None:
            cap = min(cap, int(query_cap_override))
        cap = max(int(cap), 0)
        queried_fresh = 0
        query_attempts = 0
        accepted_idx: int | None = None  # no-op-inclusive
        reason = "coarse_v3_no_query"
        initial_second_idx = int(order[1]) + 1 if len(order) > 1 else None
        t2 = time.perf_counter()

        while not self.stop_requested:
            exact_map = self._coarse_exact_map(seqs, lineage)
            committee = self._build_v3_committee(
                candidate_z=z,
                incumbent_z=incumbent_z,
                incumbent_scores=old_scores,
                exact_map=exact_map,
                preference=pref,
            )
            best_exact_idx, best_exact_u = best_exact_decision_index(
                exact_map, preference=pref, rho=float(a.rho)
            )
            observed = sorted(int(i) for i in exact_map)
            plausible = plausible_challenger_indices(
                committee,
                contender_indices=contender_indices,
                observed_indices=observed,
                best_exact_utility=float(best_exact_u),
                epsilon_dec=float(a.pdo_coarse_epsilon_dec),
            )

            outside = [
                i for i in range(1, len(seqs) + 1)
                if i not in contender_set
                and i not in set(observed)
                and float(committee.utility_upper[i]) > float(best_exact_u) + float(a.pdo_coarse_epsilon_dec) + 1e-12
            ]
            top_outside_upper = max((float(committee.utility_upper[i]) for i in outside), default=float("-inf"))
            self.coarse_v3_decision_history.append({
                "slate_id": slate_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence": old_sequence,
                "candidate_count_including_noop": int(len(seqs) + 1),
                "frozen_contender_cap": int(a.pdo_coarse_v3_contender_cap),
                "frozen_contender_count": int(len(contender_indices)),
                "global_committee_members": int(committee.member_scores.shape[0]),
                "global_committee_labels": ";".join(committee.member_labels),
                "exact_nonnoop_candidates": int(sum(int(i) != 0 for i in exact_map)),
                "fresh_queries_used": int(queried_fresh),
                "fresh_query_cap": int(cap),
                "best_exact_index_noop_inclusive": int(best_exact_idx),
                "best_exact_utility": float(best_exact_u),
                "best_exact_gain": float(best_exact_u - old_utility),
                "plausible_challenger_count": int(len(plausible)),
                "plausible_challenger_indices": ";".join(str(int(i)) for i in plausible.tolist()),
                "outside_pool_shadow_challenger_count": int(len(outside)),
                "outside_pool_shadow_best_upper_utility": "" if not outside else float(top_outside_upper),
                "epsilon_dec": float(a.pdo_coarse_epsilon_dec),
                "claim_level": "empirical_global_readout_committee",
                "information_score_used_for_acquisition": False,
            })

            # Query 1 is pure exploitation: initial production top-1, irrespective of
            # committee uncertainty.  Thereafter only plausible contenders can be queried.
            if query_attempts == 0 and queried_fresh == 0:
                if cap <= 0:
                    reason = "coarse_v3_zero_query_cap"
                    break
                next_idx = int(order[0]) + 1
                query_mode = "exploit_first_top1"
            else:
                if len(plausible) == 0:
                    self._coarse_v3_early_stops += 1
                    reason = "coarse_v3_no_plausible_challenger"
                    break
                if queried_fresh >= cap:
                    reason = "coarse_v3_query_cap_reached"
                    break
                next_idx = choose_exploit_first_challenger(
                    committee,
                    plausible,
                    initial_rank=rank_of,
                )
                if next_idx is None:
                    reason = "coarse_v3_no_plausible_challenger"
                    break
                query_mode = "updated_best_plausible_challenger"

            if queried_fresh >= cap:
                reason = "coarse_v3_query_cap_reached"
                break
            idx = int(next_idx)
            if idx <= 0 or idx > len(seqs):
                raise RuntimeError("v3 coarse query selected invalid candidate index")
            if idx in exact_map:
                # This should be rare because generation excludes archived sequences.  It
                # is still handled safely without consuming a paid-query slot.
                query_attempts += 1
                reason = "coarse_v3_cached_candidate_reused"
                continue

            seq = seqs[idx - 1]
            lower_gain = float(committee.utility_lower[idx] - old_utility)
            upper_gain = float(committee.utility_upper[idx] - old_utility)
            regret_upper = max(0.0, float(np.max(committee.utility_upper[list(contender_indices)]) - committee.utility_lower[idx]))
            entry, fresh = self._record_coarse_exact(
                candidate_index=idx,
                tokens=slate[idx - 1],
                z=z[idx - 1],
                seq=seq,
                chain=chain_labels[idx - 1],
                preference_id=preference_id,
                lineage=lineage,
                slate_id=slate_id,
                cycle=cycle,
                query_order=query_attempts + 1,
                source=(
                    "pdo_coarse_v3_exploit_first"
                    if query_mode == "exploit_first_top1"
                    else "pdo_coarse_v3_challenger"
                ),
                mode=query_mode,
                predicted_utility=float(committee.median_utilities[idx]),
                gain_lower=lower_gain,
                gain_upper=upper_gain,
                regret_upper=regret_upper,
                info_score=None,
            )
            if entry is None:
                reason = "global_budget_exhausted_before_coarse_v3_query"
                break
            query_attempts += 1
            queried_fresh += int(fresh)
            exact_u = float(base._utility(entry.scores, pref, a.rho))
            if idx - 1 in top_records:
                top_records[idx - 1].update({
                    "queried": 1,
                    "exact_utility": exact_u,
                    "exact_gain": exact_u - old_utility,
                })
            qrow = {
                "slate_id": slate_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "query_order_within_slate": int(query_attempts),
                "query_mode": query_mode,
                "candidate_index_noop_inclusive": int(idx),
                "candidate_index": int(idx - 1),
                "candidate_sequence": seq,
                "initial_predicted_rank": int(rank_of[idx - 1]),
                "in_frozen_contender_pool": int(idx in contender_set),
                "committee_median_utility_before_query": float(committee.median_utilities[idx]),
                "committee_lower_utility_before_query": float(committee.utility_lower[idx]),
                "committee_upper_utility_before_query": float(committee.utility_upper[idx]),
                "best_exact_utility_before_query": float(best_exact_u),
                "plausible_challenger_count_before_query": int(len(plausible)),
                "exact_utility": exact_u,
                "exact_gain": float(exact_u - old_utility),
                "fresh_oracle_query": int(fresh),
            }
            self.coarse_v3_query_history.append(qrow)
            if query_attempts == 2 and initial_second_idx is not None and idx != initial_second_idx:
                self._coarse_v3_reranked_second_queries += 1
            reason = "coarse_v3_exact_label_added_and_global_readout_refit"

        # Exact fallback is the only acceptance path.  This also handles early stopping,
        # query-cap boundaries, and global-budget exhaustion monotonically and safely.
        final_map = self._coarse_exact_map(seqs, lineage)
        final_best_idx, final_best_u = best_exact_decision_index(
            final_map, preference=pref, rho=float(a.rho)
        )
        if int(final_best_idx) != 0 and float(final_best_u - old_utility) > float(a.accept_epsilon):
            seq = seqs[int(final_best_idx) - 1]
            entry = self.archive_by_sequence.get(seq)
            if entry is None:
                raise RuntimeError("v3 exact fallback selected candidate without archive entry")
            if self._accept_coarse_entry(
                lineage,
                candidate_tokens=slate[int(final_best_idx) - 1],
                entry=entry,
                slate_id=slate_id,
                candidate_sequence=seq,
            ):
                accepted_idx = int(final_best_idx)
                reason = f"coarse_v3_best_exact_accept_after_{reason}"

        oracle_seconds = float(time.perf_counter() - t2)
        lineage.slates_attempted += 1
        if accepted_idx is not None and accepted_idx - 1 in top_records:
            top_records[accepted_idx - 1]["accepted"] = 1
        self.top_prediction_history.extend(top_records.values())
        accepted_candidate_idx = -1 if accepted_idx is None else int(accepted_idx - 1)
        accepted_gain = float(lineage.utility - old_utility)
        branch_counts = np.asarray([b.allocated_candidates for b in branches], dtype=np.int64)
        self.slate_history.append({
            "slate_id": slate_id,
            "cycle": int(cycle),
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "incumbent_sequence_before": old_sequence,
            "incumbent_utility_before": old_utility,
            "slate_size": int(len(seqs)),
            "slate_per_chain": int(a.slate_per_chain),
            "verification_k": int(cap),
            "verification_queries": int(queried_fresh),
            "accepted": int(accepted_idx is not None),
            "accepted_rank": int(rank_of[accepted_candidate_idx]) if accepted_candidate_idx >= 0 else -1,
            "accepted_sequence": seqs[accepted_candidate_idx] if accepted_candidate_idx >= 0 else "",
            "accepted_chain": chain_labels[accepted_candidate_idx] if accepted_candidate_idx >= 0 else "",
            "accepted_gain": accepted_gain,
            "incumbent_utility_after": float(lineage.utility),
            "cumulative_gain_from_lineage_start": float(lineage.utility - lineage.initial_utility),
            "readout_training_labels": readout_train_size,
            "readout_mode": str(a.readout_mode),
            "best_predicted_utility": float(pred_u[int(order[0])]),
            "best_predicted_gain": float(pred_u[int(order[0])] - old_utility),
            "proposal_seconds": proposal_seconds,
            "ranking_seconds": ranking_seconds,
            "oracle_seconds": oracle_seconds,
            "slate_wall_seconds": float(proposal_seconds + ranking_seconds + oracle_seconds),
            "proposal_physical_draws": int(sum(int(m.get("cheap_draws", 0)) for m in proposal_meta)),
            "proposal_candidates": int(len(seqs)),
            "proposal_draws_per_candidate": float(sum(int(m.get("cheap_draws", 0)) for m in proposal_meta) / max(len(seqs), 1)),
            "batched_root_generation": False,
            "branch_conditioned_generation": True,
            "koopman_A_C_used_for_branch_allocation": int(str(a.koopman_branch_allocation_mode) == "ac_opportunity"),
            "koopman_adaptive_fraction": float(a.koopman_adaptive_fraction),
            "branch_count": int(len(branches)),
            "min_branch_candidates": int(np.min(branch_counts)),
            "max_branch_candidates": int(np.max(branch_counts)),
            "coarse_pdo_used": True,
            "coarse_pdo_v3_exploit_first": True,
            "coarse_v3_contender_cap": int(a.pdo_coarse_v3_contender_cap),
            "coarse_pdo_stop_reason": reason,
            "residual_bank_used": False,
        })
        self._coarse_status[(int(preference_id), int(lineage.lineage_id))] = {
            "accepted": bool(accepted_idx is not None),
            "accepted_gain": accepted_gain,
            "best_predicted_gain": float(pred_u[int(order[0])] - old_utility),
            "slate_id": slate_id,
        }
        if int(a.save_every_slates) > 0 and slate_id % int(a.save_every_slates) == 0:
            self._save_progress()
        return {
            "slate_id": slate_id,
            "paid_queries": int(queried_fresh),
            "accepted": int(accepted_idx is not None),
            "gain": accepted_gain,
            "reason": reason,
        }

    def _run_unified_coarse(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        *,
        query_cap_override: int | None = None,
    ) -> dict[str, Any]:
        mode = str(self.args.v3_coarse_verification_mode)
        if mode == "exploit_first":
            return self._run_coarse_exploit_first_slate(
                preference_id, lineage, cycle, query_cap_override=query_cap_override
            )
        if mode == "v2_pdo":
            return super()._run_coarse_pdo_slate(
                preference_id, lineage, cycle, query_cap_override=query_cap_override
            )
        if mode == "static_topk":
            return super()._run_coarse_static_slate(
                preference_id, lineage, cycle, query_cap_override=query_cap_override
            )
        raise ValueError(f"unknown v3 coarse verification mode {mode!r}")

    def optimize(self) -> None:
        a = self.args
        total = len(self.preferences) * int(a.lineages) * int(a.slates_per_lineage)
        bar = None
        if v2.tqdm is not None and not bool(a.no_progress):
            bar = v2.tqdm(total=total, desc=METHOD_NAME, unit="turn", dynamic_ncols=True)
        try:
            for cycle in range(int(a.slates_per_lineage)):
                for pref_id in range(len(self.preferences)):
                    for lineage in self.lineages[pref_id]:
                        if self.stop_requested:
                            return
                        before_q = int(self.oracle.unique_oracle_queries)
                        before_u = float(lineage.utility)
                        mode = str(a.pdo_budget_mode)
                        if mode == "protected_baseline":
                            self._run_unified_coarse(pref_id, lineage, cycle)
                            if self.stop_requested:
                                return
                            self._run_pdo_turn(pref_id, lineage, cycle)
                        elif mode == "matched_budget":
                            total_cap = int(a.verification_k)
                            coarse = self._run_unified_coarse(
                                pref_id, lineage, cycle, query_cap_override=total_cap
                            )
                            if self.stop_requested:
                                return
                            remaining = max(0, total_cap - int(coarse.get("paid_queries", 0)))
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
        summary.pop("unified_pdo_pegasus_v2", None)
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "physical stochastic intermediate starts from frozen LKF",
            "frozen Koopman A_C conditional terminal-observable look-ahead",
            "75%-protected / 25%-adaptive model-side continuation allocation by default",
            "physical terminal continuation generation from realized starts",
            "freeze pre-query top-of-slate contender pool",
            "coarse query #1 is always the initial global-readout top candidate",
            "each exact coarse vector immediately refits a joint global KFM readout committee",
            "queries #2+ restricted to plausible challengers and chosen by updated median utility",
            "early stop when no contender can plausibly change the best exact decision",
            "best exact improving coarse candidate is the only coarse acceptance path",
            "same-turn reuse of every coarse exact vector by fine PDO",
            "fine Hamming-1 PDO in frozen LKF intervention geometry",
            "exact monotone acceptance at both scales",
        ]
        summary.setdefault("readout", {})["within_slate_global_readout_refit_after_query"] = True
        summary["readout"]["coarse_pdo_local_exact_corrections"] = False
        summary["readout"]["coarse_v3_global_committee"] = {
            "fixed_alpha": float(self.readout_alpha),
            "leave_block_out_folds": int(self.args.pdo_coarse_v3_global_jackknife_folds),
            "min_train_labels": int(self.args.pdo_coarse_v3_min_global_train_labels),
            "joint_multiobjective_models": True,
            "formal_confidence_claim": False,
        }
        summary.setdefault("search", {})["coarse_verification_mode"] = str(self.args.v3_coarse_verification_mode)
        summary["search"]["coarse_v3_contender_cap"] = int(self.args.pdo_coarse_v3_contender_cap)
        summary["unified_pdo_pegasus_v3"] = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "schedule": "coarse_then_fine_same_turn_cross_scale_reuse",
            "roles": {
                "LKF": "physical stochastic branching, terminal generation, fine H1 reachability",
                "Koopman_A_C": "conditional future observability for protected branch continuation allocation",
                "coarse_PDO": "exploit-first global task decision observability over a frozen contender pool",
                "fine_PDO": "local Hamming-1 intervention decision observability",
                "exact_oracle": "sole acceptance authority",
            },
            "coarse_pdo": {
                "mode": str(self.args.v3_coarse_verification_mode),
                "query_1": "initial global-readout top candidate",
                "later_queries": "updated highest-median-utility plausible challenger only",
                "information_first_queries": False,
                "contender_pool": "frozen pre-query top-of-slate",
                "contender_cap": int(self.args.pdo_coarse_v3_contender_cap),
                "epsilon_dec": float(self.args.pdo_coarse_epsilon_dec),
                "global_committee_folds": int(self.args.pdo_coarse_v3_global_jackknife_folds),
                "fresh_queries": int(self._coarse_pdo_fresh_queries),
                "accepted_moves": int(self._coarse_pdo_accepted_moves),
                "early_stop_slates": int(self._coarse_v3_early_stops),
                "reranked_second_query_slates": int(self._coarse_v3_reranked_second_queries),
                "slates": int(self._coarse_v3_slates),
                "same_turn_labels_reused_by_fine_PDO": True,
                "formal_certificate_claimed": False,
            },
            "koopman_branch_allocation": {
                "mode": str(self.args.koopman_branch_allocation_mode),
                "pilot_starts_per_chain": int(self.args.koopman_pilot_starts_per_chain),
                "adaptive_fraction": float(self.args.koopman_adaptive_fraction),
                "hard_pruning": False,
                "residual_bank_or_residual_control": False,
            },
        }
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] LKF physical branching -> A_C look-ahead -> exploit-first coarse PDO -> fine PDO\n"
            f"[{METHOD_NAME}] query #1=top predicted endpoint; later queries=plausible challengers only\n"
            f"[{METHOD_NAME}] exact oracle is sole acceptance authority; residual control and hard pruning disabled",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = v2.build_parser()
    p.description = __doc__
    p.add_argument(
        "--v3-coarse-verification-mode",
        choices=("exploit_first", "v2_pdo", "static_topk"),
        default="exploit_first",
        help="Production v3 uses exploit_first. Other modes are exact component controls with identical proposal code.",
    )
    p.add_argument(
        "--pdo-coarse-v3-contender-cap",
        type=int,
        default=32,
        help="Freeze this many top pre-query terminal candidates as the coarse decision-relevant contender set.",
    )
    p.add_argument(
        "--pdo-coarse-v3-global-jackknife-folds",
        type=int,
        default=4,
        help="Full global readout plus deterministic leave-one-block-out readouts; empirical committee only.",
    )
    p.add_argument(
        "--pdo-coarse-v3-min-global-train-labels",
        type=int,
        default=8,
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    v2._validate_args(args)
    if int(args.pdo_coarse_v3_contender_cap) <= 0:
        raise ValueError("--pdo-coarse-v3-contender-cap must be positive")
    if int(args.pdo_coarse_v3_global_jackknife_folds) < 1:
        raise ValueError("--pdo-coarse-v3-global-jackknife-folds must be at least 1")
    if int(args.pdo_coarse_v3_min_global_train_labels) < 2:
        raise ValueError("--pdo-coarse-v3-min-global-train-labels must be at least 2")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    UnifiedPDOPegasusV3Runner(args).run()


if __name__ == "__main__":
    main()
