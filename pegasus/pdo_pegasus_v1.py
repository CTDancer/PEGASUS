"""PEGASUS v1.3: Pareto Decision Observability for fine Hamming-1 refinement.

This module adds the evidence-aligned PDO fine-decision layer to the frozen production
Koopman Sparse-Verify coarse optimizer without changing its physical proposal law.

Frozen roles
------------
* LKF / discrete flow: physical generator and realizable fine intervention geometry.
* KFM: default coarse observable/readout head and global task prior.
* PDO: inference-time decision observability over explicit Hamming-1 actions.
* Exact vector oracle: authoritative acceptance; model predictions never commit a move.

Corrected v1 design invariants
------------------------------
1. The fine family always contains a no-op action with exact zero gain.
2. Operational PDO is robust regret across a *joint complete vector-response version set*
   under the full augmented-Tchebycheff utility, not a Cartesian product of marginal boxes.
3. Pareto-switch/critical-active-set quantities are diagnostics only.
4. Coarse labels may change the KFM prior mean, but only exact Hamming-1 observations
   from the current incumbent enter the local correction fit/information matrix. Coarse
   labels never enter those local matrices directly.
5. PDO v1.x uses only empirical-confidence wording; no calibration file can promote the
   operational committee to a formal certificate. Every paid query is decision-relevant
   from query 1: consensus winners are tested directly and committee disagreement is
   resolved only among plausible/current winner actions.
6. Fine witnesses are literal sequence edits; there is no latent actuator/decoder.
7. Acquisition probes remain information-first, but an already-paid exact improving probe
   is never discarded at a local stopping/query-cap boundary merely because PDO confidence
   is not yet strong enough to establish empirical top-epsilon confidence.
8. A failed empirical-confidence verification is a falsification event: reject exactly,
   update with the paid vector, suppress confidence stopping, and force one new decision-
   relevant recovery probe before confidence stopping can resume.
9. Existing live-lineage diversity constraints are preserved, not strengthened.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from .kfocus.objectives import OPT_NAMES
from .kfocus.regions import terminal_z
from .pdo_features import (
    FineAction,
    action_feature_matrix,
    enumerate_hamming1_actions,
    find_lkf,
)
from .pdo_math import (
    EmpiricalConfidenceRecovery,
    augmented_tchebycheff_utility,
    best_exact_improving_index,
    decision_relevant_probe_choice,
    empirical_confidence_verification_index,
    fresh_h1_query_gate,
    matched_coarse_verification_slots,
    pareto_switch_diagnostic,
    version_set_pdo_decision,
)
from .pdo_uncertainty import PDOCalibration, build_local_response_state
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-v1.3-recovery-safe"
METHOD_NAME = "PEGASUS v1"


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals:
        raise ValueError("expected at least one numeric value")
    return vals


class PDOPegasusRunner(base.KoopmanSparseVerifyRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.lkf = find_lkf(self.model)
        if self.lkf is None:
            raise RuntimeError("PEGASUS could not locate the frozen LKF under the KFM runtime")
        self.pdo_calibration: PDOCalibration | None = None
        if str(args.pdo_calibration_json).strip():
            cal = PDOCalibration.from_json(args.pdo_calibration_json)
            if tuple(cal.objective_names) != tuple(OPT_NAMES):
                raise ValueError(
                    "PDO calibration objective names do not match production optimization scores"
                )
            self.pdo_calibration = cal
        self.pdo_decision_history: list[dict[str, Any]] = []
        self.pdo_query_history: list[dict[str, Any]] = []
        self.pdo_turn_history: list[dict[str, Any]] = []
        self.pdo_turn_counter = 0
        # ``unknown`` forces a coarse opportunity check before uncertain local acquisition.
        self._coarse_status: dict[tuple[int, int], dict[str, Any]] = {}
        self._per_preference_seen_queries: dict[int, set[str]] = {
            i: set() for i in range(len(self.preferences))
        }
        self._pdo_fresh_queries = 0
        self._pdo_verification_queries = 0
        self._pdo_acquisition_queries = 0
        self._pdo_accepted_moves = 0
        self._pdo_zero_query_cache_accepts = 0
        self._pdo_failed_confidence_verifications = 0
        self._pdo_recovery_probes = 0
        # Recovery must persist across optimizer cycles when a false-confidence event
        # occurs at a turn/query boundary. Keyed by (preference, lineage, incumbent).
        self._pdo_recovery_by_state: dict[tuple[int, int, str], EmpiricalConfidenceRecovery] = {}
        # Every fresh H1 query (acquisition OR empirical-confidence verification) is
        # charged to the local-state cap. Cached exact labels remain free.
        self._pdo_local_fresh_query_counts: dict[tuple[int, int, str], int] = {}

    def _recovery_for_state(
        self, state_key: tuple[int, int, str]
    ) -> EmpiricalConfidenceRecovery:
        """Return the persistent recovery state for one exact incumbent.

        Recovery is keyed by preference, lineage, and sequence so a falsification that
        occurs at the end of one optimizer cycle cannot be forgotten on the next cycle.
        A lineage move naturally switches to a different key and therefore a fresh state.
        """
        return self._pdo_recovery_by_state.setdefault(
            state_key, EmpiricalConfidenceRecovery()
        )

    def _budget_remaining(self) -> int | None:
        """Matched-budget counts every query; protected mode excludes fine extras.

        In protected-baseline mode ``--max-unique-oracle-queries`` is interpreted as
        the ordinary PEGASUS/discovery quota.  Fresh PDO fine queries are additional
        experimental budget and therefore do not consume that protected coarse quota.
        """
        limit = int(self.args.max_unique_oracle_queries)
        if limit <= 0:
            return None
        total = int(self.oracle.unique_oracle_queries)
        if str(getattr(self.args, "pdo_budget_mode", "matched_budget")) == "protected_baseline":
            fine = int(getattr(self, "_pdo_fresh_queries", 0))
            charged = max(0, total - fine)
            return max(0, limit - charged)
        return max(0, limit - total)

    def discovery(self) -> None:
        super().discovery()
        if self.pdo_calibration is None:
            # PDO does not manufacture an operational Cartesian uncertainty box from
            # discovery residuals.  The local joint version set is generated only from
            # the KFM prior plus exact local H1 probes.  Keep zero-radius metadata solely
            # for provenance/claim-level reporting.
            self.pdo_calibration = PDOCalibration(
                objective_names=tuple(OPT_NAMES),
                base_radius=np.zeros(len(OPT_NAMES), dtype=np.float64),
                irreducible_floor_fraction=0.0,
                coverage=float(self.args.pdo_empirical_coverage),
                sequentially_validated=False,
                validation_method="joint_version_set_unvalidated",
                source="pdo-v1.3-joint-version-set-recovery-safe",
            )
        # Shared discovery labels would have to be paid independently under a per-preference
        # benchmark, so include them in the separate accounting view for every preference.
        discovery_sequences = [e.sequence for e in self.archive if e.source == "discovery"]
        for pref_id in self._per_preference_seen_queries:
            self._per_preference_seen_queries[pref_id].update(discovery_sequences)
        write_json(self.out / "pdo_calibration_used.json", self.pdo_calibration.to_dict())

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.pdo_decision_history:
            write_csv(self.out / "pdo_decision_history.csv", self.pdo_decision_history)
        if self.pdo_query_history:
            write_csv(self.out / "pdo_query_history.csv", self.pdo_query_history)
        if self.pdo_turn_history:
            write_csv(self.out / "pdo_turn_history.csv", self.pdo_turn_history)

    def _fine_actions(self, lineage: base.LineageState) -> list[FineAction]:
        actions = enumerate_hamming1_actions(self.lkf, lineage.tokens, include_noop=True)
        anchors = self._diversity_anchors(lineage.preference_id, lineage.lineage_id)
        kept: list[FineAction] = []
        for a in actions:
            if a.is_noop:
                kept.append(a)
                continue
            if all(
                base._hamming(a.tokens, other) >= float(self.args.min_lineage_hamming) - 1e-12
                for other in anchors
            ):
                kept.append(a)
        # Reindex after diversity filtering so all downstream arrays are contiguous.
        return [
            FineAction(
                index=i,
                action_id=a.action_id,
                position_0based=a.position_0based,
                source_aa=a.source_aa,
                target_aa=a.target_aa,
                target_sequence=a.target_sequence,
                tokens=a.tokens,
                is_noop=a.is_noop,
            )
            for i, a in enumerate(kept)
        ]

    def _local_exact_scores(
        self,
        actions: Sequence[FineAction],
        lineage: base.LineageState,
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for a in actions:
            if a.is_noop:
                out[int(a.index)] = np.asarray(lineage.scores, dtype=np.float64).copy()
                continue
            entry = self.archive_by_sequence.get(a.target_sequence)
            if entry is not None:
                out[int(a.index)] = np.asarray(entry.scores, dtype=np.float64).copy()
        return out

    def _fit_pdo_prior_readout(self) -> base.RidgeReadout:
        """Fit the fine-stage global prior without changing the coarse PEGASUS readout.

        ``all_paid`` is the designed default and reuses coarse labels as cross-scale prior
        information.  The other modes exist only for causal ablation of that warm start;
        none of these modes alters the local PDO information matrix.
        """
        mode = str(self.args.pdo_fine_prior_labels)
        if mode == "all_paid":
            return self._fit_current_readout()
        if mode == "discovery_only":
            entries = [e for e in self.archive if e.source == "discovery"]
        elif mode == "discovery_plus_fine":
            entries = [e for e in self.archive if e.source != "sparse_verification"]
        else:
            raise ValueError(f"unknown --pdo-fine-prior-labels {mode!r}")
        if len(entries) < 2:
            raise RuntimeError(f"PDO prior mode {mode} has too few labels")
        z = np.stack([e.z for e in entries], axis=0)
        scores = np.stack([e.scores for e in entries], axis=0)
        return base._fit_ridge(z, scores, float(self.readout_alpha))

    def _pdo_state(
        self,
        preference_id: int,
        lineage: base.LineageState,
        actions: Sequence[FineAction],
        kfm_z: np.ndarray,
        incumbent_kfm_z: np.ndarray,
        Q: np.ndarray,
        *,
        allow_empirical_confidence: bool = True,
    ) -> tuple[Any, Any, Any, base.RidgeReadout, dict[int, np.ndarray]]:
        assert self.pdo_calibration is not None
        readout = self._fit_pdo_prior_readout()
        prior = base._predict_scores(
            readout,
            kfm_z,
            mode=str(self.args.readout_mode),
            incumbent_scores=lineage.scores,
            incumbent_z=incumbent_kfm_z,
        )
        exact_map = self._local_exact_scores(actions, lineage)
        response = build_local_response_state(
            prior,
            Q,
            exact_map,
            self.pdo_calibration,
            local_ridge_alphas=_parse_float_tuple(self.args.pdo_local_ridge_alphas),
            information_lambda=float(self.args.pdo_information_lambda),
            clip_scores=not bool(self.args.no_clip_optimization_scores),
        )
        pref = self.preferences[int(preference_id)]
        decision = version_set_pdo_decision(
            response.member_scores,
            incumbent_scores=lineage.scores,
            preference=pref,
            rho=float(self.args.rho),
            epsilon_dec=float(self.args.pdo_epsilon_dec),
            enable_certification=(
                bool(allow_empirical_confidence)
                and int(response.local_nonreference_observations)
                >= int(self.args.pdo_min_local_probes_before_confidence)
            ),
        )
        switch = pareto_switch_diagnostic(
            lineage.scores,
            response.score_lower,
            response.score_upper,
            pref,
        )
        return response, decision, switch, readout, exact_map

    def _coarse_allows_acquisition(self, preference_id: int, lineage_id: int) -> bool:
        policy = str(self.args.pdo_mode_policy)
        if policy == "always_pdo":
            return True
        if policy == "local_first":
            return True
        if policy != "coarse_failure_trigger":
            raise ValueError(f"unknown PDO mode policy {policy!r}")
        status = self._coarse_status.get((int(preference_id), int(lineage_id)))
        if status is None:
            return False
        return not bool(status.get("accepted", False))

    def _record_preference_query_use(self, preference_id: int, sequence: str) -> None:
        self._per_preference_seen_queries[int(preference_id)].add(str(sequence))

    def _exact_for_fine_action(
        self,
        *,
        action: FineAction,
        z: np.ndarray,
        preference_id: int,
        lineage: base.LineageState,
        source: str,
        mode: str,
        turn_id: int,
        predicted_gain_lower: float,
        predicted_gain_upper: float,
        predicted_regret_upper: float,
        info_score: float | None = None,
    ) -> tuple[base.ArchiveEntry | None, bool]:
        """Return an exact record, or ``None`` when no fresh query budget remains.

        Budget exhaustion is a normal optimizer boundary, not an exception.  Cached exact
        labels are always reusable at zero query cost.  Every fresh H1 query is charged to
        both the global fine-query counters and the local-state fresh-query counter,
        regardless of whether it was an acquisition or a confidence verification.
        """
        existing = self.archive_by_sequence.get(action.target_sequence)
        before_q = int(self.oracle.unique_oracle_queries)
        if existing is not None:
            entry = existing
        else:
            if self._budget_remaining() == 0:
                self.stop_requested = True
                return None, False
            rec = self.oracle.evaluate_one(action.target_sequence)
            exact_u = float(base._utility(rec.scores, self.preferences[int(preference_id)], self.args.rho))
            entry = self._record_exact(
                action.tokens,
                z,
                rec,
                source=source,
                preference_id=int(preference_id),
                lineage_id=int(lineage.lineage_id),
                slate_id=None,
                verification_rank=None,
                utility_at_query=exact_u,
            )
        after_q = int(self.oracle.unique_oracle_queries)
        fresh = after_q > before_q
        if fresh:
            delta_q = int(after_q - before_q)
            self._pdo_fresh_queries += delta_q
            if mode == "verification":
                self._pdo_verification_queries += delta_q
            elif mode == "acquisition":
                self._pdo_acquisition_queries += delta_q
            state_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
            state_key = (int(preference_id), int(lineage.lineage_id), state_sequence)
            self._pdo_local_fresh_query_counts[state_key] = (
                int(self._pdo_local_fresh_query_counts.get(state_key, 0)) + delta_q
            )
        self._record_preference_query_use(preference_id, action.target_sequence)
        exact_u = float(base._utility(entry.scores, self.preferences[int(preference_id)], self.args.rho))
        self.pdo_query_history.append(
            {
                "turn_id": int(turn_id),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence": base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0],
                "mode": str(mode),
                "action_id": action.action_id,
                "candidate_sequence": action.target_sequence,
                "fresh_oracle_query": int(fresh),
                "global_unique_query_index": int(self.oracle.unique_oracle_queries),
                "predicted_gain_lower": float(predicted_gain_lower),
                "predicted_gain_upper": float(predicted_gain_upper),
                "predicted_regret_upper": float(predicted_regret_upper),
                "information_score": "" if info_score is None else float(info_score),
                "exact_utility": exact_u,
                "exact_gain": float(exact_u - lineage.utility),
                "accepted_immediately": 0,
                "acquisition_is_information_only": int(mode == "acquisition"),
                "confidence_falsified_by_exact_verification": 0,
                "is_forced_recovery_probe": 0,
            }
        )
        return entry, fresh

    def _accept_exact_entry(
        self,
        lineage: base.LineageState,
        action: FineAction,
        entry: base.ArchiveEntry,
        *,
        turn_id: int,
    ) -> bool:
        pref = self.preferences[int(lineage.preference_id)]
        exact_u = float(base._utility(entry.scores, pref, self.args.rho))
        gain = float(exact_u - lineage.utility)
        if gain <= float(self.args.accept_epsilon):
            return False
        lineage.tokens = action.tokens.detach().cpu().clone()
        lineage.raw = np.asarray(entry.raw, dtype=np.float64).copy()
        lineage.scores = np.asarray(entry.scores, dtype=np.float64).copy()
        lineage.utility = exact_u
        lineage.accepted_moves += 1
        self._pdo_accepted_moves += 1
        # New state: local response field must be recomputed; force a fresh coarse
        # opportunity assessment before uncertain acquisition at the new incumbent.
        self._coarse_status.pop((int(lineage.preference_id), int(lineage.lineage_id)), None)
        # Fallback acceptance can select an acquisition probe queried earlier in the same
        # turn, so do not assume the accepted action is the last query-history row.
        for row in reversed(self.pdo_query_history):
            if int(row.get("turn_id", -1)) != int(turn_id):
                break
            if row.get("candidate_sequence") == action.target_sequence:
                row["accepted_immediately"] = 1
                row["accepted_via_exact_fallback"] = int(row.get("mode") == "acquisition")
                break
        return True

    def _run_pdo_turn(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
    ) -> dict[str, Any]:
        self.pdo_turn_counter += 1
        turn_id = int(self.pdo_turn_counter)
        start_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        start_utility = float(lineage.utility)
        state_key = (int(preference_id), int(lineage.lineage_id), start_sequence)
        actions = self._fine_actions(lineage)
        if len(actions) <= 1:
            out = {
                "turn_id": turn_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence": start_sequence,
                "fine_actions": int(len(actions)),
                "paid_queries": 0,
                "accepted": 0,
                "reason": "no_diversity_admissible_h1_actions",
            }
            self.pdo_turn_history.append(out)
            return out

        action_tokens = torch.stack([a.tokens for a in actions], dim=0)
        kfm_z = terminal_z(self.model, action_tokens, batch_size=int(self.args.feature_batch_size))
        incumbent_kfm_z = terminal_z(
            self.model,
            lineage.tokens.reshape(1, -1),
            batch_size=int(self.args.feature_batch_size),
        )[0]
        _V, Q, _basis, representation_scale = action_feature_matrix(
            self.lkf,
            actions,
            lineage.tokens,
            batch_size=int(self.args.feature_batch_size),
            svd_rtol=float(self.args.pdo_decision_span_svd_rtol),
        )
        before_turn_q = int(self.oracle.unique_oracle_queries)
        action_attempts = 0
        accepted = False
        reason = "pdo_no_query"
        recovery = self._recovery_for_state(state_key)
        turn_failed_before = int(recovery.failed_verifications)
        turn_recovery_before = int(recovery.recovery_probes)
        if str(self.args.pdo_budget_mode) == "matched_budget":
            max_fresh = min(
                int(self.args.pdo_fine_query_cap_per_turn), int(self.args.verification_k)
            )
        else:
            max_fresh = int(self.args.pdo_fine_query_cap_per_turn)

        while not self.stop_requested:
            response, decision, switch, _readout, exact_map = self._pdo_state(
                preference_id,
                lineage,
                actions,
                kfm_z,
                incumbent_kfm_z,
                Q,
                allow_empirical_confidence=recovery.confidence_allowed(True),
            )
            no_op_idx = next(i for i, a in enumerate(actions) if a.is_noop)
            exact_nonnoop = sum(1 for i in exact_map if i != no_op_idx)
            conf_idx = decision.best_certified_index
            verify_idx = empirical_confidence_verification_index(
                decision,
                no_op_index=int(no_op_idx),
                accept_epsilon=float(self.args.accept_epsilon),
                min_confidence_gain=float(self.args.pdo_min_confidence_gain),
            )
            conf_positive = verify_idx is not None

            best_idx = int(decision.best_mean_index)
            self.pdo_decision_history.append(
                {
                    "turn_id": turn_id,
                    "cycle": int(cycle),
                    "preference_id": int(preference_id),
                    "lineage_id": int(lineage.lineage_id),
                    "incumbent_sequence": start_sequence,
                    "action_count_including_noop": int(len(actions)),
                    "decision_rank": int(response.decision_rank),
                    "representation_scale": float(representation_scale),
                    "local_exact_nonnoop_actions": int(exact_nonnoop),
                    "version_set_members": int(response.ensemble_size),
                    "distinct_member_winners": int(decision.distinct_member_winners),
                    "empirical_confidence_enabled": int(decision.certification_enabled),
                    "confidence_suppressed_for_recovery": int(recovery.pending),
                    "plausible_winner_count": int(len(decision.plausible_indices)),
                    "common_epsilon_action_count": int(len(decision.common_epsilon_indices)),
                    "best_mean_action": actions[best_idx].action_id,
                    "best_mean_gain_lower": float(decision.gain_lower[best_idx]),
                    "best_mean_gain_upper": float(decision.gain_upper[best_idx]),
                    "best_mean_regret_upper": float(decision.regret_upper[best_idx]),
                    "confidence_action": "" if conf_idx is None else actions[int(conf_idx)].action_id,
                    "confidence_positive": int(conf_positive),
                    "epsilon_dec": float(self.args.pdo_epsilon_dec),
                    "claim_level": "empirical_confidence",
                    "pareto_switch_safe_diagnostic": int(switch.switch_safe),
                    "pareto_active_objective": OPT_NAMES[int(switch.active_index)],
                    "pareto_critical_objectives": ";".join(OPT_NAMES[int(i)] for i in switch.critical_indices),
                    "pareto_switch_is_gate": False,
                    "fresh_h1_queries_for_state": int(
                        self._pdo_local_fresh_query_counts.get(state_key, 0)
                    ),
                    "failed_confidence_verifications_for_state": int(recovery.failed_verifications),
                    "recovery_probes_for_state": int(recovery.recovery_probes),
                }
            )

            if conf_positive:
                idx = int(verify_idx)
                action = actions[idx]
                was_cached = action.target_sequence in self.archive_by_sequence
                fresh_used = int(self.oracle.unique_oracle_queries) - before_turn_q
                state_fresh = int(self._pdo_local_fresh_query_counts.get(state_key, 0))
                if not was_cached:
                    gate = fresh_h1_query_gate(
                        turn_fresh_queries=fresh_used,
                        turn_cap=max_fresh,
                        state_fresh_queries=state_fresh,
                        state_cap=int(self.args.pdo_max_local_probes_per_state),
                        global_budget_remaining=self._budget_remaining(),
                    )
                    if not gate.allowed:
                        if gate.reason == "global_budget_exhausted":
                            self.stop_requested = True
                            reason = "global_budget_exhausted_before_confidence_verification"
                        elif gate.reason == "turn_fresh_query_cap_reached":
                            reason = "fine_query_cap_reached_before_confidence_verification"
                        else:
                            reason = "local_fresh_query_cap_reached_before_confidence_verification"
                        break

                entry, fresh = self._exact_for_fine_action(
                    action=action,
                    z=kfm_z[idx],
                    preference_id=preference_id,
                    lineage=lineage,
                    source="pdo_fine_confidence_verification",
                    mode="verification",
                    turn_id=turn_id,
                    predicted_gain_lower=float(decision.gain_lower[idx]),
                    predicted_gain_upper=float(decision.gain_upper[idx]),
                    predicted_regret_upper=float(decision.regret_upper[idx]),
                )
                if entry is None:
                    reason = "global_budget_exhausted_before_confidence_verification"
                    break
                action_attempts += 1
                accepted = self._accept_exact_entry(lineage, action, entry, turn_id=turn_id)
                if accepted:
                    if was_cached and not fresh:
                        self._pdo_zero_query_cache_accepts += 1
                    reason = "empirical_confidence_fine_accepted"
                    break

                # Exact verification falsified the empirical-confidence decision.  The
                # paid vector is now local information.  Do not trust the same committee
                # to stop immediately; force one new decision-relevant recovery probe.
                recovery.observe_verification(
                    float(base._utility(entry.scores, self.preferences[int(preference_id)], self.args.rho))
                    - float(lineage.utility),
                    float(self.args.accept_epsilon),
                )
                self._pdo_failed_confidence_verifications += 1
                reason = "empirical_confidence_verification_failed_recovery_required"
                if self.pdo_query_history:
                    self.pdo_query_history[-1]["confidence_falsified_by_exact_verification"] = 1
                    self.pdo_query_history[-1]["recovery_required"] = 1
                continue

            # Empirical top-epsilon stopping is only enabled when recovery is not pending.
            if conf_idx is not None:
                reason = "epsilon_optimal_local_stop_empirical_confidence_without_robust_gain"
                break

            median_gain = np.median(decision.member_utilities, axis=0) - float(lineage.utility)
            nonnoop = np.asarray([i for i, a in enumerate(actions) if not a.is_noop], dtype=int)
            best_local_mean_gain = float(np.max(median_gain[nonnoop])) if nonnoop.size else float("-inf")
            best_local_upper_gain = float(np.max(decision.gain_upper[nonnoop])) if nonnoop.size else float("-inf")
            plausible_positive = (
                True
                if (recovery.pending or not bool(decision.certification_enabled))
                else (
                    best_local_upper_gain > float(self.args.pdo_min_plausible_gain)
                    and best_local_mean_gain > float(self.args.pdo_min_mean_gain_for_acquisition)
                )
            )
            coarse_allows = (
                True
                if recovery.pending
                else self._coarse_allows_acquisition(preference_id, lineage.lineage_id)
            )
            if not plausible_positive:
                reason = "no_plausible_positive_local_opportunity"
                break
            if not coarse_allows:
                reason = "coarse_opportunity_preferred_before_uncertain_acquisition"
                break

            state_fresh = int(self._pdo_local_fresh_query_counts.get(state_key, 0))
            turn_fresh = int(self.oracle.unique_oracle_queries) - before_turn_q
            gate = fresh_h1_query_gate(
                turn_fresh_queries=turn_fresh,
                turn_cap=max_fresh,
                state_fresh_queries=state_fresh,
                state_cap=int(self.args.pdo_max_local_probes_per_state),
                global_budget_remaining=self._budget_remaining(),
            )
            if not gate.allowed:
                if gate.reason == "global_budget_exhausted":
                    self.stop_requested = True
                    reason = (
                        "global_budget_exhausted_during_recovery"
                        if recovery.pending
                        else "global_budget_exhausted_before_fine_acquisition"
                    )
                elif gate.reason == "turn_fresh_query_cap_reached":
                    reason = (
                        "fine_query_cap_reached_during_recovery"
                        if recovery.pending
                        else "fine_query_cap_reached"
                    )
                else:
                    reason = (
                        "local_fresh_query_cap_reached_during_recovery"
                        if recovery.pending
                        else "local_fresh_query_cap_reached_fallback_to_coarse"
                    )
                break

            observed = set(int(i) for i in exact_map)
            idx, info_score, acquisition_mode, target_size = decision_relevant_probe_choice(
                Q,
                decision,
                observed_indices=observed,
                no_op_index=int(no_op_idx),
                ridge_lambda=float(self.args.pdo_information_lambda),
                epsilon_dec=float(self.args.pdo_epsilon_dec),
                plausible_cap=int(self.args.pdo_acquisition_plausible_cap),
                min_information_score=float(self.args.pdo_min_information_score),
            )
            if idx is None:
                reason = (
                    "no_unobserved_decision_relevant_action_during_recovery"
                    if recovery.pending
                    else "all_decision_relevant_actions_already_exact"
                )
                break

            action = actions[int(idx)]
            was_recovery = bool(recovery.pending)
            entry, fresh = self._exact_for_fine_action(
                action=action,
                z=kfm_z[int(idx)],
                preference_id=preference_id,
                lineage=lineage,
                source="pdo_fine_acquisition",
                mode="acquisition",
                turn_id=turn_id,
                predicted_gain_lower=float(decision.gain_lower[int(idx)]),
                predicted_gain_upper=float(decision.gain_upper[int(idx)]),
                predicted_regret_upper=float(decision.regret_upper[int(idx)]),
                info_score=float(info_score),
            )
            if entry is None:
                reason = "global_budget_exhausted_before_fine_acquisition"
                break
            action_attempts += 1
            if self.pdo_query_history:
                mode_label = str(acquisition_mode)
                if was_recovery:
                    mode_label = f"recovery_after_failed_confidence__{mode_label}"
                self.pdo_query_history[-1]["acquisition_mode"] = mode_label
                self.pdo_query_history[-1]["information_target_size"] = int(target_size)
                self.pdo_query_history[-1]["is_forced_recovery_probe"] = int(was_recovery)
            if was_recovery:
                if not fresh:
                    raise RuntimeError(
                        "PDO recovery invariant violation: forced recovery action was already exact"
                    )
                recovery.observe_recovery_probe(fresh=True)
                self._pdo_recovery_probes += 1
            reason = "acquired_local_decision_information"
            continue

        # Rebuild from the authoritative archive so the final paid query can never be
        # omitted from zero-cost fallback by a stale pre-query local map.
        if not accepted:
            final_exact_map = self._local_exact_scores(actions, lineage)
            fallback_idx, _fallback_gain = best_exact_improving_index(
                final_exact_map,
                incumbent_scores=lineage.scores,
                preference=self.preferences[int(preference_id)],
                rho=float(self.args.rho),
                accept_epsilon=float(self.args.accept_epsilon),
            )
            if fallback_idx is not None and not actions[int(fallback_idx)].is_noop:
                fallback_action = actions[int(fallback_idx)]
                fallback_entry = self.archive_by_sequence.get(fallback_action.target_sequence)
                if fallback_entry is None:
                    raise RuntimeError(
                        "PDO exact fallback selected a non-noop action without an archive entry"
                    )
                stop_reason = reason
                accepted = self._accept_exact_entry(
                    lineage,
                    fallback_action,
                    fallback_entry,
                    turn_id=turn_id,
                )
                if accepted:
                    self._pdo_zero_query_cache_accepts += 1
                    reason = f"best_exact_fallback_accepted_after_{stop_reason}"

        paid = int(self.oracle.unique_oracle_queries) - before_turn_q
        out = {
            "turn_id": turn_id,
            "cycle": int(cycle),
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "incumbent_sequence": start_sequence,
            "incumbent_utility_before": start_utility,
            "incumbent_utility_after": float(lineage.utility),
            "fine_actions": int(len(actions)),
            "action_attempts": int(action_attempts),
            "fresh_h1_queries_for_state": int(
                self._pdo_local_fresh_query_counts.get(state_key, 0)
            ),
            "failed_confidence_verifications": int(recovery.failed_verifications - turn_failed_before),
            "forced_recovery_probes": int(recovery.recovery_probes - turn_recovery_before),
            "failed_confidence_verifications_for_state": int(recovery.failed_verifications),
            "forced_recovery_probes_for_state": int(recovery.recovery_probes),
            "recovery_pending_at_stop": int(recovery.pending),
            "paid_queries": int(paid),
            "accepted": int(accepted),
            "gain": float(lineage.utility - start_utility),
            "reason": reason,
        }
        self.pdo_turn_history.append(out)
        if int(self.args.save_every_slates) > 0 and turn_id % int(self.args.save_every_slates) == 0:
            self._save_progress()
        return out

    def _run_coarse_and_track(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        *,
        verification_k_override: int | None = None,
    ) -> None:
        before = len(self.slate_history)
        original_k = int(self.args.verification_k)
        if verification_k_override is not None:
            k = int(verification_k_override)
            if k <= 0:
                return
            if k > original_k:
                raise ValueError("coarse verification override cannot exceed configured verification_k")
            self.args.verification_k = k
        try:
            super()._run_one_slate(preference_id, lineage, cycle)
        finally:
            self.args.verification_k = original_k
        if len(self.slate_history) > before:
            row = self.slate_history[-1]
            self._coarse_status[(int(preference_id), int(lineage.lineage_id))] = {
                "accepted": bool(int(row.get("accepted", 0))),
                "accepted_gain": float(row.get("accepted_gain", 0.0)),
                "best_predicted_gain": float(row.get("best_predicted_gain", float("nan"))),
                "slate_id": int(row.get("slate_id", -1)),
            }
            # Track amortized versus per-preference query accounting using actual coarse
            # query-history rows emitted by the base runner.
            slate_id = int(row.get("slate_id", -1))
            for q in reversed(self.query_history):
                if int(q.get("slate_id", -2)) != slate_id:
                    if int(q.get("slate_id", -2)) < slate_id:
                        break
                    continue
                self._record_preference_query_use(preference_id, q.get("candidate_sequence", ""))

    def optimize(self) -> None:
        a = self.args
        total = len(self.preferences) * int(a.lineages) * int(a.slates_per_lineage)
        bar = None
        if tqdm is not None and not bool(a.no_progress):
            bar = tqdm(total=total, desc="PEGASUS optimization", unit="turn", dynamic_ncols=True)
        try:
            for cycle in range(int(a.slates_per_lineage)):
                for pref_id in range(len(self.preferences)):
                    for lineage in self.lineages[pref_id]:
                        if self.stop_requested:
                            return
                        before_q = int(self.oracle.unique_oracle_queries)
                        before_u = float(lineage.utility)
                        fine = self._run_pdo_turn(pref_id, lineage, cycle)
                        if self.stop_requested:
                            return
                        mode = str(a.pdo_budget_mode)
                        if mode == "protected_baseline":
                            self._run_coarse_and_track(pref_id, lineage, cycle)
                        elif mode == "matched_budget":
                            # Match the base PEGASUS verification allocation per turn.  A fine
                            # acceptance ends the turn just like an early coarse acceptance.
                            # If fine queries failed, give the frozen coarse branch only the
                            # *remaining* verification slots instead of discarding them.
                            if int(fine["accepted"]) == 0:
                                remaining = matched_coarse_verification_slots(
                                    verification_k=int(a.verification_k),
                                    fine_paid_queries=int(fine["paid_queries"]),
                                    fine_accepted=False,
                                )
                                if remaining > 0:
                                    self._run_coarse_and_track(
                                        pref_id, lineage, cycle, verification_k_override=remaining
                                    )
                        else:
                            raise ValueError(f"unknown PDO budget mode {mode!r}")
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

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        assert self.pdo_calibration is not None

        per_pref_fine: dict[str, dict[str, int]] = {}
        for pref_id in range(len(self.preferences)):
            qrows = [
                r for r in self.pdo_query_history
                if int(r.get("preference_id", -1)) == int(pref_id)
                and int(r.get("fresh_oracle_query", 0)) == 1
            ]
            trows = [
                r for r in self.pdo_turn_history
                if int(r.get("preference_id", -1)) == int(pref_id)
            ]
            per_pref_fine[str(pref_id)] = {
                "fresh_fine_queries": int(len(qrows)),
                "fresh_confidence_verification_queries": int(
                    sum(str(r.get("mode", "")) == "verification" for r in qrows)
                ),
                "fresh_acquisition_queries": int(
                    sum(str(r.get("mode", "")) == "acquisition" for r in qrows)
                ),
                "fine_accepted_moves": int(sum(int(r.get("accepted", 0)) for r in trows)),
                "failed_confidence_verifications": int(
                    sum(int(r.get("confidence_falsified_by_exact_verification", 0)) for r in qrows)
                ),
            }

        pdo = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "budget_mode": str(self.args.pdo_budget_mode),
            "mode_policy": str(self.args.pdo_mode_policy),
            "epsilon_dec": float(self.args.pdo_epsilon_dec),
            "fine_query_cap_per_turn": int(self.args.pdo_fine_query_cap_per_turn),
            "max_local_fresh_h1_queries_per_state": int(self.args.pdo_max_local_probes_per_state),
            "fine_fresh_queries": int(self._pdo_fresh_queries),
            "fine_verification_fresh_queries": int(self._pdo_verification_queries),
            "fine_acquisition_fresh_queries": int(self._pdo_acquisition_queries),
            "fine_accepted_moves": int(self._pdo_accepted_moves),
            "fine_zero_query_cache_accepts": int(self._pdo_zero_query_cache_accepts),
            "failed_empirical_confidence_verifications": int(self._pdo_failed_confidence_verifications),
            "forced_recovery_probes": int(self._pdo_recovery_probes),
            "calibration": self.pdo_calibration.to_dict(),
            "fine_prior_label_mode": str(self.args.pdo_fine_prior_labels),
            "cross_scale_coarse_labels_used_in_fine_prior": bool(
                str(self.args.pdo_fine_prior_labels) == "all_paid"
            ),
            "claim_level": "empirical_confidence",
            "formal_certificate_claimed": False,
            "operational_scalarization": (
                "robust full augmented-Tchebycheff regret over joint vector-response version set"
            ),
            "preference_weight_normalization": "sum_to_one_exactly_matching_frozen_PEGASUS",
            "no_op_in_action_family": True,
            "pareto_switch_used_as_gate": False,
            "coarse_labels_directly_enter_local_information_matrix": False,
            "cross_scale_prior_can_change_member_means": True,
            "cross_scale_confidence_claim": (
                "empirical only; confidence requires local H1 evidence and no theorem-level "
                "cross-scale uncertainty transfer is claimed"
            ),
            "static_marginal_boxes_used_operationally": False,
            "failed_confidence_recovery_rule": (
                "reject exactly; add paid vector; suppress confidence stopping; force one new "
                "decision-relevant recovery probe before confidence stopping can resume"
            ),
            "min_local_probes_before_confidence": int(
                self.args.pdo_min_local_probes_before_confidence
            ),
            "fine_geometry": "pooled frozen LKF shared-block displacement at t=1",
            "fine_action_family": "explicit diversity-admissible Hamming-1 substitutions + no-op",
            "acquisition_probe_acceptance": (
                "information-first; an already-paid exact improving probe may be accepted later "
                "by empirical-confidence verification or zero-query exact fallback"
            ),
            "budget_accounting": {
                "actual_global_unique_queries_fine_inclusive": int(self.oracle.unique_oracle_queries),
                "legacy_coarse_verification_query_history_rows": int(len(self.query_history)),
                "fine_fresh_queries": int(self._pdo_fresh_queries),
                "protected_coarse_charged_queries": int(
                    self.oracle.unique_oracle_queries - self._pdo_fresh_queries
                ) if str(self.args.pdo_budget_mode) == "protected_baseline" else int(self.oracle.unique_oracle_queries),
                "fine_queries_are_extra_in_protected_mode": bool(
                    str(self.args.pdo_budget_mode) == "protected_baseline"
                ),
                "matched_budget_semantics": (
                    "fine failed queries consume verification slots; unused slots are returned to "
                    "the frozen coarse branch; fine acceptance ends the turn"
                ),
                "note": (
                    "Legacy PEGASUS per-preference verification counters exclude PDO fine queries. "
                    "Use the explicit fine-inclusive PDO accounting in this block."
                ),
            },
            "per_preference_fine_accounting": per_pref_fine,
            "cross_preference_accounting": {
                "global_amortized_unique_queries": int(self.oracle.unique_oracle_queries),
                "per_preference_distinct_sequences_used": {
                    str(k): int(len(v)) for k, v in self._per_preference_seen_queries.items()
                },
                "cross_preference_cache_reuse_enabled": bool(len(self.preferences) > 1),
                "note": (
                    "The production vector-oracle cache/readout is shared across preferences, matching legacy PEGASUS. "
                    "Per-preference counts above record sequences directly charged/used by that preference plus shared discovery; "
                    "they are NOT a counterfactual independent-run oracle cost. For strict per-preference budgets, run each "
                    "preference in a separate PEGASUS process/output directory."
                ),
            },
        }
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["pdo"] = pdo
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] coarse_base={base.IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] budget_mode={self.args.pdo_budget_mode}, "
            f"mode_policy={self.args.pdo_mode_policy}, epsilon_dec={self.args.pdo_epsilon_dec}\n"
            f"[{METHOD_NAME}] fine geometry=explicit H1 + no-op in frozen LKF shared-block space; "
            f"KFM remains coarse/global prior; exact verification is authoritative",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = base.build_parser()
    p.description = __doc__
    p.set_defaults(
        residual_checkpoint="./results/residual_distribution_koopman_calibration/checkpoints/calibrated.pt",
        terminal_checkpoint="./results/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt",
        lkf_checkpoint="./checkpoints/M8.ckpt",
    )
    for _action in p._actions:
        if _action.dest == "residual_checkpoint":
            _action.required = False

    g = p.add_argument_group("PEGASUS v1 fine decision")
    g.add_argument(
        "--pdo-budget-mode",
        choices=("matched_budget", "protected_baseline"),
        default="matched_budget",
        help=(
            "matched_budget: fresh PDO fine queries consume that turn's coarse verification slots; "
            "unused slots remain available to the frozen coarse branch; protected_baseline: "
            "always retain the ordinary coarse slate in addition to PDO."
        ),
    )
    g.add_argument(
        "--pdo-mode-policy",
        choices=("coarse_failure_trigger", "always_pdo", "local_first"),
        default="coarse_failure_trigger",
        help=(
            "Default is opportunity-based: uncertain local acquisition starts after the last "
            "coarse slate on this lineage failed, while empirical-confidence fine actions may be verified anytime."
        ),
    )
    g.add_argument(
        "--pdo-fine-prior-labels",
        choices=("all_paid", "discovery_only", "discovery_plus_fine"),
        default="all_paid",
        help=(
            "Controls only the KFM prior used by PDO. all_paid is the designed cross-scale warm start; "
            "other modes are causal ablations. Local confidence still shrinks only from exact H1 actions."
        ),
    )
    g.add_argument("--pdo-epsilon-dec", type=float, default=0.002)
    g.add_argument(
        "--pdo-min-confidence-gain",
        "--pdo-min-certified-gain",
        dest="pdo_min_confidence_gain",
        type=float,
        default=0.0,
        help="Minimum worst-member predicted gain for empirical-confidence verification.",
    )
    g.add_argument("--pdo-min-plausible-gain", type=float, default=0.0)
    g.add_argument("--pdo-min-mean-gain-for-acquisition", type=float, default=0.0)
    g.add_argument("--pdo-fine-query-cap-per-turn", type=int, default=3)
    g.add_argument("--pdo-max-local-probes-per-state", type=int, default=6)
    g.add_argument("--pdo-acquisition-plausible-cap", type=int, default=32)
    g.add_argument(
        "--pdo-initial-candidate-cap",
        type=int,
        default=-1,
        help="Deprecated compatibility option; PDO v1.3 has no geometry warm-up slate.",
    )
    g.add_argument("--pdo-min-local-probes-before-confidence", type=int, default=2)
    g.add_argument("--pdo-min-information-score", type=float, default=1e-12)
    g.add_argument("--pdo-information-lambda", type=float, default=1.0)
    g.add_argument("--pdo-local-ridge-alphas", default="0.03,0.1,0.3,1,3")
    g.add_argument("--pdo-decision-span-svd-rtol", type=float, default=1e-8)

    c = p.add_argument_group("PDO confidence metadata")
    c.add_argument(
        "--pdo-calibration-json",
        default="",
        help=(
            "Optional PDO calibration/provenance metadata. Marginal objective radii are not "
            "used operationally by PDO. PDO v1.x always reports empirical confidence, never a formal certificate."
        ),
    )
    c.add_argument("--pdo-empirical-coverage", type=float, default=0.95)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """Validate inherited PEGASUS invariants plus PDO-specific safety constraints."""
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
    if int(args.verification_k) > int(args.slate_per_chain) * len(base.parse_chains(args.chains)):
        raise ValueError("--verification-k exceeds total slate size")
    if (not math.isfinite(float(args.accept_epsilon))) or float(args.accept_epsilon) < 0:
        raise ValueError("--accept-epsilon must be finite and nonnegative")
    if (not math.isfinite(float(args.pdo_epsilon_dec))) or float(args.pdo_epsilon_dec) < 0:
        raise ValueError("--pdo-epsilon-dec must be finite and nonnegative")
    if (not math.isfinite(float(args.rho))) or float(args.rho) < 0:
        raise ValueError("--rho must be finite and nonnegative")
    if (not math.isfinite(float(args.strong_gain))) or float(args.strong_gain) <= 0.0:
        raise ValueError("--strong-gain must be finite and positive")
    if not (0.0 <= float(args.min_lineage_hamming) <= 1.0):
        raise ValueError("--min-lineage-hamming must lie in [0,1]")
    if int(args.pdo_fine_query_cap_per_turn) <= 0:
        raise ValueError("--pdo-fine-query-cap-per-turn must be positive")
    if int(args.pdo_max_local_probes_per_state) < 0:
        raise ValueError("--pdo-max-local-probes-per-state cannot be negative")
    if int(args.pdo_min_local_probes_before_confidence) < 1:
        raise ValueError("--pdo-min-local-probes-before-confidence must be at least 1")
    if int(args.pdo_acquisition_plausible_cap) == 0:
        raise ValueError(
            "--pdo-acquisition-plausible-cap cannot be zero; use a negative value for unlimited"
        )
    for _name in (
        "pdo_min_confidence_gain",
        "pdo_min_plausible_gain",
        "pdo_min_mean_gain_for_acquisition",
    ):
        if not math.isfinite(float(getattr(args, _name))):
            raise ValueError(f"--{_name.replace('_', '-')} must be finite")
    local_alphas = _parse_float_tuple(args.pdo_local_ridge_alphas)
    if any((not math.isfinite(float(a))) or float(a) <= 0.0 for a in local_alphas):
        raise ValueError("--pdo-local-ridge-alphas values must all be finite and positive")
    if (not math.isfinite(float(args.pdo_information_lambda))) or float(args.pdo_information_lambda) <= 0:
        raise ValueError("--pdo-information-lambda must be finite and positive")
    if (not math.isfinite(float(args.pdo_min_information_score))) or float(args.pdo_min_information_score) < 0:
        raise ValueError("--pdo-min-information-score must be finite and nonnegative")
    if (not math.isfinite(float(args.pdo_decision_span_svd_rtol))) or float(args.pdo_decision_span_svd_rtol) <= 0:
        raise ValueError("--pdo-decision-span-svd-rtol must be finite and positive")
    if not 0.0 < float(args.pdo_empirical_coverage) < 1.0:
        raise ValueError("--pdo-empirical-coverage must lie in (0,1)")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    PDOPegasusRunner(args).run()


if __name__ == "__main__":
    main()
