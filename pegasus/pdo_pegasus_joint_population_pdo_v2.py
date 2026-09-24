"""PEGASUS joint Population PDO v2: shared acquisition, exact-leader stopping, exact acceptance.

This revision keeps the successful frozen protected PEGASUS decision boundary and changes
only HOW fine-stage exact challenger queries are allocated across the population.

The v1 joint implementation demonstrated real cross-lineage shared decision geometry, but
its empirical robust-set certification of unseen actions was contradicted by exact
verification on most attempts.  v2 therefore removes unqueried-action certification
entirely.

Fine-stage design
-----------------
1. Realize the same finite multiscale Hamming action families (H1/H2/H4/H8/...) as the
   frozen protected method.  Scale is an action attribute, never a latent scale choice.
2. Preserve one protected exploit-first exact fine exposure per active lineage.
3. For every lineage, define the current exact leader from no-op plus exactly evaluated
   fine candidates.  An unqueried candidate is PDO-active only when the empirical global
   readout committee says its utility upper bound can exceed the exact leader by more than
   the PDO query-resolution tolerance.  This is the successful protected stopping rule.
4. Pool one exploit-first representative from every active lineage-scale.  Buy ONE exact
   population-wide query at a time.  The production scheduler lexicographically maximizes
   shared residual decision-rank reduction across all lineages; value and contraction are
   only tie-breaks.  If numerical rank gain is zero, the highest-value active challenger is
   still queried rather than pruned, so approximate geometry cannot suppress opportunity.
5. Refit the shared readout immediately and recompute all lineage decisions after every
   purchased label.  A query in one lineage may therefore resolve challengers elsewhere.
6. Stop only when each lineage has no plausible unqueried challenger to its exact leader,
   or its existing fine-query cap is reached.
7. Final action choice uses ONLY no-op plus exactly evaluated candidates.  Each lineage
   forms its exact near-optimal set using the existing split of the total decision epsilon.
   Coordinate descent then minimizes SIGNED raw reachable contraction inside the Cartesian
   product of those exact near-optimal sets.  Hence the selected exact tuple cannot have
   worse one-step mean pairwise Hamming diversity than the independent exact-best tuple,
   while exact utility sacrifice is bounded by the configured exact-tie tolerance.
8. Exact oracle values alone authorize accepted moves.  There is no empirical certification
   of an unseen action and no certification/verification loop.

The exact-linear population-PDO theorem remains conditional on a valid shared response
class.  The deployed committee is still empirical confidence, not formal coverage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_joint_population_pdo as v1
from . import pdo_pegasus_multiscale_population_pdo_protected as protected
from .kfocus.objectives import OPT_NAMES, RAW_NAMES
from .pdo_contraction_allocation import AllocationOption
from .pdo_joint_population import (
    build_fast_shared_rank_context,
    coordinate_minimize_raw_contraction,
    fast_shared_rank_gain,
    feature_scale,
    joint_span_statistics,
    scale_unique_rank_contributions,
)
from .pdo_pegasus_v3 import plausible_challenger_indices
from .utils import write_json

IMPLEMENTATION_VERSION = "pegasus-joint-population-pdo-v2"
METHOD_NAME = "PEGASUS Joint Population PDO v2"

@dataclass
class ExactLeaderJointLineageView:
    """One lineage's unified multiscale exact-leader challenger problem."""

    state: protected.PopulationFineState
    committee: Any | None
    plausible_indices: np.ndarray
    by_unified_index: dict[int, v1.JointBinding]
    by_key: dict[tuple[int, int], v1.JointBinding]
    exact_indices: set[int]
    protected_exposure: bool
    best_exact_utility: float
    best_exact_uid: int

    @property
    def empirical_resolved(self) -> bool:
        # Name retained only for compatibility with inherited geometry helpers.  The
        # condition is exact-leader challenger resolution, not empirical certification.
        return bool(self.protected_exposure and len(self.plausible_indices) == 0)

    @property
    def plausible_nonnoop(self) -> list[v1.JointBinding]:
        return [
            self.by_unified_index[int(i)]
            for i in self.plausible_indices
            if int(i) > 0 and int(i) in self.by_unified_index
        ]


class JointPopulationPDORunnerV2(v1.JointPopulationPDORunner):
    """Protected PEGASUS with shared-rank population acquisition and exact-only acceptance."""

    # -------------------------------------------------------------------------
    # Exact-leader challenger semantics
    # -------------------------------------------------------------------------
    def _build_joint_view(self, state: protected.PopulationFineState) -> ExactLeaderJointLineageView:
        """Build one unified multiscale committee, but resolve against an exact leader.

        ``plausible_indices`` means unqueried candidates whose empirical upper utility can
        still beat the current exact leader by more than the PDO query tolerance.  This is
        a challenger-stopping rule, not an unseen-action certificate.
        """
        rows: list[tuple[int, Any, int]] = []
        for pool_key, pool in enumerate(state.pools):
            for local in sorted(set(int(i) for i in pool.contender_indices.tolist())):
                if local <= 0 or local > len(pool.actions):
                    raise RuntimeError("invalid frozen contender index")
                rows.append((int(pool_key), pool, int(local)))

        protected_exposure = bool(self._has_exact_fine_exposure(state))
        if not rows:
            return ExactLeaderJointLineageView(
                state=state,
                committee=None,
                plausible_indices=np.zeros(0, dtype=np.int64),
                by_unified_index={},
                by_key={},
                exact_indices={0},
                protected_exposure=True,
                best_exact_utility=float(state.start_utility),
                best_exact_uid=0,
            )

        candidate_z = np.stack([pool.z[local - 1] for _, pool, local in rows], axis=0)
        exact_map: dict[int, np.ndarray] = {
            0: np.asarray(state.lineage.scores, dtype=np.float64).copy()
        }
        exact_indices: set[int] = {0}
        for uid, (_pool_key, pool, local) in enumerate(rows, start=1):
            action = pool.actions[local - 1]
            entry = self.archive_by_sequence.get(action.target_sequence)
            if entry is not None:
                exact_map[int(uid)] = np.asarray(entry.scores, dtype=np.float64).copy()
                exact_indices.add(int(uid))

        pref = self.preferences[int(state.preference_id)]
        committee = self._build_v3_committee(
            candidate_z=candidate_z,
            incumbent_z=state.incumbent_z,
            incumbent_scores=np.asarray(state.lineage.scores, dtype=np.float64),
            exact_map=exact_map,
            preference=pref,
        )

        exact_u = {
            int(uid): float(base._utility(score, pref, self.args.rho))
            for uid, score in exact_map.items()
        }
        exact_leader_uid = min(
            exact_u,
            key=lambda uid: (-float(exact_u[uid]), int(uid)),
        )
        best_exact_u = float(exact_u[int(exact_leader_uid)])
        contender_uids = list(range(1, len(rows) + 1))
        plausible = plausible_challenger_indices(
            committee,
            contender_indices=contender_uids,
            observed_indices=sorted(exact_indices),
            best_exact_utility=best_exact_u,
            epsilon_dec=float(self._pdo_query_epsilon()),
        )

        by_uid: dict[int, v1.JointBinding] = {}
        by_key: dict[tuple[int, int], v1.JointBinding] = {}
        for uid, (pool_key, pool, local) in enumerate(rows, start=1):
            action = pool.actions[local - 1]
            binding = v1.JointBinding(
                state=state,
                pool=pool,
                pool_key=int(pool_key),
                local_index=int(local),
                unified_index=int(uid),
                delta_z=np.asarray(pool.z[local - 1] - state.incumbent_z, dtype=np.float64),
                option=AllocationOption(
                    lineage_id=int(state.lineage_id),
                    preference_id=int(state.preference_id),
                    sequence=str(action.target_sequence),
                    incumbent_sequence=str(state.start_sequence),
                    scale_k=int(pool.k),
                    predicted_utility=float(committee.median_utilities[uid]),
                    predicted_upper=float(committee.utility_upper[uid]),
                    pool_key=int(pool_key),
                    local_index=int(local),
                    initial_rank=int(pool.initial_rank[local - 1]),
                ),
            )
            by_uid[int(uid)] = binding
            by_key[binding.key] = binding

        return ExactLeaderJointLineageView(
            state=state,
            committee=committee,
            plausible_indices=np.asarray(plausible, dtype=np.int64),
            by_unified_index=by_uid,
            by_key=by_key,
            exact_indices=exact_indices,
            protected_exposure=protected_exposure,
            best_exact_utility=float(best_exact_u),
            best_exact_uid=int(exact_leader_uid),
        )

    def _rank_context(
        self,
        views: Mapping[int, ExactLeaderJointLineageView],
        *,
        include_resolved: bool = False,
    ) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, dict[str, object], dict[int, int]]:
        """Joint unresolved geometry over query-active lineages only.

        Exact directions from every lineage remain shared observations.  A lineage that has
        already stopped (resolved or capped) is not allowed to inflate shared-rank gain.
        """
        observed = self._observed_fine_directions(views)
        decision: dict[int, np.ndarray] = {}
        scale_rows: dict[int, list[np.ndarray]] = {}
        all_for_scale: list[np.ndarray] = [r for r in observed]
        for lid, view in views.items():
            if (not include_resolved) and (view.state.done or view.empirical_resolved):
                continue
            mat = self._decision_rows(view)
            decision[int(lid)] = mat
            for b in view.plausible_nonnoop:
                scale_rows.setdefault(int(b.pool.k), []).append(np.asarray(b.delta_z, dtype=np.float64))
                all_for_scale.append(np.asarray(b.delta_z, dtype=np.float64))
        dim = observed.shape[1] if observed.ndim == 2 and observed.shape[1] else (
            next((m.shape[1] for m in decision.values() if m.shape[1] > 0), 0)
        )
        scale = feature_scale(
            np.stack(all_for_scale, axis=0)
            if all_for_scale
            else np.zeros((0, dim), dtype=np.float64)
        ) if dim > 0 else np.ones(0, dtype=np.float64)
        stats = joint_span_statistics(
            observed,
            decision,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        )
        by_scale = {
            k: np.stack(v, axis=0) if v else np.zeros((0, dim), dtype=np.float64)
            for k, v in scale_rows.items()
        }
        unique = scale_unique_rank_contributions(
            observed,
            by_scale,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        ) if by_scale else {}
        return observed, decision, scale, stats, unique

    def _scale_representatives(self, view: ExactLeaderJointLineageView) -> list[v1.JointBinding]:
        """One exploit-first representative from each currently active scale."""
        grouped: dict[int, list[v1.JointBinding]] = defaultdict(list)
        for b in view.plausible_nonnoop:
            if int(b.unified_index) in view.exact_indices:
                continue
            grouped[int(b.pool.k)].append(b)
        reps: list[v1.JointBinding] = []
        for k in sorted(grouped):
            rows = sorted(
                grouped[k],
                key=lambda b: (
                    -float(b.option.predicted_utility),
                    -float(b.option.predicted_upper),
                    int(b.option.initial_rank),
                    str(b.option.sequence),
                ),
            )
            reps.append(rows[0])
        return reps

    def _choose_joint_rank_query(
        self,
        views: Mapping[int, ExactLeaderJointLineageView],
        states: Sequence[protected.PopulationFineState],
    ) -> tuple[v1.JointBinding | None, dict[str, Any]]:
        """Choose one active scale representative population-wide.

        Shared rank is a lexicographic acquisition priority, not a pruning rule.  Therefore
        gain-zero representatives remain eligible and fall back to exploit-first value
        ordering when the numerical shared geometry cannot identify an informative basis
        direction.
        """
        observed, decision, scale, stats, _unique = self._rank_context(
            views, include_resolved=False
        )
        fast_context = build_fast_shared_rank_context(
            observed,
            decision,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        )
        state_by_id = {int(s.lineage_id): s for s in states}
        scored: list[tuple[tuple[Any, ...], v1.JointBinding, dict[int, int], int, float]] = []
        for lid, view in views.items():
            state = state_by_id[int(lid)]
            if state.done or view.empirical_resolved:
                continue
            if self._fresh_used(state) >= int(state.fresh_cap):
                continue
            for b in self._scale_representatives(view):
                gain, gains = fast_shared_rank_gain(fast_context, b.delta_z)
                contraction = (
                    self._query_contraction_tiebreak(b, states)
                    if str(self.args.pdo_population_mode) == "contraction"
                    else 0.0
                )
                rank_key = int(gain) if str(self.args.pdo_joint_acquisition_mode) == "shared_rank" else 0
                key = (
                    -int(rank_key),
                    -float(b.option.predicted_utility),
                    -float(b.option.predicted_upper),
                    float(contraction),
                    int(b.option.initial_rank),
                    int(b.option.scale_k),
                    int(b.option.lineage_id),
                    str(b.option.sequence),
                )
                scored.append((key, b, gains, int(gain), float(contraction)))
        if not scored:
            return None, {"rank_stats": stats, "eligible_query_count": 0}
        scored.sort(key=lambda x: x[0])
        _key, chosen, gains, gain, contraction = scored[0]
        return chosen, {
            "rank_stats": stats,
            "eligible_query_count": int(len(scored)),
            "shared_rank_gain": int(gain),
            "per_lineage_rank_gain": gains,
            "query_contraction_tiebreak": float(contraction),
            "gain_zero_fallback": int(gain <= 0),
        }

    def _log_joint_state(
        self,
        views: Mapping[int, ExactLeaderJointLineageView],
        *,
        iteration: int,
        selection_reason: str,
    ) -> dict[str, object]:
        observed, _decision, _scale, stats, unique = self._rank_context(
            views, include_resolved=False
        )
        row: dict[str, Any] = {
            "cycle": int(next(iter(views.values())).state.cycle) if views else -1,
            "preference_id": int(next(iter(views.values())).state.preference_id) if views else -1,
            "iteration": int(iteration),
            "selection_reason": str(selection_reason),
            "lineage_count": int(len(views)),
            "protected_lineages": int(sum(v.protected_exposure for v in views.values())),
            "exact_leader_resolved_lineages": int(sum(v.empirical_resolved for v in views.values())),
            "unresolved_lineages": int(sum(not v.empirical_resolved for v in views.values())),
            "observed_fine_direction_count": int(observed.shape[0]),
            "sum_lineage_unresolved_dimension": int(stats["sum_lineage_dimension"]),
            "joint_unresolved_dimension": int(stats["joint_dimension"]),
            "sharing_factor": float(stats["sharing_factor"]),
            "plausible_exact_leader_challenger_count": int(
                sum(len(v.plausible_nonnoop) for v in views.values() if not v.empirical_resolved)
            ),
            "pdo_query_epsilon": float(self._pdo_query_epsilon()),
            "exact_tie_epsilon": float(self._exact_tie_epsilon()),
            "rank_rtol": float(self._joint_rank_rtol()),
        }
        self.joint_population_pdo_history.append(row)
        for k, val in sorted(unique.items()):
            self.joint_population_scale_rank_history.append(
                {
                    "cycle": int(row["cycle"]),
                    "preference_id": int(row["preference_id"]),
                    "iteration": int(iteration),
                    "scale_k": int(k),
                    "unique_unresolved_rank_contribution": int(val),
                    "joint_unresolved_dimension": int(stats["joint_dimension"]),
                }
            )
        return stats

    # -------------------------------------------------------------------------
    # Exact-only final population decision
    # -------------------------------------------------------------------------
    def _joint_exact_acceptance_v2(
        self,
        states: Sequence[protected.PopulationFineState],
        *,
        iteration: int,
    ) -> None:
        alternatives: dict[int, list[AllocationOption]] = {}
        bindings: dict[int, dict[tuple[int, int], protected.ExactBinding]] = {}
        best_binding: dict[int, protected.ExactBinding] = {}
        state_by_id = {int(s.lineage_id): s for s in states}

        for state in states:
            eligible = self._exact_near_optimal_bindings(state)
            lid = int(state.lineage_id)
            alternatives[lid] = [r.option for r in eligible]
            bindings[lid] = {
                (int(r.option.pool_key), int(r.option.local_index)): r for r in eligible
            }
            best_binding[lid] = eligible[0]

        selected_opts, raw0, raw1 = coordinate_minimize_raw_contraction(
            alternatives,
            passes=int(self.args.pdo_population_coordinate_passes),
            use_diversity=(str(self.args.pdo_population_mode) == "contraction"),
        )
        pair_count = max(1, len(selected_opts) * (len(selected_opts) - 1) // 2)
        L = max(1, len(next(iter(states)).start_sequence)) if states else 1
        changed_total = 0

        for lid in sorted(state_by_id):
            state = state_by_id[lid]
            best = best_binding[lid]
            opt = selected_opts[lid]
            chosen = bindings[lid][(int(opt.pool_key), int(opt.local_index))]
            changed = int(
                (best.option.pool_key, best.option.local_index)
                != (chosen.option.pool_key, chosen.option.local_index)
            )
            changed_total += changed
            self._population_exact_tiebreak_changes += changed
            sacrifice = float(best.exact_utility - chosen.exact_utility)
            if sacrifice > float(self._exact_tie_epsilon()) + 1e-10:
                raise RuntimeError("exact acceptance sacrifice exceeds configured tie epsilon")

            accepted = False
            accepted_k = -1
            accepted_seq = ""
            if (
                not chosen.is_noop
                and float(chosen.exact_utility - state.start_utility)
                > float(self.args.accept_epsilon)
            ):
                if chosen.pool is None or int(chosen.local_index) <= 0:
                    raise RuntimeError("non-noop exact acceptance missing scale binding")
                action = chosen.pool.actions[int(chosen.local_index) - 1]
                entry = self.archive_by_sequence.get(action.target_sequence)
                if entry is None:
                    raise RuntimeError("joint exact acceptance selected action without archive entry")
                accepted = self._accept_exact_entry(
                    state.lineage,
                    action.as_fine_action(),
                    entry,
                    turn_id=int(state.turn_id),
                )
                if accepted:
                    self._multiscale_accepted_moves += 1
                    accepted_k = int(chosen.pool.k)
                    accepted_seq = str(action.target_sequence)
                    self._mark_population_accept(state.turn_id, accepted_seq)

            query_stop_reason = str(state.reason)
            self.population_acceptance_history.append(
                {
                    "turn_id": int(state.turn_id),
                    "cycle": int(state.cycle),
                    "preference_id": int(state.preference_id),
                    "lineage_id": int(lid),
                    "total_decision_epsilon": float(self.args.pdo_multiscale_epsilon_dec),
                    "pdo_query_epsilon": float(self._pdo_query_epsilon()),
                    "exact_tie_epsilon": float(self._exact_tie_epsilon()),
                    "empirical_unqueried_action_certification": 0,
                    "exact_near_optimal_count": int(len(alternatives[lid])),
                    "best_exact_sequence": str(best.option.sequence),
                    "best_exact_scale_k": int(best.option.scale_k),
                    "best_exact_utility": float(best.exact_utility),
                    "selected_sequence": str(chosen.option.sequence),
                    "selected_scale_k": int(chosen.option.scale_k),
                    "selected_exact_utility": float(chosen.exact_utility),
                    "selected_noop": int(chosen.is_noop),
                    "exact_utility_sacrifice": float(sacrifice),
                    "acceptance_changed_by_raw_contraction": int(changed),
                    "population_raw_contraction_exact_best": float(raw0),
                    "population_raw_contraction_exact_selected": float(raw1),
                    "population_raw_contraction_reduction": float(raw0 - raw1),
                    "query_stop_reason": query_stop_reason,
                    "accepted": int(accepted),
                }
            )

            exact_scales = sum(
                any(int(i) != 0 for i in self._candidate_exact_map(pool, state.lineage))
                for pool in state.pools
            )
            self.multiscale_turn_history.append(
                {
                    "turn_id": int(state.turn_id),
                    "cycle": int(state.cycle),
                    "preference_id": int(state.preference_id),
                    "lineage_id": int(state.lineage_id),
                    "incumbent_sequence": str(state.start_sequence),
                    "incumbent_utility_before": float(state.start_utility),
                    "incumbent_utility_after": float(state.lineage.utility),
                    "configured_radii": ";".join(str(p.k) for p in state.pools),
                    "scale_count": int(len(state.pools)),
                    "scales_with_exact_candidate": int(exact_scales),
                    "candidate_count_total": int(sum(len(p.actions) for p in state.pools)),
                    "contender_count_total": int(sum(len(p.contender_indices) for p in state.pools)),
                    "fresh_query_cap": int(state.fresh_cap),
                    "paid_queries": int(self._fresh_used(state)),
                    "accepted": int(accepted),
                    "accepted_scale_k": int(accepted_k),
                    "accepted_sequence": str(accepted_seq),
                    "gain": float(state.lineage.utility - state.start_utility),
                    "reason": "joint_exact_queried_set_acceptance",
                    "query_stop_reason": query_stop_reason,
                    "active_scale_rounds": ";".join(
                        f"H{k}:{v}" for k, v in sorted(state.active_scale_rounds.items())
                    ),
                    "queried_by_scale": ";".join(
                        f"H{k}:{v}" for k, v in sorted(state.queried_by_scale.items())
                    ),
                    "unresolved_scales_at_cap": ";".join(
                        str(k) for k in state.unresolved_scales_at_cap
                    ),
                    "exact_near_optimal_count": int(len(alternatives[lid])),
                    "exact_acceptance_utility_sacrifice": float(sacrifice),
                    "empirical_unqueried_action_certification": 0,
                }
            )

        self.joint_population_diversity_history.append(
            {
                "cycle": int(next(iter(states)).cycle) if states else -1,
                "preference_id": int(next(iter(states)).preference_id) if states else -1,
                "iteration": int(iteration),
                "lineage_count": int(len(selected_opts)),
                "exact_queried_tuple": 1,
                "primary_raw_contraction": float(raw0),
                "selected_raw_contraction": float(raw1),
                "raw_contraction_reduction": float(raw0 - raw1),
                "exact_proposal_pairwise_hamming_improvement_vs_exact_best": float(
                    (raw0 - raw1) / (pair_count * L)
                ),
                "lineage_choices_changed_for_diversity": int(changed_total),
                "population_mode": str(self.args.pdo_population_mode),
            }
        )

    # -------------------------------------------------------------------------
    # Population fine loop
    # -------------------------------------------------------------------------
    def _run_population_fine(self, states: list[protected.PopulationFineState]) -> None:
        """Protected anchors -> shared exact-leader challengers -> exact population choice."""
        if not states:
            return
        self._run_protected_anchors(states)
        if self.stop_requested:
            for s in states:
                if s.reason == "prepared":
                    s.reason = "global_budget_after_protected_anchor"
            self._joint_exact_acceptance_v2(states, iteration=0)
            return

        max_iterations = 4 + sum(max(0, int(s.fresh_cap)) for s in states)
        iteration = 0
        while not self.stop_requested:
            views = self._build_joint_views(states)
            self._log_joint_state(
                views,
                iteration=iteration,
                selection_reason="exact_leader_challenger_recompute",
            )

            active = 0
            for lid, view in views.items():
                state = view.state
                if state.done:
                    continue
                if not view.protected_exposure and len(view.by_unified_index) > 0:
                    # This can occur only if the cap/budget prevented the protected anchor.
                    state.reason = "protected_anchor_unavailable"
                    state.done = True
                    continue
                if view.empirical_resolved:
                    state.reason = "no_plausible_exact_leader_challenger"
                    state.done = True
                    self._multiscale_early_stops += 1
                    continue
                if self._fresh_used(state) >= int(state.fresh_cap):
                    state.unresolved_scales_at_cap = tuple(
                        sorted({int(b.pool.k) for b in view.plausible_nonnoop})
                    )
                    state.reason = "fine_query_cap_reached_with_plausible_challengers"
                    state.done = True
                    continue
                for k in sorted({int(b.pool.k) for b in view.plausible_nonnoop}):
                    state.active_scale_rounds[k] = int(state.active_scale_rounds.get(k, 0) + 1)
                active += 1

            if active == 0:
                self._joint_exact_acceptance_v2(states, iteration=iteration)
                return

            # Views were built before marking resolved/capped states.  Rebuild so rank
            # accounting and acquisition contain only truly active lineages.
            views = self._build_joint_views(states)
            chosen, meta = self._choose_joint_rank_query(views, states)
            if chosen is None:
                # No candidate is silently pruned.  If an active lineage remains but no
                # representative can be formed, terminate safely on exact evidence.
                for s in states:
                    if not s.done:
                        s.reason = "no_queryable_exact_leader_challenger"
                        s.done = True
                self._joint_exact_acceptance_v2(states, iteration=iteration)
                return

            gains = meta.get("per_lineage_rank_gain", {})
            chosen.state.query_order += 1
            _entry, fresh = self._query_joint_binding(
                chosen,
                phase="joint_rank_acquisition",
                iteration=iteration,
                shared_rank_gain_value=int(meta.get("shared_rank_gain", 0)),
                per_lineage_rank_gain=gains if isinstance(gains, Mapping) else {},
                query_contraction=float(meta.get("query_contraction_tiebreak", 0.0)),
            )
            self._population_rounds += 1
            self.population_round_history.append(
                {
                    "cycle": int(chosen.state.cycle),
                    "preference_id": int(chosen.state.preference_id),
                    "round_index": int(iteration),
                    "allocation_mode": "joint_exact_leader_shared_rank_one_query",
                    "active_lineages_queried": 1,
                    "fresh_queries": int(fresh),
                    "cached_queries": int(not fresh),
                    "global_unique_queries_added": int(fresh),
                    "shared_rank_gain": int(meta.get("shared_rank_gain", 0)),
                    "gain_zero_fallback": int(meta.get("gain_zero_fallback", 0)),
                    "sum_lineage_unresolved_dimension_before": int(
                        meta.get("rank_stats", {}).get("sum_lineage_dimension", 0)
                    ),
                    "joint_unresolved_dimension_before": int(
                        meta.get("rank_stats", {}).get("joint_dimension", 0)
                    ),
                    "sharing_factor_before": float(
                        meta.get("rank_stats", {}).get("sharing_factor", 1.0)
                    ),
                    "query_contraction_tiebreak": float(
                        meta.get("query_contraction_tiebreak", 0.0)
                    ),
                    "total_lineages": int(len(states)),
                }
            )
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("joint population v2 exceeded deterministic fine-query bound")

        for s in states:
            if not s.done:
                s.reason = "global_budget_exhausted"
                s.done = True
        self._joint_exact_acceptance_v2(states, iteration=iteration)

    def finalize(self) -> dict[str, Any]:
        # Reuse all inherited base metrics, then overwrite every v1-specific semantic field.
        summary = super().finalize()
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "production v3 coarse reachability/KFM look-ahead/exploit-first exact coarse verification unchanged",
            "same finite realized multiscale Hamming action families as frozen protected Population PDO",
            "one protected exact non-noop fine exposure per active lineage",
            "current exact fine leader is no-op plus all exactly evaluated multiscale candidates",
            "unqueried actions remain active only if empirical utility upper can beat the exact leader by the PDO query epsilon",
            "one exploit-first representative per active lineage-scale enters population-wide acquisition",
            "one exact population-wide challenger is purchased at a time, prioritizing shared unresolved-rank reduction",
            "rank-zero candidates are not pruned; exploit-first value fallback preserves opportunity under approximate geometry",
            "every paid label immediately refits the shared global readout and all lineage decisions are recomputed",
            "there is no empirical certification or verification loop for unseen final actions",
            "final population choice uses only exact no-op-inclusive near-optimal queried sets",
            "signed raw reachable contraction is coordinate-minimized only inside those exact near-optimal sets",
            "exact oracle remains sole acceptance authority",
        ]

        sharing = [
            float(r.get("sharing_factor", np.nan))
            for r in self.joint_population_pdo_history
            if np.isfinite(float(r.get("sharing_factor", np.nan)))
            and int(r.get("joint_unresolved_dimension", 0)) > 0
        ]
        rank_rows = [
            r for r in self.joint_population_query_history
            if str(r.get("phase", "")) == "joint_rank_acquisition"
            and int(r.get("fresh_oracle_query", 0)) == 1
        ]
        gains = [int(r.get("shared_rank_gain", 0)) for r in rank_rows]
        div = [
            float(r.get("exact_proposal_pairwise_hamming_improvement_vs_exact_best", 0.0))
            for r in self.joint_population_diversity_history
        ]
        jp = summary.setdefault("joint_population_pdo", {})
        jp.update(
            {
                "formal_certificate_claimed": False,
                "empirical_unqueried_action_certification": False,
                "shared_response_geometry": "current-cycle KFM terminal displacement directions over exact-leader-plausible realized actions",
                "historical_labels_reduce_formal_style_rank": False,
                "decision_stop_rule": "no unqueried contender has empirical utility_upper > current exact leader + PDO query epsilon, or explicit fine cap",
                "protected_exact_exposure": True,
                "query_scheduler": "one population-wide exploit-first lineage-scale representative at a time; shared residual decision rank is the production acquisition priority",
                "acquisition_mode": str(self.args.pdo_joint_acquisition_mode),
                "rank_rtol": float(self._joint_rank_rtol()),
                "rank_acquisition_fresh_queries": int(self._joint_rank_queries),
                "exact_population_acceptance_cycles": int(len(self.joint_population_diversity_history)),
                "certified_verification_fresh_queries": 0,
                "verification_contradictions": 0,
                "fallback_population_cycles": 0,
                "mean_sharing_factor_when_unresolved": float(np.mean(sharing)) if sharing else 1.0,
                "max_sharing_factor_when_unresolved": float(np.max(sharing)) if sharing else 1.0,
                "mean_lineages_helped_per_rank_query": float(np.mean(gains)) if gains else 0.0,
                "max_lineages_helped_by_one_rank_query": int(max(gains)) if gains else 0,
                "gain_zero_fallback_queries": int(sum(int(r.get("shared_rank_gain", 0)) <= 0 for r in rank_rows)),
                "mean_exact_proposal_diversity_improvement_vs_exact_best": float(np.mean(div)) if div else 0.0,
                "diversity_rule": "coordinate-minimize signed raw reachable contraction inside exact queried near-optimal sets",
                "scale_handling": "all configured scales share one decision span; no scale selector or scale-specific query budget",
                "acceptance_authority": "exact oracle only; no unseen action is selectable",
            }
        )
        if "population_pdo" in summary:
            summary["population_pdo"].update(
                {
                    "query_scheduler": "joint exact-leader shared-rank one-query-at-a-time scheduler",
                    "exact_acceptance": "exact queried near-optimal sets -> signed raw contraction coordination -> exact monotone acceptance",
                    "total_decision_epsilon": float(self.args.pdo_multiscale_epsilon_dec),
                    "pdo_query_epsilon": float(self._pdo_query_epsilon()),
                    "exact_tie_epsilon": float(self._exact_tie_epsilon()),
                    "formal_certificate_claimed": False,
                }
            )
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] coarse stage: unchanged production v3\n"
            f"[{METHOD_NAME}] fine: protected anchor -> exact-leader plausible challengers -> one shared-rank query -> exact queried-set diversity acceptance\n"
            f"[{METHOD_NAME}] NO empirical unseen-action certification; NO certification verification loop; exact oracle alone authorizes acceptance",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = v1.build_parser()
    p.description = __doc__
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    v1._validate_args(args)
    JointPopulationPDORunnerV2(args).run()


if __name__ == "__main__":
    main()
