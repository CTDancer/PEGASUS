"""PEGASUS joint Population PDO: shared decision observability across lineages and scales.

This implementation changes only the fine population decision stage of the frozen protected
multiscale PEGASUS/PEGASUS optimizer.  The production v3 coarse stage, physical LKF
reachability, KFM terminal coordinates, objective oracle, protected exploit-first exposure,
and exact acceptance authority are unchanged.

Fine-stage design
-----------------
1. Realize the same finite multiscale Hamming action families for every lineage.
2. Preserve one protected exploit-first exact fine exposure per active lineage before an
   empirical committee may certify that lineage.
3. Build ONE no-op-inclusive multiscale committee per lineage from the shared KFM task
   readout.  PDO uses the union/intersection of epsilon-good actions across coherent
   committee members, not a predicted latent scale choice.
4. Treat current fine response directions from all lineages/scales in one shared decision
   geometry.  A population-wide acquisition queries one PDO-relevant candidate at a time,
   choosing the direction that reduces the unresolved decision dimension of the largest
   number of lineages.  Scale labels never enter the rank rule.
5. When every protected lineage has a nonempty empirical robust PDO set, choose one action
   per lineage from those sets by coordinate-minimizing SIGNED reachable contraction.  This
   is exactly a monotone one-step pairwise-Hamming diversity improvement relative to the
   independent robust-primary tuple; no diversity reward/weight is introduced.
6. Any selected unqueried action is exact-verified.  A verification contradiction updates
   the shared archive and the entire population PDO problem is recomputed before acceptance.
7. Exact oracle values alone authorize accepted moves.  If empirical joint PDO cannot be
   resolved within the existing per-lineage fine caps, the implementation falls back to the
   frozen protected exact queried-set acceptance rather than making an unsupported claim.

The joint-rank accounting intentionally uses only exact fine-response directions available
in the CURRENT population cycle.  Historical discovery/coarse labels still train the
existing empirical KFM readout, but they are not allowed to make the formal-style joint
rank diagnostic look solved without validated cross-context response transfer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_multiscale_population_pdo_protected as protected
from . import pdo_pegasus_multiscale_tournament as ms
from .kfocus.objectives import OPT_NAMES, RAW_NAMES
from .pdo_contraction_allocation import AllocationOption, normalized_reachable_contraction
from .pdo_joint_population import (
    EmpiricalPDOSets,
    build_fast_shared_rank_context,
    coordinate_minimize_raw_contraction,
    empirical_pdo_sets,
    fast_shared_rank_gain,
    feature_scale,
    joint_span_statistics,
    scale_unique_rank_contributions,
)
from .pdo_pegasus_v3 import GlobalCoarseCommittee
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-joint-population-pdo-v1"
METHOD_NAME = "PEGASUS Joint Population PDO"


@dataclass
class JointBinding:
    state: protected.PopulationFineState
    pool: ms.ScalePool
    pool_key: int
    local_index: int
    unified_index: int
    delta_z: np.ndarray
    option: AllocationOption

    @property
    def key(self) -> tuple[int, int]:
        return int(self.pool_key), int(self.local_index)


@dataclass
class JointLineageView:
    state: protected.PopulationFineState
    committee: GlobalCoarseCommittee | None
    pdo_sets: EmpiricalPDOSets
    by_unified_index: dict[int, JointBinding]
    by_key: dict[tuple[int, int], JointBinding]
    exact_indices: set[int]
    protected_exposure: bool

    @property
    def empirical_resolved(self) -> bool:
        return bool(self.protected_exposure and self.pdo_sets.resolved)

    @property
    def robust_nonnoop(self) -> list[JointBinding]:
        return [
            self.by_unified_index[int(i)]
            for i in self.pdo_sets.robust_indices
            if int(i) > 0 and int(i) in self.by_unified_index
        ]

    @property
    def plausible_nonnoop(self) -> list[JointBinding]:
        return [
            self.by_unified_index[int(i)]
            for i in self.pdo_sets.plausible_indices
            if int(i) > 0 and int(i) in self.by_unified_index
        ]


class JointPopulationPDORunner(protected.PopulationPDORunner):
    """Protected multiscale PEGASUS with one shared population fine-PDO acquisition loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.joint_population_pdo_history: list[dict[str, Any]] = []
        self.joint_population_query_history: list[dict[str, Any]] = []
        self.joint_population_diversity_history: list[dict[str, Any]] = []
        self.joint_population_verification_history: list[dict[str, Any]] = []
        self.joint_population_scale_rank_history: list[dict[str, Any]] = []
        self._joint_rank_queries = 0
        self._joint_verification_queries = 0
        self._joint_fallback_cycles = 0
        self._joint_certified_cycles = 0
        self._joint_verification_contradictions = 0
        self._pending_verification: dict[str, Any] | None = None

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.joint_population_pdo_history:
            write_csv(self.out / "joint_population_pdo_history.csv", self.joint_population_pdo_history)
        if self.joint_population_query_history:
            write_csv(self.out / "joint_population_query_history.csv", self.joint_population_query_history)
        if self.joint_population_diversity_history:
            write_csv(self.out / "joint_population_diversity_history.csv", self.joint_population_diversity_history)
        if self.joint_population_verification_history:
            write_csv(self.out / "joint_population_verification_history.csv", self.joint_population_verification_history)
        if self.joint_population_scale_rank_history:
            write_csv(self.out / "joint_population_scale_rank_history.csv", self.joint_population_scale_rank_history)

    def _joint_epsilon(self) -> float:
        return float(self.args.pdo_multiscale_epsilon_dec)

    def _joint_rank_rtol(self) -> float:
        return float(self.args.pdo_joint_rank_rtol)

    def _noop_option(self, state: protected.PopulationFineState) -> AllocationOption:
        u = float(state.start_utility)
        return AllocationOption(
            lineage_id=int(state.lineage_id),
            preference_id=int(state.preference_id),
            sequence=str(state.start_sequence),
            incumbent_sequence=str(state.start_sequence),
            scale_k=0,
            predicted_utility=u,
            predicted_upper=u,
            pool_key=-1,
            local_index=0,
            initial_rank=0,
        )

    def _build_joint_view(self, state: protected.PopulationFineState) -> JointLineageView:
        """One no-op-inclusive multiscale PDO committee for a lineage.

        Only the frozen contender prefix from each realized scale is included.  This keeps
        the empirical decision family exactly aligned with the existing multiscale design.
        """
        rows: list[tuple[int, ms.ScalePool, int]] = []
        for pool_key, pool in enumerate(state.pools):
            for local in sorted(set(int(i) for i in pool.contender_indices.tolist())):
                if local <= 0 or local > len(pool.actions):
                    raise RuntimeError("invalid frozen contender index")
                rows.append((int(pool_key), pool, int(local)))

        protected_exposure = bool(self._has_exact_fine_exposure(state))
        if not rows:
            # No non-noop fine decision exists.  The protected-exposure rule applies only
            # to active lineages with at least one non-noop candidate, so no-op is exactly
            # resolved here without forcing a meaningless oracle query.
            sets = EmpiricalPDOSets(
                plausible_indices=np.asarray([0], dtype=np.int64),
                robust_indices=np.asarray([0], dtype=np.int64),
                epsilon=self._joint_epsilon(),
            )
            return JointLineageView(state, None, sets, {}, {}, {0}, True)

        candidate_z = np.stack([pool.z[local - 1] for _, pool, local in rows], axis=0)
        exact_map: dict[int, np.ndarray] = {0: np.asarray(state.lineage.scores, dtype=np.float64).copy()}
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
        sets = empirical_pdo_sets(committee.member_utilities, self._joint_epsilon())
        by_uid: dict[int, JointBinding] = {}
        by_key: dict[tuple[int, int], JointBinding] = {}
        for uid, (pool_key, pool, local) in enumerate(rows, start=1):
            action = pool.actions[local - 1]
            binding = JointBinding(
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
        return JointLineageView(
            state=state,
            committee=committee,
            pdo_sets=sets,
            by_unified_index=by_uid,
            by_key=by_key,
            exact_indices=exact_indices,
            protected_exposure=protected_exposure,
        )

    def _build_joint_views(self, states: Sequence[protected.PopulationFineState]) -> dict[int, JointLineageView]:
        return {int(s.lineage_id): self._build_joint_view(s) for s in states}

    def _observed_fine_directions(self, views: Mapping[int, JointLineageView]) -> np.ndarray:
        """Current-cycle exact fine response directions only (not historical archive rank)."""
        rows: list[np.ndarray] = []
        dim = 0
        for view in views.values():
            for uid, binding in view.by_unified_index.items():
                dim = len(binding.delta_z)
                if int(uid) in view.exact_indices:
                    rows.append(np.asarray(binding.delta_z, dtype=np.float64))
        if rows:
            return np.stack(rows, axis=0)
        if dim <= 0:
            dim = int(next(iter(views.values())).state.incumbent_z.shape[0]) if views else 0
        return np.zeros((0, dim), dtype=np.float64)

    def _decision_rows(self, view: JointLineageView) -> np.ndarray:
        rows = [np.asarray(b.delta_z, dtype=np.float64) for b in view.plausible_nonnoop]
        dim = int(view.state.incumbent_z.shape[0])
        return np.stack(rows, axis=0) if rows else np.zeros((0, dim), dtype=np.float64)

    def _rank_context(
        self,
        views: Mapping[int, JointLineageView],
        *,
        include_resolved: bool = False,
    ) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, dict[str, object], dict[int, int]]:
        observed = self._observed_fine_directions(views)
        decision: dict[int, np.ndarray] = {}
        scale_rows: dict[int, list[np.ndarray]] = {}
        all_for_scale: list[np.ndarray] = [r for r in observed]
        for lid, view in views.items():
            if (not include_resolved) and view.empirical_resolved:
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
            np.stack(all_for_scale, axis=0) if all_for_scale else np.zeros((0, dim), dtype=np.float64)
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

    def _query_contraction_tiebreak(self, binding: JointBinding, states: Sequence[protected.PopulationFineState]) -> float:
        total = 0.0
        for other in states:
            if int(other.lineage_id) == int(binding.state.lineage_id):
                continue
            total += normalized_reachable_contraction(
                str(binding.state.start_sequence),
                str(other.start_sequence),
                str(binding.option.sequence),
                str(other.start_sequence),
            )
        return float(total)

    def _choose_joint_rank_query(
        self,
        views: Mapping[int, JointLineageView],
        states: Sequence[protected.PopulationFineState],
    ) -> tuple[JointBinding | None, dict[str, Any]]:
        observed, decision, scale, stats, _unique = self._rank_context(views, include_resolved=False)
        fast_context = build_fast_shared_rank_context(
            observed,
            decision,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        )
        candidates: list[tuple[tuple[Any, ...], JointBinding, dict[int, int], int, float]] = []
        state_by_id = {int(s.lineage_id): s for s in states}
        for lid, view in views.items():
            if view.empirical_resolved:
                continue
            state = state_by_id[int(lid)]
            if self._fresh_used(state) >= int(state.fresh_cap):
                continue
            for b in view.plausible_nonnoop:
                if int(b.unified_index) in view.exact_indices:
                    continue
                gain, gains = fast_shared_rank_gain(fast_context, b.delta_z)
                if gain <= 0:
                    continue
                contraction = (
                    self._query_contraction_tiebreak(b, states)
                    if str(self.args.pdo_population_mode) == "contraction"
                    else 0.0
                )
                acquisition_gain_key = int(gain) if str(self.args.pdo_joint_acquisition_mode) == "shared_rank" else 0
                key = (
                    -int(acquisition_gain_key),
                    -float(b.option.predicted_utility),
                    -float(b.option.predicted_upper),
                    float(contraction),
                    int(b.option.initial_rank),
                    int(b.option.scale_k),
                    int(b.option.lineage_id),
                    str(b.option.sequence),
                )
                candidates.append((key, b, gains, int(gain), float(contraction)))
        if not candidates:
            return None, {"rank_stats": stats, "eligible_query_count": 0}
        candidates.sort(key=lambda x: x[0])
        _key, chosen, gains, gain, contraction = candidates[0]
        return chosen, {
            "rank_stats": stats,
            "eligible_query_count": int(len(candidates)),
            "shared_rank_gain": int(gain),
            "per_lineage_rank_gain": gains,
            "query_contraction_tiebreak": float(contraction),
        }

    def _query_joint_binding(
        self,
        binding: JointBinding,
        *,
        phase: str,
        iteration: int,
        shared_rank_gain_value: int = 0,
        per_lineage_rank_gain: Mapping[int, int] | None = None,
        query_contraction: float = 0.0,
    ) -> tuple[base.ArchiveEntry | None, bool]:
        state = binding.state
        pool = binding.pool
        idx = int(binding.local_index)
        action = pool.actions[idx - 1]
        existing = self.archive_by_sequence.get(action.target_sequence)
        before = int(self.oracle.unique_oracle_queries)
        if existing is not None:
            entry = existing
        else:
            if self._budget_remaining() == 0:
                self.stop_requested = True
                return None, False
            rec = self.oracle.evaluate_one(action.target_sequence)
            exact_u = float(base._utility(rec.scores, self.preferences[int(state.preference_id)], self.args.rho))
            entry = self._record_exact(
                action.tokens,
                pool.z[idx - 1],
                rec,
                source="pegasus_joint_population_pdo",
                preference_id=int(state.preference_id),
                lineage_id=int(state.lineage_id),
                slate_id=None,
                verification_rank=int(state.query_order),
                utility_at_query=exact_u,
            )
        after = int(self.oracle.unique_oracle_queries)
        fresh = after > before
        if fresh:
            delta = int(after - before)
            self._multiscale_fresh_queries += delta
            state.lineage.verification_queries += delta
            state.fine_fresh_queries += delta
            if phase == "joint_rank_acquisition":
                self._joint_rank_queries += delta
                self._population_challenger_fresh_queries += delta
                self._multiscale_challenger_queries += delta
            elif phase == "joint_certified_verification":
                self._joint_verification_queries += delta
            elif phase == "joint_protected_anchor":
                self._population_initial_fresh_queries += delta
        state.queried_by_scale[int(pool.k)] = int(state.queried_by_scale.get(int(pool.k), 0) + 1)
        self._record_preference_query_use(state.preference_id, action.target_sequence)
        exact_u = float(base._utility(entry.scores, self.preferences[int(state.preference_id)], self.args.rho))
        row: dict[str, Any] = {
            "turn_id": int(state.turn_id),
            "cycle": int(state.cycle),
            "preference_id": int(state.preference_id),
            "lineage_id": int(state.lineage_id),
            "incumbent_sequence": str(state.start_sequence),
            "query_order": int(state.query_order),
            "round_index": int(iteration),
            "phase": str(phase),
            "scale_k": int(pool.k),
            "action_id": str(action.action_id),
            "candidate_sequence": str(action.target_sequence),
            "fresh_oracle_query": int(fresh),
            "global_unique_query_index": int(self.oracle.unique_oracle_queries),
            "predicted_median_utility": float(binding.option.predicted_utility),
            "predicted_utility_upper": float(binding.option.predicted_upper),
            "exact_utility": float(exact_u),
            "exact_gain": float(exact_u - float(state.start_utility)),
            "accepted": 0,
            "shared_rank_gain": int(shared_rank_gain_value),
            "lineages_helped": ";".join(str(k) for k, v in sorted((per_lineage_rank_gain or {}).items()) if int(v) > 0),
            "query_contraction_tiebreak": float(query_contraction),
        }
        for j, name in enumerate(OPT_NAMES):
            row[f"exact_score_{name}"] = float(entry.scores[j])
        for j, name in enumerate(RAW_NAMES):
            row[f"raw_{name}"] = float(entry.raw[j])
        self.multiscale_query_history.append(row)
        self.joint_population_query_history.append(dict(row))
        return entry, fresh

    def _run_protected_anchors(self, states: Sequence[protected.PopulationFineState]) -> None:
        """Freeze all required protected anchors before purchasing any of them."""
        frozen: list[tuple[protected.PopulationFineState, protected.ChoiceBinding]] = []
        for state in states:
            if state.done or self._has_exact_fine_exposure(state):
                continue
            if self._fresh_used(state) >= int(state.fresh_cap):
                state.reason = "protected_anchor_unavailable_at_fine_cap"
                continue
            b = self._protected_first_representative(state)
            if b is None:
                state.reason = "no_unqueried_multiscale_candidate_for_protected_anchor"
                continue
            frozen.append((state, b))

        for state, old_binding in frozen:
            if self.stop_requested or self._fresh_used(state) >= int(state.fresh_cap):
                continue
            # Rebind into the current unified lineage view so logging uses the same common
            # multiscale committee as later joint PDO.
            view = self._build_joint_view(state)
            jb = view.by_key.get((int(old_binding.option.pool_key), int(old_binding.local_index)))
            if jb is None:
                # Protected candidate can be outside the frozen contender prefix only in a
                # malformed pool; fail loudly rather than silently changing the safeguard.
                raise RuntimeError("protected anchor missing from unified frozen contender family")
            state.query_order += 1
            self._query_joint_binding(jb, phase="joint_protected_anchor", iteration=0)

    def _log_joint_state(
        self,
        views: Mapping[int, JointLineageView],
        *,
        iteration: int,
        selection_reason: str,
    ) -> dict[str, object]:
        observed, decision, scale, stats, unique = self._rank_context(views, include_resolved=False)
        row: dict[str, Any] = {
            "cycle": int(next(iter(views.values())).state.cycle) if views else -1,
            "preference_id": int(next(iter(views.values())).state.preference_id) if views else -1,
            "iteration": int(iteration),
            "selection_reason": str(selection_reason),
            "lineage_count": int(len(views)),
            "protected_lineages": int(sum(v.protected_exposure for v in views.values())),
            "empirically_resolved_lineages": int(sum(v.empirical_resolved for v in views.values())),
            "unresolved_lineages": int(sum(not v.empirical_resolved for v in views.values())),
            "observed_fine_direction_count": int(observed.shape[0]),
            "observed_fine_direction_rank": int(
                joint_span_statistics(
                    np.zeros((0, observed.shape[1]), dtype=np.float64),
                    {0: observed},
                    rtol=self._joint_rank_rtol(),
                    scale=scale if len(scale) else None,
                )["joint_dimension"]
            ) if observed.shape[1] else 0,
            "sum_lineage_unresolved_dimension": int(stats["sum_lineage_dimension"]),
            "joint_unresolved_dimension": int(stats["joint_dimension"]),
            "sharing_factor": float(stats["sharing_factor"]),
            "plausible_action_count": int(sum(len(v.plausible_nonnoop) for v in views.values() if not v.empirical_resolved)),
            "robust_action_count": int(sum(len(v.pdo_sets.robust_indices) for v in views.values())),
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

    def _close_pending_verification(self, views: Mapping[int, JointLineageView]) -> None:
        pending = self._pending_verification
        if pending is None:
            return
        lid = int(pending["lineage_id"])
        view = views.get(lid)
        remains = False
        exact_u = float("nan")
        if view is not None:
            b = view.by_key.get((int(pending["pool_key"]), int(pending["local_index"])))
            if b is not None:
                remains = int(b.unified_index) in set(int(i) for i in view.pdo_sets.robust_indices.tolist())
                entry = self.archive_by_sequence.get(str(b.option.sequence))
                if entry is not None:
                    exact_u = float(base._utility(entry.scores, self.preferences[int(view.state.preference_id)], self.args.rho))
        contradicted = int(not remains)
        self._joint_verification_contradictions += contradicted
        row = dict(pending)
        row.update(
            {
                "exact_utility": float(exact_u),
                "remains_empirically_robust_after_exact": int(remains),
                "verification_contradicted_empirical_robust_set": int(contradicted),
                "all_lineages_resolved_after_verification": int(all(v.empirical_resolved for v in views.values())),
            }
        )
        self.joint_population_verification_history.append(row)
        self._pending_verification = None

    def _certified_options(
        self,
        view: JointLineageView,
    ) -> tuple[list[AllocationOption], dict[tuple[int, int], JointBinding | None]]:
        if not view.empirical_resolved:
            return [], {}
        options: list[AllocationOption] = []
        lookup: dict[tuple[int, int], JointBinding | None] = {}
        robust = set(int(i) for i in view.pdo_sets.robust_indices.tolist())
        if 0 in robust:
            noop = self._noop_option(view.state)
            options.append(noop)
            lookup[(-1, 0)] = None
        for uid in sorted(i for i in robust if i > 0):
            b = view.by_unified_index[int(uid)]
            options.append(b.option)
            lookup[b.key] = b
        if not options:
            raise RuntimeError("resolved PDO view has an empty robust option set")
        # Primary tuple = independent robust point-estimate choice.  Diversity coordination
        # is guaranteed not to reduce exact one-step pairwise Hamming relative to this tuple.
        options.sort(
            key=lambda o: (
                -float(o.predicted_utility),
                -float(o.predicted_upper),
                int(o.initial_rank),
                int(o.scale_k),
                str(o.sequence),
            )
        )
        return options, lookup

    def _select_certified_population(
        self,
        views: Mapping[int, JointLineageView],
        *,
        iteration: int,
    ) -> tuple[dict[int, AllocationOption], dict[int, JointBinding | None], float, float] | None:
        alternatives: dict[int, list[AllocationOption]] = {}
        bindings: dict[int, dict[tuple[int, int], JointBinding | None]] = {}
        for lid, view in views.items():
            opts, look = self._certified_options(view)
            if not opts:
                return None
            alternatives[int(lid)] = opts
            bindings[int(lid)] = look
        selected, cost0, cost1 = coordinate_minimize_raw_contraction(
            alternatives,
            passes=int(self.args.pdo_population_coordinate_passes),
            use_diversity=(str(self.args.pdo_population_mode) == "contraction"),
        )
        selected_bindings: dict[int, JointBinding | None] = {}
        changes = 0
        for lid, opt in selected.items():
            primary = alternatives[lid][0]
            changes += int((primary.pool_key, primary.local_index) != (opt.pool_key, opt.local_index))
            selected_bindings[int(lid)] = bindings[lid][(int(opt.pool_key), int(opt.local_index))]
        self._population_exact_tiebreak_changes += int(changes)
        pair_count = max(1, len(selected) * (len(selected) - 1) // 2)
        L = max(1, len(next(iter(views.values())).state.start_sequence)) if views else 1
        self.joint_population_diversity_history.append(
            {
                "cycle": int(next(iter(views.values())).state.cycle) if views else -1,
                "preference_id": int(next(iter(views.values())).state.preference_id) if views else -1,
                "iteration": int(iteration),
                "lineage_count": int(len(selected)),
                "certified_tuple": 1,
                "primary_raw_contraction": float(cost0),
                "selected_raw_contraction": float(cost1),
                "raw_contraction_reduction": float(cost0 - cost1),
                "certified_proposal_pairwise_hamming_improvement_vs_primary": float((cost0 - cost1) / (pair_count * L)),
                "lineage_choices_changed_for_diversity": int(changes),
                "population_mode": str(self.args.pdo_population_mode),
            }
        )
        return selected, selected_bindings, float(cost0), float(cost1)

    def _choose_verification_binding(
        self,
        selected_bindings: Mapping[int, JointBinding | None],
        views: Mapping[int, JointLineageView],
    ) -> JointBinding | None:
        candidates = [b for b in selected_bindings.values() if b is not None and b.option.sequence not in self.archive_by_sequence]
        candidates = [b for b in candidates if self._fresh_used(b.state) < int(b.state.fresh_cap)]
        if not candidates:
            return None
        # Verification is mandatory for acceptance.  If several selected actions are unseen,
        # verify the one with the largest potential shared rank reuse first; a contradiction
        # can then prevent wasteful verification of the remaining frozen tuple.
        observed, decision, scale, _stats, _unique = self._rank_context(views, include_resolved=True)
        fast_context = build_fast_shared_rank_context(
            observed,
            decision,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        )
        scored: list[tuple[tuple[Any, ...], JointBinding, int, dict[int, int]]] = []
        for b in candidates:
            gain, gains = fast_shared_rank_gain(fast_context, b.delta_z)
            key = (
                -int(gain),
                -float(b.option.predicted_utility),
                -float(b.option.predicted_upper),
                int(b.option.initial_rank),
                int(b.option.scale_k),
                int(b.option.lineage_id),
                str(b.option.sequence),
            )
            scored.append((key, b, int(gain), gains))
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    def _accept_certified_population(
        self,
        states: Sequence[protected.PopulationFineState],
        views: Mapping[int, JointLineageView],
        selected: Mapping[int, AllocationOption],
        selected_bindings: Mapping[int, JointBinding | None],
        *,
        raw_cost_primary: float,
        raw_cost_selected: float,
    ) -> None:
        """Accept a frozen jointly certified tuple only after every selected action is exact."""
        for lid in sorted(selected):
            state = views[int(lid)].state
            opt = selected[int(lid)]
            binding = selected_bindings[int(lid)]
            if binding is None:
                exact_u = float(state.start_utility)
                is_noop = True
            else:
                entry = self.archive_by_sequence.get(str(opt.sequence))
                if entry is None:
                    raise RuntimeError("certified population acceptance received an unverified action")
                exact_u = float(base._utility(entry.scores, self.preferences[int(state.preference_id)], self.args.rho))
                is_noop = False

            exact_rows = self._exact_near_optimal_bindings(state)
            best_exact_u = float(exact_rows[0].exact_utility)
            sacrifice = float(best_exact_u - exact_u)
            # If a selected action remains in the empirical robust intersection after its
            # exact value is clamped, it must be within the TOTAL PDO epsilon of every exact
            # candidate already in the common action family.  Enforce this implementation
            # invariant; it catches index/set bugs without claiming committee coverage.
            if sacrifice > self._joint_epsilon() + 1e-8:
                raise RuntimeError("empirical robust selected action exceeds total PDO epsilon from best exact action")

            accepted = False
            accepted_k = -1
            accepted_seq = ""
            if (not is_noop) and float(exact_u - state.start_utility) > float(self.args.accept_epsilon):
                assert binding is not None
                action = binding.pool.actions[int(binding.local_index) - 1]
                entry = self.archive_by_sequence.get(action.target_sequence)
                if entry is None:
                    raise RuntimeError("verified selected action disappeared from archive")
                accepted = self._accept_exact_entry(
                    state.lineage,
                    action.as_fine_action(),
                    entry,
                    turn_id=int(state.turn_id),
                )
                if accepted:
                    self._multiscale_accepted_moves += 1
                    accepted_k = int(binding.pool.k)
                    accepted_seq = str(action.target_sequence)
                    self._mark_population_accept(state.turn_id, accepted_seq)

            state.reason = "joint_pdo_certified_exact_acceptance"
            self.population_acceptance_history.append(
                {
                    "turn_id": int(state.turn_id),
                    "cycle": int(state.cycle),
                    "preference_id": int(state.preference_id),
                    "lineage_id": int(lid),
                    "total_decision_epsilon": float(self._joint_epsilon()),
                    "pdo_query_epsilon": float(self._joint_epsilon()),
                    "exact_tie_epsilon": 0.0,
                    "empirical_joint_pdo_certified": 1,
                    "robust_set_size": int(len(views[int(lid)].pdo_sets.robust_indices)),
                    "best_exact_utility": float(best_exact_u),
                    "selected_sequence": str(opt.sequence),
                    "selected_scale_k": int(opt.scale_k),
                    "selected_exact_utility": float(exact_u),
                    "selected_noop": int(is_noop),
                    "exact_utility_sacrifice": float(sacrifice),
                    "population_raw_contraction_primary": float(raw_cost_primary),
                    "population_raw_contraction_selected": float(raw_cost_selected),
                    "population_raw_contraction_reduction": float(raw_cost_primary - raw_cost_selected),
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
                    "reason": str(state.reason),
                    "active_scale_rounds": ";".join(f"H{k}:{v}" for k, v in sorted(state.active_scale_rounds.items())),
                    "queried_by_scale": ";".join(f"H{k}:{v}" for k, v in sorted(state.queried_by_scale.items())),
                    "unresolved_scales_at_cap": ";".join(str(k) for k in state.unresolved_scales_at_cap),
                    "empirical_joint_pdo_certified": 1,
                    "robust_set_size": int(len(views[int(lid)].pdo_sets.robust_indices)),
                    "exact_acceptance_utility_sacrifice": float(sacrifice),
                }
            )

    def _fallback_exact_population(self, states: Sequence[protected.PopulationFineState], *, reason: str) -> None:
        self._joint_fallback_cycles += 1
        for state in states:
            if state.reason in {"prepared", "joint_pdo_running"}:
                state.reason = f"joint_pdo_fallback:{reason}"
        self._joint_exact_acceptance(states)

    def _run_population_fine(self, states: list[protected.PopulationFineState]) -> None:
        """Protected anchors -> one-query-at-a-time shared PDO -> certified diversity -> exact."""
        if not states:
            return
        self._run_protected_anchors(states)
        if self.stop_requested:
            self._fallback_exact_population(states, reason="global_budget_after_protected_anchor")
            return

        max_iterations = 4 + sum(max(0, int(s.fresh_cap)) for s in states) + 2 * len(states)
        iteration = 0
        while not self.stop_requested:
            views = self._build_joint_views(states)
            self._close_pending_verification(views)
            stats = self._log_joint_state(views, iteration=iteration, selection_reason="joint_pdo_recompute")

            all_resolved = all(v.empirical_resolved for v in views.values())
            if all_resolved:
                selection = self._select_certified_population(views, iteration=iteration)
                if selection is None:
                    self._fallback_exact_population(states, reason="certified_selection_empty")
                    return
                selected, selected_bindings, cost0, cost1 = selection
                unseen = [
                    b for b in selected_bindings.values()
                    if b is not None and str(b.option.sequence) not in self.archive_by_sequence
                ]
                if not unseen:
                    self._joint_certified_cycles += 1
                    self._accept_certified_population(
                        states,
                        views,
                        selected,
                        selected_bindings,
                        raw_cost_primary=cost0,
                        raw_cost_selected=cost1,
                    )
                    return

                b = self._choose_verification_binding(selected_bindings, views)
                if b is None:
                    self._fallback_exact_population(states, reason="certified_selected_action_unverifiable_at_fine_cap")
                    return
                view = views[int(b.state.lineage_id)]
                uid = int(b.unified_index)
                if view.committee is None:
                    raise RuntimeError("verification candidate missing committee")
                pre_lower = float(view.committee.utility_lower[uid])
                pre_upper = float(view.committee.utility_upper[uid])
                b.state.query_order += 1
                entry, fresh = self._query_joint_binding(
                    b,
                    phase="joint_certified_verification",
                    iteration=iteration,
                )
                if entry is None:
                    self._fallback_exact_population(states, reason="global_budget_during_certified_verification")
                    return
                self._pending_verification = {
                    "cycle": int(b.state.cycle),
                    "preference_id": int(b.state.preference_id),
                    "iteration": int(iteration),
                    "lineage_id": int(b.state.lineage_id),
                    "pool_key": int(b.pool_key),
                    "local_index": int(b.local_index),
                    "scale_k": int(b.pool.k),
                    "sequence": str(b.option.sequence),
                    "fresh_oracle_query": int(fresh),
                    "preverification_median_utility": float(b.option.predicted_utility),
                    "preverification_utility_lower": float(pre_lower),
                    "preverification_utility_upper": float(pre_upper),
                    "was_in_empirical_robust_set": 1,
                }
                iteration += 1
                if iteration > max_iterations:
                    raise RuntimeError("joint population PDO exceeded deterministic iteration bound")
                continue

            # At least one lineage remains empirically unresolved.  Purchase ONE population-
            # wide informative direction, immediately refit the shared archive, then let all
            # lineages reconsider their decision before another query is chosen.
            chosen, meta = self._choose_joint_rank_query(views, states)
            if chosen is None:
                self._fallback_exact_population(states, reason="no_informative_joint_pdo_query_within_caps")
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
                    "allocation_mode": "joint_shared_rank_one_query",
                    "active_lineages_queried": 1,
                    "fresh_queries": int(fresh),
                    "cached_queries": int(not fresh),
                    "global_unique_queries_added": int(fresh),
                    "shared_rank_gain": int(meta.get("shared_rank_gain", 0)),
                    "sum_lineage_unresolved_dimension_before": int(meta.get("rank_stats", {}).get("sum_lineage_dimension", 0)),
                    "joint_unresolved_dimension_before": int(meta.get("rank_stats", {}).get("joint_dimension", 0)),
                    "sharing_factor_before": float(meta.get("rank_stats", {}).get("sharing_factor", 1.0)),
                    "query_contraction_tiebreak": float(meta.get("query_contraction_tiebreak", 0.0)),
                    "total_lineages": int(len(states)),
                }
            )
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("joint population PDO exceeded deterministic iteration bound")

        self._fallback_exact_population(states, reason="global_budget_exhausted")

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "production v3 coarse reachability/KFM look-ahead/exploit-first exact coarse verification unchanged",
            "same finite realized multiscale Hamming action families as protected Population PDO",
            "one protected exact fine exposure per active lineage before empirical joint PDO certification",
            "one unified no-op-inclusive multiscale committee per lineage; scale is an action attribute, never a latent choice",
            "one shared current-cycle fine decision geometry across lineages and scales in KFM terminal displacement coordinates",
            "one population-wide PDO-relevant exact acquisition at a time maximizing shared unresolved-rank reduction",
            "historical discovery/coarse labels train the empirical readout but do not shrink the joint fine-rank accounting",
            "joint robust PDO sets are intersections of epsilon-good actions across coherent committee members",
            "signed reachable contraction coordinates only actions already inside empirical robust PDO sets",
            "coordinate diversity selection never worsens one-step pairwise Hamming relative to independent robust-primary choices",
            "selected unseen certified actions are exact-verified and any contradiction triggers a full joint PDO recomputation",
            "exact oracle remains sole acceptance authority; unresolved empirical cases fall back to frozen exact queried-set acceptance",
        ]
        sharing = [
            float(r.get("sharing_factor", np.nan))
            for r in self.joint_population_pdo_history
            if np.isfinite(float(r.get("sharing_factor", np.nan))) and int(r.get("joint_unresolved_dimension", 0)) > 0
        ]
        gains = [
            int(r.get("shared_rank_gain", 0))
            for r in self.joint_population_query_history
            if str(r.get("phase", "")) == "joint_rank_acquisition" and int(r.get("fresh_oracle_query", 0)) == 1
        ]
        div_improve = [
            float(r.get("certified_proposal_pairwise_hamming_improvement_vs_primary", 0.0))
            for r in self.joint_population_diversity_history
        ]
        summary["joint_population_pdo"] = {
            "formal_certificate_claimed": False,
            "shared_response_geometry": "current-cycle KFM terminal displacement directions over PDO-relevant realized actions",
            "historical_labels_reduce_formal_style_rank": False,
            "empirical_committee": "shared v3 KFM global task-readout jackknife committee",
            "pdo_set_definition": "union/intersection of full nonlinear augmented-Tchebycheff epsilon-good actions",
            "protected_exact_exposure": True,
            "query_scheduler": "one population-wide PDO-relevant direction at a time; default max shared residual decision-rank reduction, with exploit-first control available",
            "acquisition_mode": str(self.args.pdo_joint_acquisition_mode),
            "rank_rtol": float(self._joint_rank_rtol()),
            "rank_acquisition_fresh_queries": int(self._joint_rank_queries),
            "certified_verification_fresh_queries": int(self._joint_verification_queries),
            "certified_population_cycles": int(self._joint_certified_cycles),
            "fallback_population_cycles": int(self._joint_fallback_cycles),
            "verification_contradictions": int(self._joint_verification_contradictions),
            "mean_sharing_factor_when_unresolved": float(np.mean(sharing)) if sharing else 1.0,
            "max_sharing_factor_when_unresolved": float(np.max(sharing)) if sharing else 1.0,
            "mean_lineages_helped_per_rank_query": float(np.mean(gains)) if gains else 0.0,
            "max_lineages_helped_by_one_rank_query": int(max(gains)) if gains else 0,
            "mean_certified_proposal_diversity_improvement_vs_primary": float(np.mean(div_improve)) if div_improve else 0.0,
            "diversity_rule": "coordinate-minimize signed raw reachable contraction inside empirical robust PDO sets",
            "scale_handling": "all configured scales share one joint decision span; no scale selector or scale-specific query budget",
            "acceptance_authority": "exact oracle only",
        }
        if "population_pdo" in summary:
            summary["population_pdo"]["superseded_fine_scheduler"] = summary["population_pdo"].get("query_scheduler", "")
            summary["population_pdo"].update(
                {
                    "query_scheduler": "joint shared-rank one-query-at-a-time scheduler",
                    "exact_acceptance": "empirical robust-set diversity coordination -> exact verification -> exact acceptance, with exact-set fallback",
                    "total_decision_epsilon": float(self._joint_epsilon()),
                    "pdo_query_epsilon": float(self._joint_epsilon()),
                    "exact_tie_epsilon": 0.0,
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
            f"[{METHOD_NAME}] fine: protected anchor -> joint multiscale PDO -> one shared-rank query -> certified diversity tuple -> exact verification\n"
            f"[{METHOD_NAME}] no scale controller, no diversity reward, no historical-label rank shrinkage, exact oracle remains authoritative",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = protected.build_parser()
    p.description = __doc__
    g = p.add_argument_group("Joint population PDO")
    g.add_argument(
        "--pdo-joint-rank-rtol",
        type=float,
        default=1e-7,
        help="Relative SVD tolerance for numerical joint decision-rank accounting.",
    )
    g.add_argument(
        "--pdo-joint-acquisition-mode",
        choices=("shared_rank", "exploit_first"),
        default="shared_rank",
        help=(
            "shared_rank is the production rule; exploit_first is a matched one-query-at-a-time "
            "ablation that still requires an informative PDO direction but does not prioritize cross-lineage reuse."
        ),
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    protected._validate_args(args)
    rtol = float(args.pdo_joint_rank_rtol)
    if not np.isfinite(rtol) or rtol <= 0 or rtol >= 1:
        raise ValueError("--pdo-joint-rank-rtol must be finite in (0,1)")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    JointPopulationPDORunner(args).run()


if __name__ == "__main__":
    main()
