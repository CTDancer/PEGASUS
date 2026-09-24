"""Population-aware multiscale PEGASUS.

This frozen implementation is a direct refinement of the successful production multiscale-v3
optimizer with one protected exact fine exposure before empirical PDO may eliminate a turn.  It does not add a new predictor, uncertainty model, diversity objective,
basin model, or hard reachability filter.

The v3 architecture remains Reach -> Observe -> Decide:

* Reach: expose concrete exact-Hamming actions independently at every configured radius.
* Observe: use the existing KFM terminal representation + v3 global task-readout committee.
* Decide: maintain a protected PDO-active challenger set inside every scale.  In each
  synchronous population query round, every unresolved scale contributes exactly one
  exploit-first PDO challenger. Reachable contraction is used only to choose the order in
  which those already-PDO-relevant challengers are queried across the population.
* Exact acceptance: after PDO querying ends, each lineage forms an exact near-optimal set
  from no-op plus exactly evaluated candidates.  Reachable contraction only breaks ties
  inside that exact set; no action outside the exact tolerance can be selected.

Thus contraction never removes a candidate from the PDO action family and never declares
candidate-level decision redundancy.  It only allocates equivalent/decision-relevant
population search effort nonredundantly.

The contraction geometry uses actual realized edit radius, not H1/H2/H4 labels, so the
same code applies unchanged to H8/H16/H32/... and percentage-based scales on longer
sequences.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_multiscale_tournament as ms
from .kfocus.objectives import OPT_NAMES, RAW_NAMES
from .pdo_contraction_allocation import (
    AllocationOption,
    coordinate_minimize_contraction,
    hamming,
)
from .pdo_pegasus_v3 import (
    GlobalCoarseCommittee,
    choose_exploit_first_challenger,
    plausible_challenger_indices,
)
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-multiscale-population-pdo-protected-v1"
METHOD_NAME = "PEGASUS Multiscale Population PDO (Protected)"


@dataclass
class ChoiceBinding:
    option: AllocationOption
    pool: ms.ScalePool
    local_index: int
    committee: GlobalCoarseCommittee


@dataclass
class PopulationFineState:
    turn_id: int
    cycle: int
    preference_id: int
    lineage: base.LineageState
    start_sequence: str
    start_utility: float
    pools: list[ms.ScalePool]
    incumbent_z: np.ndarray
    fresh_cap: int
    before_fine_q: int
    coarse_paid_queries: int
    query_order: int = 0
    round_index: int = 0
    done: bool = False
    reason: str = "prepared"
    fine_fresh_queries: int = 0
    active_scale_rounds: dict[int, int] = field(default_factory=dict)
    queried_by_scale: dict[int, int] = field(default_factory=dict)
    unresolved_scales_at_cap: tuple[int, ...] = field(default_factory=tuple)

    @property
    def lineage_id(self) -> int:
        return int(self.lineage.lineage_id)


@dataclass
class ExactBinding:
    option: AllocationOption
    pool: ms.ScalePool | None
    local_index: int
    exact_utility: float
    is_noop: bool


class PopulationPDORunner(ms.ScaleFactoredTournamentRunner):
    """Multiscale-v3 with PDO-active population query ordering and exact tie acceptance."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.population_query_allocation_history: list[dict[str, Any]] = []
        self.population_round_history: list[dict[str, Any]] = []
        self.population_active_scale_history: list[dict[str, Any]] = []
        self.population_acceptance_history: list[dict[str, Any]] = []
        self.population_cycle_history: list[dict[str, Any]] = []
        self._population_rounds = 0
        self._population_query_tiebreak_changes = 0
        self._population_exact_tiebreak_changes = 0
        self._population_initial_fresh_queries = 0
        self._population_challenger_fresh_queries = 0

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.population_query_allocation_history:
            write_csv(self.out / "population_query_allocation_history.csv", self.population_query_allocation_history)
        if self.population_round_history:
            write_csv(self.out / "population_round_history.csv", self.population_round_history)
        if self.population_active_scale_history:
            write_csv(self.out / "population_active_scale_history.csv", self.population_active_scale_history)
        if self.population_acceptance_history:
            write_csv(self.out / "population_acceptance_history.csv", self.population_acceptance_history)
        if self.population_cycle_history:
            write_csv(self.out / "population_cycle_history.csv", self.population_cycle_history)

    def _exact_tie_epsilon(self) -> float:
        """Exact acceptance share of the existing total fine decision tolerance."""
        x = float(self.args.pdo_population_exact_tie_epsilon)
        if x < 0:
            return 0.5 * float(self.args.pdo_multiscale_epsilon_dec)
        return x

    def _pdo_query_epsilon(self) -> float:
        """Remaining decision tolerance assigned to PDO query resolution."""
        return max(0.0, float(self.args.pdo_multiscale_epsilon_dec) - self._exact_tie_epsilon())

    def _fresh_used(self, state: PopulationFineState) -> int:
        return int(state.fine_fresh_queries)

    def _prepare_fine_state(
        self,
        *,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        fresh_cap: int,
        coarse_paid_queries: int,
    ) -> PopulationFineState:
        self.pdo_turn_counter += 1
        self._multiscale_turns += 1
        turn_id = int(self.pdo_turn_counter)
        start_seq = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        start_u = float(lineage.utility)
        before = int(self.oracle.unique_oracle_queries)
        pools, incumbent_z = self._build_scale_pools(
            preference_id=preference_id,
            lineage=lineage,
            cycle=cycle,
        )
        state = PopulationFineState(
            turn_id=turn_id,
            cycle=int(cycle),
            preference_id=int(preference_id),
            lineage=lineage,
            start_sequence=start_seq,
            start_utility=start_u,
            pools=pools,
            incumbent_z=np.asarray(incumbent_z, dtype=np.float64),
            fresh_cap=max(0, int(fresh_cap)),
            before_fine_q=before,
            coarse_paid_queries=int(coarse_paid_queries),
        )
        if not pools:
            state.done = True
            state.reason = "no_multiscale_actions"
        elif state.fresh_cap <= 0:
            state.done = True
            state.reason = "no_fine_query_budget"
        return state

    def _binding_from_index(
        self,
        state: PopulationFineState,
        pool_key: int,
        pool: ms.ScalePool,
        local_index: int,
        committee: GlobalCoarseCommittee,
    ) -> ChoiceBinding:
        idx = int(local_index)
        action = pool.actions[idx - 1]
        return ChoiceBinding(
            option=AllocationOption(
                lineage_id=int(state.lineage_id),
                preference_id=int(state.preference_id),
                sequence=str(action.target_sequence),
                incumbent_sequence=str(state.start_sequence),
                scale_k=int(pool.k),
                predicted_utility=float(committee.median_utilities[idx]),
                predicted_upper=float(committee.utility_upper[idx]),
                pool_key=int(pool_key),
                local_index=int(idx),
                initial_rank=int(pool.initial_rank[idx - 1]),
            ),
            pool=pool,
            local_index=idx,
            committee=committee,
        )

    def _has_exact_fine_exposure(self, state: PopulationFineState) -> bool:
        """Whether this turn already has any exact non-noop fine action available.

        Cached exact labels count as exposure: the goal is to prevent zero-evidence
        empirical PDO elimination, not to force a redundant fresh oracle call.
        """
        for pool in state.pools:
            exact = self._candidate_exact_map(pool, state.lineage)
            if any(int(i) != 0 for i in exact):
                return True
        return False

    def _protected_first_representative(self, state: PopulationFineState) -> ChoiceBinding | None:
        """Global exploit-first candidate across all scales before empirical PDO may stop.

        This deliberately ignores the committee upper-bound elimination test for the first
        fine observation only.  It never changes feasibility or exact acceptance.
        """
        if state.done or self._fresh_used(state) >= int(state.fresh_cap):
            return None
        pref = self.preferences[int(state.preference_id)]
        reps: list[ChoiceBinding] = []
        for pool_key, pool in enumerate(state.pools):
            committee = self._build_scale_committee(pool, state.lineage, state.incumbent_z, pref)
            exact_map = self._candidate_exact_map(pool, state.lineage)
            observed = set(int(i) for i in exact_map)
            candidates = [int(i) for i in pool.contender_indices if int(i) not in observed]
            chosen = choose_exploit_first_challenger(
                committee,
                candidates,
                initial_rank=pool.initial_rank,
            )
            if chosen is not None:
                reps.append(self._binding_from_index(state, pool_key, pool, int(chosen), committee))
        if not reps:
            return None
        reps.sort(
            key=lambda b: (
                -float(b.option.predicted_utility),
                -float(b.option.predicted_upper),
                int(b.option.initial_rank),
                int(b.option.scale_k),
                str(b.option.sequence),
            )
        )
        return reps[0]

    def _active_scale_representatives(
        self,
        state: PopulationFineState,
        *,
        log_active: bool,
        ignore_query_cap: bool = False,
    ) -> list[ChoiceBinding]:
        """Return exactly one exploit-first challenger from every PDO-active scale.

        A scale is active iff at least one frozen unqueried contender has utility_upper
        above the current exact global leader + the PDO-resolution share of epsilon_dec
        under the existing v3 global readout committee. No point-estimate allocation
        slack is used.
        """
        if state.done:
            return []
        if (not ignore_query_cap) and self._fresh_used(state) >= int(state.fresh_cap):
            return []
        pref = self.preferences[int(state.preference_id)]
        _best_pool, _best_local, best_exact_u = self._global_best_exact(
            state.pools, state.lineage, pref
        )
        reps: list[ChoiceBinding] = []
        for pool_key, pool in enumerate(state.pools):
            committee = self._build_scale_committee(pool, state.lineage, state.incumbent_z, pref)
            exact_map = self._candidate_exact_map(pool, state.lineage)
            observed = sorted(int(i) for i in exact_map)
            plausible = plausible_challenger_indices(
                committee,
                contender_indices=pool.contender_indices,
                observed_indices=observed,
                best_exact_utility=float(best_exact_u),
                epsilon_dec=float(self._pdo_query_epsilon()),
            )
            chosen = choose_exploit_first_challenger(
                committee,
                plausible,
                initial_rank=pool.initial_rank,
            )
            active = chosen is not None
            if active:
                if log_active:
                    state.active_scale_rounds[int(pool.k)] = int(state.active_scale_rounds.get(int(pool.k), 0) + 1)
                reps.append(self._binding_from_index(state, pool_key, pool, int(chosen), committee))
            if log_active:
                self.population_active_scale_history.append(
                    {
                        "turn_id": int(state.turn_id),
                        "cycle": int(state.cycle),
                        "preference_id": int(state.preference_id),
                        "lineage_id": int(state.lineage_id),
                        "round_index": int(state.round_index),
                        "scale_k": int(pool.k),
                        "active": int(active),
                        "plausible_count": int(len(plausible)),
                        "queried_exact_count": int(max(0, len(exact_map) - 1)),
                        "best_exact_global_utility": float(best_exact_u),
                        "representative_sequence": "" if chosen is None else str(pool.actions[int(chosen)-1].target_sequence),
                        "representative_predicted_utility": float("nan") if chosen is None else float(committee.median_utilities[int(chosen)]),
                        "representative_predicted_upper": float("nan") if chosen is None else float(committee.utility_upper[int(chosen)]),
                        "query_count_scale_this_turn": int(state.queried_by_scale.get(int(pool.k), 0)),
                    }
                )
        reps.sort(
            key=lambda b: (
                -float(b.option.predicted_utility),
                -float(b.option.predicted_upper),
                int(b.option.initial_rank),
                int(b.option.scale_k),
                str(b.option.sequence),
            )
        )
        return reps

    def _fixed_option_for_state(self, state: PopulationFineState) -> AllocationOption:
        pref = self.preferences[int(state.preference_id)]
        pool, local, u = self._global_best_exact(state.pools, state.lineage, pref)
        if pool is None or int(local) <= 0:
            seq = str(state.start_sequence)
            k = 0
        else:
            seq = str(pool.actions[int(local) - 1].target_sequence)
            k = hamming(state.start_sequence, seq)
        return AllocationOption(
            lineage_id=int(state.lineage_id),
            preference_id=int(state.preference_id),
            sequence=seq,
            incumbent_sequence=str(state.start_sequence),
            scale_k=int(k),
            predicted_utility=float(u),
            predicted_upper=float(u),
            pool_key=-1,
            local_index=0,
            initial_rank=10**9,
        )

    def _allocate_query_round(
        self,
        states: Sequence[PopulationFineState],
    ) -> tuple[dict[int, ChoiceBinding], float, float]:
        """One query per unresolved lineage from its protected PDO-active scale reps."""
        by_lineage: dict[int, list[ChoiceBinding]] = {}
        primary: dict[int, ChoiceBinding] = {}
        fixed: dict[int, AllocationOption] = {}
        state_by_id = {int(s.lineage_id): s for s in states}

        for state in states:
            if state.done or self._fresh_used(state) >= int(state.fresh_cap):
                if not state.done:
                    state.done = True
                    state.reason = "fine_query_cap_reached"
                fixed[int(state.lineage_id)] = self._fixed_option_for_state(state)
                continue
            if not self._has_exact_fine_exposure(state):
                protected = self._protected_first_representative(state)
                if protected is None:
                    state.done = True
                    state.reason = "no_unqueried_multiscale_candidate"
                    fixed[int(state.lineage_id)] = self._fixed_option_for_state(state)
                    continue
                reps = [protected]
            else:
                reps = self._active_scale_representatives(state, log_active=True)
                if not reps:
                    state.done = True
                    state.reason = "no_scale_has_plausible_global_challenger"
                    self._multiscale_early_stops += 1
                    fixed[int(state.lineage_id)] = self._fixed_option_for_state(state)
                    continue
            by_lineage[int(state.lineage_id)] = reps
            primary[int(state.lineage_id)] = reps[0]

        alternatives = {lid: [b.option for b in rows] for lid, rows in by_lineage.items()}
        selected_opts, initial_cost, final_cost = coordinate_minimize_contraction(
            alternatives,
            fixed=fixed,
            passes=int(self.args.pdo_population_coordinate_passes),
            use_contraction=(str(self.args.pdo_population_mode) == "contraction"),
        )
        selected: dict[int, ChoiceBinding] = {}
        for lid, opt in selected_opts.items():
            lookup = {(b.option.pool_key, b.option.local_index): b for b in by_lineage[lid]}
            selected[lid] = lookup[(opt.pool_key, opt.local_index)]

        for lid in sorted(selected):
            state = state_by_id[lid]
            p = primary[lid]
            c = selected[lid]
            changed = int(
                (p.option.pool_key, p.option.local_index)
                != (c.option.pool_key, c.option.local_index)
            )
            self._population_query_tiebreak_changes += changed
            self.population_query_allocation_history.append(
                {
                    "turn_id": int(state.turn_id),
                    "cycle": int(state.cycle),
                    "preference_id": int(state.preference_id),
                    "lineage_id": int(lid),
                    "round_index": int(state.round_index),
                    "active_scale_count": int(len(by_lineage[lid])),
                    "active_scales": ";".join(str(int(b.pool.k)) for b in by_lineage[lid]),
                    "primary_scale_k": int(p.pool.k),
                    "primary_sequence": str(p.option.sequence),
                    "primary_predicted_utility": float(p.option.predicted_utility),
                    "selected_scale_k": int(c.pool.k),
                    "selected_sequence": str(c.option.sequence),
                    "selected_predicted_utility": float(c.option.predicted_utility),
                    "selected_predicted_upper": float(c.option.predicted_upper),
                    "query_order_changed_by_contraction": int(changed),
                    "population_contraction_cost_primary": float(initial_cost),
                    "population_contraction_cost_selected": float(final_cost),
                    "population_contraction_cost_reduction": float(initial_cost - final_cost),
                }
            )
        return selected, float(initial_cost), float(final_cost)

    def _query_population_action(
        self,
        *,
        state: PopulationFineState,
        binding: ChoiceBinding,
    ) -> tuple[base.ArchiveEntry | None, bool]:
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
                source="pdo_multiscale_population_pdo",
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
            if int(state.round_index) == 0:
                self._population_initial_fresh_queries += delta
            else:
                self._population_challenger_fresh_queries += delta
                self._multiscale_challenger_queries += delta
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
            "round_index": int(state.round_index),
            "phase": "population_pdo_initial" if int(state.round_index) == 0 else "population_pdo_challenger",
            "scale_k": int(pool.k),
            "action_id": str(action.action_id),
            "candidate_sequence": str(action.target_sequence),
            "fresh_oracle_query": int(fresh),
            "global_unique_query_index": int(self.oracle.unique_oracle_queries),
            "predicted_median_utility": float(binding.committee.median_utilities[idx]),
            "predicted_utility_lower": float(binding.committee.utility_lower[idx]),
            "predicted_utility_upper": float(binding.committee.utility_upper[idx]),
            "exact_utility": float(exact_u),
            "exact_gain": float(exact_u - float(state.start_utility)),
            "accepted": 0,
        }
        for j, name in enumerate(OPT_NAMES):
            row[f"exact_score_{name}"] = float(entry.scores[j])
        for j, name in enumerate(RAW_NAMES):
            row[f"raw_{name}"] = float(entry.raw[j])
        self.multiscale_query_history.append(row)
        return entry, fresh

    def _mark_population_accept(self, turn_id: int, sequence: str) -> None:
        for row in reversed(self.multiscale_query_history):
            if int(row.get("turn_id", -1)) == int(turn_id) and str(row.get("candidate_sequence", "")) == str(sequence):
                row["accepted"] = 1
                return

    def _exact_near_optimal_bindings(self, state: PopulationFineState) -> list[ExactBinding]:
        """Exact no-op-inclusive tie set within the existing PDO decision tolerance."""
        pref = self.preferences[int(state.preference_id)]
        rows: list[ExactBinding] = []
        # no-op is an ordinary exact action
        noop_u = float(base._utility(state.lineage.scores, pref, self.args.rho))
        rows.append(
            ExactBinding(
                option=AllocationOption(
                    lineage_id=int(state.lineage_id),
                    preference_id=int(state.preference_id),
                    sequence=str(state.start_sequence),
                    incumbent_sequence=str(state.start_sequence),
                    scale_k=0,
                    predicted_utility=float(noop_u),
                    predicted_upper=float(noop_u),
                    pool_key=-1,
                    local_index=0,
                    initial_rank=0,
                ),
                pool=None,
                local_index=0,
                exact_utility=float(noop_u),
                is_noop=True,
            )
        )
        for pool_key, pool in enumerate(state.pools):
            exact = self._candidate_exact_map(pool, state.lineage)
            for idx, score in exact.items():
                if int(idx) == 0:
                    continue
                u = float(base._utility(score, pref, self.args.rho))
                action = pool.actions[int(idx) - 1]
                rows.append(
                    ExactBinding(
                        option=AllocationOption(
                            lineage_id=int(state.lineage_id),
                            preference_id=int(state.preference_id),
                            sequence=str(action.target_sequence),
                            incumbent_sequence=str(state.start_sequence),
                            scale_k=int(pool.k),
                            predicted_utility=float(u),
                            predicted_upper=float(u),
                            pool_key=int(pool_key),
                            local_index=int(idx),
                            initial_rank=int(pool.initial_rank[int(idx) - 1]),
                        ),
                        pool=pool,
                        local_index=int(idx),
                        exact_utility=float(u),
                        is_noop=False,
                    )
                )
        rows.sort(
            key=lambda r: (
                -float(r.exact_utility),
                int(r.option.scale_k),
                int(r.option.initial_rank),
                str(r.option.sequence),
            )
        )
        best_u = float(rows[0].exact_utility)
        eps = float(self._exact_tie_epsilon())
        eligible = [r for r in rows if float(r.exact_utility) >= best_u - eps - 1e-15]
        if not eligible:
            raise RuntimeError("exact near-optimal set unexpectedly empty")
        # Preserve exact-best-first ordering; value_only therefore accepts exact best.
        eligible.sort(
            key=lambda r: (
                -float(r.exact_utility),
                int(r.option.scale_k),
                int(r.option.initial_rank),
                str(r.option.sequence),
            )
        )
        return eligible

    def _joint_exact_acceptance(self, states: Sequence[PopulationFineState]) -> None:
        alternatives: dict[int, list[AllocationOption]] = {}
        bindings: dict[int, dict[tuple[int, int], ExactBinding]] = {}
        best_binding: dict[int, ExactBinding] = {}
        state_by_id = {int(s.lineage_id): s for s in states}

        for state in states:
            eligible = self._exact_near_optimal_bindings(state)
            lid = int(state.lineage_id)
            alternatives[lid] = [r.option for r in eligible]
            bindings[lid] = {(r.option.pool_key, r.option.local_index): r for r in eligible}
            best_binding[lid] = eligible[0]

        selected_opts, cost0, cost1 = coordinate_minimize_contraction(
            alternatives,
            fixed=None,
            passes=int(self.args.pdo_population_coordinate_passes),
            use_contraction=(str(self.args.pdo_population_mode) == "contraction"),
        )

        for lid in sorted(state_by_id):
            state = state_by_id[lid]
            best = best_binding[lid]
            opt = selected_opts[lid]
            chosen = bindings[lid][(opt.pool_key, opt.local_index)]
            changed = int(
                (best.option.pool_key, best.option.local_index)
                != (chosen.option.pool_key, chosen.option.local_index)
            )
            self._population_exact_tiebreak_changes += changed
            sacrifice = float(best.exact_utility - chosen.exact_utility)
            if sacrifice > float(self._exact_tie_epsilon()) + 1e-10:
                raise RuntimeError("exact acceptance sacrifice exceeds configured tie epsilon")

            accepted = False
            accepted_k = -1
            accepted_seq = ""
            if (
                not chosen.is_noop
                and float(chosen.exact_utility - state.start_utility) > float(self.args.accept_epsilon)
            ):
                if chosen.pool is None or int(chosen.local_index) <= 0:
                    raise RuntimeError("non-noop exact acceptance missing scale binding")
                action = chosen.pool.actions[int(chosen.local_index) - 1]
                entry = self.archive_by_sequence.get(action.target_sequence)
                if entry is None:
                    raise RuntimeError("population PDO exact acceptance selected action without archive entry")
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

            self.population_acceptance_history.append(
                {
                    "turn_id": int(state.turn_id),
                    "cycle": int(state.cycle),
                    "preference_id": int(state.preference_id),
                    "lineage_id": int(lid),
                    "total_decision_epsilon": float(self.args.pdo_multiscale_epsilon_dec),
                    "pdo_query_epsilon": float(self._pdo_query_epsilon()),
                    "exact_tie_epsilon": float(self._exact_tie_epsilon()),
                    "exact_near_optimal_count": int(len(alternatives[lid])),
                    "best_exact_sequence": str(best.option.sequence),
                    "best_exact_scale_k": int(best.option.scale_k),
                    "best_exact_utility": float(best.exact_utility),
                    "selected_sequence": str(chosen.option.sequence),
                    "selected_scale_k": int(chosen.option.scale_k),
                    "selected_exact_utility": float(chosen.exact_utility),
                    "selected_noop": int(chosen.is_noop),
                    "exact_utility_sacrifice": float(sacrifice),
                    "acceptance_changed_by_contraction": int(changed),
                    "population_contraction_cost_exact_best": float(cost0),
                    "population_contraction_cost_exact_selected": float(cost1),
                    "population_contraction_cost_reduction": float(cost0 - cost1),
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
                    "exact_near_optimal_count": int(len(alternatives[lid])),
                    "exact_acceptance_utility_sacrifice": float(sacrifice),
                }
            )

    def _run_population_fine(self, states: list[PopulationFineState]) -> None:
        if not states:
            return
        round_idx = 0
        while not self.stop_requested:
            for state in states:
                state.round_index = int(round_idx)
            selected, cost0, cost1 = self._allocate_query_round(states)
            if not selected:
                break
            self._population_rounds += 1
            before = int(self.oracle.unique_oracle_queries)
            fresh = 0
            cached = 0
            state_lookup = {int(s.lineage_id): s for s in states}

            # Selection is frozen above; exact labels acquired here cannot alter any other
            # same-round choice. They become available only to the next PDO round.
            for lid in sorted(selected):
                state = state_lookup[lid]
                if state.done or self._fresh_used(state) >= int(state.fresh_cap):
                    continue
                state.query_order += 1
                entry, was_fresh = self._query_population_action(state=state, binding=selected[lid])
                if entry is None:
                    state.done = True
                    state.reason = "global_budget_exhausted"
                    break
                fresh += int(was_fresh)
                cached += int(not was_fresh)
                if self._fresh_used(state) >= int(state.fresh_cap):
                    # Record which protected scale sets remain decision-active at the cap.
                    remaining = self._active_scale_representatives(
                        state, log_active=False, ignore_query_cap=True
                    )
                    state.unresolved_scales_at_cap = tuple(sorted(set(int(b.pool.k) for b in remaining)))
                    state.done = True
                    state.reason = "fine_query_cap_reached"

            resolved = 0
            for state in states:
                if state.done:
                    resolved += 1
                    continue
                remaining = self._active_scale_representatives(state, log_active=False)
                if not remaining:
                    state.done = True
                    state.reason = "no_scale_has_plausible_global_challenger"
                    self._multiscale_early_stops += 1
                    resolved += 1

            self.population_round_history.append(
                {
                    "cycle": int(states[0].cycle),
                    "preference_id": int(states[0].preference_id),
                    "round_index": int(round_idx),
                    "allocation_mode": str(self.args.pdo_population_mode),
                    "active_lineages_queried": int(len(selected)),
                    "fresh_queries": int(fresh),
                    "cached_queries": int(cached),
                    "global_unique_queries_added": int(self.oracle.unique_oracle_queries - before),
                    "contraction_cost_value_primary": float(cost0),
                    "contraction_cost_population_selected": float(cost1),
                    "contraction_cost_reduction": float(cost0 - cost1),
                    "resolved_lineages_after_round": int(resolved),
                    "total_lineages": int(len(states)),
                }
            )
            if self.stop_requested:
                break
            round_idx += 1
            if round_idx > max(int(s.fresh_cap) for s in states) + 1:
                raise RuntimeError("population PDO fine rounds exceeded per-lineage query-cap bound")

        # The exact near-optimal sets and contraction tie-break are frozen jointly before
        # any lineage accepts, preventing acceptance-order artifacts.
        self._joint_exact_acceptance(states)

    def optimize(self) -> None:
        """Unchanged v3 coarse sweep + synchronous population-aware multiscale PDO fine stage."""
        a = self.args
        total = len(self.preferences) * int(a.lineages) * int(a.slates_per_lineage)
        bar = None
        # v2 is re-exported in the multiscale module's v3 dependency; avoid a new runtime dependency.
        try:
            from . import pdo_pegasus_v2 as v2
            if v2.tqdm is not None and not bool(a.no_progress):
                bar = v2.tqdm(total=total, desc=METHOD_NAME, unit="turn", dynamic_ncols=True)
        except Exception:
            bar = None
        try:
            for cycle in range(int(a.slates_per_lineage)):
                for pref_id in range(len(self.preferences)):
                    states: list[PopulationFineState] = []
                    cycle_start_q = int(self.oracle.unique_oracle_queries)
                    cycle_start_u: dict[int, float] = {}
                    fine_caps: dict[int, int] = {}
                    coarse_paid: dict[int, int] = {}

                    # Production v3 coarse stage is untouched. All coarse updates happen
                    # before any fine pool is realized, giving the population fine stage a
                    # common post-coarse information snapshot.
                    for lineage in self.lineages[pref_id]:
                        if self.stop_requested:
                            return
                        lid = int(lineage.lineage_id)
                        cycle_start_u[lid] = float(lineage.utility)
                        before_q = int(self.oracle.unique_oracle_queries)
                        mode = str(a.pdo_budget_mode)
                        if mode == "protected_baseline":
                            coarse = self._run_unified_coarse(pref_id, lineage, cycle)
                            fine_cap = int(a.pdo_multiscale_query_cap_per_turn)
                        elif mode == "matched_budget":
                            total_cap = int(a.verification_k)
                            coarse = self._run_unified_coarse(pref_id, lineage, cycle, query_cap_override=total_cap)
                            fine_cap = min(
                                int(a.pdo_multiscale_query_cap_per_turn),
                                max(0, total_cap - int(coarse.get("paid_queries", 0))),
                            )
                        else:
                            raise ValueError(f"unknown PDO budget mode {mode!r}")
                        if self.stop_requested:
                            return
                        coarse_paid[lid] = int(self.oracle.unique_oracle_queries) - before_q
                        fine_caps[lid] = int(fine_cap)

                    pre_fine_sequences: dict[int, str] = {}
                    for lineage in self.lineages[pref_id]:
                        lid = int(lineage.lineage_id)
                        pre_fine_sequences[lid] = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
                        states.append(
                            self._prepare_fine_state(
                                preference_id=pref_id,
                                lineage=lineage,
                                cycle=cycle,
                                fresh_cap=fine_caps[lid],
                                coarse_paid_queries=coarse_paid[lid],
                            )
                        )

                    self._run_population_fine(states)
                    post_fine_sequences = {
                        int(s.lineage_id): base.decode_esm_tokens(s.lineage.tokens.reshape(1, -1))[0]
                        for s in states
                    }
                    ids = sorted(pre_fine_sequences)
                    pre_vals: list[float] = []
                    post_vals: list[float] = []
                    for ii in range(len(ids)):
                        for jj in range(ii + 1, len(ids)):
                            li, lj = ids[ii], ids[jj]
                            L = len(pre_fine_sequences[li])
                            pre_vals.append(hamming(pre_fine_sequences[li], pre_fine_sequences[lj]) / L)
                            post_vals.append(hamming(post_fine_sequences[li], post_fine_sequences[lj]) / L)
                    cycle_turns = [
                        r for r in self.multiscale_turn_history
                        if int(r.get("cycle", -1)) == int(cycle)
                        and int(r.get("preference_id", -1)) == int(pref_id)
                    ]
                    self.population_cycle_history.append(
                        {
                            "cycle": int(cycle),
                            "preference_id": int(pref_id),
                            "lineage_count": int(len(ids)),
                            "pre_fine_pairwise_hamming_mean": float(np.mean(pre_vals)) if pre_vals else 0.0,
                            "post_fine_pairwise_hamming_mean": float(np.mean(post_vals)) if post_vals else 0.0,
                            "fine_pairwise_diversity_change": float(np.mean(post_vals) - np.mean(pre_vals)) if pre_vals else 0.0,
                            "cycle_unique_queries": int(self.oracle.unique_oracle_queries - cycle_start_q),
                            "accepted_fine_moves": int(sum(int(r.get("accepted", 0)) for r in cycle_turns)),
                            "turns_hitting_fine_query_cap": int(sum(str(r.get("reason", "")) == "fine_query_cap_reached" for r in cycle_turns)),
                            "turns_with_unresolved_scale_at_cap": int(sum(bool(str(r.get("unresolved_scales_at_cap", ""))) for r in cycle_turns)),
                        }
                    )
                    if bar is not None:
                        for state in states:
                            bar.update(1)
                            bar.set_postfix(
                                unique_q=self.oracle.unique_oracle_queries,
                                last_gain=f"{state.lineage.utility-cycle_start_u[state.lineage_id]:+.4f}",
                                fine_q=self._fresh_used(state),
                            )
                    if int(a.save_every_slates) > 0 and (cycle + 1) % max(1, int(a.save_every_slates)) == 0:
                        self._save_progress()
        finally:
            if bar is not None:
                bar.close()

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "production PEGASUS v3 coarse physical branching and exploit-first coarse PDO unchanged",
            "concrete multiscale fine action exposure inherited from the scale-factored tournament",
            "each unresolved scale contributes one exploit-first PDO-active challenger per query round",
            "reachable contraction only orders already-PDO-active scale challengers across the population",
            "no predicted utility allocation slack and no contraction-based candidate pruning",
            "all scale challenger sets remain active until PDO eliminates them or the fine query cap is reached",
            "exact no-op plus queried candidates form an exact near-optimal set within the existing decision tolerance",
            "reachable contraction only breaks ties inside that exact near-optimal set for joint population acceptance",
            "exact oracle remains sole acceptance authority; no diversity reward or new uncertainty model",
            "contraction geometry uses actual realized edit radius and is scale-label agnostic",
        ]
        summary["population_pdo"] = {
            "mode": str(self.args.pdo_population_mode),
            "query_scheduler": "one protected global exploit-first exact fine exposure before empirical PDO elimination; then PDO-active scale representatives with contraction tie-break",
            "predicted_allocation_slack_used": False,
            "hard_contraction_pruning": False,
            "scale_protection": "scale remains an active challenger source until PDO eliminates its frozen contender set",
            "total_decision_epsilon": float(self.args.pdo_multiscale_epsilon_dec),
            "pdo_query_epsilon": float(self._pdo_query_epsilon()),
            "exact_tie_epsilon": float(self._exact_tie_epsilon()),
            "exact_acceptance": "joint contraction tie-break inside exact no-op-inclusive near-optimal queried sets",
            "coordinate_passes": int(self.args.pdo_population_coordinate_passes),
            "population_schedule": "v3 coarse sweep then synchronous fine PDO query rounds per preference",
            "query_rounds": int(self._population_rounds),
            "query_order_changes_by_contraction": int(self._population_query_tiebreak_changes),
            "exact_acceptance_changes_by_contraction": int(self._population_exact_tiebreak_changes),
            "fresh_initial_queries": int(self._population_initial_fresh_queries),
            "fresh_challenger_queries": int(self._population_challenger_fresh_queries),
            "actual_radius_geometry": True,
            "configured_radius_spec": str(self.args.pdo_multiscale_radius_spec),
            "formal_certificate_claimed": False,
        }
        if "multiscale_tournament" in summary:
            summary["multiscale_tournament"].update(
                {
                    "initial_champion_from_every_radius": False,
                    "fine_query_scheduler": "population-aware PDO-active round scheduler",
                    "fresh_initial_queries": int(self._population_initial_fresh_queries),
                    "challenger_fresh_queries": int(self._population_challenger_fresh_queries),
                    "final_arbitration": "joint exact near-optimal population acceptance plus no-op",
                }
            )
        if "unified_pdo_pegasus_v3" in summary:
            summary["unified_pdo_pegasus_v3"]["schedule"] = (
                "v3_coarse_population_sweep_then_synchronous_population-aware_multiscale_PDO_fine_rounds"
            )
            summary["unified_pdo_pegasus_v3"].setdefault("roles", {})["fine_PDO"] = (
                "scale-protected multiscale PDO with contraction-aware population query ordering and exact tie acceptance"
            )
        if "pdo" in summary:
            summary["pdo"]["acquisition_probe_acceptance"] = (
                "existing PDO defines active scale challengers; contraction only orders active queries and exact near-optimal ties"
            )
            summary["pdo"]["fine_action_family"] = (
                "all configured realized exact-Hamming candidate families plus no-op"
            )
            summary["pdo"]["formal_certificate_claimed"] = False
            per_pref: dict[str, dict[str, int]] = {}
            for pref_id in range(len(self.preferences)):
                qrows = [
                    r for r in self.multiscale_query_history
                    if int(r.get("preference_id", -1)) == int(pref_id)
                    and int(r.get("fresh_oracle_query", 0)) == 1
                ]
                trows = [
                    r for r in self.multiscale_turn_history
                    if int(r.get("preference_id", -1)) == int(pref_id)
                ]
                per_pref[str(pref_id)] = {
                    "fresh_fine_queries": int(len(qrows)),
                    "fresh_population_initial_queries": int(sum(str(r.get("phase", "")) == "population_pdo_initial" for r in qrows)),
                    "fresh_population_challenger_queries": int(sum(str(r.get("phase", "")) == "population_pdo_challenger" for r in qrows)),
                    "fine_accepted_moves": int(sum(int(r.get("accepted", 0)) for r in trows)),
                }
            summary["pdo"]["per_preference_fine_accounting"] = per_pref
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] coarse stage: unchanged production v3\n"
            f"[{METHOD_NAME}] fine PDO: protected exact fine exposure -> protected scale-active sets -> contraction query tie-break -> exact near-optimal acceptance tie-break\n"
            f"[{METHOD_NAME}] no allocation slack, no hard pruning, no diversity objective, exact oracle remains authoritative",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = ms.build_parser()
    p.description = __doc__
    g = p.add_argument_group("Population-aware PDO")
    g.add_argument(
        "--pdo-population-mode",
        choices=("contraction", "value_only"),
        default="contraction",
        help=(
            "contraction uses reachable contraction only as PDO query/exact-tie ordering; "
            "value_only is the matched synchronous schedule control."
        ),
    )
    g.add_argument(
        "--pdo-population-exact-tie-epsilon",
        type=float,
        default=-1.0,
        help=(
            "Exact utility share of the existing total fine decision tolerance. Negative "
            "uses half of --pdo-multiscale-epsilon-dec; PDO resolution uses the remainder."
        ),
    )
    g.add_argument(
        "--pdo-population-coordinate-passes",
        type=int,
        default=3,
        help="Deterministic coordinate-descent passes for population contraction tie-breaking.",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    ms._validate_args(args)
    if int(args.pdo_population_coordinate_passes) < 0:
        raise ValueError("--pdo-population-coordinate-passes cannot be negative")
    if float(args.pdo_population_exact_tie_epsilon) >= 0 and not np.isfinite(float(args.pdo_population_exact_tie_epsilon)):
        raise ValueError("--pdo-population-exact-tie-epsilon must be finite or negative sentinel")
    if float(args.pdo_population_exact_tie_epsilon) < -1:
        raise ValueError("--pdo-population-exact-tie-epsilon must be >= -1")
    tie = (
        0.5 * float(args.pdo_multiscale_epsilon_dec)
        if float(args.pdo_population_exact_tie_epsilon) < 0
        else float(args.pdo_population_exact_tie_epsilon)
    )
    if tie > float(args.pdo_multiscale_epsilon_dec) + 1e-15:
        raise ValueError("population exact tie epsilon cannot exceed total multiscale epsilon_dec")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    PopulationPDORunner(args).run()


if __name__ == "__main__":
    main()
