"""PEGASUS joint Population PDO v3: lazy protected batches with shared-rank scheduling.

v3 freezes the architecture around a strict separation of roles established by the
length-40 two-seed evidence:

* protected PEGASUS decides WHICH optimization-relevant candidate each unresolved
  lineage would query in the current synchronous round;
* joint population rank decides only WHICH member of that frozen protected batch is
  purchased first;
* after every purchased exact vector, the ordinary protected-PDO stopping rule is
  recomputed for every lineage, allowing still-unpurchased protected queries to be
  lazily skipped when shared evidence has made them unnecessary;
* cancellation is provisional until the frozen batch ends: if subsequent shared labels
  reopen a lineage under the ordinary protected rule, its original frozen query becomes
  eligible again.  No new candidate is substituted inside the batch;
* final population choice remains no-op plus exactly evaluated near-optimal actions,
  coordinated by signed reachable contraction and authorized only by the exact oracle.

Joint information geometry therefore schedules and removes redundancy among queries that
protected PEGASUS already judged optimization-relevant.  It never chooses a replacement
optimization action, never certifies an unseen action, and never decides by itself that a
lineage is resolved.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import pdo_pegasus_joint_population_pdo_v2 as v2
from . import pdo_pegasus_multiscale_population_pdo_protected as protected
from .pdo_joint_population import build_fast_shared_rank_context, fast_shared_rank_gain
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-joint-population-pdo-v3"
METHOD_NAME = "PEGASUS Joint Population PDO v3"



def frozen_protected_scheduler_key(
    *,
    shared_rank_gain: int,
    predicted_utility: float,
    predicted_upper: float,
    frozen_rank: int,
    lineage_id: int,
    sequence: str,
    mode: str,
) -> tuple[Any, ...]:
    """Deterministic ordering key over candidate identities already frozen by protected PEGASUS."""
    rank_key = int(shared_rank_gain) if str(mode) == "shared_rank" else 0
    return (
        -int(rank_key),
        -float(predicted_utility),
        -float(predicted_upper),
        int(frozen_rank),
        int(lineage_id),
        str(sequence),
    )


@dataclass(frozen=True)
class FrozenProtectedQuery:
    """One exact candidate selected by the frozen protected PEGASUS batch."""

    lineage_id: int
    state: protected.PopulationFineState
    binding: protected.ChoiceBinding
    frozen_rank: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            int(self.lineage_id),
            int(self.binding.option.pool_key),
            int(self.binding.option.local_index),
        )


class JointPopulationPDORunnerV3(v2.JointPopulationPDORunnerV2):
    """Protected candidate choice + joint-rank ordering + lazy cancellation."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.joint_population_batch_history: list[dict[str, Any]] = []
        self.joint_population_cancellation_history: list[dict[str, Any]] = []
        self.joint_population_scheduler_history: list[dict[str, Any]] = []
        self._protected_batch_planned_queries = 0
        self._protected_batch_purchased_queries = 0
        self._protected_batch_final_cancellations = 0
        self._protected_batch_fresh_queries_saved = 0
        self._protected_batch_provisional_cancellation_events = 0
        self._protected_batch_reactivation_events = 0
        self._protected_batch_bound_violations = 0

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.joint_population_batch_history:
            write_csv(
                self.out / "joint_population_batch_history.csv",
                self.joint_population_batch_history,
            )
        if self.joint_population_cancellation_history:
            write_csv(
                self.out / "joint_population_cancellation_history.csv",
                self.joint_population_cancellation_history,
            )
        if self.joint_population_scheduler_history:
            write_csv(
                self.out / "joint_population_scheduler_history.csv",
                self.joint_population_scheduler_history,
            )

    # ------------------------------------------------------------------
    # Frozen protected batch construction and exact protected stopping
    # ------------------------------------------------------------------
    def _freeze_protected_batch(
        self,
        states: Sequence[protected.PopulationFineState],
        *,
        batch_round: int,
    ) -> tuple[dict[int, FrozenProtectedQuery], float, float]:
        """Construct exactly the batch the frozen protected method would query.

        The inherited protected allocator chooses one candidate per unresolved lineage from
        its ordinary PDO-active scale representatives, including the existing contraction
        allocation across scales/lineages.  Joint rank has no influence here.
        """
        for state in states:
            state.round_index = int(batch_round)
        selected, cost0, cost1 = self._allocate_query_round(states)
        frozen: dict[int, FrozenProtectedQuery] = {}
        ordered = sorted(
            selected.items(),
            key=lambda kv: (
                -float(kv[1].option.predicted_utility),
                -float(kv[1].option.predicted_upper),
                int(kv[1].option.initial_rank),
                int(kv[0]),
                str(kv[1].option.sequence),
            ),
        )
        for rank, (lid, binding) in enumerate(ordered, start=1):
            frozen[int(lid)] = FrozenProtectedQuery(
                lineage_id=int(lid),
                state=next(s for s in states if int(s.lineage_id) == int(lid)),
                binding=binding,
                frozen_rank=int(rank),
            )
        self._protected_batch_planned_queries += int(len(frozen))
        return frozen, float(cost0), float(cost1)

    def _protected_currently_resolved(
        self,
        state: protected.PopulationFineState,
    ) -> tuple[bool, str]:
        """Use only the ordinary protected-PDO rule as cancellation authority."""
        if state.done:
            return True, str(state.reason or "state_done")
        if self._fresh_used(state) >= int(state.fresh_cap):
            return True, "fine_query_cap_reached"
        if not self._has_exact_fine_exposure(state):
            # Protected exposure is mandatory; absence never licenses lazy cancellation.
            return False, "protected_anchor_missing"
        remaining = self._active_scale_representatives(
            state,
            log_active=False,
            ignore_query_cap=False,
        )
        if remaining:
            return False, "protected_pdo_unresolved"
        return True, "protected_pdo_resolved"

    def _binding_is_already_exact(self, q: FrozenProtectedQuery) -> bool:
        return str(q.binding.option.sequence) in self.archive_by_sequence

    def _current_batch_activity(
        self,
        frozen: Mapping[int, FrozenProtectedQuery],
        purchased: set[int],
        previous_provisional: set[int],
        *,
        batch_round: int,
        after_purchase_index: int,
    ) -> tuple[set[int], set[int], set[int]]:
        """Recompute pending/provisionally-cancelled queries after shared evidence.

        A protected query can switch between pending and provisional-cancelled several
        times while other frozen queries are purchased.  This prevents an early model
        update from permanently suppressing a lineage that a later same-batch label would
        reopen.  Only unpurchased queries that are still resolved when the batch terminates
        count as final lazy cancellations.
        """
        pending: set[int] = set()
        provisional: set[int] = set()
        became_exact: set[int] = set()
        for lid, q in frozen.items():
            lid = int(lid)
            if lid in purchased:
                continue
            state = q.state
            if self._binding_is_already_exact(q):
                became_exact.add(lid)
                continue
            resolved, reason = self._protected_currently_resolved(state)
            if resolved and reason == "protected_pdo_resolved":
                provisional.add(lid)
            elif resolved and reason == "fine_query_cap_reached":
                provisional.add(lid)
            else:
                pending.add(lid)

        new_cancel = provisional - previous_provisional
        reactivated = previous_provisional - provisional - became_exact
        self._protected_batch_provisional_cancellation_events += int(len(new_cancel))
        self._protected_batch_reactivation_events += int(len(reactivated))
        for lid in sorted(new_cancel):
            q = frozen[lid]
            self.joint_population_cancellation_history.append(
                {
                    "cycle": int(q.state.cycle),
                    "preference_id": int(q.state.preference_id),
                    "batch_round": int(batch_round),
                    "after_purchase_index": int(after_purchase_index),
                    "lineage_id": int(lid),
                    "candidate_sequence": str(q.binding.option.sequence),
                    "scale_k": int(q.binding.pool.k),
                    "event": "provisional_cancel",
                    "authority": "ordinary_protected_pdo_stopping_rule",
                    "final": 0,
                }
            )
        for lid in sorted(reactivated):
            q = frozen[lid]
            self.joint_population_cancellation_history.append(
                {
                    "cycle": int(q.state.cycle),
                    "preference_id": int(q.state.preference_id),
                    "batch_round": int(batch_round),
                    "after_purchase_index": int(after_purchase_index),
                    "lineage_id": int(lid),
                    "candidate_sequence": str(q.binding.option.sequence),
                    "scale_k": int(q.binding.pool.k),
                    "event": "reactivated_after_later_shared_label",
                    "authority": "ordinary_protected_pdo_stopping_rule",
                    "final": 0,
                }
            )
        return pending, provisional, became_exact

    # ------------------------------------------------------------------
    # Shared rank is ONLY an ordering rule over the frozen batch
    # ------------------------------------------------------------------
    def _choose_frozen_batch_query(
        self,
        frozen: Mapping[int, FrozenProtectedQuery],
        pending: set[int],
        states: Sequence[protected.PopulationFineState],
    ) -> tuple[FrozenProtectedQuery | None, dict[str, Any]]:
        if not pending:
            return None, {"eligible_query_count": 0}

        views = self._build_joint_views(states)
        observed, decision, scale, stats, _unique = self._rank_context(
            views, include_resolved=False
        )
        fast_context = build_fast_shared_rank_context(
            observed,
            decision,
            rtol=self._joint_rank_rtol(),
            scale=scale if len(scale) else None,
        )

        scored: list[
            tuple[tuple[Any, ...], FrozenProtectedQuery, dict[int, int], int]
        ] = []
        for lid in sorted(pending):
            q = frozen[int(lid)]
            state = q.state
            delta_z = np.asarray(
                q.binding.pool.z[int(q.binding.local_index) - 1] - state.incumbent_z,
                dtype=np.float64,
            )
            gain, per = fast_shared_rank_gain(fast_context, delta_z)
            # Candidate identity is already frozen by protected PEGASUS.  Rank only orders
            # those fixed candidates; exploit-first value is the deterministic tie-break.
            key = frozen_protected_scheduler_key(
                shared_rank_gain=int(gain),
                predicted_utility=float(q.binding.option.predicted_utility),
                predicted_upper=float(q.binding.option.predicted_upper),
                frozen_rank=int(q.frozen_rank),
                lineage_id=int(q.lineage_id),
                sequence=str(q.binding.option.sequence),
                mode=str(self.args.pdo_joint_acquisition_mode),
            )
            scored.append((key, q, per, int(gain)))

        if not scored:
            return None, {"eligible_query_count": 0, "rank_stats": stats}
        scored.sort(key=lambda x: x[0])
        _key, chosen, per, gain = scored[0]
        return chosen, {
            "eligible_query_count": int(len(scored)),
            "rank_stats": stats,
            "shared_rank_gain": int(gain),
            "per_lineage_rank_gain": per,
            "scheduler_mode": str(self.args.pdo_joint_acquisition_mode),
        }

    def _query_frozen_member(
        self,
        q: FrozenProtectedQuery,
        *,
        states: Sequence[protected.PopulationFineState],
        batch_round: int,
        purchase_index: int,
        batch_size: int,
        meta: Mapping[str, Any],
    ) -> tuple[Any | None, bool]:
        """Purchase exactly the frozen protected candidate, never a rank-selected replacement."""
        views = self._build_joint_views(states)
        view = views.get(int(q.lineage_id))
        if view is None:
            raise RuntimeError("frozen protected query lineage missing from current joint view")
        jb = view.by_key.get(
            (int(q.binding.option.pool_key), int(q.binding.option.local_index))
        )
        if jb is None:
            raise RuntimeError(
                "frozen protected candidate missing from immutable fine contender family"
            )
        q.state.query_order += 1
        gains = meta.get("per_lineage_rank_gain", {})
        entry, fresh = self._query_joint_binding(
            jb,
            phase="joint_rank_acquisition",
            iteration=int(batch_round),
            shared_rank_gain_value=int(meta.get("shared_rank_gain", 0)),
            per_lineage_rank_gain=gains if isinstance(gains, Mapping) else {},
            query_contraction=0.0,
        )
        # Annotate the inherited query row with the strict v3 semantics.  Keep the phase
        # name for backwards-compatible accounting of joint-rank-scheduled exact queries.
        for hist in (self.joint_population_query_history, self.multiscale_query_history):
            if hist:
                hist[-1].update(
                    {
                        "acquisition_role": "frozen_protected_batch_member",
                        "protected_batch_round": int(batch_round),
                        "protected_batch_purchase_index": int(purchase_index),
                        "protected_batch_size": int(batch_size),
                        "protected_candidate_replaced_by_joint_rank": 0,
                        "joint_rank_role": "ordering_only",
                    }
                )
        self.joint_population_scheduler_history.append(
            {
                "cycle": int(q.state.cycle),
                "preference_id": int(q.state.preference_id),
                "batch_round": int(batch_round),
                "purchase_index": int(purchase_index),
                "lineage_id": int(q.lineage_id),
                "candidate_sequence": str(q.binding.option.sequence),
                "scale_k": int(q.binding.pool.k),
                "frozen_protected_rank": int(q.frozen_rank),
                "shared_rank_gain": int(meta.get("shared_rank_gain", 0)),
                "lineages_helped": ";".join(
                    str(k)
                    for k, val in sorted(
                        (gains if isinstance(gains, Mapping) else {}).items()
                    )
                    if int(val) > 0
                ),
                "eligible_pending_queries": int(meta.get("eligible_query_count", 0)),
                "scheduler_mode": str(meta.get("scheduler_mode", "shared_rank")),
                "candidate_identity_source": "frozen_protected_pegasus_batch",
                "fresh_oracle_query": int(fresh),
            }
        )
        self._protected_batch_purchased_queries += 1
        return entry, bool(fresh)

    def _joint_exact_acceptance_v3(
        self,
        states: Sequence[protected.PopulationFineState],
        *,
        iteration: int,
    ) -> None:
        """Run inherited exact-only acceptance and assert the diversity invariant."""
        before = len(self.joint_population_diversity_history)
        self._joint_exact_acceptance_v2(states, iteration=iteration)
        if len(self.joint_population_diversity_history) != before + 1:
            raise RuntimeError("v3 exact population acceptance did not emit one diversity row")
        row = self.joint_population_diversity_history[-1]
        raw0 = float(row.get("primary_raw_contraction", 0.0))
        raw1 = float(row.get("selected_raw_contraction", 0.0))
        if raw1 > raw0 + 1e-10:
            raise RuntimeError(
                "v3 exact population diversity coordination worsened signed raw contraction"
            )

    # ------------------------------------------------------------------
    # Population fine loop: frozen protected rounds, lazy ordering/cancel
    # ------------------------------------------------------------------
    def _run_population_fine(self, states: list[protected.PopulationFineState]) -> None:
        if not states:
            return

        # Keep the empirically necessary safeguard exactly as in v2/protected PEGASUS.
        self._run_protected_anchors(states)
        if self.stop_requested:
            for s in states:
                if s.reason == "prepared":
                    s.reason = "global_budget_after_protected_anchor"
            self._joint_exact_acceptance_v3(states, iteration=0)
            return

        max_rounds = max(int(s.fresh_cap) for s in states) + 2
        batch_round = 0
        while not self.stop_requested:
            frozen, contraction0, contraction1 = self._freeze_protected_batch(
                states, batch_round=batch_round
            )
            if not frozen:
                break

            self._population_rounds += 1
            batch_start_queries = int(self.oracle.unique_oracle_queries)
            purchased: set[int] = set()
            provisional: set[int] = set()
            exact_skip_logged: set[int] = set()
            purchase_index = 0
            fresh = 0
            cached = 0
            reactivation_before = int(self._protected_batch_reactivation_events)
            provisional_before = int(self._protected_batch_provisional_cancellation_events)

            # Initial pending set is the entire protected batch.  Exact duplicates across
            # lineages can become cached only after a previous frozen member is purchased.
            pending = set(int(lid) for lid in frozen)
            while pending and not self.stop_requested:
                views = self._build_joint_views(states)
                self._log_joint_state(
                    views,
                    iteration=int(batch_round * 1000 + purchase_index),
                    selection_reason="v3_frozen_protected_batch_scheduler",
                )
                chosen, meta = self._choose_frozen_batch_query(frozen, pending, states)
                if chosen is None:
                    # Joint geometry must never block a protected query.  Deterministic
                    # exploit-first fallback over the SAME frozen candidates.
                    lid = min(
                        pending,
                        key=lambda x: (
                            -float(frozen[x].binding.option.predicted_utility),
                            -float(frozen[x].binding.option.predicted_upper),
                            int(frozen[x].frozen_rank),
                            int(x),
                        ),
                    )
                    chosen = frozen[int(lid)]
                    meta = {
                        "eligible_query_count": int(len(pending)),
                        "shared_rank_gain": 0,
                        "per_lineage_rank_gain": {},
                        "scheduler_mode": "protected_exploit_first_fallback",
                    }

                purchase_index += 1
                entry, was_fresh = self._query_frozen_member(
                    chosen,
                    states=states,
                    batch_round=batch_round,
                    purchase_index=purchase_index,
                    batch_size=len(frozen),
                    meta=meta,
                )
                if entry is None:
                    self.stop_requested = True
                    break
                purchased.add(int(chosen.lineage_id))
                fresh += int(was_fresh)
                cached += int(not was_fresh)

                # Recompute the ordinary protected stopping rule after every shared label.
                # Previously provisionally-cancelled candidates may reactivate.
                pending_now, provisional_now, became_exact = self._current_batch_activity(
                    frozen,
                    purchased,
                    provisional,
                    batch_round=batch_round,
                    after_purchase_index=purchase_index,
                )
                provisional = set(provisional_now)
                # A frozen query that became exact through another lineage does not need a
                # redundant cached purchase.  It is removed from this batch, but its lineage
                # remains eligible for a new protected candidate next round if unresolved.
                pending = set(pending_now)
                for lid in sorted(became_exact - exact_skip_logged):
                    q = frozen[lid]
                    exact_skip_logged.add(int(lid))
                    self.joint_population_cancellation_history.append(
                        {
                            "cycle": int(q.state.cycle),
                            "preference_id": int(q.state.preference_id),
                            "batch_round": int(batch_round),
                            "after_purchase_index": int(purchase_index),
                            "lineage_id": int(lid),
                            "candidate_sequence": str(q.binding.option.sequence),
                            "scale_k": int(q.binding.pool.k),
                            "event": "planned_query_became_exact_via_shared_cache",
                            "authority": "exact_shared_archive",
                            "final": 1,
                            "estimated_fresh_query_saved": 0,
                        }
                    )

            # Final batch-end protected reassessment.  Only NOW are provisional skips
            # committed as actual lazy cancellations / resolved lineages.
            final_cancelled: set[int] = set()
            already_exact_final: set[int] = set()
            for lid, q in frozen.items():
                lid = int(lid)
                if lid in purchased:
                    continue
                state = q.state
                if self._binding_is_already_exact(q):
                    already_exact_final.add(lid)
                    continue
                resolved, reason = self._protected_currently_resolved(state)
                if resolved and reason == "protected_pdo_resolved":
                    final_cancelled.add(lid)
                    state.done = True
                    state.reason = "lazy_cancelled_protected_query_after_shared_resolution"
                    self._multiscale_early_stops += 1
                elif resolved and reason == "fine_query_cap_reached":
                    state.done = True
                    remaining = self._active_scale_representatives(
                        state, log_active=False, ignore_query_cap=True
                    )
                    state.unresolved_scales_at_cap = tuple(
                        sorted(set(int(b.pool.k) for b in remaining))
                    )
                    state.reason = "fine_query_cap_reached_with_plausible_challengers"

            # Count saved oracle calls by UNIQUE canceled sequence.  If two lineages had
            # frozen the same sequence, the synchronous protected batch would pay at most
            # one fresh oracle call and the duplicate would be cached.
            saved_unique_sequences = {
                str(frozen[lid].binding.option.sequence) for lid in final_cancelled
            }
            saved_seen: set[str] = set()
            self._protected_batch_fresh_queries_saved += int(len(saved_unique_sequences))
            for lid in sorted(final_cancelled):
                q = frozen[lid]
                self._protected_batch_final_cancellations += 1
                seq = str(q.binding.option.sequence)
                saved_flag = int(seq not in saved_seen)
                saved_seen.add(seq)
                self.joint_population_cancellation_history.append(
                    {
                        "cycle": int(q.state.cycle),
                        "preference_id": int(q.state.preference_id),
                        "batch_round": int(batch_round),
                        "after_purchase_index": int(purchase_index),
                        "lineage_id": int(lid),
                        "candidate_sequence": seq,
                        "scale_k": int(q.binding.pool.k),
                        "event": "final_lazy_cancel",
                        "authority": "ordinary_protected_pdo_stopping_rule",
                        "final": 1,
                        "estimated_fresh_query_saved": int(saved_flag),
                    }
                )

            # Reassess every lineage after the whole frozen batch, exactly as protected
            # PEGASUS does after a synchronous round.  Purchased lineages can therefore
            # stop before the next batch; unresolved ones get a newly constructed protected
            # query only in the NEXT batch, never as an in-batch replacement.
            resolved_after = 0
            for state in states:
                if state.done:
                    resolved_after += 1
                    continue
                if self._fresh_used(state) >= int(state.fresh_cap):
                    remaining = self._active_scale_representatives(
                        state, log_active=False, ignore_query_cap=True
                    )
                    state.unresolved_scales_at_cap = tuple(
                        sorted(set(int(b.pool.k) for b in remaining))
                    )
                    state.done = True
                    state.reason = "fine_query_cap_reached"
                    resolved_after += 1
                    continue
                remaining = self._active_scale_representatives(state, log_active=False)
                if not remaining:
                    state.done = True
                    state.reason = "no_scale_has_plausible_global_challenger"
                    self._multiscale_early_stops += 1
                    resolved_after += 1

            planned = int(len(frozen))
            bought = int(len(purchased))
            if bought > planned:
                self._protected_batch_bound_violations += 1
                raise RuntimeError("v3 purchased more queries than the frozen protected batch")
            self.joint_population_batch_history.append(
                {
                    "cycle": int(states[0].cycle),
                    "preference_id": int(states[0].preference_id),
                    "batch_round": int(batch_round),
                    "planned_protected_queries": int(planned),
                    "purchased_protected_queries": int(bought),
                    "fresh_queries": int(fresh),
                    "cached_queries": int(cached),
                    "final_lazy_cancellations": int(len(final_cancelled)),
                    "planned_became_exact_via_shared_cache": int(len(already_exact_final)),
                    "estimated_fresh_queries_saved": int(len(saved_unique_sequences)),
                    "provisional_cancellation_events": int(
                        self._protected_batch_provisional_cancellation_events - provisional_before
                    ),
                    "reactivation_events": int(
                        self._protected_batch_reactivation_events - reactivation_before
                    ),
                    "query_bound_holds": int(bought <= planned),
                    "global_unique_queries_added": int(
                        self.oracle.unique_oracle_queries - batch_start_queries
                    ),
                    "protected_batch_contraction_cost_value_primary": float(contraction0),
                    "protected_batch_contraction_cost_selected": float(contraction1),
                    "protected_batch_contraction_cost_reduction": float(
                        contraction0 - contraction1
                    ),
                    "resolved_lineages_after_batch": int(resolved_after),
                    "total_lineages": int(len(states)),
                    "joint_rank_role": "ordering_only",
                    "candidate_replacement_allowed": 0,
                }
            )
            self.population_round_history.append(
                {
                    "cycle": int(states[0].cycle),
                    "preference_id": int(states[0].preference_id),
                    "round_index": int(batch_round),
                    "allocation_mode": "v3_lazy_frozen_protected_batch",
                    "active_lineages_queried": int(bought),
                    "planned_protected_queries": int(planned),
                    "lazy_cancelled_queries": int(len(final_cancelled)),
                    "fresh_queries": int(fresh),
                    "cached_queries": int(cached),
                    "global_unique_queries_added": int(
                        self.oracle.unique_oracle_queries - batch_start_queries
                    ),
                    "resolved_lineages_after_round": int(resolved_after),
                    "total_lineages": int(len(states)),
                }
            )

            if self.stop_requested:
                break
            batch_round += 1
            if batch_round > max_rounds:
                raise RuntimeError(
                    "joint population v3 exceeded protected per-lineage query-cap round bound"
                )

        if self.stop_requested:
            for state in states:
                if not state.done:
                    state.reason = "global_budget_exhausted"
                    state.done = True

        # Exact-only population decision/diversity logic is inherited unchanged from v2.
        self._joint_exact_acceptance_v3(states, iteration=batch_round)

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "production v3 coarse reachability/KFM look-ahead/exploit-first exact coarse verification unchanged",
            "same finite realized multiscale Hamming action families as frozen protected Population PDO",
            "one protected exact non-noop fine exposure per active lineage",
            "ordinary protected PEGASUS alone constructs one planned optimization-relevant query per unresolved lineage",
            "the entire protected query batch is frozen before any same-batch exact label is purchased",
            "joint population rank ONLY orders members of that frozen protected batch; it cannot replace a candidate",
            "after every purchased label the ordinary protected-PDO stopping rule is recomputed for every lineage",
            "unbought protected queries are lazily skipped only while that ordinary protected rule says the lineage is resolved",
            "lazy cancellation is provisional within a batch and a frozen query reactivates if later same-batch evidence reopens its lineage",
            "no new challenger is substituted inside the frozen batch; unresolved lineages receive a newly protected candidate only next round",
            "final population choice uses only exact no-op-inclusive near-optimal queried sets",
            "signed raw reachable contraction is coordinate-minimized only inside those exact near-optimal sets",
            "there is no empirical unseen-action certification and exact oracle remains sole acceptance authority",
        ]

        batches = self.joint_population_batch_history
        planned = int(sum(int(r.get("planned_protected_queries", 0)) for r in batches))
        purchased = int(sum(int(r.get("purchased_protected_queries", 0)) for r in batches))
        cancelled = int(sum(int(r.get("final_lazy_cancellations", 0)) for r in batches))
        saved = int(sum(int(r.get("estimated_fresh_queries_saved", 0)) for r in batches))
        strict_batches = int(sum(int(r.get("final_lazy_cancellations", 0)) > 0 for r in batches))
        jp = summary.setdefault("joint_population_pdo", {})
        jp.update(
            {
                "formal_certificate_claimed": False,
                "empirical_unqueried_action_certification": False,
                "protected_candidate_choice_preserved": True,
                "joint_rank_role": "ordering_only_over_frozen_protected_batch",
                "candidate_replacement_allowed": False,
                "lazy_cancellation_authority": "ordinary protected PDO stopping rule only",
                "lazy_cancellation_provisional_within_batch": True,
                "same_batch_reactivation_supported": True,
                "protected_batch_planned_queries": int(planned),
                "protected_batch_purchased_queries": int(purchased),
                "final_lazy_cancelled_protected_queries": int(cancelled),
                "estimated_fresh_protected_queries_saved": int(saved),
                "batches_with_strict_lazy_cancellation": int(strict_batches),
                "protected_batch_query_bound_violations": int(
                    self._protected_batch_bound_violations
                ),
                "provisional_cancellation_events": int(
                    self._protected_batch_provisional_cancellation_events
                ),
                "same_batch_reactivation_events": int(
                    self._protected_batch_reactivation_events
                ),
                "lazy_cancellation_fraction_of_planned": float(cancelled / planned)
                if planned > 0
                else 0.0,
                "fixed_batch_query_bound": "purchased protected batch members <= planned protected batch members",
                "query_scheduler": "protected candidate batch first; shared rank orders only; protected PDO lazily cancels redundant pending queries",
                "scale_handling": "all configured scales remain ordinary protected action families; no scale selector or scale-specific query budget",
                "acceptance_authority": "exact oracle only; final action must already be in exact queried set",
            }
        )
        if "population_pdo" in summary:
            summary["population_pdo"].update(
                {
                    "query_scheduler": "v3 frozen protected batch -> shared-rank order -> protected-PDO lazy cancellation",
                    "joint_rank_role": "ordering_only",
                    "candidate_replacement_allowed": False,
                    "exact_acceptance": "exact queried near-optimal sets -> signed raw contraction coordination -> exact monotone acceptance",
                    "formal_certificate_claimed": False,
                }
            )
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] fine: protected anchor -> FROZEN protected query batch -> joint-rank ORDER only -> protected-PDO lazy cancellation -> exact queried-set acceptance\n"
            f"[{METHOD_NAME}] joint rank CANNOT replace protected candidates and CANNOT declare resolution\n"
            f"[{METHOD_NAME}] cancellation authority = ordinary protected PDO stopping rule; cancellation is provisional until batch end\n"
            f"[{METHOD_NAME}] unseen actions are NEVER selectable; exact oracle alone authorizes acceptance",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = v2.build_parser()
    p.description = __doc__
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Keep v1/v2 validation because the CLI and all physical/model-side settings are
    # intentionally unchanged.
    from . import pdo_pegasus_joint_population_pdo as v1

    v1._validate_args(args)
    JointPopulationPDORunnerV3(args).run()


if __name__ == "__main__":
    main()
