"""PEGASUS Scale-Factored Tournament, implemented directly from production v3.

The coarse optimizer is *exactly* PEGASUS v3.  Only the v3 fine Hamming-1 turn is
replaced by a scale-factored tournament over concrete exact-Hamming action families.
No code from the experimental H2/multiscale diagnostic branches is imported.

Fine-stage principle
--------------------
1. Expose actual actions independently at each requested Hamming radius.  H1 uses the
   original exhaustive v3 action family when tractable; larger radii use a bounded set
   of uniformly sampled compatible exact-k edits.  No objective-response additivity is
   assumed at any radius.
2. Compute frozen KFM terminal features for every actual candidate and use the same
   inference-time global task readout family used by v3's successful exploit-first
   coarse verifier.
3. Freeze a small contender set independently inside every radius.
4. Query the initially top-ranked candidate from each radius (cached labels are free).
5. The best exact candidate across all radii becomes the global leader.  A radius remains
   active only while some coherent v3 global-readout member says one of its frozen
   unqueried contenders can beat that exact leader by epsilon_dec.
6. Query only the highest-value challenger among the still-active radii, refit the global
   task readout from the shared exact archive, and repeat to a small fresh-query cap.
7. Accept only the best exactly evaluated improving action across every radius; otherwise
   take no-op.

Thus scale is never predicted or committed to.  Scales compete through exact champions,
while PDO is the stopping/elimination rule.  Cross-scale local response interpolation is
absent; the only shared information is the ordinary v3 global KFM task readout/cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from . import pdo_pegasus_v3 as v3
from .kfocus.objectives import OPT_NAMES, RAW_NAMES
from .kfocus.regions import terminal_z
from .pdo_pegasus_v3 import (
    GlobalCoarseCommittee,
    choose_exploit_first_challenger,
    plausible_challenger_indices,
)
from .pdo_multiscale_actions import (
    MultiscaleAction,
    expose_exact_hamming_actions,
    parse_radius_spec,
    verify_action_family,
)
from .utils import write_csv, write_json

IMPLEMENTATION_VERSION = "pegasus-scale-factored-tournament-v1"
METHOD_NAME = "PEGASUS Scale-Factored Tournament"


@dataclass
class ScalePool:
    k: int
    label: str
    actions: list[MultiscaleAction]
    z: np.ndarray
    initial_committee: GlobalCoarseCommittee
    initial_median_utilities: np.ndarray  # candidate-only, no-op excluded
    initial_rank: np.ndarray              # candidate indices ordered by initial median utility
    contender_indices: np.ndarray        # no-op-inclusive local indices 1..N
    cached_local_indices: tuple[int, ...]


class ScaleFactoredTournamentRunner(v3.UnifiedPDOPegasusV3Runner):
    """Original v3 runner with only the fine turn replaced by multiscale tournament PDO."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.multiscale_turn_history: list[dict[str, Any]] = []
        self.multiscale_scale_history: list[dict[str, Any]] = []
        self.multiscale_query_history: list[dict[str, Any]] = []
        self._multiscale_fresh_queries = 0
        self._multiscale_accepted_moves = 0
        self._multiscale_turns = 0
        self._multiscale_early_stops = 0
        self._multiscale_challenger_queries = 0

    def _save_progress(self) -> None:
        super()._save_progress()
        if self.multiscale_turn_history:
            write_csv(self.out / "multiscale_turn_history.csv", self.multiscale_turn_history)
        if self.multiscale_scale_history:
            write_csv(self.out / "multiscale_scale_history.csv", self.multiscale_scale_history)
        if self.multiscale_query_history:
            write_csv(self.out / "multiscale_query_history.csv", self.multiscale_query_history)

    def _resolved_radii(self, lineage: base.LineageState) -> tuple[int, ...]:
        seq = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        return parse_radius_spec(str(self.args.pdo_multiscale_radius_spec), len(seq))

    def _scale_rng(
        self,
        *,
        cycle: int,
        preference_id: int,
        lineage_id: int,
        incumbent_sequence: str,
        k: int,
    ) -> np.random.Generator:
        seed = base._stable_seed(
            "pdo_multiscale_tournament",
            int(self.args.seed),
            int(cycle),
            int(preference_id),
            int(lineage_id),
            str(incumbent_sequence),
            int(k),
        )
        return np.random.default_rng(int(seed))

    def _candidate_exact_map(
        self,
        pool: ScalePool,
        lineage: base.LineageState,
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {0: np.asarray(lineage.scores, dtype=np.float64).copy()}
        for j, action in enumerate(pool.actions, start=1):
            entry = self.archive_by_sequence.get(action.target_sequence)
            if entry is not None:
                out[int(j)] = np.asarray(entry.scores, dtype=np.float64).copy()
        return out

    def _build_scale_committee(
        self,
        pool: ScalePool,
        lineage: base.LineageState,
        incumbent_z: np.ndarray,
        preference: np.ndarray,
    ) -> GlobalCoarseCommittee:
        return self._build_v3_committee(
            candidate_z=pool.z,
            incumbent_z=incumbent_z,
            incumbent_scores=np.asarray(lineage.scores, dtype=np.float64),
            exact_map=self._candidate_exact_map(pool, lineage),
            preference=preference,
        )

    def _build_scale_pools(
        self,
        *,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
    ) -> tuple[list[ScalePool], np.ndarray]:
        incumbent_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        radii = self._resolved_radii(lineage)
        anchors = self._diversity_anchors(preference_id, lineage.lineage_id)
        incumbent_z = terminal_z(
            self.model,
            lineage.tokens.reshape(1, -1),
            batch_size=int(self.args.feature_batch_size),
        )[0]
        pref = self.preferences[int(preference_id)]
        pools: list[ScalePool] = []
        for k in radii:
            actions = expose_exact_hamming_actions(
                self.lkf,
                lineage.tokens,
                k=int(k),
                candidate_cap=int(self.args.pdo_multiscale_candidates_per_scale),
                rng=self._scale_rng(
                    cycle=cycle,
                    preference_id=preference_id,
                    lineage_id=lineage.lineage_id,
                    incumbent_sequence=incumbent_sequence,
                    k=int(k),
                ),
                anchors=anchors,
                min_lineage_hamming=float(self.args.min_lineage_hamming),
                exhaustive_h1_max_actions=int(self.args.pdo_multiscale_exhaustive_h1_max_actions),
                max_sampling_attempts=int(self.args.pdo_multiscale_max_sampling_attempts),
            )
            verify_action_family(actions, incumbent_sequence, k=int(k))
            if not actions:
                self.multiscale_scale_history.append(
                    {
                        "cycle": int(cycle),
                        "preference_id": int(preference_id),
                        "lineage_id": int(lineage.lineage_id),
                        "incumbent_sequence": incumbent_sequence,
                        "scale_k": int(k),
                        "candidate_count": 0,
                        "status": "no_diversity_admissible_actions",
                    }
                )
                continue
            toks = torch.stack([a.tokens for a in actions], dim=0)
            z = terminal_z(self.model, toks, batch_size=int(self.args.feature_batch_size))
            # Initial ranking is frozen before any new multiscale query this turn, exactly
            # like v3 freezes its coarse contender pool before exploit-first verification.
            exact_map: dict[int, np.ndarray] = {0: np.asarray(lineage.scores, dtype=np.float64).copy()}
            cached_local: list[int] = []
            for j, action in enumerate(actions, start=1):
                entry = self.archive_by_sequence.get(action.target_sequence)
                if entry is not None:
                    exact_map[int(j)] = np.asarray(entry.scores, dtype=np.float64).copy()
                    cached_local.append(int(j))
            committee = self._build_v3_committee(
                candidate_z=z,
                incumbent_z=incumbent_z,
                incumbent_scores=np.asarray(lineage.scores, dtype=np.float64),
                exact_map=exact_map,
                preference=pref,
            )
            med = np.asarray(committee.median_utilities[1:], dtype=np.float64)
            order0 = np.argsort(-med, kind="mergesort").astype(np.int64)
            cap = min(int(self.args.pdo_multiscale_contender_cap_per_scale), len(actions))
            cont = (order0[:cap] + 1).astype(np.int64)
            # Cached exact actions are free and authoritative; never exclude one merely
            # because the global prior ranks it outside the frozen prefix.
            if cached_local:
                cont = np.asarray(sorted(set(cont.tolist()) | set(cached_local)), dtype=np.int64)
            rank_of = np.empty(len(actions), dtype=np.int64)
            rank_of[order0] = np.arange(len(actions), dtype=np.int64)
            pools.append(
                ScalePool(
                    k=int(k),
                    label=f"H{k}",
                    actions=actions,
                    z=np.asarray(z, dtype=np.float64),
                    initial_committee=committee,
                    initial_median_utilities=med,
                    initial_rank=rank_of,
                    contender_indices=cont,
                    cached_local_indices=tuple(cached_local),
                )
            )
            self.multiscale_scale_history.append(
                {
                    "cycle": int(cycle),
                    "preference_id": int(preference_id),
                    "lineage_id": int(lineage.lineage_id),
                    "incumbent_sequence": incumbent_sequence,
                    "scale_k": int(k),
                    "candidate_count": int(len(actions)),
                    "contender_count": int(len(cont)),
                    "cached_exact_candidates": int(len(cached_local)),
                    "initial_top_candidate": actions[int(order0[0])].target_sequence,
                    "initial_top_predicted_utility": float(med[int(order0[0])]),
                    "status": "ready",
                }
            )
        return pools, np.asarray(incumbent_z, dtype=np.float64)

    def _global_best_exact(
        self,
        pools: Sequence[ScalePool],
        lineage: base.LineageState,
        preference: np.ndarray,
    ) -> tuple[ScalePool | None, int, float]:
        best_pool: ScalePool | None = None
        best_local = 0
        best_u = float(base._utility(lineage.scores, preference, self.args.rho))
        for pool in pools:
            exact = self._candidate_exact_map(pool, lineage)
            for local_idx, score in exact.items():
                if int(local_idx) == 0:
                    continue
                u = float(base._utility(score, preference, self.args.rho))
                if u > best_u + 1e-15:
                    best_u = u
                    best_pool = pool
                    best_local = int(local_idx)
        return best_pool, best_local, best_u

    def _query_multiscale_action(
        self,
        *,
        pool: ScalePool,
        local_index: int,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
        turn_id: int,
        query_order: int,
        phase: str,
        committee: GlobalCoarseCommittee,
    ) -> tuple[base.ArchiveEntry | None, bool]:
        idx = int(local_index)
        if idx <= 0 or idx > len(pool.actions):
            raise ValueError("multiscale local candidate index out of range")
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
            exact_u = float(base._utility(rec.scores, self.preferences[int(preference_id)], self.args.rho))
            entry = self._record_exact(
                action.tokens,
                pool.z[idx - 1],
                rec,
                source="pdo_multiscale_tournament",
                preference_id=int(preference_id),
                lineage_id=int(lineage.lineage_id),
                slate_id=None,
                verification_rank=int(query_order),
                utility_at_query=exact_u,
            )
        after = int(self.oracle.unique_oracle_queries)
        fresh = after > before
        if fresh:
            delta = int(after - before)
            self._multiscale_fresh_queries += delta
            lineage.verification_queries += delta
        self._record_preference_query_use(preference_id, action.target_sequence)
        exact_u = float(base._utility(entry.scores, self.preferences[int(preference_id)], self.args.rho))
        row: dict[str, Any] = {
            "turn_id": int(turn_id),
            "cycle": int(cycle),
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "incumbent_sequence": base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0],
            "query_order": int(query_order),
            "phase": str(phase),
            "scale_k": int(pool.k),
            "action_id": str(action.action_id),
            "candidate_sequence": str(action.target_sequence),
            "fresh_oracle_query": int(fresh),
            "global_unique_query_index": int(self.oracle.unique_oracle_queries),
            "predicted_median_utility": float(committee.median_utilities[idx]),
            "predicted_utility_lower": float(committee.utility_lower[idx]),
            "predicted_utility_upper": float(committee.utility_upper[idx]),
            "exact_utility": float(exact_u),
            "exact_gain": float(exact_u - float(lineage.utility)),
            "accepted": 0,
        }
        for j, name in enumerate(OPT_NAMES):
            row[f"exact_score_{name}"] = float(entry.scores[j])
        for j, name in enumerate(RAW_NAMES):
            row[f"raw_{name}"] = float(entry.raw[j])
        self.multiscale_query_history.append(row)
        return entry, fresh

    def _mark_multiscale_accept(self, turn_id: int, sequence: str) -> None:
        for row in reversed(self.multiscale_query_history):
            if int(row.get("turn_id", -1)) != int(turn_id):
                break
            if str(row.get("candidate_sequence", "")) == str(sequence):
                row["accepted"] = 1
                return

    def _run_pdo_turn(
        self,
        preference_id: int,
        lineage: base.LineageState,
        cycle: int,
    ) -> dict[str, Any]:
        """Scale-factored tournament replacing only v3's original fine H1 PDO turn."""
        self.pdo_turn_counter += 1
        self._multiscale_turns += 1
        turn_id = int(self.pdo_turn_counter)
        start_sequence = base.decode_esm_tokens(lineage.tokens.reshape(1, -1))[0]
        start_utility = float(lineage.utility)
        before_q = int(self.oracle.unique_oracle_queries)
        pref = self.preferences[int(preference_id)]

        pools, incumbent_z = self._build_scale_pools(
            preference_id=preference_id,
            lineage=lineage,
            cycle=cycle,
        )
        if not pools:
            out = {
                "turn_id": turn_id,
                "cycle": int(cycle),
                "preference_id": int(preference_id),
                "lineage_id": int(lineage.lineage_id),
                "incumbent_sequence": start_sequence,
                "scale_count": 0,
                "paid_queries": 0,
                "accepted": 0,
                "gain": 0.0,
                "reason": "no_multiscale_actions",
            }
            self.multiscale_turn_history.append(out)
            return out

        if str(self.args.pdo_budget_mode) == "matched_budget":
            fresh_cap = min(
                int(self.args.pdo_multiscale_query_cap_per_turn),
                int(self.args.verification_k),
            )
        else:
            fresh_cap = int(self.args.pdo_multiscale_query_cap_per_turn)
        if fresh_cap <= 0:
            raise RuntimeError("multiscale fine query cap resolved to zero")

        # Freeze one initial champion per scale before any new query this turn.  Champion
        # phase ordering is exploit-first across scales only to make budget truncation
        # graceful; with the default cap every configured scale receives a champion.
        champion_rows: list[tuple[float, int, ScalePool, int]] = []
        for pool in pools:
            if len(pool.actions) == 0:
                continue
            local_idx = int(np.argmax(pool.initial_median_utilities)) + 1
            champion_rows.append(
                (
                    float(pool.initial_median_utilities[local_idx - 1]),
                    -int(pool.k),
                    pool,
                    local_idx,
                )
            )
        champion_rows.sort(key=lambda x: (-x[0], -x[1]))

        query_order = 0
        champion_scales_touched: set[int] = set()
        reason = "champion_phase_complete"
        for _pred, _negk, pool, local_idx in champion_rows:
            # Cached champions are exact and cost no fresh budget, so always register them.
            committee = self._build_scale_committee(pool, lineage, incumbent_z, pref)
            action = pool.actions[int(local_idx) - 1]
            if action.target_sequence in self.archive_by_sequence:
                champion_scales_touched.add(int(pool.k))
                continue
            fresh_used = int(self.oracle.unique_oracle_queries) - before_q
            if fresh_used >= fresh_cap:
                reason = "query_cap_during_scale_champion_phase"
                break
            query_order += 1
            entry, _fresh = self._query_multiscale_action(
                pool=pool,
                local_index=local_idx,
                preference_id=preference_id,
                lineage=lineage,
                cycle=cycle,
                turn_id=turn_id,
                query_order=query_order,
                phase="scale_champion",
                committee=committee,
            )
            if entry is None:
                reason = "global_budget_exhausted_during_scale_champion_phase"
                break
            champion_scales_touched.add(int(pool.k))
            if self.stop_requested:
                break

        # Tournament phase: a scale is eliminated only when none of its frozen unqueried
        # contenders can beat the current exact global leader under any coherent v3
        # global-readout member.  This is PDO over scale-factored contender sets.
        while not self.stop_requested:
            fresh_used = int(self.oracle.unique_oracle_queries) - before_q
            if fresh_used >= fresh_cap:
                reason = "multiscale_query_cap_reached"
                break
            best_pool, best_local, best_exact_u = self._global_best_exact(pools, lineage, pref)
            candidate_challengers: list[tuple[float, float, int, ScalePool, int, GlobalCoarseCommittee]] = []
            active_scale_count = 0
            for pool in pools:
                committee = self._build_scale_committee(pool, lineage, incumbent_z, pref)
                exact_map = self._candidate_exact_map(pool, lineage)
                observed = sorted(int(i) for i in exact_map)
                plausible = plausible_challenger_indices(
                    committee,
                    contender_indices=pool.contender_indices,
                    observed_indices=observed,
                    best_exact_utility=float(best_exact_u),
                    epsilon_dec=float(self.args.pdo_multiscale_epsilon_dec),
                )
                if len(plausible) == 0:
                    continue
                active_scale_count += 1
                chosen = choose_exploit_first_challenger(
                    committee,
                    plausible,
                    initial_rank=pool.initial_rank,
                )
                if chosen is None:
                    continue
                idx = int(chosen)
                candidate_challengers.append(
                    (
                        float(committee.median_utilities[idx]),
                        float(committee.utility_upper[idx]),
                        -int(pool.initial_rank[idx - 1]),
                        pool,
                        idx,
                        committee,
                    )
                )
            if not candidate_challengers:
                self._multiscale_early_stops += 1
                reason = "no_scale_has_plausible_global_challenger"
                break
            # Global tournament arbitration among one best current challenger per active
            # scale: highest updated median utility, then upper value, then initial rank.
            candidate_challengers.sort(key=lambda x: (-x[0], -x[1], -x[2], int(x[3].k)))
            _med, _upper, _negrank, pool, idx, committee = candidate_challengers[0]
            query_order += 1
            entry, fresh = self._query_multiscale_action(
                pool=pool,
                local_index=idx,
                preference_id=preference_id,
                lineage=lineage,
                cycle=cycle,
                turn_id=turn_id,
                query_order=query_order,
                phase="global_challenger",
                committee=committee,
            )
            if entry is None:
                reason = "global_budget_exhausted_during_multiscale_challenger"
                break
            if fresh:
                self._multiscale_challenger_queries += 1
            reason = "exact_challenger_added_and_global_readout_refit"

        # Exact global arbitration: no prediction can accept a move.
        best_pool, best_local, best_exact_u = self._global_best_exact(pools, lineage, pref)
        accepted = False
        accepted_k = -1
        accepted_sequence = ""
        if (
            best_pool is not None
            and int(best_local) > 0
            and float(best_exact_u - start_utility) > float(self.args.accept_epsilon)
        ):
            action = best_pool.actions[int(best_local) - 1]
            entry = self.archive_by_sequence.get(action.target_sequence)
            if entry is None:
                raise RuntimeError("multiscale exact arbitration selected action without archive entry")
            accepted = self._accept_exact_entry(
                lineage,
                action.as_fine_action(),
                entry,
                turn_id=turn_id,
            )
            if accepted:
                self._multiscale_accepted_moves += 1
                accepted_k = int(best_pool.k)
                accepted_sequence = str(action.target_sequence)
                self._mark_multiscale_accept(turn_id, accepted_sequence)
                reason = f"best_exact_multiscale_action_accepted_after_{reason}"

        paid = int(self.oracle.unique_oracle_queries) - before_q
        exact_scales = 0
        for pool in pools:
            exact_map = self._candidate_exact_map(pool, lineage)
            if any(int(i) != 0 for i in exact_map):
                exact_scales += 1
        out = {
            "turn_id": int(turn_id),
            "cycle": int(cycle),
            "preference_id": int(preference_id),
            "lineage_id": int(lineage.lineage_id),
            "incumbent_sequence": start_sequence,
            "incumbent_utility_before": float(start_utility),
            "incumbent_utility_after": float(lineage.utility),
            "configured_radii": ";".join(str(p.k) for p in pools),
            "scale_count": int(len(pools)),
            "scales_with_exact_candidate": int(exact_scales),
            "champion_scales_touched": int(len(champion_scales_touched)),
            "candidate_count_total": int(sum(len(p.actions) for p in pools)),
            "contender_count_total": int(sum(len(p.contender_indices) for p in pools)),
            "fresh_query_cap": int(fresh_cap),
            "paid_queries": int(paid),
            "accepted": int(accepted),
            "accepted_scale_k": int(accepted_k),
            "accepted_sequence": accepted_sequence,
            "gain": float(lineage.utility - start_utility),
            "reason": str(reason),
        }
        self.multiscale_turn_history.append(out)
        if int(self.args.save_every_slates) > 0 and turn_id % int(self.args.save_every_slates) == 0:
            self._save_progress()
        return out

    def finalize(self) -> dict[str, Any]:
        summary = super().finalize()
        summary["method"] = METHOD_NAME
        summary["implementation_version"] = IMPLEMENTATION_VERSION
        summary["core_algorithm"] = [
            "unchanged PEGASUS v3 physical coarse branching and A_C allocation",
            "unchanged v3 exploit-first global coarse PDO",
            "same-turn exact-label cache reused by multiscale fine tournament",
            "independent concrete exact-Hamming action exposure at each configured fine radius",
            "KFM terminal task readout ranks actual candidates within each radius",
            "frozen top contender set independently inside every radius",
            "initial exact champion from every radius subject only to the shared fresh-query cap",
            "global exact leader maintained across radii",
            "radius eliminated when no coherent v3 global-readout member contains a challenger",
            "challenger queries are exploit-first across the still-active radii",
            "exact best queried action across all radii is the only fine acceptance path",
            "no Hk objective additivity, no cross-scale local response interpolation, no new uncertainty module",
        ]
        summary.setdefault("multiscale_tournament", {})
        summary["multiscale_tournament"].update(
            {
                "radius_spec": str(self.args.pdo_multiscale_radius_spec),
                "candidates_per_sampled_scale": int(self.args.pdo_multiscale_candidates_per_scale),
                "exhaustive_h1_max_actions": int(self.args.pdo_multiscale_exhaustive_h1_max_actions),
                "contender_cap_per_scale": int(self.args.pdo_multiscale_contender_cap_per_scale),
                "epsilon_dec": float(self.args.pdo_multiscale_epsilon_dec),
                "fresh_query_cap_per_turn": int(self.args.pdo_multiscale_query_cap_per_turn),
                "turns": int(self._multiscale_turns),
                "fresh_queries": int(self._multiscale_fresh_queries),
                "challenger_fresh_queries": int(self._multiscale_challenger_queries),
                "accepted_moves": int(self._multiscale_accepted_moves),
                "early_stops_no_global_challenger": int(self._multiscale_early_stops),
                "fine_h1_local_linear_response_model_used": False,
                "new_uncertainty_model_added": False,
                "cross_scale_local_confidence_transfer": False,
                "final_arbitration": "exact_global_best_among_queried_scale_candidates_plus_noop",
            }
        )
        # Correct inherited v1/v3 metadata that otherwise describes the disabled local-H1
        # response model.  The coarse-v3 block itself remains authoritative and unchanged.
        if "unified_pdo_pegasus_v3" in summary:
            u = summary["unified_pdo_pegasus_v3"]
            u["schedule"] = "coarse_then_scale_factored_fine_tournament_same_turn"
            roles = u.setdefault("roles", {})
            roles["LKF"] = (
                "physical stochastic coarse branching plus explicit discrete fine intervention reachability"
            )
            roles["fine_PDO"] = (
                "scale-factored exploit-first tournament PDO over concrete exact-Hamming actions"
            )
            if "coarse_pdo" in u:
                u["coarse_pdo"]["same_turn_labels_reused_by_fine_PDO"] = True
        if "pdo" in summary:
            pdo = summary["pdo"]
            pdo["implementation_version"] = IMPLEMENTATION_VERSION
            pdo["epsilon_dec"] = float(self.args.pdo_multiscale_epsilon_dec)
            pdo["fine_query_cap_per_turn"] = int(self.args.pdo_multiscale_query_cap_per_turn)
            pdo["fine_fresh_queries"] = int(self._multiscale_fresh_queries)
            pdo["fine_verification_fresh_queries"] = int(self._multiscale_fresh_queries)
            pdo["fine_acquisition_fresh_queries"] = 0
            pdo["fine_accepted_moves"] = int(self._multiscale_accepted_moves)
            pdo["fine_action_family"] = (
                "scale-factored concrete exact-Hamming candidate families + one global no-op"
            )
            pdo["fine_geometry"] = (
                "KFM terminal features for ranking; original H1 local-linear response geometry disabled"
            )
            pdo["acquisition_probe_acceptance"] = (
                "exploit-first exact scale champions followed by only plausible global challengers"
            )
            pdo["coarse_labels_directly_enter_local_information_matrix"] = False
            pdo["cross_scale_prior_can_change_member_means"] = True
            pdo["cross_scale_confidence_claim"] = (
                "no cross-scale local response/confidence transfer; scales share only the ordinary global KFM task archive"
            )
            pdo["local_h1_response_model_enabled"] = False
            pdo["no_op_in_action_family"] = True
            pdo["formal_certificate_claimed"] = False
            # Replace inherited H1 per-preference accounting with the actual tournament logs.
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
                    "fresh_scale_champion_queries": int(sum(str(r.get("phase", "")) == "scale_champion" for r in qrows)),
                    "fresh_global_challenger_queries": int(sum(str(r.get("phase", "")) == "global_challenger" for r in qrows)),
                    "fine_accepted_moves": int(sum(int(r.get("accepted", 0)) for r in trows)),
                }
            pdo["per_preference_fine_accounting"] = per_pref
            if "budget_accounting" in pdo:
                ba = pdo["budget_accounting"]
                ba["fine_fresh_queries"] = int(self._multiscale_fresh_queries)
                ba["actual_global_unique_queries_fine_inclusive"] = int(self.oracle.unique_oracle_queries)
                if str(self.args.pdo_budget_mode) == "protected_baseline":
                    ba["protected_coarse_charged_queries"] = int(
                        self.oracle.unique_oracle_queries - self._multiscale_fresh_queries
                    )
                ba["note"] = (
                    "Fine-query accounting in this block is scale-factored tournament accounting; "
                    "the original local-H1 PDO query history is disabled."
                )
        write_json(self.out / "summary.json", summary)
        self._save_progress()
        return summary

    def run(self) -> dict[str, Any]:
        print(
            f"[{METHOD_NAME}] implementation={IMPLEMENTATION_VERSION}\n"
            f"[{METHOD_NAME}] coarse stage is unchanged production v3\n"
            f"[{METHOD_NAME}] fine stage: scale champions -> global exact leader -> plausible challengers\n"
            f"[{METHOD_NAME}] no Hk response additivity; exact oracle remains sole acceptance authority",
            flush=True,
        )
        self.discovery()
        self.optimize()
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = v3.build_parser()
    p.description = __doc__
    g = p.add_argument_group("Scale-factored fine PDO tournament")
    g.add_argument(
        "--pdo-multiscale-radius-spec",
        default="1,2,4",
        help=(
            "Comma-separated exact or length-normalized intervention radii. Examples: "
            "'1,2,4' or '1%%,2%%,5%%'. H1 is exhaustive when tractable; larger sets are sampled."
        ),
    )
    g.add_argument(
        "--pdo-multiscale-candidates-per-scale",
        type=int,
        default=256,
        help="Number of sampled concrete candidates for non-exhaustive radii.",
    )
    g.add_argument(
        "--pdo-multiscale-exhaustive-h1-max-actions",
        type=int,
        default=2048,
        help="Enumerate all H1 actions only when L*19 is at most this value; otherwise sample H1 too.",
    )
    g.add_argument(
        "--pdo-multiscale-contender-cap-per-scale",
        type=int,
        default=32,
        help="Freeze this many top pre-query candidates independently inside each radius.",
    )
    g.add_argument("--pdo-multiscale-epsilon-dec", type=float, default=0.002)
    g.add_argument(
        "--pdo-multiscale-query-cap-per-turn",
        type=int,
        default=8,
        help="Fresh exact fine queries shared by all radii in one tournament turn; cached labels are free.",
    )
    g.add_argument(
        "--pdo-multiscale-max-sampling-attempts",
        type=int,
        default=100000,
        help="Maximum proposal attempts per sampled Hk action family after diversity filtering.",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    v3._validate_args(args)
    radii = parse_radius_spec(str(args.pdo_multiscale_radius_spec), int(args.peptide_length))
    if int(args.pdo_multiscale_candidates_per_scale) <= 0:
        raise ValueError("--pdo-multiscale-candidates-per-scale must be positive")
    if int(args.pdo_multiscale_exhaustive_h1_max_actions) < 0:
        raise ValueError("--pdo-multiscale-exhaustive-h1-max-actions cannot be negative")
    if int(args.pdo_multiscale_contender_cap_per_scale) <= 0:
        raise ValueError("--pdo-multiscale-contender-cap-per-scale must be positive")
    if int(args.pdo_multiscale_query_cap_per_turn) <= 0:
        raise ValueError("--pdo-multiscale-query-cap-per-turn must be positive")
    if int(args.pdo_multiscale_max_sampling_attempts) <= 0:
        raise ValueError("--pdo-multiscale-max-sampling-attempts must be positive")
    if (not math.isfinite(float(args.pdo_multiscale_epsilon_dec))) or float(args.pdo_multiscale_epsilon_dec) < 0:
        raise ValueError("--pdo-multiscale-epsilon-dec must be finite and nonnegative")
    # This is a warning-worthy configuration, not an error: matched-budget coarse queries
    # can reduce the fine cap further.  The implementation handles truncation gracefully.
    if len(radii) > int(args.pdo_multiscale_query_cap_per_turn):
        print(
            "[Scale-Factored PDO warning] more configured radii than fresh fine queries; "
            "champion phase may be truncated by exploit-first priority.",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    ScaleFactoredTournamentRunner(args).run()


if __name__ == "__main__":
    main()
