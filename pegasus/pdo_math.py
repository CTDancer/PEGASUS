"""Pure mathematical utilities for PEGASUS v1.

Operational PDO uses a *joint vector-response version set*.  Each member predicts the
full objective vector for every realizable action; augmented-Tchebycheff utility is then
evaluated inside each member.  This preserves cross-objective and cross-action structure
and avoids the unrealistically large Cartesian uncertainty set induced by independent
coordinate-wise score boxes.

Axis-aligned box utilities are retained only for diagnostics/backward compatibility.
They are not the operational PDO decision rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PDODecision:
    """Joint-version-set PDO state for one fine action family."""

    utility_lower: np.ndarray
    utility_upper: np.ndarray
    gain_lower: np.ndarray
    gain_upper: np.ndarray
    regret_upper: np.ndarray
    plausible_indices: np.ndarray
    best_certified_index: int | None
    best_mean_index: int
    epsilon_dec: float
    member_utilities: np.ndarray
    member_regrets: np.ndarray
    member_best_indices: np.ndarray
    common_epsilon_indices: np.ndarray
    distinct_member_winners: int
    calibration_slack: float
    certification_enabled: bool


def empirical_confidence_verification_index(
    decision: PDODecision,
    *,
    no_op_index: int,
    accept_epsilon: float,
    min_confidence_gain: float = 0.0,
) -> int | None:
    """Return the empirical-confidence action only when it is worth exact verification.

    The action must be non-noop and its worst-member gain must exceed both the configured
    confidence threshold and the optimizer's exact acceptance margin.  Sharing this helper
    between replay and production prevents a subtle policy mismatch where an action could be
    called confidently positive yet be too small to satisfy exact acceptance.
    """
    ci = decision.best_certified_index
    if ci is None:
        return None
    idx = int(ci)
    if idx == int(no_op_index):
        return None
    accept = float(accept_epsilon)
    minimum = float(min_confidence_gain)
    if not np.isfinite(accept) or accept < 0.0:
        raise ValueError("accept_epsilon must be finite and nonnegative")
    if not np.isfinite(minimum):
        raise ValueError("min_confidence_gain must be finite")
    threshold = max(accept, minimum)
    if float(decision.gain_lower[idx]) <= threshold:
        return None
    return idx


@dataclass(frozen=True)
class ParetoSwitchDiagnostic:
    deficits: np.ndarray
    active_index: int
    active_gap: float
    critical_indices: np.ndarray
    switch_safe: bool
    per_objective_reachable_weighted_change: np.ndarray


@dataclass
class EmpiricalConfidenceRecovery:
    """Small state machine for falsified empirical-confidence decisions.

    A failed exact verification suppresses confidence stopping until one *fresh*
    decision-relevant recovery probe has been observed. Cached labels cannot clear the
    recovery state because they add no new information.
    """

    pending: bool = False
    failed_verifications: int = 0
    recovery_probes: int = 0

    def confidence_allowed(self, base_enabled: bool) -> bool:
        return bool(base_enabled and not self.pending)

    def observe_verification(self, exact_gain: float, accept_epsilon: float) -> bool:
        gain = float(exact_gain)
        accept = float(accept_epsilon)
        if not np.isfinite(gain):
            raise ValueError("exact_gain must be finite")
        if not np.isfinite(accept) or accept < 0.0:
            raise ValueError("accept_epsilon must be finite and nonnegative")
        failed = gain <= accept
        if failed:
            self.pending = True
            self.failed_verifications += 1
        return failed

    def observe_recovery_probe(self, *, fresh: bool) -> None:
        if not self.pending:
            return
        if not bool(fresh):
            raise RuntimeError("a cached label cannot clear PDO recovery state")
        self.pending = False
        self.recovery_probes += 1


@dataclass(frozen=True)
class FreshQueryGate:
    allowed: bool
    reason: str


def fresh_h1_query_gate(
    *,
    turn_fresh_queries: int,
    turn_cap: int,
    state_fresh_queries: int,
    state_cap: int,
    global_budget_remaining: int | None,
) -> FreshQueryGate:
    """Check all fresh-query caps before paying for an H1 objective vector.

    ``global_budget_remaining=None`` means unlimited.  Cached exact labels bypass this
    helper entirely because they cost zero queries.
    """
    if int(turn_fresh_queries) < 0 or int(state_fresh_queries) < 0:
        raise ValueError("fresh-query counts cannot be negative")
    if int(turn_cap) < 0 or int(state_cap) < 0:
        raise ValueError("fresh-query caps cannot be negative")
    if int(turn_fresh_queries) >= int(turn_cap):
        return FreshQueryGate(False, "turn_fresh_query_cap_reached")
    if int(state_fresh_queries) >= int(state_cap):
        return FreshQueryGate(False, "state_fresh_query_cap_reached")
    if global_budget_remaining is not None and int(global_budget_remaining) <= 0:
        return FreshQueryGate(False, "global_budget_exhausted")
    return FreshQueryGate(True, "allowed")


def matched_coarse_verification_slots(
    *,
    verification_k: int,
    fine_paid_queries: int,
    fine_accepted: bool,
) -> int:
    """Remaining frozen-coarse verification slots in matched-budget mode."""
    k = int(verification_k)
    q = int(fine_paid_queries)
    if k <= 0:
        raise ValueError("verification_k must be positive")
    if q < 0:
        raise ValueError("fine_paid_queries cannot be negative")
    if q > k:
        raise ValueError("fine_paid_queries cannot exceed verification_k in matched-budget mode")
    if bool(fine_accepted):
        return 0
    return max(0, k - q)


def augmented_tchebycheff_utility(
    scores: np.ndarray,
    preference: np.ndarray,
    *,
    rho: float = 0.05,
    reference: np.ndarray | None = None,
) -> np.ndarray | float:
    """Maximize-oriented augmented Tchebycheff utility.

    ``scores`` may be ``[m]`` or ``[n,m]``. Larger scores are always better.
    Preference weights are normalized to sum one exactly as in the frozen production
    PEGASUS scalarization, so ``epsilon_dec`` lives on the same absolute utility scale.
    """
    x = np.asarray(scores, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("scores must be [m] or [n,m]")
    if np.any(~np.isfinite(x)):
        raise ValueError("scores must be finite")
    w = np.asarray(preference, dtype=np.float64).reshape(-1)
    if x.shape[1] != len(w):
        raise ValueError("score/preference dimension mismatch")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("preference weights must be finite and nonnegative")
    if not np.any(w > 0):
        raise ValueError("at least one preference weight must be positive")
    # Match the frozen production PEGASUS scalarization exactly.  The base
    # implementation normalizes preferences to the simplex before applying the
    # augmented Tchebycheff utility.  This matters not only for ranking but for
    # the absolute decision-regret tolerance epsilon_dec.
    w = w / float(np.sum(w))
    rrho = float(rho)
    if not np.isfinite(rrho) or rrho < 0.0:
        raise ValueError("rho must be finite and nonnegative")
    z = np.ones_like(w) if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1)
    if z.shape != w.shape:
        raise ValueError("reference/preference dimension mismatch")
    if np.any(~np.isfinite(z)):
        raise ValueError("reference must be finite")
    deficit = (z.reshape(1, -1) - x) * w.reshape(1, -1)
    out = -np.max(deficit, axis=1) - rrho * np.sum(deficit, axis=1)
    return float(out[0]) if scalar else out


def version_set_pdo_decision(
    score_members: np.ndarray,
    *,
    incumbent_scores: np.ndarray,
    preference: np.ndarray,
    rho: float,
    epsilon_dec: float,
    reference: np.ndarray | None = None,
    calibration_slack: float = 0.0,
    enable_certification: bool = True,
) -> PDODecision:
    """PDO decision under a coherent joint vector-response version set.

    Parameters
    ----------
    score_members:
        Array ``[C, A, m]``.  Member ``c`` is one complete plausible vector-response
        model over all ``A`` actions.  Correlations between objectives and between
        actions are preserved by evaluating each member as a whole.

    The operational robust decision regret of action ``a`` is

        max_c [ max_b U_c(b) - U_c(a) ] + calibration_slack.

    Hence an action is empirically PDO epsilon-good when this quantity is <=
    ``epsilon_dec``.  ``calibration_slack`` is reserved for a separately validated
    sequential misspecification allowance; v1 replay defaults it to zero and reports
    empirical coverage rather than pretending a formal guarantee.
    """
    members = np.asarray(score_members, dtype=np.float64)
    if members.ndim != 3:
        raise ValueError("score_members must be [members, actions, objectives]")
    c, n, m = members.shape
    if c <= 0 or n <= 0 or m <= 0:
        raise ValueError("score_members cannot be empty")
    eps = float(epsilon_dec)
    slack = float(calibration_slack)
    if (not np.isfinite(eps)) or (not np.isfinite(slack)):
        raise ValueError("epsilon_dec/calibration_slack must be finite")
    if eps < 0 or slack < 0:
        raise ValueError("epsilon_dec/calibration_slack cannot be negative")
    incumbent = np.asarray(incumbent_scores, dtype=np.float64).reshape(-1)
    if len(incumbent) != m:
        raise ValueError("incumbent objective dimension mismatch")

    flat_u = augmented_tchebycheff_utility(
        members.reshape(c * n, m), preference, rho=rho, reference=reference
    )
    util = np.asarray(flat_u, dtype=np.float64).reshape(c, n)
    incumbent_u = float(
        augmented_tchebycheff_utility(incumbent, preference, rho=rho, reference=reference)
    )
    gains = util - incumbent_u

    member_best_u = np.max(util, axis=1)
    member_best = np.argmax(util, axis=1).astype(np.int64)
    member_regret = member_best_u.reshape(-1, 1) - util
    robust_regret = np.max(member_regret, axis=0) + slack

    # Union of model-wise epsilon-good actions: every action that could still matter
    # under at least one plausible complete response model.
    plausible = np.flatnonzero(np.min(member_regret, axis=0) <= eps + slack + 1e-12)
    common = np.flatnonzero(robust_regret <= eps + 1e-12)

    cert: int | None = None
    if bool(enable_certification) and common.size:
        # Among jointly epsilon-good actions, prefer the action whose worst member still
        # predicts the largest exact-incumbent-relative gain.
        worst_gain = np.min(gains[:, common], axis=0)
        cert = int(common[int(np.argmax(worst_gain))])

    median_u = np.median(util, axis=0)
    best_mean = int(np.argmax(median_u))
    return PDODecision(
        utility_lower=np.min(util, axis=0),
        utility_upper=np.max(util, axis=0),
        gain_lower=np.min(gains, axis=0),
        gain_upper=np.max(gains, axis=0),
        regret_upper=robust_regret,
        plausible_indices=plausible.astype(np.int64),
        best_certified_index=cert,
        best_mean_index=best_mean,
        epsilon_dec=eps,
        member_utilities=util,
        member_regrets=member_regret,
        member_best_indices=member_best,
        common_epsilon_indices=common.astype(np.int64),
        distinct_member_winners=int(len(set(int(i) for i in member_best.tolist()))),
        calibration_slack=slack,
        certification_enabled=bool(enable_certification),
    )


def version_set_pair_disagreement_weights(
    member_utilities: np.ndarray,
    plausible_indices: Sequence[int],
    *,
    epsilon_dec: float,
    near_tie_weight: float = 0.25,
) -> np.ndarray:
    """Pair weights for decision-focused transductive acquisition.

    A pair receives high weight when committee members disagree about its ordering.
    Near ties receive a smaller weight because resolving them may change the top-epsilon
    set even when all current members have the same sign.  The returned matrix can be
    supplied to :func:`decision_information_scores`.
    """
    u = np.asarray(member_utilities, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("member_utilities must be [members, actions]")
    c, n = u.shape
    inds = np.asarray(sorted(set(int(i) for i in plausible_indices)), dtype=int)
    inds = inds[(inds >= 0) & (inds < n)]
    out = np.zeros((n, n), dtype=np.float64)
    eps = max(float(epsilon_dec), 1e-12)
    for ia, a in enumerate(inds):
        for b in inds[ia + 1 :]:
            d = u[:, int(a)] - u[:, int(b)]
            p = float(np.mean(d >= 0.0))
            flip = 4.0 * p * (1.0 - p)  # 0 if unanimous, 1 at a 50/50 split.
            near = float(np.mean(np.abs(d) <= eps))
            w = flip + float(near_tie_weight) * near
            out[int(a), int(b)] = out[int(b), int(a)] = w
    return out


def version_set_action_disagreement(member_utilities: np.ndarray) -> np.ndarray:
    """Per-action utility disagreement used only as an acquisition tie breaker."""
    u = np.asarray(member_utilities, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("member_utilities must be [members, actions]")
    if u.shape[0] <= 1:
        return np.zeros(u.shape[1], dtype=np.float64)
    return np.std(u, axis=0, ddof=0)


def decision_relevant_probe_choice(
    action_coordinates: np.ndarray,
    decision: PDODecision,
    *,
    observed_indices: Iterable[int] = (),
    no_op_index: int = 0,
    ridge_lambda: float = 1.0,
    epsilon_dec: float | None = None,
    plausible_cap: int = 32,
    min_information_score: float = 0.0,
) -> tuple[int | None, float, str, int]:
    """Choose the next PDO probe from the *current decision-relevant set*.

    This is the operational acquisition policy used from the very first local query.
    It deliberately avoids a mandatory geometry-spanning warm-up:

    * if all version-set members agree on one unobserved non-noop winner, query that
      winner directly;
    * if they agree on no-op, query the strongest predicted non-noop challenger rather
      than spending a generic action-span probe;
    * if members disagree, target only distinctions among model-wise winners and
      epsilon-plausible actions and choose the realizable probe that most reduces those
      decision-relevant distinctions;
    * if the transductive information score is numerically zero, fall back to the most
      disputed/high-value *decision-relevant* action, never to a global geometry probe.

    The candidate query itself is restricted to the decision-relevant action slate.  This
    matches PDO's principle that an oracle call should be purchased only when it can
    plausibly affect the action decision.
    """
    q = np.asarray(action_coordinates, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("action_coordinates must be [actions, rank]")
    n = q.shape[0]
    noop = int(no_op_index)
    if noop < 0 or noop >= n:
        raise ValueError("no_op_index out of range")
    obs = set(int(i) for i in observed_indices)
    if any(i < 0 or i >= n for i in obs):
        raise ValueError("observed action index out of range")
    if decision.member_utilities.shape[1] != n:
        raise ValueError("decision/action-coordinate size mismatch")

    eps = float(decision.epsilon_dec if epsilon_dec is None else epsilon_dec)
    if (not np.isfinite(eps)) or eps < 0:
        raise ValueError("epsilon_dec must be finite and nonnegative")
    cap = int(plausible_cap)
    min_info = float(min_information_score)
    if (not np.isfinite(min_info)) or min_info < 0:
        raise ValueError("min_information_score must be finite and nonnegative")
    lam = float(ridge_lambda)
    if (not np.isfinite(lam)) or lam <= 0.0:
        raise ValueError("ridge_lambda must be finite and positive")

    median_u = np.median(decision.member_utilities, axis=0)
    disagreement = version_set_action_disagreement(decision.member_utilities)
    nonnoop = [i for i in range(n) if i != noop]
    unobserved_nonnoop = [i for i in nonnoop if i not in obs]
    if not unobserved_nonnoop:
        return None, 0.0, "no_unobserved_nonnoop_action", 1

    member_winners = sorted(set(int(i) for i in decision.member_best_indices.tolist()))
    # A unanimous current winner is itself the most decision-relevant experiment.  This
    # is especially important at t=0, when all committee members may coincide with the
    # KFM prior and generic geometry would waste the first paid query.
    if len(member_winners) == 1:
        winner = int(member_winners[0])
        if winner != noop and winner not in obs:
            target_count = len(set(int(i) for i in decision.plausible_indices.tolist()) | {noop, winner})
            return winner, 0.0, "consensus_winner_probe", int(target_count)

        # If no-op is the unanimous prior winner, certification may still be disabled
        # because the prior has not been locally tested.  Probe the strongest non-noop
        # challenger directly.  If the unanimous winner is already exact, the same rule
        # supplies the next most relevant challenger needed to test the ordering.
        plausible_nonnoop = [
            int(i)
            for i in decision.plausible_indices.tolist()
            if int(i) != noop and int(i) not in obs
        ]
        pool = plausible_nonnoop if plausible_nonnoop else unobserved_nonnoop
        challenger = max(
            pool,
            key=lambda i: (
                float(median_u[int(i)]),
                float(decision.utility_upper[int(i)]),
                -int(i),
            ),
        )
        mode = "noop_challenger_probe" if winner == noop else "consensus_exact_winner_challenger"
        target_count = len(set(int(i) for i in decision.plausible_indices.tolist()) | {noop, winner, challenger})
        return int(challenger), 0.0, mode, int(target_count)

    # Disagreement case: retain every model-wise winner, then fill the remaining slate
    # with epsilon-plausible actions in optimistic-utility order.  Never cap away a
    # model-wise winner, because that would remove a live decision hypothesis.
    relevant_nonnoop = [i for i in member_winners if i != noop]
    extras = [
        int(i)
        for i in decision.plausible_indices.tolist()
        if int(i) != noop and int(i) not in relevant_nonnoop
    ]
    extras.sort(
        key=lambda i: (
            float(decision.utility_upper[int(i)]),
            float(disagreement[int(i)]),
            -int(i),
        ),
        reverse=True,
    )
    if cap > 0:
        room = max(0, cap - len(relevant_nonnoop))
        extras = extras[:room]
    relevant_nonnoop.extend(extras)
    # It is possible for all member-wise winners to be no-op while some non-noop action
    # is epsilon-plausible.  If not, add the strongest non-noop challenger so that PDO
    # can still test whether staying put is genuinely best.
    if not relevant_nonnoop:
        relevant_nonnoop = [
            max(
                unobserved_nonnoop,
                key=lambda i: (
                    float(median_u[int(i)]),
                    float(decision.utility_upper[int(i)]),
                    -int(i),
                ),
            )
        ]

    targets = list(dict.fromkeys(relevant_nonnoop + [noop]))
    candidates = [i for i in relevant_nonnoop if i not in obs]
    if not candidates:
        # All currently plausible/winning mutations are already exact but the version set
        # still disagrees.  Add the next best unobserved challenger rather than reverting
        # to a geometry-wide probe.
        challenger = max(
            unobserved_nonnoop,
            key=lambda i: (
                float(median_u[int(i)]),
                float(decision.utility_upper[int(i)]),
                -int(i),
            ),
        )
        candidates = [int(challenger)]
        targets = list(dict.fromkeys(targets + [int(challenger)]))

    pair_weights = version_set_pair_disagreement_weights(
        decision.member_utilities,
        targets,
        epsilon_dec=eps,
    )
    info = decision_information_scores(
        q,
        targets,
        local_observed_indices=obs,
        ridge_lambda=lam,
        candidate_indices=candidates,
        pair_weights=pair_weights,
    )
    best_info = max(float(info[int(i)]) for i in candidates)
    if best_info > min_info:
        idx = max(
            candidates,
            key=lambda i: (
                float(info[int(i)]),
                float(disagreement[int(i)]),
                float(decision.utility_upper[int(i)]),
                -int(i),
            ),
        )
        return int(idx), float(info[int(idx)]), "version_set_disagreement", int(len(targets))

    # Numerical/degenerate fallback: still query only a live decision hypothesis.
    idx = max(
        candidates,
        key=lambda i: (
            float(disagreement[int(i)]),
            float(decision.utility_upper[int(i)]),
            float(median_u[int(i)]),
            -int(i),
        ),
    )
    return int(idx), float(info[int(idx)]), "decision_relevant_direct_fallback", int(len(targets))


def best_exact_improving_index(
    exact_scores_by_action: dict[int, np.ndarray] | Sequence[tuple[int, np.ndarray]],
    *,
    incumbent_scores: np.ndarray,
    preference: np.ndarray,
    rho: float,
    accept_epsilon: float,
    reference: np.ndarray | None = None,
) -> tuple[int | None, float]:
    """Return the best already-exact action if it strictly improves the incumbent.

    This is a zero-additional-query fallback.  It is intentionally independent of PDO
    confidence: once an action's full objective vector has already been paid for, there
    is no statistical reason to discard a known improvement merely because the current
    version set cannot certify that it is globally/top-epsilon optimal within the H1
    neighborhood.
    """
    items = (
        exact_scores_by_action.items()
        if hasattr(exact_scores_by_action, "items")
        else exact_scores_by_action
    )
    accept = float(accept_epsilon)
    if not np.isfinite(accept) or accept < 0.0:
        raise ValueError("accept_epsilon must be finite and nonnegative")
    incumbent_u = float(
        augmented_tchebycheff_utility(
            np.asarray(incumbent_scores, dtype=np.float64),
            preference,
            rho=float(rho),
            reference=reference,
        )
    )
    best_idx: int | None = None
    best_gain = float("-inf")
    for idx_raw, scores_raw in items:
        idx = int(idx_raw)
        u = float(
            augmented_tchebycheff_utility(
                np.asarray(scores_raw, dtype=np.float64),
                preference,
                rho=float(rho),
                reference=reference,
            )
        )
        gain = float(u - incumbent_u)
        if gain > best_gain + 1e-15 or (
            abs(gain - best_gain) <= 1e-15 and best_idx is not None and idx < best_idx
        ):
            best_idx = idx
            best_gain = gain
    if best_idx is None or best_gain <= accept:
        return None, 0.0 if best_idx is None else float(best_gain)
    return int(best_idx), float(best_gain)


# ---------------------------------------------------------------------------
# Diagnostic/backward-compatible box utilities. Not used for operational PDO.
# ---------------------------------------------------------------------------

def utility_bounds_from_score_boxes(
    score_lower: np.ndarray,
    score_upper: np.ndarray,
    preference: np.ndarray,
    *,
    rho: float,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(score_lower, dtype=np.float64)
    hi = np.asarray(score_upper, dtype=np.float64)
    if lo.shape != hi.shape or lo.ndim != 2:
        raise ValueError("score boxes must be aligned [n,m] arrays")
    if np.any(lo > hi + 1e-12):
        raise ValueError("score lower bound exceeds upper bound")
    ulo = np.asarray(
        augmented_tchebycheff_utility(lo, preference, rho=rho, reference=reference),
        dtype=np.float64,
    ).reshape(-1)
    uhi = np.asarray(
        augmented_tchebycheff_utility(hi, preference, rho=rho, reference=reference),
        dtype=np.float64,
    ).reshape(-1)
    return ulo, uhi


def robust_pdo_decision(
    score_lower: np.ndarray,
    score_upper: np.ndarray,
    *,
    incumbent_scores: np.ndarray,
    preference: np.ndarray,
    rho: float,
    epsilon_dec: float,
    reference: np.ndarray | None = None,
    mean_scores: np.ndarray | None = None,
) -> PDODecision:
    """Legacy box-hull decision helper retained only for diagnostics/tests.

    New PDO code must call :func:`version_set_pdo_decision` instead.
    """
    lo = np.asarray(score_lower, dtype=np.float64)
    hi = np.asarray(score_upper, dtype=np.float64)
    if mean_scores is None:
        mid = 0.5 * (lo + hi)
    else:
        mid = np.asarray(mean_scores, dtype=np.float64)
        if mid.shape != lo.shape:
            raise ValueError("mean_scores shape mismatch")
    # Three coherent members are deliberately *not* claimed to represent the full box;
    # this wrapper preserves the historical interface for tests/diagnostics only.
    members = np.stack([lo, mid, hi], axis=0)
    return version_set_pdo_decision(
        members,
        incumbent_scores=incumbent_scores,
        preference=preference,
        rho=rho,
        epsilon_dec=epsilon_dec,
        reference=reference,
        enable_certification=True,
    )


def pareto_switch_diagnostic(
    incumbent_scores: np.ndarray,
    score_lower: np.ndarray,
    score_upper: np.ndarray,
    preference: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    tolerance_multiplier: float = 1.0,
) -> ParetoSwitchDiagnostic:
    """Report active-deficit/switch quantities without gating the PDO decision."""
    cur = np.asarray(incumbent_scores, dtype=np.float64).reshape(-1)
    lo = np.asarray(score_lower, dtype=np.float64)
    hi = np.asarray(score_upper, dtype=np.float64)
    w = np.asarray(preference, dtype=np.float64).reshape(-1)
    if lo.ndim != 2 or lo.shape != hi.shape or lo.shape[1] != len(cur) or len(w) != len(cur):
        raise ValueError("shape mismatch in Pareto-switch diagnostic")
    z = np.ones_like(cur) if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1)
    deficits = w * (z - cur)
    active = int(np.argmax(deficits))
    order = np.argsort(-deficits)
    gap = float(deficits[order[0]] - deficits[order[1]]) if len(order) > 1 else float("inf")

    delta_lo = lo - cur.reshape(1, -1)
    delta_hi = hi - cur.reshape(1, -1)
    reachable = np.max(np.maximum(np.abs(delta_lo), np.abs(delta_hi)), axis=0) * w
    tau = float(tolerance_multiplier) * float(np.max(reachable[active] + reachable))
    max_def = float(np.max(deficits))
    critical = np.flatnonzero(max_def - deficits <= tau + 1e-12)
    safe = True
    for k in range(len(cur)):
        if k == active:
            continue
        if deficits[active] - deficits[k] <= reachable[active] + reachable[k] + 1e-12:
            safe = False
            break
    return ParetoSwitchDiagnostic(
        deficits=deficits,
        active_index=active,
        active_gap=gap,
        critical_indices=critical.astype(np.int64),
        switch_safe=bool(safe),
        per_objective_reachable_weighted_change=reachable,
    )



def decision_span_information_scores(
    action_coordinates: np.ndarray,
    *,
    local_observed_indices: Iterable[int] = (),
    ridge_lambda: float = 1.0,
    target_indices: Sequence[int] | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Scalable transductive information score for initialization.

    Before local labels create a meaningful response-model committee, PDO should not
    pretend the single KFM prior is certain.  This criterion asks which realizable probe
    most reduces uncertainty over the *centered action-difference span* as a whole.  It
    avoids explicit O(|A|^2) pair enumeration: if T is the centered target-coordinate
    matrix, the summed rank-one variance reduction is

        q^T G^-1 (T^T T) G^-1 q / (1 + q^T G^-1 q).

    Thus all primitive actions can participate in initialization even for substantially
    larger neighborhoods.
    """
    q_all = np.asarray(action_coordinates, dtype=np.float64)
    if q_all.ndim != 2:
        raise ValueError("action_coordinates must be [n,r]")
    if np.any(~np.isfinite(q_all)):
        raise ValueError("action_coordinates must be finite")
    n, r = q_all.shape
    if r == 0:
        return np.zeros(n, dtype=np.float64)
    lam = float(ridge_lambda)
    if (not np.isfinite(lam)) or lam <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    obs = np.asarray(sorted(set(int(i) for i in local_observed_indices)), dtype=int)
    if obs.size and (obs.min() < 0 or obs.max() >= n):
        raise ValueError("observed action index out of range")
    G = lam * np.eye(r, dtype=np.float64)
    if obs.size:
        G += q_all[obs].T @ q_all[obs]
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)

    targets = np.arange(n, dtype=int) if target_indices is None else np.asarray(target_indices, dtype=int)
    targets = targets[(targets >= 0) & (targets < n)]
    if len(targets) <= 1:
        return np.zeros(n, dtype=np.float64)
    T = q_all[targets]
    T = T - np.mean(T, axis=0, keepdims=True)
    scatter = T.T @ T
    H = Ginv @ scatter @ Ginv

    candidates = np.arange(n, dtype=int) if candidate_indices is None else np.asarray(candidate_indices, dtype=int)
    out = np.zeros(n, dtype=np.float64)
    for qi in candidates:
        if qi < 0 or qi >= n:
            continue
        q = q_all[int(qi)]
        gq = Ginv @ q
        denom = 1.0 + float(q @ gq)
        if denom <= 0:
            continue
        out[int(qi)] = float(q @ H @ q / denom)
    return out

def decision_information_scores(
    action_coordinates: np.ndarray,
    plausible_indices: Sequence[int],
    *,
    local_observed_indices: Iterable[int] = (),
    ridge_lambda: float = 1.0,
    candidate_indices: Sequence[int] | None = None,
    pair_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Transductive rank-one variance-reduction score for realizable probes.

    Only exact local H1 observations enter the information matrix.  ``pair_weights``
    should encode current version-set disagreement; therefore the acquisition targets
    distinctions that can actually change the current Pareto decision rather than global
    regression error.
    """
    q_all = np.asarray(action_coordinates, dtype=np.float64)
    if q_all.ndim != 2:
        raise ValueError("action_coordinates must be [n,r]")
    if np.any(~np.isfinite(q_all)):
        raise ValueError("action_coordinates must be finite")
    n, r = q_all.shape
    if r == 0:
        return np.zeros(n, dtype=np.float64)
    lam = float(ridge_lambda)
    if (not np.isfinite(lam)) or lam <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    obs = np.asarray(sorted(set(int(i) for i in local_observed_indices)), dtype=int)
    if obs.size and (obs.min() < 0 or obs.max() >= n):
        raise ValueError("observed action index out of range")
    G = lam * np.eye(r, dtype=np.float64)
    if obs.size:
        G += q_all[obs].T @ q_all[obs]
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)

    plausible = np.asarray(sorted(set(int(i) for i in plausible_indices)), dtype=int)
    plausible = plausible[(plausible >= 0) & (plausible < n)]
    if len(plausible) < 2:
        return np.zeros(n, dtype=np.float64)
    pw = None if pair_weights is None else np.asarray(pair_weights, dtype=np.float64)
    if pw is not None and pw.shape != (n, n):
        raise ValueError("pair_weights must be [n,n]")
    if pw is not None and np.any(~np.isfinite(pw)):
        raise ValueError("pair_weights must be finite")

    diffs: list[np.ndarray] = []
    weights: list[float] = []
    for ia, a in enumerate(plausible):
        for ib in range(ia + 1, len(plausible)):
            b = int(plausible[ib])
            d = q_all[int(a)] - q_all[b]
            if float(np.dot(d, d)) <= 1e-18:
                continue
            w = 1.0 if pw is None else float(pw[int(a), b])
            if w <= 0:
                continue
            diffs.append(d)
            weights.append(w)
    if not diffs:
        return np.zeros(n, dtype=np.float64)
    D = np.stack(diffs, axis=0)
    W = np.asarray(weights, dtype=np.float64)
    DG = D @ Ginv

    candidates = np.arange(n, dtype=int) if candidate_indices is None else np.asarray(candidate_indices, dtype=int)
    out = np.zeros(n, dtype=np.float64)
    for qi in candidates:
        if qi < 0 or qi >= n:
            continue
        q = q_all[int(qi)]
        gq = Ginv @ q
        denom = 1.0 + float(q @ gq)
        if denom <= 0:
            continue
        cross = DG @ q
        out[int(qi)] = float(np.sum(W * cross * cross) / denom)
    return out


__all__ = [
    "PDODecision",
    "EmpiricalConfidenceRecovery",
    "FreshQueryGate",
    "ParetoSwitchDiagnostic",
    "empirical_confidence_verification_index",
    "fresh_h1_query_gate",
    "matched_coarse_verification_slots",
    "augmented_tchebycheff_utility",
    "version_set_pdo_decision",
    "version_set_pair_disagreement_weights",
    "version_set_action_disagreement",
    "decision_relevant_probe_choice",
    "best_exact_improving_index",
    "utility_bounds_from_score_boxes",
    "robust_pdo_decision",
    "pareto_switch_diagnostic",
    "decision_span_information_scores",
    "decision_information_scores",
]
