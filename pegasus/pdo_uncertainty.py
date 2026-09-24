"""Joint local response version set for PEGASUS v1.

The operational uncertainty state is a *committee of complete vector-response models*.
This is intentionally different from the first implementation, which converted marginal
per-objective residual quantiles into an axis-aligned score box.  That box destroyed the
cross-objective/action correlations that PDO is supposed to exploit and was far too
conservative in replay.

Design invariants
-----------------
* KFM supplies only the global/coarse prior mean.
* The fine statistical problem is parent-relative objective change.
* Only exact Hamming-1 observations from the current incumbent update local response
  models and the local information matrix.
* Version-set members are coherent complete response models (zero correction, ridge
  corrections with multiple regularization strengths, and deterministic leave-one-probe
  variants when enough local observations exist).
* Every exact local observation is clamped to its exact full objective vector in every
  member.
* Static calibration metadata is diagnostic only unless separately validated by adaptive
  held-out replay; it no longer creates coordinate-wise operational uncertainty boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PDOCalibration:
    """Calibration metadata retained for provenance/claim-level accounting.

    ``base_radius`` is kept for backward compatibility with existing calibration JSONs,
    but PDO v1.3 does not use those marginal radii to form the operational decision set.
    """

    objective_names: tuple[str, ...]
    base_radius: np.ndarray
    irreducible_floor_fraction: float = 0.0
    coverage: float = 0.95
    sequentially_validated: bool = False
    validation_method: str = "unvalidated"
    source: str = "empirical"

    def __post_init__(self) -> None:
        r = np.asarray(self.base_radius, dtype=np.float64).reshape(-1)
        if len(r) != len(self.objective_names):
            raise ValueError("calibration radius/objective mismatch")
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("calibration radii must be finite and nonnegative")
        if not (0.0 < float(self.coverage) < 1.0):
            raise ValueError("calibration coverage must lie in (0,1)")
        if not (0.0 <= float(self.irreducible_floor_fraction) <= 1.0):
            raise ValueError("irreducible_floor_fraction must lie in [0,1]")
        object.__setattr__(self, "base_radius", r)

    @property
    def claim_level(self) -> str:
        """PDO v1.x never promotes empirical committee agreement to a formal certificate.

        Historical calibration JSONs may contain ``sequentially_validated`` metadata, but
        the current operational version set has not established theorem-level sequential
        coverage under adaptive acquisition.  Claim hygiene therefore stays conservative
        by construction.
        """
        return "empirical_confidence"

    @classmethod
    def from_json(cls, path: str | Path) -> "PDOCalibration":
        p = Path(path).expanduser().resolve()
        payload = json.loads(p.read_text())
        names = tuple(str(x) for x in payload["objective_names"])
        radii_raw = payload.get("base_radius", [0.0] * len(names))
        if isinstance(radii_raw, Mapping):
            radii = np.asarray([float(radii_raw[n]) for n in names], dtype=np.float64)
        else:
            radii = np.asarray(radii_raw, dtype=np.float64)
        return cls(
            objective_names=names,
            base_radius=radii,
            irreducible_floor_fraction=float(payload.get("irreducible_floor_fraction", 0.0)),
            coverage=float(payload.get("coverage", 0.95)),
            sequentially_validated=bool(payload.get("sequentially_validated", False)),
            validation_method=str(payload.get("validation_method", "unvalidated")),
            source=str(payload.get("source", str(p))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_names": list(self.objective_names),
            "base_radius": {
                n: float(self.base_radius[i]) for i, n in enumerate(self.objective_names)
            },
            "irreducible_floor_fraction": float(self.irreducible_floor_fraction),
            "coverage": float(self.coverage),
            "sequentially_validated": bool(self.sequentially_validated),
            "sequentially_validated_metadata": bool(self.sequentially_validated),
            "formal_certificate_supported": False,
            "validation_method": str(self.validation_method),
            "claim_level": self.claim_level,
            "source": self.source,
            "operational_use": (
                "metadata/diagnostic only in PDO v1.3; marginal radii do not define the "
                "joint response version set"
            ),
        }


@dataclass(frozen=True)
class PDOResponseState:
    mean_scores: np.ndarray
    member_scores: np.ndarray
    member_labels: tuple[str, ...]
    score_lower: np.ndarray
    score_upper: np.ndarray
    width: np.ndarray
    local_correction: np.ndarray
    leverage: np.ndarray
    baseline_leverage: np.ndarray
    local_observed_indices: tuple[int, ...]
    local_nonreference_observations: int
    ensemble_size: int
    decision_rank: int
    claim_level: str


def _ridge_correction(
    Q: np.ndarray,
    obs: np.ndarray,
    residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Multi-output ridge correction in dual form; returns [n,m]."""
    n, r = Q.shape
    m = residual.shape[1]
    if obs.size == 0 or r == 0:
        return np.zeros((n, m), dtype=np.float64)
    X = Q[obs]
    yy = residual
    gram = X @ X.T + float(alpha) * np.eye(len(obs), dtype=np.float64)
    try:
        dual = np.linalg.solve(gram, yy)
    except np.linalg.LinAlgError:
        dual = np.linalg.pinv(gram) @ yy
    return Q @ X.T @ dual


def _leverage(Q: np.ndarray, obs: np.ndarray, ridge_lambda: float) -> np.ndarray:
    if Q.shape[1] == 0:
        return np.zeros(Q.shape[0], dtype=np.float64)
    G = float(ridge_lambda) * np.eye(Q.shape[1], dtype=np.float64)
    if obs.size:
        G += Q[obs].T @ Q[obs]
    try:
        sol = np.linalg.solve(G, Q.T).T
    except np.linalg.LinAlgError:
        sol = Q @ np.linalg.pinv(G)
    lev = np.sum(Q * sol, axis=1)
    return np.maximum(lev, 0.0)


def build_local_response_state(
    prior_scores: np.ndarray,
    action_coordinates: np.ndarray,
    exact_scores_by_action: Mapping[int, np.ndarray],
    calibration: PDOCalibration,
    *,
    reference_action_index: int = 0,
    local_ridge_alphas: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    information_lambda: float = 1.0,
    clip_scores: bool = True,
    include_leave_one_probe_members: bool = True,
) -> PDOResponseState:
    """Build coherent complete vector-response members over the fine action set.

    The KFM prior is first converted into parent-relative objective changes.  Each local
    model then predicts a *joint* multi-objective correction as a function of the LKF
    action coordinates.  No marginal objective error radius is Cartesian-producted into
    the decision set.
    """
    prior = np.asarray(prior_scores, dtype=np.float64)
    Q = np.asarray(action_coordinates, dtype=np.float64)
    if prior.ndim != 2 or Q.ndim != 2 or prior.shape[0] != Q.shape[0]:
        raise ValueError("prior/action-coordinate shape mismatch")
    if np.any(~np.isfinite(prior)) or np.any(~np.isfinite(Q)):
        raise ValueError("prior scores and action coordinates must be finite")
    if not np.isfinite(float(information_lambda)) or float(information_lambda) <= 0.0:
        raise ValueError("information_lambda must be finite and positive")
    n, m = prior.shape
    if m != len(calibration.objective_names):
        raise ValueError("objective dimension does not match calibration")
    ref = int(reference_action_index)
    if ref < 0 or ref >= n:
        raise ValueError("reference action index out of range")
    if ref not in exact_scores_by_action:
        raise ValueError("reference/no-op action must have an exact score vector")
    exact_ref = np.asarray(exact_scores_by_action[ref], dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(exact_ref)):
        raise ValueError("reference exact scores must be finite")
    if len(exact_ref) != m:
        raise ValueError("reference exact score dimension mismatch")

    # Remove any absolute parent bias in the coarse readout.  Fine PDO works entirely in
    # action-induced change coordinates with the exact incumbent as the origin.
    prior_delta = prior - prior[ref].reshape(1, -1)

    obs_list = sorted(set(int(i) for i in exact_scores_by_action))
    if any(i < 0 or i >= n for i in obs_list):
        raise ValueError("exact local action index out of range")
    obs = np.asarray(obs_list, dtype=int)
    exact = np.stack(
        [np.asarray(exact_scores_by_action[int(i)], dtype=np.float64).reshape(-1) for i in obs],
        axis=0,
    )
    if exact.shape[1] != m:
        raise ValueError("exact local score dimension mismatch")
    if np.any(~np.isfinite(exact)):
        raise ValueError("exact local scores must be finite")
    exact_delta = exact - exact_ref.reshape(1, -1)
    residual = exact_delta - prior_delta[obs]

    nonref_mask = obs != ref
    fit_obs = obs[nonref_mask]
    fit_residual = residual[nonref_mask]

    alphas = tuple(float(a) for a in local_ridge_alphas)
    if not alphas or any((not np.isfinite(a)) or a <= 0.0 for a in alphas):
        raise ValueError("local ridge alphas must all be finite and positive")

    corrections: list[np.ndarray] = [np.zeros_like(prior_delta)]
    labels: list[str] = ["kfm_prior_no_local_correction"]
    for alpha in alphas:
        corrections.append(_ridge_correction(Q, fit_obs, fit_residual, alpha))
        labels.append(f"ridge_alpha={alpha:g}")

    # Deterministic jackknife members expose local model sensitivity without pretending
    # bootstrap randomness is a probability distribution.  They are especially useful
    # once several adaptive probes have been collected.
    if bool(include_leave_one_probe_members) and len(fit_obs) >= 2:
        alpha_mid = float(alphas[len(alphas) // 2])
        for leave_pos in range(len(fit_obs)):
            mask = np.ones(len(fit_obs), dtype=bool)
            mask[leave_pos] = False
            corrections.append(
                _ridge_correction(Q, fit_obs[mask], fit_residual[mask], alpha_mid)
            )
            labels.append(f"jackknife_leave_action={int(fit_obs[leave_pos])}_alpha={alpha_mid:g}")

    delta_members = np.stack([prior_delta + c for c in corrections], axis=0)
    members = exact_ref.reshape(1, 1, -1) + delta_members

    # Every paid local label is exact and therefore must agree across every model member.
    for k, idx in enumerate(obs):
        members[:, int(idx), :] = exact[k].reshape(1, -1)

    if clip_scores:
        members = np.clip(members, 0.0, 1.0)

    center = np.median(members, axis=0)
    prior_anchored = exact_ref.reshape(1, -1) + prior_delta
    if clip_scores:
        prior_anchored = np.clip(prior_anchored, 0.0, 1.0)
    local_correction = center - prior_anchored

    lo = np.min(members, axis=0)
    hi = np.max(members, axis=0)
    width = 0.5 * (hi - lo)  # descriptive hull only; not Cartesian operational uncertainty.

    baseline = _leverage(Q, np.zeros(0, dtype=int), float(information_lambda))
    current = _leverage(Q, fit_obs, float(information_lambda))

    return PDOResponseState(
        mean_scores=center,
        member_scores=members,
        member_labels=tuple(labels),
        score_lower=lo,
        score_upper=hi,
        width=width,
        local_correction=local_correction,
        leverage=current,
        baseline_leverage=baseline,
        local_observed_indices=tuple(int(i) for i in obs_list),
        local_nonreference_observations=int(len(fit_obs)),
        ensemble_size=int(len(labels)),
        decision_rank=int(Q.shape[1]),
        claim_level=calibration.claim_level,
    )


__all__ = [
    "PDOCalibration",
    "PDOResponseState",
    "build_local_response_state",
]
