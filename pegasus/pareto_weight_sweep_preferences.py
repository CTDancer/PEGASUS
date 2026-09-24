"""Canonical preference vectors for the PEGASUS Pareto weight sweep.

Single source of truth for the seven paper-space preference vectors.
Larger weight = greater importance for that maximize-oriented objective.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .kfocus.objectives import OPT_NAMES

OBJECTIVE_NAMES: tuple[str, ...] = tuple(OPT_NAMES)
assert OBJECTIVE_NAMES == (
    "Non-Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
), "objective ordering must match production OPT_NAMES"

WEIGHT_TOL = 1e-12

_EQUAL = 1.0 / 6.0

PARETO_SWEEP_WEIGHTS: dict[str, tuple[float, ...]] = {
    "equal": (_EQUAL, _EQUAL, _EQUAL, _EQUAL, _EQUAL, _EQUAL),
    "nonhem": (0.5, 0.1, 0.1, 0.1, 0.1, 0.1),
    "nonfouling": (0.1, 0.5, 0.1, 0.1, 0.1, 0.1),
    "solubility": (0.1, 0.1, 0.5, 0.1, 0.1, 0.1),
    "permeability": (0.1, 0.1, 0.1, 0.5, 0.1, 0.1),
    "halflife": (0.1, 0.1, 0.1, 0.1, 0.5, 0.1),
    "affinity": (0.1, 0.1, 0.1, 0.1, 0.1, 0.5),
}

PREFERENCE_NAMES: tuple[str, ...] = tuple(PARETO_SWEEP_WEIGHTS.keys())
EMPHASIZED_PREFERENCE_NAMES: tuple[str, ...] = tuple(
    n for n in PREFERENCE_NAMES if n != "equal"
)

SWEEP_TARGETS: tuple[str, ...] = ("3IDJ", "AMHR2", "MYC")
SWEEP_SEEDS: tuple[int, ...] = (42, 73, 101)
SWEEP_LENGTH: int = 12
SWEEP_HORIZONS: tuple[int, ...] = (1, 2, 4)
SWEEP_PROTECTED_FRACTION: float = 0.5
SWEEP_TERMINAL_SIZE: int = 100
SWEEP_EXPECTED_RUNS: int = (
    len(SWEEP_TARGETS) * len(SWEEP_SEEDS) * len(PREFERENCE_NAMES)
)
SWEEP_EXPECTED_TERMINAL_ROWS: int = SWEEP_EXPECTED_RUNS * SWEEP_TERMINAL_SIZE

WEIGHT_COLUMN_NAMES: tuple[str, ...] = (
    "w_nonhem",
    "w_nonfouling",
    "w_solubility",
    "w_permeability",
    "w_halflife",
    "w_affinity",
)
SCORE_COLUMN_NAMES: tuple[str, ...] = (
    "nonhem",
    "nonfouling",
    "solubility",
    "permeability",
    "halflife",
    "affinity",
)
GENERATED_SCORE_COLUMNS: tuple[str, ...] = tuple(
    f"score_{name}" for name in OBJECTIVE_NAMES
)


def validate_weight_vector(
    weights: Sequence[float],
    *,
    name: str = "weights",
    tol: float = WEIGHT_TOL,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != 6:
        raise ValueError(f"{name}: expected length 6, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name}: weights must be finite")
    if np.any(w < 0):
        raise ValueError(f"{name}: weights must be nonnegative")
    s = float(w.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(f"{name}: weights must sum to 1.0 (got {s})")
    return w


def validate_registry(tol: float = WEIGHT_TOL) -> None:
    if set(PARETO_SWEEP_WEIGHTS) != set(PREFERENCE_NAMES):
        raise ValueError("preference registry key mismatch")
    if OBJECTIVE_NAMES != OPT_NAMES:
        raise ValueError("OBJECTIVE_NAMES must equal production OPT_NAMES")
    for name, vec in PARETO_SWEEP_WEIGHTS.items():
        validate_weight_vector(vec, name=name, tol=tol)


def get_preference_vector(name: str) -> np.ndarray:
    key = str(name).strip().lower()
    if key not in PARETO_SWEEP_WEIGHTS:
        raise KeyError(
            f"unknown preference name {name!r}; "
            f"expected one of {list(PREFERENCE_NAMES)}"
        )
    return validate_weight_vector(PARETO_SWEEP_WEIGHTS[key], name=key)


def preference_csv(name: str) -> str:
    w = get_preference_vector(name)
    return ",".join(f"{float(x):.17g}" for x in w)


def preference_weight_fields(name: str) -> dict[str, float]:
    w = get_preference_vector(name)
    return {col: float(w[i]) for i, col in enumerate(WEIGHT_COLUMN_NAMES)}


def run_id(target: str, seed: int, preference_name: str, *, length: int = SWEEP_LENGTH) -> str:
    return f"L{int(length)}__{target}__seed{int(seed)}__{preference_name}"


def expected_output_dir(
    root: str | object,
    target: str,
    seed: int,
    preference_name: str,
    *,
    length: int = SWEEP_LENGTH,
) -> str:
    from pathlib import Path

    base = Path(root)
    return str(
        base
        / f"L{int(length)}"
        / str(target)
        / f"seed_{int(seed)}"
        / str(preference_name)
    )


def iter_sweep_grid() -> list[dict[str, object]]:
    validate_registry()
    rows: list[dict[str, object]] = []
    for target in SWEEP_TARGETS:
        for seed in SWEEP_SEEDS:
            for pref in PREFERENCE_NAMES:
                w = preference_weight_fields(pref)
                rows.append(
                    {
                        "run_id": run_id(target, seed, pref),
                        "target": target,
                        "seed": int(seed),
                        "length": int(SWEEP_LENGTH),
                        "preference_name": pref,
                        **w,
                    }
                )
    if len(rows) != SWEEP_EXPECTED_RUNS:
        raise RuntimeError(
            f"expected {SWEEP_EXPECTED_RUNS} runs, built {len(rows)}"
        )
    return rows


def resolve_preferences_cli(
    *,
    preference_name: str = "",
    weights: str = "",
    preferences: str = "1,1,1,1,1,1",
) -> tuple[str, str | None]:
    """Resolve CLI preference inputs to a comma-separated vector.

    Returns (preferences_csv, canonical_preference_name_or_None).
    Priority: preference_name > weights > preferences.
    """
    name = str(preference_name or "").strip()
    wtxt = str(weights or "").strip()
    pref = str(preferences or "").strip()
    if name and wtxt:
        raise ValueError("pass only one of --preference-name or --weights")
    if name:
        validate_registry()
        return preference_csv(name), name
    if wtxt:
        parts = [float(x.strip()) for x in wtxt.split(",") if x.strip()]
        validate_weight_vector(parts, name="--weights")
        return ",".join(f"{float(x):.17g}" for x in parts), None
    if not pref:
        raise ValueError("must supply --preference-name, --weights, or --preferences")
    return pref, None


# Validate at import so misconfiguration fails early.
validate_registry()
