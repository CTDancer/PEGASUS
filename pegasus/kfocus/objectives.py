"""Six-property cached peptide oracle used by K-FOCUS.

The optimization-facing conventions intentionally match the current Koopman/
RAPTOR PeptiVerse evaluation code:

* Hemolysis -> Non-Hemolysis = 1 - raw hemolysis;
* Non-Fouling, Solubility and Permeability use the predictor score directly;
* Half-Life = clip(raw, 0, 2) / 2;
* Affinity = raw / 10.

Raw predictor values are retained in every archive/result file.  One *sequence*
evaluation returns the complete six-objective vector and counts as one oracle
query; deterministic duplicate sequences are cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math

import numpy as np

from ..probes import load_peptiverse_predictor

RAW_NAMES = (
    "Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
)
OPT_NAMES = (
    "Non-Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
)
_PROPERTY_KEYS = {
    "Hemolysis": "hemolysis",
    "Non-Fouling": "nf",
    "Solubility": "solubility",
    "Permeability": "permeability_penetrance",
    "Half-Life": "halflife",
}


def orient_and_normalize(raw: Sequence[float], *, clip: bool = True) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if x.size != 6:
        raise ValueError("raw six-objective vector must have length 6")
    y = np.asarray(
        [
            1.0 - x[0],
            x[1],
            x[2],
            x[3],
            max(0.0, min(float(x[4]), 2.0)) / 2.0,
            x[5] / 10.0,
        ],
        dtype=np.float64,
    )
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y


@dataclass(frozen=True)
class ObjectiveRecord:
    sequence: str
    raw: np.ndarray
    scores: np.ndarray
    fresh_query: bool

    def row(self) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = {
            "sequence": self.sequence,
            "fresh_query": bool(self.fresh_query),
        }
        for j, name in enumerate(RAW_NAMES):
            out[f"raw_{name}"] = float(self.raw[j])
        for j, name in enumerate(OPT_NAMES):
            out[f"score_{name}"] = float(self.scores[j])
        return out


class CachedSixPropertyOracle:
    """Evaluate and cache the complete AMHR2 six-property vector."""

    def __init__(
        self,
        predictor: Any,
        *,
        target: str,
        clip_optimization_scores: bool = True,
    ) -> None:
        self.pred = predictor
        self.target = str(target)
        if len(self.target) < 10:
            raise ValueError("Affinity requires a valid target protein sequence")
        self.clip_optimization_scores = bool(clip_optimization_scores)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.attempted_sequence_evaluations = 0
        self.unique_oracle_queries = 0
        self.cache_hits = 0
        self.predictor_calls = 0

    @classmethod
    def from_peptiverse(
        cls,
        root: str,
        *,
        target: str,
        manifest_path: str | None = None,
        device: str = "cuda",
        clip_optimization_scores: bool = True,
    ) -> "CachedSixPropertyOracle":
        predictor = load_peptiverse_predictor(
            root,
            manifest_path=manifest_path or None,
            device=device,
        )
        return cls(
            predictor,
            target=target,
            clip_optimization_scores=clip_optimization_scores,
        )

    def _evaluate_uncached(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        raw: list[float] = []
        for name in RAW_NAMES:
            if name == "Affinity":
                value = float(
                    self.pred.predict_binding_affinity(
                        col="wt", target_seq=self.target, binder_str=sequence
                    )["affinity"]
                )
            else:
                value = float(
                    self.pred.predict_property(
                        _PROPERTY_KEYS[name], col="wt", input_str=sequence
                    )["score"]
                )
            if not math.isfinite(value):
                raise ValueError(f"non-finite {name} score for {sequence!r}")
            raw.append(value)
        raw_arr = np.asarray(raw, dtype=np.float64)
        score_arr = orient_and_normalize(raw_arr, clip=self.clip_optimization_scores)
        self.predictor_calls += 6
        self.unique_oracle_queries += 1
        self.cache[sequence] = (raw_arr.copy(), score_arr.copy())
        return raw_arr, score_arr

    def evaluate_one(self, sequence: str) -> ObjectiveRecord:
        seq = str(sequence)
        self.attempted_sequence_evaluations += 1
        if seq in self.cache:
            self.cache_hits += 1
            raw, scores = self.cache[seq]
            return ObjectiveRecord(seq, raw.copy(), scores.copy(), False)
        raw, scores = self._evaluate_uncached(seq)
        return ObjectiveRecord(seq, raw.copy(), scores.copy(), True)

    def evaluate(self, sequences: Iterable[str]) -> list[ObjectiveRecord]:
        return [self.evaluate_one(seq) for seq in sequences]

    def inject_record(self, sequence: str, raw: Sequence[float]) -> None:
        """Add a previously paid exact record without incrementing query counts."""
        seq = str(sequence)
        raw_arr = np.asarray(raw, dtype=np.float64).reshape(6)
        self.cache[seq] = (
            raw_arr.copy(),
            orient_and_normalize(raw_arr, clip=self.clip_optimization_scores),
        )

    def accounting(self) -> dict[str, int]:
        return {
            "attempted_sequence_evaluations": int(self.attempted_sequence_evaluations),
            "unique_oracle_queries": int(self.unique_oracle_queries),
            "cache_hits": int(self.cache_hits),
            "predictor_calls": int(self.predictor_calls),
        }


def objective_summary(raw: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    r = np.asarray(raw, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if r.ndim != 2 or s.ndim != 2 or r.shape != s.shape or r.shape[1] != 6:
        raise ValueError("objective arrays must both be [N,6]")
    out: dict[str, dict[str, float]] = {}
    for j, raw_name in enumerate(RAW_NAMES):
        raw_best = float(np.min(r[:, j])) if raw_name == "Hemolysis" else float(np.max(r[:, j]))
        out[raw_name] = {
            "raw_mean": float(np.mean(r[:, j])),
            "raw_best": raw_best,
            "optimization_mean": float(np.mean(s[:, j])),
            "optimization_best": float(np.max(s[:, j])),
        }
    return out


__all__ = [
    "RAW_NAMES",
    "OPT_NAMES",
    "ObjectiveRecord",
    "CachedSixPropertyOracle",
    "orient_and_normalize",
    "objective_summary",
]
