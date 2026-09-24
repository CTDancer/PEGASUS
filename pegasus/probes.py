"""Few-shot objective linear probes for K-LKF evaluation.

This module is evaluation-only.  Objective labels are never imported by the
training program.  The default `current_module` score convention matches the
current peptide code's maximize directions and historical normalizations:
Non-Hemolysis=1-hemolysis, Half-Life=clip(raw,0,2)/2, Affinity=raw/10, while
Non-Fouling/Solubility/Permeability use their predictor scores directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from .model import KoopmanRegularizedLKF


OBJECTIVE_NAMES = (
    "Hemolysis",
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


def canonical_objectives(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    else:
        raw = [str(x).strip() for x in value]
    aliases = {
        "hemolysis": "Hemolysis",
        "nonhemolysis": "Hemolysis",
        "non-hemolysis": "Hemolysis",
        "nonfouling": "Non-Fouling",
        "non-fouling": "Non-Fouling",
        "solubility": "Solubility",
        "permeability": "Permeability",
        "halflife": "Half-Life",
        "half-life": "Half-Life",
        "affinity": "Affinity",
    }
    out = []
    for name in raw:
        key = name.lower().replace("_", "-").replace(" ", "")
        key = key.replace("non-fouling", "nonfouling").replace("half-life", "halflife")
        if key not in aliases:
            raise ValueError(f"unsupported objective {name!r}")
        out.append(aliases[key])
    if not out:
        raise ValueError("no objectives selected")
    return tuple(out)


def load_peptiverse_predictor(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    device: str = "cuda",
):
    """Load PeptiVerse directly, without depending on the TCFM objective module.

    The PeptiVerse repository remains an optional external evaluation dependency;
    Koopman training itself never imports it.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from inference import PeptiVersePredictor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not import PeptiVersePredictor from {root}. "
            "Set --peptiverse-root to the PeptiVerse repository."
        ) from exc
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root / "best_models.txt"
    )
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    return PeptiVersePredictor(
        manifest_path=str(manifest),
        classifier_weight_root=str(root) + "/",
        device=str(device),
    )


class PeptideObjectiveOracle:
    def __init__(
        self,
        predictor: Any,
        *,
        target: str,
        objectives: Sequence[str],
        score_mode: str = "current_module",
    ):
        self.pred = predictor
        self.target = str(target)
        self.objectives = tuple(objectives)
        self.score_mode = str(score_mode)
        if self.score_mode not in {"current_module", "raw"}:
            raise ValueError("score_mode must be current_module or raw")
        if "Affinity" in self.objectives and len(self.target) < 10:
            raise ValueError("Affinity requires a valid target protein sequence")
        self.objective_vector_queries = 0
        self.predictor_calls = 0

    def evaluate(self, sequences: Sequence[str]) -> Tensor:
        rows = []
        for seq in sequences:
            row = []
            for name in self.objectives:
                if name == "Affinity":
                    raw = float(
                        self.pred.predict_binding_affinity(
                            col="wt", target_seq=self.target, binder_str=seq
                        )["affinity"]
                    )
                    value = raw / 10.0 if self.score_mode == "current_module" else raw
                else:
                    raw = float(
                        self.pred.predict_property(
                            _PROPERTY_KEYS[name], col="wt", input_str=seq
                        )["score"]
                    )
                    if name == "Hemolysis":
                        value = 1.0 - raw
                    elif name == "Half-Life" and self.score_mode == "current_module":
                        value = max(0.0, min(raw, 2.0)) / 2.0
                    else:
                        value = raw
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} score for {seq!r}")
                row.append(value)
            rows.append(row)
        self.objective_vector_queries += len(rows)
        self.predictor_calls += len(rows) * len(self.objectives)
        return torch.tensor(rows, dtype=torch.float32)


ESM_ID_TO_AA = {
    4: "L", 5: "A", 6: "G", 7: "V", 8: "S", 9: "E", 10: "R", 11: "T",
    12: "I", 13: "D", 14: "P", 15: "K", 16: "Q", 17: "N", 18: "F",
    19: "Y", 20: "M", 21: "H", 22: "W", 23: "C",
}


def decode_esm_tokens(tokens: Tensor, model_name: str = "facebook/esm2_t33_650M_UR50D") -> list[str]:
    """Decode the fixed ESM IDs used by the scratch Uniform-LKF.

    `model_name` is retained for API compatibility but no tokenizer download is
    required.  Valid peptide residues are exactly IDs 4..23.
    """
    del model_name
    x = torch.as_tensor(tokens).detach().cpu()
    if x.ndim != 2:
        raise ValueError(f"tokens must be [batch,length], got {tuple(x.shape)}")
    out: list[str] = []
    for row in x.tolist():
        if len(row) < 3 or int(row[0]) != 0 or int(row[-1]) != 2:
            raise ValueError("expected ESM <cls> ... <eos> token framing")
        try:
            seq = "".join(ESM_ID_TO_AA[int(tok)] for tok in row[1:-1])
        except KeyError as exc:
            raise ValueError(f"non-canonical peptide token id {exc.args[0]}") from exc
        out.append(seq)
    return out


@torch.no_grad()
def feature_sets(
    model: KoopmanRegularizedLKF,
    tokens: Tensor,
    *,
    batch_size: int,
    include_esm: bool,
    esm_model_name: str,
    random_seed: int,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    z_chunks: list[Tensor] = []
    h_chunks: list[Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        x = tokens[start : start + batch_size].to(device)
        z_chunks.append(model.features(x, 1.0, update_whitener=False).cpu())
        h_chunks.append(model.lkf_hidden(x, 1.0).float().cpu())
    z = torch.cat(z_chunks, dim=0)
    hidden = torch.cat(h_chunks, dim=0)
    result: dict[str, np.ndarray] = {
        "koopman": z.numpy(),
        "lkf_hidden": hidden.numpy(),
    }

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(random_seed))
    d_in = hidden.shape[1]
    d_out = model.feature_dim
    random_matrix = torch.randn(d_in, d_out, generator=gen) / math.sqrt(float(d_out))
    result["lkf_random_projection"] = (hidden @ random_matrix).numpy()

    if include_esm:
        try:
            from transformers import EsmModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for --include-esm") from exc
        esm = EsmModel.from_pretrained(esm_model_name).to(device)
        esm.eval()
        chunks = []
        for start in range(0, tokens.shape[0], batch_size):
            x = tokens[start : start + batch_size].to(device)
            mask = torch.ones_like(x)
            output = esm(input_ids=x, attention_mask=mask).last_hidden_state
            chunks.append(output[:, 1:-1].mean(dim=1).float().cpu())
        esm_hidden = torch.cat(chunks, dim=0)
        result["esm_hidden"] = esm_hidden.numpy()
        d_esm = esm_hidden.shape[1]
        gen.manual_seed(int(random_seed) + 17)
        rp = torch.randn(d_esm, d_out, generator=gen) / math.sqrt(float(d_out))
        result["esm_random_projection"] = (esm_hidden @ rp).numpy()
        del esm
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return float("nan")
    try:
        return float(fn(a, b).statistic)
    except Exception:
        return float("nan")


def run_few_shot_probes(
    features: Mapping[str, np.ndarray],
    labels: Tensor | np.ndarray,
    *,
    objective_names: Sequence[str],
    q_values: Sequence[int],
    test_size: int,
    repeats: int,
    ridge_alpha: float,
    seed: int,
) -> tuple[list[dict[str, Any]], Ridge]:
    y = np.asarray(labels, dtype=np.float64)
    n = y.shape[0]
    if y.ndim != 2 or y.shape[1] != len(objective_names):
        raise ValueError("labels shape does not match objective_names")
    if test_size < 8 or test_size >= n:
        raise ValueError("test_size must leave a non-empty training reservoir")
    max_q = max(int(q) for q in q_values)
    if max_q > n - test_size:
        raise ValueError(
            f"largest Q={max_q} exceeds training reservoir {n-test_size}; enlarge probe pool"
        )
    rows: list[dict[str, Any]] = []
    for rep in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + 1009 * rep)
        perm = rng.permutation(n)
        test_idx = perm[-test_size:]
        reservoir = perm[:-test_size]
        for q in q_values:
            train_idx = reservoir[: int(q)]
            for feature_name, x in features.items():
                scaler = StandardScaler().fit(x[train_idx])
                x_train = scaler.transform(x[train_idx])
                x_test = scaler.transform(x[test_idx])
                reg = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
                reg.fit(x_train, y[train_idx])
                pred = reg.predict(x_test)
                for j, objective in enumerate(objective_names):
                    true_j = y[test_idx, j]
                    pred_j = pred[:, j]
                    rmse = float(math.sqrt(mean_squared_error(true_j, pred_j)))
                    denom = float(np.std(true_j))
                    pred_std = float(np.std(pred_j))
                    if pred_std > 0 and true_j.size >= 2:
                        calibration_slope, calibration_intercept = np.polyfit(pred_j, true_j, 1)
                    else:
                        calibration_slope, calibration_intercept = float("nan"), float("nan")
                    rows.append(
                        {
                            "feature": feature_name,
                            "feature_dim": int(x.shape[1]),
                            "Q": int(q),
                            "repeat": rep,
                            "objective": objective,
                            "pearson": _safe_corr(pearsonr, true_j, pred_j),
                            "spearman": _safe_corr(spearmanr, true_j, pred_j),
                            "rmse": rmse,
                            "normalized_rmse": rmse / denom if denom > 0 else float("nan"),
                            "mae": float(np.mean(np.abs(true_j - pred_j))),
                            "r2": float(r2_score(true_j, pred_j)),
                            "prediction_mean": float(np.mean(pred_j)),
                            "target_mean": float(np.mean(true_j)),
                            "prediction_std": pred_std,
                            "target_std": denom,
                            "mean_bias": float(np.mean(pred_j - true_j)),
                            "calibration_slope_true_on_pred": float(calibration_slope),
                            "calibration_intercept_true_on_pred": float(calibration_intercept),
                        }
                    )

    # A full-data readout in the native whitened Koopman coordinates is useful
    # for reachability projection.  Koopman features are already normalized, so
    # no extra StandardScaler is used here.
    full_readout = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    full_readout.fit(np.asarray(features["koopman"], dtype=np.float64), y)
    return rows, full_readout


__all__ = [
    "OBJECTIVE_NAMES",
    "canonical_objectives",
    "load_peptiverse_predictor",
    "PeptideObjectiveOracle",
    "decode_esm_tokens",
    "feature_sets",
    "run_few_shot_probes",
]
