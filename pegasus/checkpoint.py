"""Checkpoint/provenance helpers for K-LKF."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .lkf.uniform_lkf import UniformLKF, load_uniform_lkf_checkpoint

from .model import KoopmanConfig, KoopmanRegularizedLKF


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch_load(path: str | Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, Mapping):
        raise TypeError("checkpoint must be a mapping")
    return obj


def save_koopman_checkpoint(
    path: str | Path,
    model: KoopmanRegularizedLKF,
    *,
    base_lkf_checkpoint: str | Path,
    stage: str,
    epoch: int,
    global_step: int,
    training_args: Mapping[str, Any],
    metrics: Mapping[str, float],
    include_lkf_state: bool,
    base_lkf_sha256: Optional[str] = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    payload: dict[str, Any] = {
        "model_type": "koopman_regularized_uniform_lkf",
        "checkpoint_format_version": 2,
        "stage": str(stage).upper(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "koopman_config": model.config.to_dict(),
        "koopman_state_dict": {
            key: value.detach().cpu() for key, value in model.auxiliary_state_dict().items()
        },
        "base_lkf_checkpoint": str(base),
        "base_lkf_sha256": str(base_lkf_sha256) if base_lkf_sha256 is not None else sha256_file(base),
        "lkf_model_config": model.lkf.model_config,
        "training_args": dict(training_args),
        "metrics": dict(metrics),
        "objective_free_training": True,
        "operator_parameterization": "A_st=matrix_exp((t-s)L)",
        "feature_source": model.config.feature_source,
        "representation_normalization": model.config.representation_norm,
    }
    if include_lkf_state:
        payload["lkf_state_dict"] = {
            key: value.detach().cpu() for key, value in model.lkf.state_dict().items()
        }
    torch.save(payload, path)
    return path


def load_koopman_checkpoint(
    koopman_checkpoint: str | Path,
    *,
    base_lkf_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    strict_base_sha: bool = True,
    eval_mode: bool = True,
) -> tuple[KoopmanRegularizedLKF, Mapping[str, Any]]:
    ckpt_path = Path(koopman_checkpoint).expanduser().resolve()
    ckpt = _torch_load(ckpt_path, map_location="cpu")
    if ckpt.get("model_type") != "koopman_regularized_uniform_lkf":
        raise ValueError("not a K-LKF checkpoint")
    state = ckpt.get("koopman_state_dict", {})
    if isinstance(state, Mapping) and any(str(k).startswith("whitener.") for k in state):
        raise ValueError(
            "This checkpoint uses the deprecated dynamic-whitening Koopman representation. "
            "It is intentionally incompatible with the fixed-L2 formulation; rerun Stage A from M8."
        )
    if base_lkf_checkpoint is None:
        recorded = ckpt.get("base_lkf_checkpoint")
        if not recorded:
            raise ValueError("base LKF checkpoint must be supplied")
        base_lkf_checkpoint = recorded
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(base)
    expected_sha = ckpt.get("base_lkf_sha256")
    actual_sha = sha256_file(base)
    if strict_base_sha and expected_sha and str(expected_sha) != actual_sha:
        raise ValueError(
            "base LKF SHA256 does not match the immutable checkpoint used by Koopman training"
        )
    lkf: UniformLKF = load_uniform_lkf_checkpoint(base, map_location="cpu", eval_mode=False)
    if "lkf_state_dict" in ckpt:
        lkf.load_state_dict(ckpt["lkf_state_dict"], strict=True)
    config = KoopmanConfig.from_mapping(ckpt["koopman_config"])
    model = KoopmanRegularizedLKF(lkf, config)
    model.load_auxiliary_state_dict(ckpt["koopman_state_dict"], strict=True)
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, ckpt


__all__ = ["sha256_file", "save_koopman_checkpoint", "load_koopman_checkpoint"]
