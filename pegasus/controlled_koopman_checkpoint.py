"""Checkpoint helpers for the final controlled Koopman flow map."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .controlled_koopman_model import ControlledKoopmanConfig, ControlledKoopmanFlowMap
from .ip_checkpoint import load_ip_koopman_checkpoint, sha256_file, torch_load_mapping
from .lkf.uniform_lkf import load_uniform_lkf_checkpoint

MODEL_TYPE = "information_preserving_controlled_koopman_flow_map"
FORMAT_VERSION = 2


def initialize_controlled_from_ip_checkpoint(
    ip_checkpoint: str | Path,
    *,
    base_lkf_checkpoint: str | Path,
    config: ControlledKoopmanConfig | None = None,
    strict_base_sha: bool = True,
) -> tuple[ControlledKoopmanFlowMap, Mapping[str, Any]]:
    ip_model, meta = load_ip_koopman_checkpoint(
        ip_checkpoint,
        base_lkf_checkpoint=base_lkf_checkpoint,
        device="cpu",
        strict_base_sha=bool(strict_base_sha),
        eval_mode=False,
    )
    model = ControlledKoopmanFlowMap.from_ip_model(ip_model, config=config)
    return model, meta


def save_controlled_koopman_checkpoint(
    path: str | Path,
    model: ControlledKoopmanFlowMap,
    *,
    base_lkf_checkpoint: str | Path,
    source_ip_checkpoint: str | Path,
    stage: str,
    epoch: int,
    global_step: int,
    training_args: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    source = Path(source_ip_checkpoint).expanduser().resolve()
    payload: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "checkpoint_format_version": FORMAT_VERSION,
        "stage": str(stage).upper(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "controlled_koopman_config": model.config.to_dict(),
        "controlled_koopman_state_dict": {
            k: v.detach().cpu() for k, v in model.auxiliary_state_dict().items()
        },
        "lkf_state_dict": {k: v.detach().cpu() for k, v in model.lkf.state_dict().items()},
        "lkf_model_config": model.lkf.model_config,
        "base_lkf_checkpoint": str(base),
        "base_lkf_sha256": sha256_file(base),
        "source_ip_checkpoint": str(source),
        "source_ip_sha256": sha256_file(source),
        "training_args": dict(training_args),
        "metrics": dict(metrics),
        "objective_free_training": True,
        "representation": "fixed information-preserving anchor only (no learned lift)",
        "operator_parameterization": "piecewise-constant noncommuting L(t), exact ordered matrix-exponential composition",
        "control_parameterization": "global knot-specific J tied to physical Gaussianized categorical innovations",
    }
    torch.save(payload, path)
    return path


def load_controlled_koopman_checkpoint(
    checkpoint: str | Path,
    *,
    base_lkf_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    strict_base_sha: bool = True,
    eval_mode: bool = True,
) -> tuple[ControlledKoopmanFlowMap, Mapping[str, Any]]:
    ckpt = torch_load_mapping(checkpoint, map_location="cpu")
    if ckpt.get("model_type") != MODEL_TYPE:
        raise ValueError("not a controlled Koopman flow-map checkpoint")
    if int(ckpt.get("checkpoint_format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported controlled Koopman checkpoint format")
    if base_lkf_checkpoint is None:
        base_lkf_checkpoint = ckpt.get("base_lkf_checkpoint")
    if not base_lkf_checkpoint:
        raise ValueError("base LKF checkpoint must be supplied")
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    expected_sha = ckpt.get("base_lkf_sha256")
    if strict_base_sha and expected_sha and str(expected_sha) != sha256_file(base):
        raise ValueError("base LKF SHA256 mismatch")

    lkf = load_uniform_lkf_checkpoint(base, map_location="cpu", eval_mode=False)
    state = ckpt.get("lkf_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing lkf_state_dict")
    lkf.load_state_dict(state, strict=True)

    # The immutable anchor encoder/projection are persisted in the auxiliary state,
    # so loading does not depend on the original IP checkpoint still being present.
    cfg = ControlledKoopmanConfig.from_mapping(ckpt["controlled_koopman_config"])
    model = ControlledKoopmanFlowMap(lkf, cfg)
    model.load_auxiliary_state_dict(ckpt["controlled_koopman_state_dict"], strict=True)
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, ckpt


__all__ = [
    "MODEL_TYPE",
    "FORMAT_VERSION",
    "initialize_controlled_from_ip_checkpoint",
    "save_controlled_koopman_checkpoint",
    "load_controlled_koopman_checkpoint",
]
