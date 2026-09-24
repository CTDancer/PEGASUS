"""Checkpoint helpers for direct-terminal controlled Koopman flow maps."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .controlled_koopman_checkpoint import load_controlled_koopman_checkpoint
from .ip_checkpoint import sha256_file, torch_load_mapping
from .lkf.uniform_lkf import load_uniform_lkf_checkpoint
from .terminal_controlled_koopman_model import (
    TerminalControlledKoopmanConfig,
    TerminalControlledKoopmanFlowMap,
)

MODEL_TYPE = "information_preserving_direct_terminal_controlled_koopman_flow_map"
FORMAT_VERSION = 1


def initialize_terminal_from_controlled_checkpoint(
    controlled_checkpoint: str | Path,
    *,
    base_lkf_checkpoint: str | Path,
    config: TerminalControlledKoopmanConfig,
    strict_base_sha: bool = True,
) -> tuple[TerminalControlledKoopmanFlowMap, Mapping[str, Any]]:
    old, meta = load_controlled_koopman_checkpoint(
        controlled_checkpoint,
        base_lkf_checkpoint=base_lkf_checkpoint,
        device="cpu",
        strict_base_sha=bool(strict_base_sha),
        eval_mode=False,
    )
    model = TerminalControlledKoopmanFlowMap.from_controlled_stage_a(old, config)
    return model, meta


def save_terminal_controlled_koopman_checkpoint(
    path: str | Path,
    model: TerminalControlledKoopmanFlowMap,
    *,
    base_lkf_checkpoint: str | Path,
    source_controlled_checkpoint: str | Path,
    stage: str,
    epoch: int,
    global_step: int,
    training_args: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    source = Path(source_controlled_checkpoint).expanduser().resolve()
    payload: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "checkpoint_format_version": FORMAT_VERSION,
        "stage": str(stage).upper(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "terminal_controlled_koopman_config": model.config.to_dict(),
        "terminal_controlled_koopman_state_dict": {
            k: v.detach().cpu() for k, v in model.auxiliary_state_dict().items()
        },
        "lkf_state_dict": {k: v.detach().cpu() for k, v in model.lkf.state_dict().items()},
        "lkf_model_config": model.lkf.model_config,
        "base_lkf_checkpoint": str(base),
        "base_lkf_sha256": sha256_file(base),
        "source_controlled_checkpoint": str(source),
        "source_controlled_sha256": sha256_file(source),
        "training_args": dict(training_args),
        "metrics": dict(metrics),
        "objective_free_training": True,
        "representation": "fixed information-preserving anchor only",
        "mean_parameterization": "direct chain-specific conditional-mean operator A_C; no finite-dimensional semigroup constraint",
        "control_parameterization": "direct terminal chain pulse derivatives M_{C,k} tied to physical Gaussianized innovations",
    }
    torch.save(payload, path)
    return path


def load_terminal_controlled_koopman_checkpoint(
    checkpoint: str | Path,
    *,
    base_lkf_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    strict_base_sha: bool = True,
    eval_mode: bool = True,
) -> tuple[TerminalControlledKoopmanFlowMap, Mapping[str, Any]]:
    ckpt = torch_load_mapping(checkpoint, map_location="cpu")
    if ckpt.get("model_type") != MODEL_TYPE:
        raise ValueError("not a direct-terminal controlled Koopman checkpoint")
    if int(ckpt.get("checkpoint_format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported direct-terminal checkpoint format")
    if base_lkf_checkpoint is None:
        base_lkf_checkpoint = ckpt.get("base_lkf_checkpoint")
    if not base_lkf_checkpoint:
        raise ValueError("base LKF checkpoint must be supplied")
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    expected = ckpt.get("base_lkf_sha256")
    if strict_base_sha and expected and str(expected) != sha256_file(base):
        raise ValueError("base LKF SHA256 mismatch")

    lkf = load_uniform_lkf_checkpoint(base, map_location="cpu", eval_mode=False)
    state = ckpt.get("lkf_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint missing lkf_state_dict")
    lkf.load_state_dict(state, strict=True)
    cfg = TerminalControlledKoopmanConfig.from_mapping(ckpt["terminal_controlled_koopman_config"])
    model = TerminalControlledKoopmanFlowMap(lkf, cfg)
    model.load_auxiliary_state_dict(ckpt["terminal_controlled_koopman_state_dict"], strict=True)
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, ckpt


__all__ = [
    "MODEL_TYPE",
    "FORMAT_VERSION",
    "initialize_terminal_from_controlled_checkpoint",
    "save_terminal_controlled_koopman_checkpoint",
    "load_terminal_controlled_koopman_checkpoint",
]
