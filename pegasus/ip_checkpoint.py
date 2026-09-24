"""Checkpoint helpers for the information-preserving Koopman lift."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .ip_model import InformationPreservingKoopmanConfig, InformationPreservingKoopmanLKF
from .lkf.uniform_lkf import UniformLKF, load_uniform_lkf_checkpoint

MODEL_TYPE = "information_preserving_koopman_uniform_lkf"
FORMAT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_mapping(path: str | Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, Mapping):
        raise TypeError("checkpoint must be a mapping")
    return obj


def load_generator_from_koopman_checkpoint(
    koopman_checkpoint: str | Path,
    *,
    base_lkf_checkpoint: str | Path,
    strict_base_sha: bool = True,
) -> tuple[UniformLKF, Mapping[str, Any]]:
    """Extract only the active LKF generator from an older/newer Koopman checkpoint.

    This is the intended bridge from the successful fixed-norm Stage-B K-LKF
    checkpoint to the new information-preserving Stage A.  The old Koopman head
    is deliberately discarded.
    """
    ckpt = torch_load_mapping(koopman_checkpoint, map_location="cpu")
    state = ckpt.get("lkf_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(
            "Generator-initialization Koopman checkpoint has no lkf_state_dict. "
            "Use a Stage-B checkpoint that saved the fine-tuned LKF weights."
        )
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    expected_sha = ckpt.get("base_lkf_sha256")
    actual_sha = sha256_file(base)
    if strict_base_sha and expected_sha and str(expected_sha) != actual_sha:
        raise ValueError("base LKF SHA256 does not match generator-initialization checkpoint")
    lkf = load_uniform_lkf_checkpoint(base, map_location="cpu", eval_mode=False)
    lkf.load_state_dict(state, strict=True)
    return lkf, ckpt


def save_ip_koopman_checkpoint(
    path: str | Path,
    model: InformationPreservingKoopmanLKF,
    *,
    base_lkf_checkpoint: str | Path,
    stage: str,
    epoch: int,
    global_step: int,
    training_args: Mapping[str, Any],
    metrics: Mapping[str, Any],
    base_lkf_sha256: Optional[str] = None,
    generator_init_checkpoint: Optional[str | Path] = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    init_path = (
        str(Path(generator_init_checkpoint).expanduser().resolve())
        if generator_init_checkpoint
        else None
    )
    payload: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "checkpoint_format_version": FORMAT_VERSION,
        "stage": str(stage).upper(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "ip_koopman_config": model.config.to_dict(),
        "ip_koopman_state_dict": {
            k: v.detach().cpu() for k, v in model.auxiliary_state_dict().items()
        },
        # Always save active LKF state. Stage A may itself start from a previous
        # Stage-B generator rather than the immutable M8 checkpoint.
        "lkf_state_dict": {k: v.detach().cpu() for k, v in model.lkf.state_dict().items()},
        "base_lkf_checkpoint": str(base),
        "base_lkf_sha256": str(base_lkf_sha256) if base_lkf_sha256 else sha256_file(base),
        "generator_init_koopman_checkpoint": init_path,
        "generator_init_koopman_sha256": sha256_file(init_path) if init_path else None,
        "lkf_model_config": model.lkf.model_config,
        "training_args": dict(training_args),
        "metrics": dict(metrics),
        "objective_free_training": True,
        "representation": "phi=[fixed_random_projection(frozen_reference_lkf_hidden); learned_lift]",
        "operator_parameterization": "A_st=matrix_exp((t-s)L)",
    }
    torch.save(payload, path)
    return path


def load_ip_koopman_checkpoint(
    checkpoint: str | Path,
    *,
    base_lkf_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    strict_base_sha: bool = True,
    eval_mode: bool = True,
) -> tuple[InformationPreservingKoopmanLKF, Mapping[str, Any]]:
    ckpt = torch_load_mapping(checkpoint, map_location="cpu")
    if ckpt.get("model_type") != MODEL_TYPE:
        raise ValueError("not an information-preserving Koopman-LKF checkpoint")
    if int(ckpt.get("checkpoint_format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported information-preserving Koopman checkpoint format")
    if base_lkf_checkpoint is None:
        base_lkf_checkpoint = ckpt.get("base_lkf_checkpoint")
    if not base_lkf_checkpoint:
        raise ValueError("base LKF checkpoint must be supplied")
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    expected_sha = ckpt.get("base_lkf_sha256")
    actual_sha = sha256_file(base)
    if strict_base_sha and expected_sha and str(expected_sha) != actual_sha:
        raise ValueError("base LKF SHA256 mismatch")
    lkf = load_uniform_lkf_checkpoint(base, map_location="cpu", eval_mode=False)
    state = ckpt.get("lkf_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing lkf_state_dict")
    lkf.load_state_dict(state, strict=True)
    config = InformationPreservingKoopmanConfig.from_mapping(ckpt["ip_koopman_config"])
    model = InformationPreservingKoopmanLKF(lkf, config)
    model.load_auxiliary_state_dict(ckpt["ip_koopman_state_dict"], strict=True)
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, ckpt


__all__ = [
    "MODEL_TYPE",
    "FORMAT_VERSION",
    "sha256_file",
    "torch_load_mapping",
    "load_generator_from_koopman_checkpoint",
    "save_ip_koopman_checkpoint",
    "load_ip_koopman_checkpoint",
]
