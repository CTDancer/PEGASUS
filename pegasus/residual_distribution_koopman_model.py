"""Shared-residual distributional Koopman model.

This is intentionally *not* a second generator and not a controller.  It wraps
our strongest validated direct finite-time mean Koopman model and augments it
with an empirical, chain-specific residual law:

    z_T = mu_C(x_s) + eps_C,

where z is the fixed information-normalized anchor.  The residual law is stored
nonparametrically so multimodality and heavy tails are retained.

Two residual banks are kept:
- intrinsic: z_T - empirical E[z_T | x_s], isolating stochastic shape;
- predictive: z_T - mu_C(x_s), representing end-to-end predictive error/tails.

No residual vector is decoded back to a peptide.  This model is descriptive and
is used to forecast reachable terminal distributions/tails and allocate future
exploration, not to invent an expectation-to-sequence decoder.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np
import torch

from .terminal_controlled_koopman_checkpoint import load_terminal_controlled_koopman_checkpoint
from .terminal_controlled_koopman_geometry import chain_name
from .checkpoint import sha256_file


@dataclass(frozen=True)
class ResidualDistributionConfig:
    chains: tuple[tuple[float, ...], ...]
    direction_count: int = 32
    direction_seed: int = 84217
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
    betas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    max_bank_size: int = 16384

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chains"] = [list(c) for c in self.chains]
        d["quantiles"] = list(self.quantiles)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidualDistributionConfig":
        return cls(
            chains=tuple(tuple(float(x) for x in c) for c in value["chains"]),
            direction_count=int(value.get("direction_count", 32)),
            direction_seed=int(value.get("direction_seed", 84217)),
            quantiles=tuple(float(x) for x in value.get("quantiles", (0.05,0.1,0.5,0.8,0.9,0.95))),
            betas=tuple(float(x) for x in value.get("betas", (0.5,1.0,2.0,4.0))),
            max_bank_size=int(value.get("max_bank_size", 16384)),
        )


class SharedResidualKoopman:
    def __init__(
        self,
        base_model,
        config: ResidualDistributionConfig,
        laws: Mapping[str, Mapping[str, Any]],
        *,
        checkpoint_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_model = base_model
        self.config = config
        self.laws = dict(laws)
        self.checkpoint_payload = checkpoint_payload
        for c in self.config.chains:
            self.base_model.match_chain(c)
            if chain_name(c) not in self.laws:
                raise KeyError(f"missing residual law for {chain_name(c)}")

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @property
    def anchor_dim(self) -> int:
        return int(self.base_model.anchor_dim)

    @property
    def chains(self) -> tuple[tuple[float, ...], ...]:
        return self.config.chains

    def match_chain(self, chain: Sequence[float]) -> tuple[float, ...]:
        c = self.base_model.match_chain(chain)
        if chain_name(c) not in self.laws:
            raise KeyError(f"chain {chain_name(c)} was not residual-calibrated")
        return c

    @torch.no_grad()
    def predict_mean_z(self, x_start: torch.Tensor, chain: Sequence[float]) -> torch.Tensor:
        c = self.match_chain(chain)
        r0 = self.base_model.anchor_features(x_start, float(c[0]))
        pred_r = self.base_model.predict_uncontrolled_mean(r0, c)
        return self.base_model.information_normalize(pred_r)

    def residual_bank(self, chain: Sequence[float], *, kind: str = "predictive") -> np.ndarray:
        c = self.match_chain(chain)
        law = self.laws[chain_name(c)]
        key = f"{kind}_bank"
        if key not in law:
            raise KeyError(f"law has no {key}")
        return np.asarray(law[key], dtype=np.float32)

    def source_summary(self, chain: Sequence[float]) -> Mapping[str, Any]:
        c = self.match_chain(chain)
        return self.laws[chain_name(c)]["state_summaries"]

    def sample_terminal_z(
        self,
        x_start: torch.Tensor,
        chain: Sequence[float],
        *,
        samples: int,
        kind: str = "predictive",
        seed: int = 0,
    ) -> np.ndarray:
        """Sample approximate terminal information vectors from mean + residual bank."""
        mu = self.predict_mean_z(x_start.to(self.device), chain).detach().cpu().numpy().astype(np.float64)
        bank = self.residual_bank(chain, kind=kind).astype(np.float64)
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, bank.shape[0], size=(mu.shape[0], int(samples)))
        eps = bank[idx]
        return mu[:, None, :] + eps

    def chain_law(self, chain: Sequence[float]) -> Mapping[str, Any]:
        c = self.match_chain(chain)
        return self.laws[chain_name(c)]


MODEL_TYPE = "shared_residual_distributional_koopman"
FORMAT_VERSION = 1


def save_residual_distribution_checkpoint(
    path: str | Path,
    *,
    terminal_checkpoint: str | Path,
    base_lkf_checkpoint: str | Path,
    config: ResidualDistributionConfig,
    laws: Mapping[str, Mapping[str, Any]],
    calibration_args: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    terminal = Path(terminal_checkpoint).expanduser().resolve()
    base = Path(base_lkf_checkpoint).expanduser().resolve()
    serial_laws: dict[str, dict[str, Any]] = {}
    for name, law in laws.items():
        serial_laws[name] = {}
        for k, v in law.items():
            if isinstance(v, np.ndarray):
                # Banks can be large; fp16 is sufficient for empirical resampling,
                # while summaries/source features remain fp32.
                dtype = torch.float16 if k.endswith("_bank") else torch.float32
                serial_laws[name][k] = torch.as_tensor(v, dtype=dtype)
            elif isinstance(v, Mapping):
                nested = {}
                for nk, nv in v.items():
                    nested[nk] = torch.as_tensor(nv, dtype=torch.float32) if isinstance(nv, np.ndarray) else nv
                serial_laws[name][k] = nested
            else:
                serial_laws[name][k] = v
    payload = {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "terminal_checkpoint": str(terminal),
        "terminal_checkpoint_sha256": sha256_file(terminal),
        "base_lkf_checkpoint": str(base),
        "base_lkf_sha256": sha256_file(base),
        "config": config.to_dict(),
        "laws": serial_laws,
        "calibration_args": dict(calibration_args),
        "calibration_summary": dict(calibration_summary),
        "scientific_model": "z_T = direct_Koopman_mean(x_s, chain) + chain_shared_empirical_residual",
        "objective_free": True,
        "does_not_decode_anchor_to_sequence": True,
    }
    torch.save(payload, p)
    return p


def load_residual_distribution_checkpoint(
    checkpoint: str | Path,
    *,
    terminal_checkpoint: str | Path | None = None,
    base_lkf_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict_sha: bool = True,
) -> tuple[SharedResidualKoopman, Mapping[str, Any]]:
    p = Path(checkpoint).expanduser().resolve()
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("model_type") != MODEL_TYPE or int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("not a supported shared-residual Koopman checkpoint")
    terminal = Path(terminal_checkpoint or payload["terminal_checkpoint"]).expanduser().resolve()
    base = Path(base_lkf_checkpoint or payload["base_lkf_checkpoint"]).expanduser().resolve()
    if strict_sha:
        if payload.get("terminal_checkpoint_sha256") and sha256_file(terminal) != payload["terminal_checkpoint_sha256"]:
            raise ValueError("terminal checkpoint SHA256 mismatch")
        if payload.get("base_lkf_sha256") and sha256_file(base) != payload["base_lkf_sha256"]:
            raise ValueError("base LKF checkpoint SHA256 mismatch")
    base_model, _ = load_terminal_controlled_koopman_checkpoint(
        terminal, base_lkf_checkpoint=base, device=device, strict_base_sha=strict_sha, eval_mode=True
    )
    laws: dict[str, dict[str, Any]] = {}
    for name, law in payload["laws"].items():
        out: dict[str, Any] = {}
        for k, v in law.items():
            if torch.is_tensor(v):
                out[k] = v.cpu().numpy().astype(np.float32)
            elif isinstance(v, Mapping):
                out[k] = {nk: (nv.cpu().numpy().astype(np.float32) if torch.is_tensor(nv) else nv) for nk, nv in v.items()}
            else:
                out[k] = v
        laws[name] = out
    cfg = ResidualDistributionConfig.from_mapping(payload["config"])
    return SharedResidualKoopman(base_model, cfg, laws, checkpoint_payload=payload), payload


__all__ = [
    "ResidualDistributionConfig", "SharedResidualKoopman", "MODEL_TYPE", "FORMAT_VERSION",
    "save_residual_distribution_checkpoint", "load_residual_distribution_checkpoint",
]
