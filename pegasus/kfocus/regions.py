"""Residual witness regions and cheap conditional generation for K-FOCUS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
import torch

from ..residual_distribution_koopman_model import SharedResidualKoopman
from ..terminal_controlled_koopman_geometry import chain_name, prepare_chain_start


@dataclass(frozen=True)
class ResidualMetric:
    scale: np.ndarray

    @classmethod
    def from_bank(cls, bank: np.ndarray) -> "ResidualMetric":
        b = np.asarray(bank, dtype=np.float64)
        scale = np.std(b, axis=0)
        positive = scale[scale > 1e-8]
        floor = float(np.median(positive) * 0.05) if positive.size else 1.0
        scale = np.where(scale > floor, scale, floor)
        return cls(scale=scale.astype(np.float64))

    def distance(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        cc = np.asarray(center, dtype=np.float64).reshape(-1)
        z = (xx - cc) / self.scale
        return np.sqrt(np.mean(z*z, axis=-1))


@dataclass
class WitnessRegion:
    chain: tuple[float, ...]
    witness_sequence: str
    witness_id: str
    target_residual: np.ndarray
    radius: float
    target_mass: float
    metric_scale: np.ndarray
    randomized_target: bool = False

    def contains(self, residuals: np.ndarray) -> np.ndarray:
        metric = ResidualMetric(self.metric_scale)
        return metric.distance(residuals, self.target_residual) <= float(self.radius)


@torch.no_grad()
def terminal_z(model: SharedResidualKoopman, tokens: torch.Tensor, *, batch_size: int = 256) -> np.ndarray:
    base = model.base_model
    out = []
    for i in range(0, int(tokens.shape[0]), int(batch_size)):
        x = tokens[i:i+batch_size].to(model.device)
        z = base.information_normalize(base.anchor_features(x, 1.0))
        out.append(z.cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def make_chain_start(
    model: SharedResidualKoopman,
    incumbent_tokens: torch.Tensor,
    chain: Sequence[float],
    *,
    seed: int,
) -> torch.Tensor:
    c = model.match_chain(chain)
    g = torch.Generator(device=model.device); g.manual_seed(int(seed))
    return prepare_chain_start(
        model.base_model,
        incumbent_tokens.reshape(1, -1).to(model.device),
        c,
        generator=g,
    )


@torch.no_grad()
def predicted_mean_z_for_start(
    model: SharedResidualKoopman,
    x_start: torch.Tensor,
    chain: Sequence[float],
) -> np.ndarray:
    x = torch.as_tensor(x_start).detach().to(device=model.device, dtype=torch.long)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"x_start must be [length] or [batch,length], got shape {tuple(x.shape)}")
    z = model.predict_mean_z(x, chain)
    return z.detach().cpu().numpy().astype(np.float64)[0]


@torch.no_grad()
def build_witness_region(
    model: SharedResidualKoopman,
    x_start: torch.Tensor,
    chain: Sequence[float],
    witness_tokens: torch.Tensor,
    witness_sequence: str,
    witness_id: str,
    *,
    target_mass: float,
    randomized_target: bool = False,
    seed: int = 0,
) -> WitnessRegion:
    if not (0.0 < float(target_mass) < 1.0):
        raise ValueError("target_mass must lie in (0,1)")
    c = model.match_chain(chain)
    bank = model.residual_bank(c, kind="predictive").astype(np.float64)
    metric = ResidualMetric.from_bank(bank)
    mu = predicted_mean_z_for_start(model, x_start, c)
    if randomized_target:
        rng = np.random.default_rng(int(seed))
        target = bank[int(rng.integers(bank.shape[0]))].copy()
    else:
        wz = terminal_z(model, witness_tokens.reshape(1, -1))[0]
        target = wz - mu
    distances = metric.distance(bank, target)
    radius = float(np.quantile(distances, float(target_mass), method="higher"))
    radius = max(radius, 1e-8)
    return WitnessRegion(
        chain=tuple(c),
        witness_sequence=str(witness_sequence),
        witness_id=str(witness_id),
        target_residual=target.astype(np.float64),
        radius=radius,
        target_mass=float(target_mass),
        metric_scale=metric.scale.copy(),
        randomized_target=bool(randomized_target),
    )


@torch.no_grad()
def sample_candidate(
    model: SharedResidualKoopman,
    x_start: torch.Tensor,
    chain: Sequence[float],
    *,
    seed: int,
    region: WitnessRegion | None = None,
    diversity_tokens: Sequence[torch.Tensor] = (),
    min_hamming: float = 0.0,
    batch_size: int = 64,
    max_draws: int = 4096,
) -> tuple[torch.Tensor | None, dict[str, float | int | str]]:
    """Rejection sample one candidate from a root or witness-conditioned arm.

    The *first* eligible sequence in generator order is returned, avoiding any
    objective-dependent pre-query ranking.  Diversity and residual membership
    are both oracle-free filters.
    """
    base = model.base_model
    c = model.match_chain(chain)
    mu = predicted_mean_z_for_start(model, x_start, c)
    draws = 0
    accepted_region = 0
    accepted_diversity = 0
    round_id = 0
    while draws < int(max_draws):
        b = min(int(batch_size), int(max_draws)-draws)
        xr = x_start.expand(b, -1).contiguous()
        g = torch.Generator(device=model.device)
        g.manual_seed(int(seed) + 104729 * round_id)
        terminal = base.sample_chain(xr, c, generator=g)
        z = terminal_z(model, terminal)
        residual = z - mu[None, :]
        if region is None:
            in_region = np.ones(b, dtype=bool)
        else:
            in_region = region.contains(residual)
        for j in range(b):
            draws += 1
            if not bool(in_region[j]):
                continue
            accepted_region += 1
            row = terminal[j].detach().cpu()
            interior = row[1:-1].numpy()
            diverse = True
            for other in diversity_tokens:
                oi = torch.as_tensor(other).detach().cpu()[1:-1].numpy()
                if float(np.mean(interior != oi)) < float(min_hamming):
                    diverse = False; break
            if not diverse:
                continue
            accepted_diversity += 1
            return row, {
                "cheap_draws": int(draws),
                "region_accepts": int(accepted_region),
                "diversity_accepts": int(accepted_diversity),
                "empirical_region_acceptance": float(accepted_region / max(draws, 1)),
                "chain": chain_name(c),
            }
        round_id += 1
    return None, {
        "cheap_draws": int(draws),
        "region_accepts": int(accepted_region),
        "diversity_accepts": int(accepted_diversity),
        "empirical_region_acceptance": float(accepted_region / max(draws, 1)),
        "chain": chain_name(c),
    }


__all__ = [
    "ResidualMetric", "WitnessRegion", "terminal_z", "make_chain_start",
    "predicted_mean_z_for_start", "build_witness_region", "sample_candidate",
]
