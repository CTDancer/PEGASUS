"""Core model for Koopman-regularized scratch Uniform-LKF.

This revision deliberately avoids batch/EMA whitening in the Koopman forward
path.  The observable dictionary is put in one fixed coordinate system by
pointwise L2 normalization,

    phi(x,t) = sqrt(d) q(x,t) / ||q(x,t)||_2,

so uniform feature-scale collapse is impossible by parameterization and the
basis cannot change from batch to batch.  Dimensional collapse across samples
is controlled separately by a hinge on covariance effective rank.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lkf.uniform_lkf import UniformLKF

Tensor = torch.Tensor


@dataclass(frozen=True)
class KoopmanConfig:
    feature_dim: int = 64
    head_hidden_dim: int = 256
    # Retained only so older CLI/config files remain parseable.  They are not
    # used by the fixed-norm representation.
    whitening_momentum: float = 0.99
    whitening_eps: float = 1e-8
    whitening_eig_floor: float = 1e-3
    clock: str = "linear"
    operator_init_scale: float = 1e-3
    feature_source: str = "shared_pooled_plus_time"
    representation_norm: str = "fixed_l2"

    def __post_init__(self) -> None:
        if self.feature_dim < 2:
            raise ValueError("feature_dim must be >=2")
        if self.head_hidden_dim < self.feature_dim:
            raise ValueError("head_hidden_dim must be >= feature_dim")
        if self.clock != "linear":
            raise ValueError("v1 supports only the native linear Uniform-LKF clock")
        if self.feature_source not in {"shared_pooled_plus_time", "shared_pooled"}:
            raise ValueError("unsupported feature_source")
        if self.representation_norm != "fixed_l2":
            raise ValueError("this implementation supports only fixed_l2 normalization")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KoopmanConfig":
        valid = asdict(cls()).keys()
        return cls(**{k: value[k] for k in valid if k in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EMAWhitener(nn.Module):
    """Deprecated compatibility shim.

    Old checkpoints used an EMA whitener.  New K-LKF checkpoints do not attach
    this module to :class:`KoopmanRegularizedLKF`; the class remains importable
    only so external code does not fail at import time.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        raise RuntimeError(
            "EMAWhitener is deprecated. New K-LKF uses fixed per-sample L2 normalization."
        )


def fixed_l2_normalize(raw: Tensor) -> Tensor:
    """Map every sample to a fixed-radius sphere without a learned/batch basis.

    For d features, ||phi||_2=sqrt(d), hence per-sample feature RMS is exactly 1
    up to floating-point error.  A uniform raw scaling q->alpha q cancels
    exactly except at numerical underflow.
    """
    q = raw.float()
    if q.ndim != 2:
        raise ValueError("raw features must have shape [N,D]")
    d = q.shape[-1]
    norm = torch.linalg.vector_norm(q, ord=2, dim=-1, keepdim=True)
    norm = norm.clamp_min(torch.finfo(q.dtype).tiny)
    return q * (math.sqrt(float(d)) / norm)


def covariance_spectrum(features: Tensor) -> tuple[Tensor, Tensor]:
    """Return centered covariance and nonnegative eigenvalues."""
    x = torch.as_tensor(features).float()
    if x.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    if x.shape[0] < 2:
        raise ValueError("at least two samples are required")
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / float(x.shape[0])
    cov = 0.5 * (cov + cov.T)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    return cov, eig


def effective_rank_from_features(features: Tensor, *, detach: bool = True) -> Tensor:
    """Entropy effective rank exp(-sum p log p) of centered feature covariance."""
    x = features.detach() if detach else features
    _, eig = covariance_spectrum(x)
    total = eig.sum()
    tiny = torch.finfo(eig.dtype).tiny
    p = eig / total.clamp_min(tiny)
    # 0*log(0) is defined as zero.  clamp is only for evaluating log.
    entropy = -(p * p.clamp_min(tiny).log()).sum()
    rank = torch.exp(entropy)
    # Exactly zero covariance should be treated as complete collapse.
    rank = torch.where(total > tiny, rank, torch.zeros_like(rank))
    return rank


def effective_rank_hinge_loss(
    features: Tensor,
    *,
    target_rank: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Penalize only loss of dimensionality below a specified effective rank.

    L_rank = relu(log(r_target) - log(r_eff)).

    Unlike isotropy regularization, this does *not* force equal eigenvalues and
    exerts exactly zero pressure whenever the representation is sufficiently
    rich.  The target is supplied from the held-out initialization reference.

    Batches with fewer than two samples have an undefined centered covariance;
    the hinge contributes exactly zero there (still attached to the graph so DDP
    ranks never diverge on whether the regularizer ran).
    """
    target = float(target_rank)
    if target < 1.0:
        raise ValueError("target_rank must be >=1")
    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    n, d = int(features.shape[0]), int(features.shape[1])
    if n < 2:
        zero = features.float().sum() * 0.0
        metrics = {
            "effective_rank": zero.detach(),
            "rank_hinge_loss": zero.detach(),
            "rank_target_used": torch.tensor(1.0, device=features.device),
            "cov_mean_eigenvalue": zero.detach(),
            "cov_min_eigenvalue": zero.detach(),
            "cov_max_eigenvalue": zero.detach(),
        }
        return zero, metrics
    # Centered covariance from N samples has rank at most N-1.  Capping the
    # local training target makes the regularizer feasible even in Stage B,
    # where the per-GPU batch can be smaller than the held-out reference rank.
    feasible_target = min(target, float(d), float(max(n - 1, 1)))
    rank = effective_rank_from_features(features, detach=False)
    tiny = torch.finfo(rank.dtype).tiny
    shortfall = F.relu(
        torch.tensor(math.log(feasible_target), device=rank.device, dtype=rank.dtype)
        - rank.clamp_min(tiny).log()
    )
    # Linear hinge: zero above the target, but non-vanishing restoring gradient
    # immediately after rank crosses below it. A squared hinge was too weak near
    # the boundary for this use case.
    loss = shortfall
    with torch.no_grad():
        _, eig = covariance_spectrum(features.detach())
        metrics = {
            "effective_rank": rank.detach(),
            "rank_hinge_loss": loss.detach(),
            "rank_target_used": torch.tensor(feasible_target, device=rank.device),
            "cov_mean_eigenvalue": eig.mean().detach(),
            "cov_min_eigenvalue": eig.min().detach(),
            "cov_max_eigenvalue": eig.max().detach(),
        }
    return loss, metrics


class KoopmanRegularizedLKF(nn.Module):
    """Auxiliary Koopman observable model attached to a UniformLKF generator."""

    def __init__(self, lkf: UniformLKF, config: KoopmanConfig):
        super().__init__()
        self.lkf = lkf
        self.config = config
        d_in = int(lkf.model_dim)
        d_hidden = int(config.head_hidden_dim)
        d_out = int(config.feature_dim)
        self.observable_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )
        for module in self.observable_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.operator_generator = nn.Parameter(torch.empty(d_out, d_out, dtype=torch.float32))
        nn.init.normal_(
            self.operator_generator,
            mean=0.0,
            std=float(config.operator_init_scale),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    def _batch_time(self, value: float | Tensor, x: Tensor, *, name: str) -> Tensor:
        return self.lkf._batch_time(
            value,
            batch_size=x.shape[0],
            device=x.device,
            dtype=self.lkf.pos_embedder.dtype,
            name=name,
        )

    def lkf_hidden(self, x: Tensor, time: float | Tensor) -> Tensor:
        self.lkf.validate_peptide_tokens(x, name="koopman_x")
        t = self._batch_time(time, x, name="koopman_time")
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError("Koopman feature time requires 0<=t<=1")
        condition = self.lkf.time_embedder(t)
        hidden = self.lkf.token_embedder(x) + self.lkf.pos_embedder[:, : x.shape[1]]
        for block in self.lkf.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:, 1:-1].mean(dim=1)
        if self.config.feature_source == "shared_pooled_plus_time":
            pooled = pooled + condition
        return pooled

    def raw_features(self, x: Tensor, time: float | Tensor) -> Tensor:
        return self.observable_head(self.lkf_hidden(x, time)).float()

    def features(
        self,
        x: Tensor,
        time: float | Tensor,
        *,
        update_whitener: bool = False,
    ) -> Tensor:
        # update_whitener is accepted for backward call-site compatibility; no
        # batch-dependent state is updated in the fixed-coordinate formulation.
        del update_whitener
        return fixed_l2_normalize(self.raw_features(x, time))

    def tau(self, time: Tensor | float) -> Tensor:
        t = torch.as_tensor(time, dtype=torch.float32, device=self.operator_generator.device)
        if self.config.clock == "linear":
            return t
        raise RuntimeError("unsupported Koopman clock")

    def operator(self, s: Tensor | float, t: Tensor | float) -> Tensor:
        s_t, t_t = self.tau(s), self.tau(t)
        if torch.any(t_t < s_t):
            raise ValueError("Koopman operator requires s<=t")
        delta = t_t - s_t
        L = self.operator_generator.float()
        if delta.ndim == 0:
            return torch.matrix_exp(delta * L)
        return torch.matrix_exp(delta[..., None, None] * L)

    def predict_features(self, z_s: Tensor, s: Tensor | float, t: Tensor | float) -> Tensor:
        A = self.operator(s, t)
        z = z_s.float()
        if A.ndim == 2:
            return z @ A.T
        if A.ndim != 3 or A.shape[0] != z.shape[0]:
            raise ValueError("batched operator must match feature batch")
        return torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)

    def closure_loss(
        self,
        x_s: Tensor,
        x_t_samples: Tensor,
        s: Tensor | float,
        t: Tensor | float,
        *,
        update_whitener: bool = False,
        rank_regularization_weight: float = 0.0,
        rank_target: float = 1.0,
        # Backward-compatible alias from the previous patch.
        covariance_isotropy_weight: float | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del update_whitener
        if covariance_isotropy_weight is not None:
            # Old call sites can still run, but the value now weights the rank
            # hinge rather than an isotropy objective.
            rank_regularization_weight = float(covariance_isotropy_weight)
        if x_t_samples.ndim != 3 or x_t_samples.shape[1:] != x_s.shape:
            raise ValueError("x_t_samples must have shape [M,B,L] matching x_s")
        m, b, length = x_t_samples.shape
        z_s = self.features(x_s, s)
        flat_t = x_t_samples.reshape(m * b, length)
        if torch.as_tensor(t).ndim == 0:
            t_flat: Tensor | float = t
        else:
            t_batch = self._batch_time(t, x_s, name="t")
            t_flat = t_batch.repeat(m)
        z_t_flat = self.features(flat_t, t_flat)
        z_t = z_t_flat.reshape(m, b, self.feature_dim)
        z_bar_t = z_t.mean(dim=0)

        pred = self.predict_features(z_s, s, t)
        residual = z_bar_t - pred
        closure_loss = residual.square().mean()

        weight = float(rank_regularization_weight)
        if weight < 0:
            raise ValueError("rank_regularization_weight must be nonnegative")
        src_rank_loss, src_rank_metrics = effective_rank_hinge_loss(
            z_s, target_rank=float(rank_target)
        )
        tgt_rank_loss, tgt_rank_metrics = effective_rank_hinge_loss(
            z_t_flat, target_rank=float(rank_target)
        )
        rank_loss = 0.5 * (src_rank_loss + tgt_rank_loss)
        regularizer_loss = closure_loss + weight * rank_loss

        with torch.no_grad():
            target_rms = z_bar_t.square().mean().sqrt()
            residual_rms = residual.square().mean().sqrt()
            cosine = F.cosine_similarity(pred, z_bar_t, dim=-1).mean()
            metrics = {
                "koopman_loss": closure_loss.detach(),
                "koopman_regularizer_loss": regularizer_loss.detach(),
                "rank_hinge_loss": rank_loss.detach(),
                "rank_regularization_weight": torch.tensor(weight, device=closure_loss.device),
                "rank_target": torch.tensor(float(rank_target), device=closure_loss.device),
                "source_effective_rank": src_rank_metrics["effective_rank"],
                "target_effective_rank": tgt_rank_metrics["effective_rank"],
                "source_rank_hinge_loss": src_rank_metrics["rank_hinge_loss"],
                "target_rank_hinge_loss": tgt_rank_metrics["rank_hinge_loss"],
                "source_cov_min_eigenvalue": src_rank_metrics["cov_min_eigenvalue"],
                "source_cov_max_eigenvalue": src_rank_metrics["cov_max_eigenvalue"],
                "target_cov_min_eigenvalue": tgt_rank_metrics["cov_min_eigenvalue"],
                "target_cov_max_eigenvalue": tgt_rank_metrics["cov_max_eigenvalue"],
                "closure_rmse": residual_rms.detach(),
                "closure_relative_rmse": (residual_rms / target_rms.clamp_min(1e-8)).detach(),
                "closure_cosine": cosine.detach(),
                "source_feature_rms": z_s.square().mean().sqrt().detach(),
                "target_individual_feature_rms": z_t_flat.square().mean().sqrt().detach(),
                "target_mean_feature_rms": target_rms.detach(),
            }
        return regularizer_loss, metrics

    @torch.no_grad()
    def semigroup_error(self, s: float, u: float, t: float) -> float:
        if not (0.0 <= s <= u <= t <= 1.0):
            raise ValueError("requires 0<=s<=u<=t<=1")
        direct = self.operator(s, t)
        composed = self.operator(u, t) @ self.operator(s, u)
        denom = torch.linalg.norm(direct).clamp_min(1e-12)
        return float((torch.linalg.norm(composed - direct) / denom).item())

    def auxiliary_state_dict(self) -> dict[str, Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lkf.")}

    def load_auxiliary_state_dict(self, state: Mapping[str, Tensor], *, strict: bool = True) -> None:
        current = self.state_dict()
        merged = dict(current)
        for key, value in state.items():
            if key.startswith("lkf."):
                raise ValueError("auxiliary Koopman state must not contain lkf.* keys")
            if key not in merged:
                if strict:
                    raise KeyError(f"unexpected Koopman state key: {key}")
                continue
            merged[key] = value
        expected_aux = {k for k in current if not k.startswith("lkf.")}
        missing = expected_aux.difference(state.keys())
        if strict and missing:
            raise KeyError(f"missing Koopman state keys: {sorted(missing)[:8]}")
        self.load_state_dict(merged, strict=True)


def configure_lkf_trainability(
    lkf: UniformLKF,
    *,
    stage: str,
    unfreeze_shared_blocks: int = 2,
    unfreeze_downstream: bool = True,
) -> list[str]:
    stage = str(stage).upper()
    for parameter in lkf.parameters():
        parameter.requires_grad_(False)
    if stage == "A":
        lkf.eval()
        return []
    if stage != "B":
        raise ValueError("stage must be A or B")
    k = int(unfreeze_shared_blocks)
    if k < 0 or k > len(lkf.shared_blocks):
        raise ValueError("unfreeze_shared_blocks lies outside available shared blocks")
    modules: list[tuple[str, nn.Module]] = []
    if k > 0:
        start = len(lkf.shared_blocks) - k
        for i in range(start, len(lkf.shared_blocks)):
            modules.append((f"shared_blocks.{i}", lkf.shared_blocks[i]))
    if unfreeze_downstream:
        modules.extend(
            [
                ("latent_blocks", lkf.latent_blocks),
                ("latent_embedder", lkf.latent_embedder),
                ("router", lkf.router),
                ("final_norm", lkf.final_norm),
                ("lm_head", lkf.lm_head),
            ]
        )
    names: list[str] = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters(prefix=prefix):
            parameter.requires_grad_(True)
            names.append(name)
    lkf.train()
    return names


__all__ = [
    "KoopmanConfig",
    "EMAWhitener",
    "fixed_l2_normalize",
    "covariance_spectrum",
    "effective_rank_from_features",
    "effective_rank_hinge_loss",
    "KoopmanRegularizedLKF",
    "configure_lkf_trainability",
]
