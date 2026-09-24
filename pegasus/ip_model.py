"""Information-preserving Koopman lift for Uniform-LKF.

The Koopman state is split into two blocks

    phi(x,t) = [ r(x,t) ; psi(x,t) ],

where r is a *fixed* Gaussian random projection of a frozen reference LKF
shared representation and psi is a learned nonlinear lift.  This preserves the
linear objective information already present in the LKF hidden state while
allowing extra coordinates to make the dynamics approximately Koopman-linear.

The anchor encoder and projection are frozen in both Stage A and Stage B.  Thus
Stage-B generator fine-tuning cannot erase the objective-rich anchor space.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lkf.uniform_lkf import UniformLKF
from .model import effective_rank_hinge_loss, fixed_l2_normalize, configure_lkf_trainability

Tensor = torch.Tensor


@dataclass(frozen=True)
class InformationPreservingKoopmanConfig:
    anchor_dim: int = 64
    lift_dim: int = 64
    lift_hidden_dim: int = 256
    anchor_seed: int = 60042
    clock: str = "linear"
    operator_init_scale: float = 1e-3
    feature_source: str = "shared_pooled_plus_time"
    lift_norm: str = "fixed_l2"
    anchor_projection: str = "gaussian_fixed"
    anchor_closure_weight: float = 1.0
    lift_closure_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.anchor_dim < 1:
            raise ValueError("anchor_dim must be >=1")
        if self.lift_dim < 1:
            raise ValueError("lift_dim must be >=1")
        if self.lift_hidden_dim < self.lift_dim:
            raise ValueError("lift_hidden_dim must be >= lift_dim")
        if self.clock != "linear":
            raise ValueError("v1 supports only the native linear time clock")
        if self.feature_source not in {"shared_pooled_plus_time", "shared_pooled"}:
            raise ValueError("unsupported feature_source")
        if self.lift_norm != "fixed_l2":
            raise ValueError("lift_norm must be fixed_l2")
        if self.anchor_projection != "gaussian_fixed":
            raise ValueError("anchor_projection must be gaussian_fixed")
        if self.anchor_closure_weight <= 0 or self.lift_closure_weight <= 0:
            raise ValueError("closure block weights must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.anchor_dim + self.lift_dim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InformationPreservingKoopmanConfig":
        valid = asdict(cls()).keys()
        return cls(**{k: value[k] for k in valid if k in value})


class FrozenSharedLKFEncoder(nn.Module):
    """Frozen copy of the objective-rich LKF shared representation.

    Keeping a separate immutable encoder is deliberate.  In Stage B the active
    generator may change, but the anchor coordinates r(x,t) remain the same
    objective-free feature map that passed the linear-probe test in Stage A.
    """

    def __init__(self, lkf: UniformLKF, *, feature_source: str):
        super().__init__()
        self.model_dim = int(lkf.model_dim)
        self.seq_len = int(lkf.seq_len)
        self.feature_source = str(feature_source)
        self.token_embedder = copy.deepcopy(lkf.token_embedder)
        self.pos_embedder = nn.Parameter(lkf.pos_embedder.detach().clone(), requires_grad=False)
        self.time_embedder = copy.deepcopy(lkf.time_embedder)
        self.shared_blocks = copy.deepcopy(lkf.shared_blocks)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # keep immutable encoder deterministic
        super().train(False)
        return self

    def _batch_time(self, value: float | Tensor, x: Tensor) -> Tensor:
        t = torch.as_tensor(value, device=x.device, dtype=self.pos_embedder.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == x.shape[0]:
            pass
        else:
            raise ValueError("time must be scalar or have shape [batch]")
        return t

    @torch.no_grad()
    def forward(self, x: Tensor, time: float | Tensor) -> Tensor:
        t = self._batch_time(time, x)
        condition = self.time_embedder(t)
        hidden = self.token_embedder(x) + self.pos_embedder[:, : x.shape[1]]
        for block in self.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:, 1:-1].mean(dim=1)
        if self.feature_source == "shared_pooled_plus_time":
            pooled = pooled + condition
        return pooled.float()


class InformationPreservingKoopmanLKF(nn.Module):
    """Uniform-LKF with a fixed information anchor plus learned Koopman lift."""

    def __init__(self, lkf: UniformLKF, config: InformationPreservingKoopmanConfig):
        super().__init__()
        self.lkf = lkf
        self.config = config
        d_in = int(lkf.model_dim)

        # Immutable objective-rich reference representation.
        self.anchor_encoder = FrozenSharedLKFEncoder(lkf, feature_source=config.feature_source)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(config.anchor_seed))
        projection = torch.randn(d_in, int(config.anchor_dim), generator=gen, dtype=torch.float32)
        projection = projection / math.sqrt(float(config.anchor_dim))
        self.register_buffer("anchor_projection", projection, persistent=True)

        self.lift_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, int(config.lift_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(config.lift_hidden_dim), int(config.lift_dim)),
        )
        for module in self.lift_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        d = int(config.feature_dim)
        self.operator_generator = nn.Parameter(torch.empty(d, d, dtype=torch.float32))
        nn.init.normal_(self.operator_generator, mean=0.0, std=float(config.operator_init_scale))

    @property
    def anchor_dim(self) -> int:
        return int(self.config.anchor_dim)

    @property
    def lift_dim(self) -> int:
        return int(self.config.lift_dim)

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

    def active_lkf_hidden(self, x: Tensor, time: float | Tensor) -> Tensor:
        self.lkf.validate_peptide_tokens(x, name="ip_koopman_x")
        t = self._batch_time(time, x, name="ip_koopman_time")
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError("feature time requires 0<=t<=1")
        condition = self.lkf.time_embedder(t)
        hidden = self.lkf.token_embedder(x) + self.lkf.pos_embedder[:, : x.shape[1]]
        for block in self.lkf.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:, 1:-1].mean(dim=1)
        if self.config.feature_source == "shared_pooled_plus_time":
            pooled = pooled + condition
        return pooled.float()

    # Compatibility with existing probe code terminology.
    def lkf_hidden(self, x: Tensor, time: float | Tensor) -> Tensor:
        return self.active_lkf_hidden(x, time)

    @torch.no_grad()
    def anchor_hidden(self, x: Tensor, time: float | Tensor) -> Tensor:
        self.lkf.validate_peptide_tokens(x, name="ip_anchor_x")
        return self.anchor_encoder(x, time)

    @torch.no_grad()
    def anchor_features(self, x: Tensor, time: float | Tensor) -> Tensor:
        return self.anchor_hidden(x, time) @ self.anchor_projection

    def raw_lift_features(self, x: Tensor, time: float | Tensor) -> Tensor:
        return self.lift_head(self.active_lkf_hidden(x, time)).float()

    def lift_features(self, x: Tensor, time: float | Tensor) -> Tensor:
        return fixed_l2_normalize(self.raw_lift_features(x, time))

    def features(self, x: Tensor, time: float | Tensor, **_: Any) -> Tensor:
        anchor = self.anchor_features(x, time)
        lift = self.lift_features(x, time)
        return torch.cat([anchor, lift], dim=-1)

    def split_features(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z[..., : self.anchor_dim], z[..., self.anchor_dim :]

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
        rank_regularization_weight: float = 1.0,
        rank_target: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
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
        z_bar = z_t.mean(dim=0)
        pred = self.predict_features(z_s, s, t)

        anchor_target, lift_target = self.split_features(z_bar)
        anchor_pred, lift_pred = self.split_features(pred)
        anchor_res = anchor_target - anchor_pred
        lift_res = lift_target - lift_pred

        # Block-normalized closure prevents the arbitrary scale of the fixed
        # linear anchor from dominating the unit-radius learned lift (or vice versa).
        eps = 1e-8
        anchor_target_mse = anchor_target.detach().square().mean().clamp_min(eps)
        lift_target_mse = lift_target.detach().square().mean().clamp_min(eps)
        anchor_loss = anchor_res.square().mean() / anchor_target_mse
        lift_loss = lift_res.square().mean() / lift_target_mse
        wa = float(self.config.anchor_closure_weight)
        wl = float(self.config.lift_closure_weight)
        closure = (wa * anchor_loss + wl * lift_loss) / (wa + wl)

        # Only the learned lift can collapse.  The anchor is fixed by construction.
        z_s_lift = self.split_features(z_s)[1]
        z_t_lift = self.split_features(z_t_flat)[1]
        src_rank_loss, src_rank = effective_rank_hinge_loss(z_s_lift, target_rank=rank_target)
        tgt_rank_loss, tgt_rank = effective_rank_hinge_loss(z_t_lift, target_rank=rank_target)
        rank_loss = 0.5 * (src_rank_loss + tgt_rank_loss)
        beta = float(rank_regularization_weight)
        if beta < 0:
            raise ValueError("rank_regularization_weight must be nonnegative")
        regularizer = closure + beta * rank_loss

        with torch.no_grad():
            full_res = z_bar - pred
            full_target_rms = z_bar.square().mean().sqrt()
            full_rmse = full_res.square().mean().sqrt()
            anchor_target_rms = anchor_target.square().mean().sqrt()
            lift_target_rms = lift_target.square().mean().sqrt()
            anchor_rmse = anchor_res.square().mean().sqrt()
            lift_rmse = lift_res.square().mean().sqrt()
            metrics = {
                "koopman_loss": closure.detach(),
                "koopman_regularizer_loss": regularizer.detach(),
                "rank_hinge_loss": rank_loss.detach(),
                "source_lift_effective_rank": src_rank["effective_rank"],
                "target_lift_effective_rank": tgt_rank["effective_rank"],
                "closure_rmse": full_rmse.detach(),
                "closure_relative_rmse": (full_rmse / full_target_rms.clamp_min(eps)).detach(),
                "closure_cosine": F.cosine_similarity(pred, z_bar, dim=-1).mean().detach(),
                "anchor_closure_loss": anchor_loss.detach(),
                "anchor_closure_rmse": anchor_rmse.detach(),
                "anchor_closure_relative_rmse": (anchor_rmse / anchor_target_rms.clamp_min(eps)).detach(),
                "anchor_closure_cosine": F.cosine_similarity(anchor_pred, anchor_target, dim=-1).mean().detach(),
                "lift_closure_loss": lift_loss.detach(),
                "lift_closure_rmse": lift_rmse.detach(),
                "lift_closure_relative_rmse": (lift_rmse / lift_target_rms.clamp_min(eps)).detach(),
                "lift_closure_cosine": F.cosine_similarity(lift_pred, lift_target, dim=-1).mean().detach(),
                "anchor_target_rms": anchor_target_rms.detach(),
                "lift_target_rms": lift_target_rms.detach(),
                "lift_individual_rms": z_t_lift.square().mean().sqrt().detach(),
                "full_target_rms": full_target_rms.detach(),
            }
        return regularizer, metrics

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
                raise ValueError("auxiliary state must not contain lkf.* keys")
            if key not in merged:
                if strict:
                    raise KeyError(f"unexpected IP-Koopman state key: {key}")
                continue
            merged[key] = value
        expected = {k for k in current if not k.startswith("lkf.")}
        missing = expected.difference(state.keys())
        if strict and missing:
            raise KeyError(f"missing IP-Koopman state keys: {sorted(missing)[:8]}")
        self.load_state_dict(merged, strict=True)
        self.anchor_encoder.eval()


__all__ = [
    "InformationPreservingKoopmanConfig",
    "FrozenSharedLKFEncoder",
    "InformationPreservingKoopmanLKF",
    "configure_lkf_trainability",
]
