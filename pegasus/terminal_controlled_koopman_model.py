"""Direct-terminal information-preserving controlled Koopman flow map.

This module implements the consolidated revision motivated by the completed
Stage-A diagnostics.  It deliberately removes two falsified approximations:

1. no finite-dimensional exact semigroup is imposed on the fixed anchor;
2. multi-pulse terminal response is never constructed as A @ J.

For each predeclared finite-time chain C=(t0,...,tH=1), the scientific model is

    E[r(X_1^v) | X_t0=x, C]
        = A_C r(x) + sum_k M_{C,k} v_k + O(||v||^2),

where A_C is a *direct* chain-specific conditional-mean regression and each
M_{C,k} is the direct derivative of the terminal controlled Koopman expectation
with respect to the physical Gaussian innovation mean at pulse k.  Under the
Gaussian base-noise parameterization,

    M_{C,k} = E[(r(X_1)-b(X_t0)) xi_k^T],

so A_C and M_C can be estimated objective-free from hard chain rollouts.

Stage-B gradients use a straight-through Gumbel-softmax chain surrogate whose
forward pass is exactly the hard categorical chain for fixed Gaussian noise.
All checkpoint gates and scientific diagnostics remain hard-sampling based.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controlled_koopman_model import ControlledKoopmanFlowMap
from .explicit_innovation import (
    normal_to_gumbel,
    sample_transition_with_gaussian_innovation,
    transition_innovation_layout,
)
from .ip_model import FrozenSharedLKFEncoder
from .lkf.uniform_lkf import UniformLKF, UniformLKFProcess

Tensor = torch.Tensor


def _as_chain(value: Iterable[float]) -> tuple[float, ...]:
    out = tuple(float(x) for x in value)
    if len(out) < 2 or not math.isclose(out[-1], 1.0, abs_tol=1e-8):
        raise ValueError("every terminal chain must contain >=2 times and end at 1")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError("terminal chain times must be strictly increasing")
    if out[0] < 0 or out[-1] > 1:
        raise ValueError("terminal chain must lie in [0,1]")
    return out


def _chain_key(chain: Sequence[float]) -> str:
    def f(x: float) -> str:
        return f"{float(x):.6f}".replace("-", "m").replace(".", "p")
    return "c_" + "_".join(f(x) for x in chain)


def _pulse_key(chain: Sequence[float], pulse: int) -> str:
    return f"{_chain_key(chain)}__p{int(pulse)}"


@dataclass(frozen=True)
class TerminalControlledKoopmanConfig:
    anchor_dim: int = 64
    anchor_seed: int = 60042
    feature_source: str = "shared_pooled_plus_time"
    chains: tuple[tuple[float, ...], ...] = field(
        default_factory=lambda: (
            (0.0, 1.0),
            (0.25, 1.0),
            (0.5, 1.0),
            (0.75, 1.0),
            (0.0, 0.25, 0.5, 1.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.75, 1.0),
            (0.25, 0.5, 0.75, 1.0),
            (0.25, 0.75, 1.0),
            (0.5, 0.75, 1.0),
        )
    )
    information_eigen_floor_relative: float = 1e-5
    information_epsilon: float = 1e-6
    soft_temperature: float = 0.5
    direct_operator_ridge: float = 1e-4

    def __post_init__(self) -> None:
        if self.anchor_dim < 1:
            raise ValueError("anchor_dim must be positive")
        if self.feature_source not in {"shared_pooled_plus_time", "shared_pooled"}:
            raise ValueError("unsupported feature_source")
        if self.soft_temperature <= 0 or self.direct_operator_ridge < 0:
            raise ValueError("invalid soft_temperature/direct_operator_ridge")
        normalized = tuple(_as_chain(c) for c in self.chains)
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate terminal chains")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["chains"] = [list(c) for c in self.chains]
        return out

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TerminalControlledKoopmanConfig":
        valid = set(asdict(cls()).keys())
        kwargs = {k: value[k] for k in valid if k in value}
        if "chains" in kwargs:
            kwargs["chains"] = tuple(_as_chain(c) for c in kwargs["chains"])
        return cls(**kwargs)


class TerminalControlledKoopmanFlowMap(nn.Module):
    """LKF + immutable information anchor + direct chain terminal geometry."""

    def __init__(
        self,
        lkf: UniformLKF,
        config: TerminalControlledKoopmanConfig,
        *,
        anchor_encoder: FrozenSharedLKFEncoder | None = None,
        anchor_projection: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lkf = lkf
        self.config = config
        if anchor_encoder is None:
            anchor_encoder = FrozenSharedLKFEncoder(lkf, feature_source=config.feature_source)
        self.anchor_encoder = copy.deepcopy(anchor_encoder)
        for p in self.anchor_encoder.parameters():
            p.requires_grad_(False)
        self.anchor_encoder.eval()

        d_in = int(lkf.model_dim)
        if anchor_projection is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(config.anchor_seed))
            projection = torch.randn(d_in, int(config.anchor_dim), generator=gen) / math.sqrt(float(config.anchor_dim))
        else:
            projection = torch.as_tensor(anchor_projection, dtype=torch.float32).detach().clone()
        if tuple(projection.shape) != (d_in, int(config.anchor_dim)):
            raise ValueError("anchor_projection has incompatible shape")
        self.register_buffer("anchor_projection", projection.float(), persistent=True)

        d = int(config.anchor_dim)
        self._chains = tuple(_as_chain(c) for c in config.chains)
        self._chain_keys = tuple(_chain_key(c) for c in self._chains)
        self._pulse_keys: list[str] = []
        self._pulse_meta: list[tuple[int, int]] = []

        # Direct chain mean operators and direct terminal pulse derivatives are
        # estimators, not trainable actuators.  ParameterDict is used only for
        # convenient state_dict/checkpoint handling; requires_grad stays False.
        ops: dict[str, nn.Parameter] = {}
        for c in self._chains:
            ops[_chain_key(c)] = nn.Parameter(torch.eye(d, dtype=torch.float32), requires_grad=False)
        self.chain_mean_operators = nn.ParameterDict(ops)

        max_layout = transition_innovation_layout(UniformLKFProcess(lkf), int(lkf.seq_len))
        self.max_innovation_dim = int(max_layout.total_dim)
        responses: dict[str, nn.Parameter] = {}
        for ci, c in enumerate(self._chains):
            for pi in range(len(c) - 1):
                key = _pulse_key(c, pi)
                responses[key] = nn.Parameter(
                    torch.zeros(d, self.max_innovation_dim, dtype=torch.float32),
                    requires_grad=False,
                )
                self._pulse_keys.append(key)
                self._pulse_meta.append((ci, pi))
        self.terminal_responses = nn.ParameterDict(responses)

        self.register_buffer("chain_mean_counts", torch.zeros(len(self._chains), dtype=torch.float64), persistent=True)
        self.register_buffer(
            "terminal_response_counts",
            torch.zeros(len(self._pulse_keys), self.max_innovation_dim, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer("geometry_calibrated", torch.tensor(False), persistent=True)

        # Replaced on migration from the validated v4 checkpoint.
        self.register_buffer("information_transform", torch.eye(d), persistent=True)
        self.register_buffer("information_covariance", torch.eye(d), persistent=True)
        self.register_buffer("information_eigenvalues", torch.ones(d), persistent=True)
        self.register_buffer("information_rank", torch.tensor(d, dtype=torch.long), persistent=True)
        self.register_buffer("information_metric_initialized", torch.tensor(False), persistent=True)

    @classmethod
    def from_controlled_stage_a(
        cls,
        old: ControlledKoopmanFlowMap,
        config: TerminalControlledKoopmanConfig,
    ) -> "TerminalControlledKoopmanFlowMap":
        if int(config.anchor_dim) != int(old.anchor_dim):
            raise ValueError("terminal anchor_dim must match validated Stage-A anchor")
        model = cls(
            old.lkf,
            config,
            anchor_encoder=old.anchor_encoder,
            anchor_projection=old.anchor_projection,
        )
        # Preserve only validated information geometry.  Old compositional A/J
        # are intentionally discarded.
        model.information_transform = old.information_transform.detach().clone()
        model.information_covariance = old.information_covariance.detach().clone()
        model.information_eigenvalues = old.information_eigenvalues.detach().clone()
        model.information_rank = old.information_rank.detach().clone()
        model.information_metric_initialized = old.information_metric_initialized.detach().clone()
        return model

    @property
    def anchor_dim(self) -> int:
        return int(self.config.anchor_dim)

    @property
    def chains(self) -> tuple[tuple[float, ...], ...]:
        return self._chains

    @property
    def process(self) -> UniformLKFProcess:
        return UniformLKFProcess(self.lkf)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor_encoder.eval()
        return self

    @torch.no_grad()
    def anchor_features(self, x: Tensor, time: float | Tensor) -> Tensor:
        self.lkf.validate_peptide_tokens(x, name="terminal_anchor_x")
        hidden = self.anchor_encoder(x, time)
        return hidden.float() @ self.anchor_projection.float()

    def information_normalize(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.anchor_dim:
            raise ValueError("last dimension must equal anchor_dim")
        return value.float() @ self.information_transform.float().T

    def match_chain(self, chain: Sequence[float], tol: float = 1e-6) -> tuple[float, ...]:
        target = tuple(float(x) for x in chain)
        for c in self._chains:
            if len(c) == len(target) and all(abs(a - b) <= tol for a, b in zip(c, target)):
                return c
        raise KeyError(f"unconfigured terminal chain {target}; configured={self._chains}")

    def chain_index(self, chain: Sequence[float]) -> int:
        return self._chains.index(self.match_chain(chain))

    def mean_operator(self, chain: Sequence[float]) -> Tensor:
        c = self.match_chain(chain)
        return self.chain_mean_operators[_chain_key(c)].float()

    def predict_uncontrolled_mean(self, r_start: Tensor, chain: Sequence[float]) -> Tensor:
        A = self.mean_operator(chain)
        return r_start.float() @ A.T

    def pulse_response_index(self, chain: Sequence[float], pulse: int) -> int:
        c = self.match_chain(chain)
        key = _pulse_key(c, int(pulse))
        return self._pulse_keys.index(key)

    def terminal_response_block(
        self,
        chain: Sequence[float],
        pulse: int,
        *,
        token_length: int | None = None,
    ) -> Tensor:
        c = self.match_chain(chain)
        pi = int(pulse)
        if pi < 0 or pi >= len(c) - 1:
            raise IndexError("pulse index out of range")
        M = self.terminal_responses[_pulse_key(c, pi)].float()
        if token_length is None:
            return M
        layout = transition_innovation_layout(self.process, int(token_length))
        return M[:, : int(layout.total_dim)]

    def aggregate_response(self, chain: Sequence[float], *, token_length: int) -> Tensor:
        c = self.match_chain(chain)
        blocks = [self.terminal_response_block(c, k, token_length=token_length) for k in range(len(c)-1)]
        return torch.cat(blocks, dim=1)

    def gramian(self, chain: Sequence[float], *, token_length: int, normalized: bool = False) -> Tensor:
        M = self.aggregate_response(chain, token_length=token_length)
        G = M @ M.T
        if normalized:
            T = self.information_transform.float()
            G = T @ G @ T.T
        return 0.5 * (G + G.T)

    @torch.no_grad()
    def sample_transition(self, x_s: Tensor, s: float, t: float, **kwargs):
        return sample_transition_with_gaussian_innovation(self.process, x_s, float(s), float(t), **kwargs)

    @torch.no_grad()
    def sample_chain(
        self,
        x_start: Tensor,
        chain: Sequence[float],
        *,
        xi_blocks: Sequence[Tensor] | None = None,
        mean_shifts: Sequence[Tensor | None] | None = None,
        generator: torch.Generator | None = None,
        return_innovations: bool = False,
    ):
        c = self.match_chain(chain)
        h = len(c) - 1
        if xi_blocks is not None and len(xi_blocks) != h:
            raise ValueError("xi_blocks length must equal number of pulses")
        if mean_shifts is None:
            mean_shifts = [None] * h
        if len(mean_shifts) != h:
            raise ValueError("mean_shifts length must equal number of pulses")
        current = x_start
        used: list[Tensor] = []
        for k, (s, t) in enumerate(zip(c[:-1], c[1:])):
            xi = None if xi_blocks is None else xi_blocks[k]
            current, xi_used = sample_transition_with_gaussian_innovation(
                self.process,
                current,
                s,
                t,
                xi=xi,
                mean_shift=mean_shifts[k],
                generator=generator,
            )
            used.append(xi_used)
        return (current, used) if return_innovations else current

    # ------------------------ differentiable Stage-B surrogate ------------------------
    def _residue_onehot(self, tokens: Tensor) -> Tensor:
        idx = self.lkf.aa_index_by_token[tokens[:, 1:-1]]
        if torch.any(idx < 0):
            raise ValueError("non-canonical residue in chain state")
        return F.one_hot(idx, num_classes=len(self.lkf.config.aa_token_ids)).float()

    def _lkf_logits_from_st_residue_probs(self, residue_probs: Tensor, time: float) -> tuple[Tensor, Tensor]:
        """Reproduce UniformLKF.forward using ST residue embeddings.

        Forward values equal the hard-token network when residue_probs has a
        hard one-hot forward value; gradients flow through residue_probs.
        """
        b, residues, aa_count = residue_probs.shape
        aa_ids = torch.tensor(self.lkf.config.aa_token_ids, device=residue_probs.device, dtype=torch.long)
        if aa_count != int(aa_ids.numel()):
            raise ValueError("AA dimension mismatch")
        emb = self.lkf.token_embedder.weight
        aa_emb = emb.index_select(0, aa_ids).float()
        interior = residue_probs.float() @ aa_emb
        cls = emb[int(self.lkf.config.cls_token_id)].float().view(1,1,-1).expand(b,1,-1)
        eos = emb[int(self.lkf.config.eos_token_id)].float().view(1,1,-1).expand(b,1,-1)
        hidden = torch.cat([cls, interior, eos], dim=1)
        t = torch.full((b,), float(time), device=residue_probs.device, dtype=self.lkf.pos_embedder.dtype)
        condition = self.lkf.time_embedder(t)
        hidden = hidden + self.lkf.pos_embedder[:, : residues + 2]
        for block in self.lkf.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:, 1:-1].mean(dim=1) + condition
        router_logits = self.lkf.router(pooled)
        m = self.lkf.latent_components
        latent_ids = torch.arange(m, device=residue_probs.device)
        latent_condition = condition[:, None, :] + self.lkf.latent_embedder(latent_ids)[None, :, :]
        branched = hidden[:, None, :, :].expand(b, m, residues + 2, self.lkf.model_dim)
        branched = branched.reshape(b * m, residues + 2, self.lkf.model_dim)
        latent_condition = latent_condition.reshape(b * m, self.lkf.model_dim)
        for block in self.lkf.latent_blocks:
            branched = block(branched, latent_condition)
        token_logits = self.lkf.lm_head(self.lkf.final_norm(branched))
        token_logits = token_logits.reshape(b, m, residues + 2, self.lkf.vocab_size)
        return router_logits, token_logits

    def _transition_log_probs_from_st_state(
        self,
        residue_probs_st: Tensor,
        s: float,
        t: float,
    ) -> tuple[Tensor, Tensor]:
        b, residues, aa_count = residue_probs_st.shape
        router_logits, clean_token_logits = self._lkf_logits_from_st_residue_probs(residue_probs_st, float(s))
        log_router = F.log_softmax(router_logits.float(), dim=-1)
        clean_logits = clean_token_logits.float()[:, :, 1:-1, :]
        clean_logits = clean_logits.masked_fill((~self.lkf.aa_output_mask).view(1,1,1,-1), float("-inf"))
        clean_full = torch.softmax(clean_logits, dim=-1)
        aa = torch.tensor(self.lkf.config.aa_token_ids, device=residue_probs_st.device, dtype=torch.long)
        p_clean = clean_full.index_select(-1, aa)
        sb = torch.full((b,), float(s), device=residue_probs_st.device, dtype=self.lkf.pos_embedder.dtype)
        tb = torch.full((b,), float(t), device=residue_probs_st.device, dtype=self.lkf.pos_embedder.dtype)
        A = float(aa_count)
        onehot_a = residue_probs_st
        denom_c = (1.0 - sb[:,None,None]) / A + sb[:,None,None] * onehot_a
        h = p_clean / denom_c[:,None,:,:].clamp_min(1e-30)
        h_sum = h.sum(dim=-1, keepdim=True)
        inner = (1.0 - tb[:,None,None,None]) / A * h_sum + tb[:,None,None,None] * h
        ratio = (sb / tb.clamp_min(1e-12)).clamp(0.0,1.0)
        q_forward = (1.0 - ratio[:,None,None]) / A + ratio[:,None,None] * onehot_a
        probs = q_forward[:,None,:,:] * inner
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return log_router, probs.clamp_min(1e-30).log()

    @staticmethod
    def _st_onehot(scores: Tensor, temperature: float) -> Tensor:
        soft = F.softmax(scores / float(temperature), dim=-1)
        hard = F.one_hot(torch.argmax(scores, dim=-1), num_classes=scores.shape[-1]).to(soft.dtype)
        return hard + soft - soft.detach()

    def _st_transition(
        self,
        hard_tokens: Tensor,
        residue_probs_st: Tensor,
        s: float,
        t: float,
        xi: Tensor,
        *,
        mean_shift: Tensor | None,
        temperature: float,
    ) -> tuple[Tensor, Tensor]:
        b, length = hard_tokens.shape
        layout = transition_innovation_layout(self.process, length)
        xi = torch.as_tensor(xi, device=hard_tokens.device, dtype=torch.float32)
        if tuple(xi.shape) != (b, layout.total_dim):
            raise ValueError("xi has incompatible shape")
        used = xi
        if mean_shift is not None:
            shift = torch.as_tensor(mean_shift, device=hard_tokens.device, dtype=torch.float32)
            if shift.ndim == 1:
                shift = shift.unsqueeze(0)
            if shift.shape[1] != layout.total_dim or shift.shape[0] not in (1,b):
                raise ValueError("mean_shift has incompatible shape")
            used = used + shift
        log_router, token_logp = self._transition_log_probs_from_st_state(residue_probs_st, s, t)
        router_score = log_router + normal_to_gumbel(used[:, :layout.router_dim])
        router_st = self._st_onehot(router_score, temperature)
        token_xi = used[:, layout.router_dim:].reshape(b, layout.residue_positions, layout.amino_acids)
        token_score = token_logp + normal_to_gumbel(token_xi)[:,None,:,:]
        token_st_by_latent = self._st_onehot(token_score, temperature)
        next_probs_st = torch.einsum("bm,bmla->bla", router_st, token_st_by_latent)
        draw_idx = torch.argmax(next_probs_st.detach(), dim=-1)
        aa = torch.tensor(self.lkf.config.aa_token_ids, device=hard_tokens.device, dtype=torch.long)
        next_hard = hard_tokens.clone()
        next_hard[:,1:-1] = aa[draw_idx]
        return next_hard, next_probs_st

    def _anchor_from_st_residue_probs(self, probs: Tensor, time: float) -> Tensor:
        b, residues, aa_count = probs.shape
        aa_ids = torch.tensor(self.lkf.config.aa_token_ids, device=probs.device, dtype=torch.long)
        if aa_count != int(aa_ids.numel()):
            raise ValueError("AA dimension mismatch")
        enc = self.anchor_encoder
        emb = enc.token_embedder.weight
        aa_emb = emb.index_select(0, aa_ids).float()
        interior = probs.float() @ aa_emb
        cls = emb[int(self.lkf.config.cls_token_id)].float().view(1,1,-1).expand(b,1,-1)
        eos = emb[int(self.lkf.config.eos_token_id)].float().view(1,1,-1).expand(b,1,-1)
        hidden = torch.cat([cls,interior,eos],dim=1)
        tv = torch.full((b,), float(time), device=probs.device, dtype=enc.pos_embedder.dtype)
        condition = enc.time_embedder(tv)
        hidden = hidden + enc.pos_embedder[:, : residues+2]
        for block in enc.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:,1:-1].mean(dim=1)
        if self.config.feature_source == "shared_pooled_plus_time":
            pooled = pooled + condition
        return pooled.float() @ self.anchor_projection.float()

    def soft_chain_terminal_anchor_samples(
        self,
        x_start: Tensor,
        chain: Sequence[float],
        *,
        xi_blocks: Sequence[Tensor],
        mean_shifts: Sequence[Tensor | None] | None = None,
        temperature: float | None = None,
    ) -> Tensor:
        """Straight-through chain terminal anchor for Stage-B gradients.

        For fixed xi, the forward terminal sequence/anchor exactly matches the
        hard categorical chain; only the backward path is relaxed.
        """
        c = self.match_chain(chain)
        h = len(c)-1
        if len(xi_blocks) != h:
            raise ValueError("xi_blocks length mismatch")
        if mean_shifts is None:
            mean_shifts = [None] * h
        if len(mean_shifts) != h:
            raise ValueError("mean_shifts length mismatch")
        hard = x_start
        probs = self._residue_onehot(hard)
        tau = float(self.config.soft_temperature if temperature is None else temperature)
        for k,(s,t) in enumerate(zip(c[:-1],c[1:])):
            # Make the state straight-through: exact hard forward, soft backward.
            hard_onehot = self._residue_onehot(hard)
            probs = hard_onehot + probs - probs.detach()
            hard, probs = self._st_transition(
                hard, probs, s, t, xi_blocks[k], mean_shift=mean_shifts[k], temperature=tau
            )
        return self._anchor_from_st_residue_probs(probs, 1.0)

    def auxiliary_state_dict(self) -> dict[str, Tensor]:
        return {k:v for k,v in self.state_dict().items() if not k.startswith("lkf.")}

    def load_auxiliary_state_dict(self, state: Mapping[str, Tensor], *, strict: bool = True) -> None:
        for name in (
            "information_transform","information_covariance","information_eigenvalues",
            "information_rank","information_metric_initialized",
        ):
            if name in state:
                setattr(self, name, torch.as_tensor(state[name]).detach().clone())
        current = self.state_dict()
        merged = dict(current)
        for key,value in state.items():
            if key.startswith("lkf."):
                raise ValueError("auxiliary state cannot contain lkf.*")
            if key not in merged:
                if strict:
                    raise KeyError(f"unexpected terminal-controlled state key: {key}")
                continue
            merged[key] = value
        if strict:
            expected = {k for k in current if not k.startswith("lkf.")}
            missing = expected.difference(state.keys())
            if missing:
                raise KeyError(f"missing terminal-controlled state keys: {sorted(missing)[:8]}")
        self.load_state_dict(merged, strict=True)
        self.anchor_encoder.eval()


def configure_terminal_stage_b_trainability(
    model: TerminalControlledKoopmanFlowMap,
    *,
    unfreeze_shared_blocks: int = 2,
    unfreeze_downstream: bool = True,
) -> list[str]:
    """Freeze geometry/anchor; conservatively adapt only the active LKF tail."""
    from .model import configure_lkf_trainability
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = configure_lkf_trainability(
        model.lkf,
        stage="B",
        unfreeze_shared_blocks=int(unfreeze_shared_blocks),
        unfreeze_downstream=bool(unfreeze_downstream),
    )
    for p in model.anchor_encoder.parameters():
        p.requires_grad_(False)
    for p in model.chain_mean_operators.parameters():
        p.requires_grad_(False)
    for p in model.terminal_responses.parameters():
        p.requires_grad_(False)
    return trainable


__all__ = [
    "TerminalControlledKoopmanConfig",
    "TerminalControlledKoopmanFlowMap",
    "configure_terminal_stage_b_trainability",
]
