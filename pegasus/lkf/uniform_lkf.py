"""Teacher-free uniform-noise Latent Kernel Flow for peptides.

This module is intentionally separate from :mod:`lkf_model` / :mod:`lkf_process`.
Those legacy modules define the PepDFM-distilled LKF used by earlier experiments.
The model here is trained *from scratch* against clean peptide sequences only.

Process convention
------------------
For a clean peptide ``X_1`` and time ``s in [0,1]`` each peptide-residue position
is sampled independently from

    q_s(x_s^i | x_1^i) = s delta_{x_1^i} + (1-s) U_AA,

where ``U_AA`` is uniform over the 20 canonical amino-acid token IDs. ESM
boundary tokens are kept fixed. Therefore ``X_0`` is an IID uniform amino-acid
sequence (with fixed <cls>/<eos>) and ``X_1`` is real data.

The neural network learns a normalized sequence-level latent clean posterior

    P_theta(x_1 | x_s, s)
      = sum_k w_k(x_s,s) prod_i p_{i,k}(x_1^i | x_s,s).

The finite-time map is the reverse kernel of the Markov uniform-noising process.
For ``0 <= s < t <= 1``, the forward/noising transition from time t back to s is

    Q_{t->s}(a | b) = (s/t) 1[a=b] + (1-s/t) U_AA(a).

Given a clean endpoint c, the exact bridge is

    R_{s,t}(b | a,c)
      = q_t(b|c) Q_{t->s}(a|b) / q_s(a|c).

The learned clean posterior is integrated through this bridge inside each latent
component, yielding an exact normalized sequence-level mixture K_theta(x_t|x_s).
Each finite-time call samples a fresh sequence-level latent; no hidden persistent
state is used.

The implementation never imports or loads PepDFM checkpoints or trajectories.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import DiTBlock, TimestepEmbedder


Tensor = torch.Tensor
TimeLike = Union[float, int, Tensor]
CheckpointPath = Union[str, Path]


@dataclass(frozen=True)
class UniformLKFConfig:
    """Architecture plus peptide-token support for scratch Uniform-LKF."""

    vocab_size: int
    seq_len: int
    model_dim: int
    n_heads: int
    n_layers: int
    latent_components: int = 8
    latent_layers: int = 4
    cls_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    aa_token_ids: tuple[int, ...] = tuple(range(4, 24))

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.seq_len < 3:
            raise ValueError("seq_len must be >= 3 to contain <cls>, residues, <eos>")
        if self.model_dim <= 0 or self.n_heads <= 0 or self.model_dim % self.n_heads != 0:
            raise ValueError("model_dim must be divisible by positive n_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.latent_components <= 0:
            raise ValueError("latent_components must be positive")
        if not (1 <= self.latent_layers <= self.n_layers):
            raise ValueError("latent_layers must lie in [1,n_layers]")
        aa = tuple(int(x) for x in self.aa_token_ids)
        if len(aa) < 2 or len(set(aa)) != len(aa):
            raise ValueError("aa_token_ids must contain unique categorical IDs")
        if min(aa) < 0 or max(aa) >= self.vocab_size:
            raise ValueError("aa_token_ids lie outside vocab_size")
        specials = (self.cls_token_id, self.pad_token_id, self.eos_token_id, self.unk_token_id)
        if any(int(x) < 0 or int(x) >= self.vocab_size for x in specials):
            raise ValueError("special token IDs lie outside vocab_size")
        if set(aa).intersection(int(x) for x in specials):
            raise ValueError("canonical amino-acid IDs overlap special token IDs")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UniformLKFConfig":
        return cls(
            vocab_size=int(value["vocab_size"]),
            seq_len=int(value["seq_len"]),
            model_dim=int(value["model_dim"]),
            n_heads=int(value["n_heads"]),
            n_layers=int(value["n_layers"]),
            latent_components=int(value.get("latent_components", 8)),
            latent_layers=int(value.get("latent_layers", 4)),
            cls_token_id=int(value.get("cls_token_id", 0)),
            pad_token_id=int(value.get("pad_token_id", 1)),
            eos_token_id=int(value.get("eos_token_id", 2)),
            unk_token_id=int(value.get("unk_token_id", 3)),
            aa_token_ids=tuple(int(x) for x in value.get("aa_token_ids", tuple(range(4, 24)))),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["aa_token_ids"] = list(self.aa_token_ids)
        return out


@dataclass
class UniformLKFOutput:
    router_logits: Tensor       # [B,M]
    clean_token_logits: Tensor  # [B,M,L,V]


@dataclass
class UniformLKFTrajectory:
    states: Tensor      # [B,NFE+1,L]
    times: Tensor       # [NFE+1]
    latent_ids: Tensor  # [B,NFE]

    @property
    def terminal(self) -> Tensor:
        return self.states[:, -1]


class UniformLKF(nn.Module):
    """Sequence-level latent clean-posterior model trained from random weights."""

    def __init__(self, config: UniformLKFConfig):
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.embedding_vocab_size = int(config.vocab_size)
        self.seq_len = int(config.seq_len)
        self.model_dim = int(config.model_dim)
        self.n_heads = int(config.n_heads)
        self.n_layers = int(config.n_layers)
        self.latent_components = int(config.latent_components)
        self.latent_layers = int(config.latent_layers)
        self.shared_layers = self.n_layers - self.latent_layers

        self.token_embedder = nn.Embedding(self.vocab_size, self.model_dim)
        self.pos_embedder = nn.Parameter(torch.empty(1, self.seq_len, self.model_dim))
        self.time_embedder = TimestepEmbedder(self.model_dim)
        self.shared_blocks = nn.ModuleList(
            [DiTBlock(self.model_dim, self.n_heads) for _ in range(self.shared_layers)]
        )
        self.latent_blocks = nn.ModuleList(
            [DiTBlock(self.model_dim, self.n_heads) for _ in range(self.latent_layers)]
        )
        self.latent_embedder = nn.Embedding(self.latent_components, self.model_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.latent_components),
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size)

        # Truly random scratch initialization. No PepDFM-copy-preserving zeros.
        def _scratch_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)

        self.apply(_scratch_init)
        nn.init.normal_(self.pos_embedder, mean=0.0, std=0.02)
        with torch.no_grad():
            if self.latent_components == 1:
                self.latent_embedder.weight.zero_()
            else:
                self.latent_embedder.weight.normal_(mean=0.0, std=0.02)
                self.latent_embedder.weight.sub_(
                    self.latent_embedder.weight.mean(dim=0, keepdim=True)
                )

        aa_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        aa_mask[list(config.aa_token_ids)] = True
        self.register_buffer("aa_output_mask", aa_mask, persistent=False)
        aa_index = torch.full((self.vocab_size,), -1, dtype=torch.long)
        for j, token_id in enumerate(config.aa_token_ids):
            aa_index[int(token_id)] = int(j)
        self.register_buffer("aa_index_by_token", aa_index, persistent=False)

    @property
    def model_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    @staticmethod
    def _batch_time(
        value: TimeLike,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        elif value.ndim == 1 and value.shape[0] == 1 and batch_size != 1:
            value = value.expand(batch_size)
        elif value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must be scalar or shape [batch={batch_size}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value

    def _validate_x(self, x: Tensor, *, name: str) -> None:
        if x.ndim != 2 or x.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long [batch,length]")
        if x.shape[1] < 3 or x.shape[1] > self.seq_len:
            raise ValueError(f"{name} length must lie in [3,{self.seq_len}]")
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= self.vocab_size):
            raise ValueError(f"{name} contains IDs outside [0,{self.vocab_size-1}]")

    def validate_peptide_tokens(self, x: Tensor, *, name: str = "x") -> None:
        """Require ESM <cls> ... canonical-AA ... <eos> layout."""
        self._validate_x(x, name=name)
        if not bool((x[:, 0] == int(self.config.cls_token_id)).all()):
            raise ValueError(f"{name} does not have cls_token_id at position 0")
        if not bool((x[:, -1] == int(self.config.eos_token_id)).all()):
            raise ValueError(f"{name} does not have eos_token_id at the final position")
        interior = x[:, 1:-1]
        allowed = self.aa_output_mask[interior]
        if not bool(allowed.all()):
            bad = interior[~allowed]
            example = int(bad.reshape(-1)[0].item()) if bad.numel() else -1
            raise ValueError(
                f"{name} contains non-canonical residue token ID {example} in an interior position"
            )

    def forward(self, x_s: Tensor, s: TimeLike) -> UniformLKFOutput:
        self.validate_peptide_tokens(x_s, name="x_s")
        batch, length = x_s.shape
        s_batch = self._batch_time(
            s,
            batch_size=batch,
            device=x_s.device,
            dtype=self.pos_embedder.dtype,
            name="s",
        )
        if torch.any(s_batch < 0) or torch.any(s_batch >= 1):
            # s=1 is an identity endpoint and should never require neural inference.
            raise ValueError("UniformLKF neural posterior requires 0 <= s < 1")
        condition = self.time_embedder(s_batch)
        hidden = self.token_embedder(x_s) + self.pos_embedder[:, :length]
        for block in self.shared_blocks:
            hidden = block(hidden, condition)
        pooled = hidden[:, 1:-1].mean(dim=1) + condition
        router_logits = self.router(pooled)

        m = self.latent_components
        latent_ids = torch.arange(m, device=x_s.device)
        latent_condition = condition[:, None, :] + self.latent_embedder(latent_ids)[None, :, :]
        branched = hidden[:, None, :, :].expand(batch, m, length, self.model_dim)
        branched = branched.reshape(batch * m, length, self.model_dim)
        latent_condition = latent_condition.reshape(batch * m, self.model_dim)
        for block in self.latent_blocks:
            branched = block(branched, latent_condition)
        token_logits = self.lm_head(self.final_norm(branched))
        token_logits = token_logits.reshape(batch, m, length, self.vocab_size)
        return UniformLKFOutput(router_logits=router_logits, clean_token_logits=token_logits)

    def clean_log_probs(self, output: UniformLKFOutput) -> Tensor:
        """Return log p(clean residue | x_s,k,s), normalized over canonical AAs."""
        logits = output.clean_token_logits.float()
        invalid = ~self.aa_output_mask
        logits = logits.masked_fill(invalid.view(1, 1, 1, -1), float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def clean_posterior_terms(
        self,
        x_s: Tensor,
        x_1: Tensor,
        s: TimeLike,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Exact clean-sequence mixture likelihood over peptide residue positions."""
        self.validate_peptide_tokens(x_s, name="x_s")
        self.validate_peptide_tokens(x_1, name="x_1")
        if x_s.shape != x_1.shape:
            raise ValueError("x_s and x_1 must have identical shapes")
        output = self(x_s, s)
        log_router = F.log_softmax(output.router_logits.float(), dim=-1)
        log_clean = self.clean_log_probs(output)[:, :, 1:-1, :]
        target = x_1[:, None, 1:-1, None].expand(-1, self.latent_components, -1, 1)
        selected = log_clean.gather(-1, target).squeeze(-1)
        per_latent_logp = selected.sum(dim=-1)
        log_mix = torch.logsumexp(log_router + per_latent_logp, dim=-1)
        return log_mix, log_router, per_latent_logp

    def clean_log_prob(self, x_s: Tensor, x_1: Tensor, s: TimeLike) -> Tensor:
        return self.clean_posterior_terms(x_s, x_1, s)[0]

    def posterior_latent_probs(self, x_s: Tensor, x_1: Tensor, s: TimeLike) -> Tensor:
        _, log_router, per_latent = self.clean_posterior_terms(x_s, x_1, s)
        return torch.softmax(log_router + per_latent, dim=-1)


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@contextmanager
def _temporary_eval(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _sample_probs(probs: Tensor, *, generator: Optional[torch.Generator]) -> Tensor:
    flat = probs.reshape(-1, probs.shape[-1])
    draws = torch.multinomial(flat, 1, replacement=True, generator=generator)
    return draws.reshape(probs.shape[:-1])


class UniformLKFProcess:
    """Analytic uniform-noise finite-time process driven by a scratch UniformLKF."""

    def __init__(self, model: UniformLKF, *, token_temperature: float = 1.0, router_temperature: float = 1.0):
        if token_temperature <= 0 or router_temperature <= 0:
            raise ValueError("temperatures must be >0")
        self.model = model
        self.token_temperature = float(token_temperature)
        self.router_temperature = float(router_temperature)
        self._aa_ids_cpu = torch.tensor(model.config.aa_token_ids, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return _model_device(self.model)

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.model.seq_len

    def _times(self, value: TimeLike, batch: int, *, name: str) -> Tensor:
        return self.model._batch_time(
            value,
            batch_size=batch,
            device=self.device,
            dtype=self.model.pos_embedder.dtype,
            name=name,
        )

    def _aa_ids(self, device: torch.device) -> Tensor:
        return self._aa_ids_cpu.to(device=device)

    @torch.no_grad()
    def sample_source(
        self,
        batch_size: int,
        seq_len: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >=1")
        if seq_len < 3 or seq_len > self.max_seq_len:
            raise ValueError(f"seq_len must lie in [3,{self.max_seq_len}]")
        aa = self._aa_ids(self.device)
        choice = torch.randint(
            0,
            len(aa),
            (int(batch_size), int(seq_len) - 2),
            device=self.device,
            generator=generator,
        )
        interior = aa[choice]
        out = torch.empty((int(batch_size), int(seq_len)), device=self.device, dtype=torch.long)
        out[:, 0] = int(self.model.config.cls_token_id)
        out[:, -1] = int(self.model.config.eos_token_id)
        out[:, 1:-1] = interior
        return out

    @torch.no_grad()
    def corrupt_clean(
        self,
        x_1: Tensor,
        s: TimeLike,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample q_s(x_s|x_1)=s delta_data +(1-s) Uniform-AA independently."""
        self.model.validate_peptide_tokens(x_1, name="x_1")
        if x_1.device != self.device:
            raise ValueError("x_1 must be on the same device as the model")
        batch, length = x_1.shape
        s_batch = self._times(s, batch, name="s")
        if torch.any(s_batch < 0) or torch.any(s_batch > 1):
            raise ValueError("corruption requires 0<=s<=1")
        if bool(torch.all(s_batch == 1)):
            return x_1.clone()
        aa = self._aa_ids(self.device)
        replacement_idx = torch.randint(
            0, len(aa), (batch, length - 2), device=self.device, generator=generator
        )
        replacement = aa[replacement_idx]
        keep = torch.rand((batch, length - 2), device=self.device, generator=generator) < s_batch[:, None]
        out = x_1.clone()
        out[:, 1:-1] = torch.where(keep, x_1[:, 1:-1], replacement)
        return out

    @torch.no_grad()
    def sample_teacher_coupled_future(
        self,
        x_s: Tensor,
        x_1: Tensor,
        s: TimeLike,
        t: TimeLike,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample X_t from the exact uniform-noising bridge given paired (X_s,X_1).

        The underlying *forward/noising* process is Markov from data (time 1)
        toward uniform noise (time 0). This conditional bridge generates valid
        held-out reverse-process pairs without any learned teacher.
        """
        self.model.validate_peptide_tokens(x_s, name="x_s")
        self.model.validate_peptide_tokens(x_1, name="x_1")
        if x_s.shape != x_1.shape:
            raise ValueError("x_s and x_1 shapes differ")
        batch = x_s.shape[0]
        s_batch = self._times(s, batch, name="s")
        t_batch = self._times(t, batch, name="t")
        if torch.any(s_batch < 0) or torch.any(t_batch > 1) or torch.any(t_batch < s_batch):
            raise ValueError("requires 0<=s<=t<=1")
        equal = torch.isclose(s_batch, t_batch, atol=1e-7, rtol=0)
        if bool(equal.all()):
            return x_s.clone()
        if bool(equal.any()):
            raise ValueError("batched bridge cannot mix identity and non-identity intervals")

        aa = self._aa_ids(self.device)
        aa_count = float(len(aa))
        a_idx = self.model.aa_index_by_token[x_s[:, 1:-1]]
        c_idx = self.model.aa_index_by_token[x_1[:, 1:-1]]
        if torch.any(a_idx < 0) or torch.any(c_idx < 0):
            raise ValueError("bridge states contain non-canonical interior tokens")
        onehot_a = F.one_hot(a_idx, num_classes=len(aa)).float()
        onehot_c = F.one_hot(c_idx, num_classes=len(aa)).float()

        # q_t(b|c) = t delta_c(b) + (1-t) U(b).
        q_t = (1.0 - t_batch[:, None, None]) / aa_count + t_batch[:, None, None] * onehot_c
        # Q_{t->s}(a|b) = (s/t) delta_a(b) + (1-s/t) U(a).
        ratio = (s_batch / t_batch.clamp_min(1e-12)).clamp(0.0, 1.0)
        q_forward = (1.0 - ratio[:, None, None]) / aa_count + ratio[:, None, None] * onehot_a
        # q_s(a|c) is the bridge normalizer.
        same_ac = (a_idx == c_idx).float()
        denom = (1.0 - s_batch[:, None]) / aa_count + s_batch[:, None] * same_ac
        probs = q_t * q_forward / denom[:, :, None].clamp_min(1e-30)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        draw_idx = _sample_probs(probs, generator=generator).long()
        out = x_s.clone()
        out[:, 1:-1] = aa[draw_idx]
        return out

    def _component_transition_log_probs(
        self,
        x_s: Tensor,
        s_batch: Tensor,
        t_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return log-router [B,M] and AA transition log probs [B,M,L-2,A].

        Each component's learned clean posterior is integrated exactly through
        the Bayes bridge of the Markov uniform-noising process.
        """
        output = self.model(x_s, s_batch)
        log_router = F.log_softmax(output.router_logits.float() / self.router_temperature, dim=-1)
        clean_logits = output.clean_token_logits.float()[:, :, 1:-1, :] / self.token_temperature
        clean_logits = clean_logits.masked_fill(
            (~self.model.aa_output_mask).view(1, 1, 1, -1), float("-inf")
        )
        clean_full = torch.softmax(clean_logits, dim=-1)
        aa = self._aa_ids(self.device)
        p_clean = clean_full.index_select(-1, aa)  # [B,M,L,A]
        aa_count = float(len(aa))
        a_idx = self.model.aa_index_by_token[x_s[:, 1:-1]]
        if torch.any(a_idx < 0):
            raise ValueError("x_s contains non-canonical interior tokens")
        onehot_a = F.one_hot(a_idx, num_classes=len(aa)).float()  # [B,L,A]

        # For c~p_clean(c|x_s,k), bridge R(b|a,c) is
        # q_t(b|c)Q(a|b)/q_s(a|c). Summing c can be simplified to:
        # Q(a|b) * [(1-t)/A * sum_c p(c)/q_s(a|c) + t*p(b)/q_s(a|b)].
        denom_c = (1.0 - s_batch[:, None, None]) / aa_count + s_batch[:, None, None] * onehot_a
        h = p_clean / denom_c[:, None, :, :].clamp_min(1e-30)
        h_sum = h.sum(dim=-1, keepdim=True)
        inner = (1.0 - t_batch[:, None, None, None]) / aa_count * h_sum + t_batch[:, None, None, None] * h

        ratio = (s_batch / t_batch.clamp_min(1e-12)).clamp(0.0, 1.0)
        q_forward = (1.0 - ratio[:, None, None]) / aa_count + ratio[:, None, None] * onehot_a
        probs = q_forward[:, None, :, :] * inner
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return log_router, probs.clamp_min(1e-30).log()

    def transition_log_prob(self, x_s: Tensor, x_t: Tensor, s: TimeLike, t: TimeLike) -> Tensor:
        """Exact normalized log K_theta(x_t|x_s,s,t), including fixed boundaries."""
        self.model.validate_peptide_tokens(x_s, name="x_s")
        self.model.validate_peptide_tokens(x_t, name="x_t")
        if x_s.shape != x_t.shape:
            raise ValueError("x_s and x_t shapes differ")
        if x_s.device != self.device or x_t.device != self.device:
            raise ValueError("states must be on model device")
        batch = x_s.shape[0]
        s_batch = self._times(s, batch, name="s")
        t_batch = self._times(t, batch, name="t")
        if torch.any(s_batch < 0) or torch.any(t_batch > 1) or torch.any(t_batch < s_batch):
            raise ValueError("requires 0<=s<=t<=1")
        equal = torch.isclose(s_batch, t_batch, atol=1e-7, rtol=0)
        if bool(equal.all()):
            return torch.where(
                (x_s == x_t).all(dim=1),
                torch.zeros(batch, device=self.device),
                torch.full((batch,), float("-inf"), device=self.device),
            )
        if bool(equal.any()):
            raise ValueError("batched transition cannot mix identity and non-identity intervals")
        if not bool((x_s[:, [0, -1]] == x_t[:, [0, -1]]).all()):
            return torch.full((batch,), float("-inf"), device=self.device)
        log_router, token_logp = self._component_transition_log_probs(x_s, s_batch, t_batch)
        target_idx = self.model.aa_index_by_token[x_t[:, 1:-1]]
        if torch.any(target_idx < 0):
            raise ValueError("x_t contains non-canonical interior tokens")
        target = target_idx[:, None, :, None].expand(-1, self.model.latent_components, -1, 1)
        selected = token_logp.gather(-1, target).squeeze(-1).sum(-1)
        return torch.logsumexp(log_router + selected, dim=-1)

    @torch.no_grad()
    def sample_transition(
        self,
        x_s: Tensor,
        s: TimeLike,
        t: TimeLike,
        *,
        generator: Optional[torch.Generator] = None,
        return_latent: bool = False,
    ):
        self.model.validate_peptide_tokens(x_s, name="x_s")
        if x_s.device != self.device:
            raise ValueError("x_s must be on model device")
        batch = x_s.shape[0]
        s_batch = self._times(s, batch, name="s")
        t_batch = self._times(t, batch, name="t")
        if torch.any(s_batch < 0) or torch.any(t_batch > 1) or torch.any(t_batch < s_batch):
            raise ValueError("requires 0<=s<=t<=1")
        equal = torch.isclose(s_batch, t_batch, atol=1e-7, rtol=0)
        if bool(equal.all()):
            latent = torch.full((batch,), -1, device=self.device, dtype=torch.long)
            return (x_s.clone(), latent) if return_latent else x_s.clone()
        if bool(equal.any()):
            raise ValueError("batched transition cannot mix identity and non-identity intervals")
        with _temporary_eval(self.model):
            log_router, token_logp = self._component_transition_log_probs(x_s, s_batch, t_batch)
            router_probs = log_router.exp()
            latent = _sample_probs(router_probs, generator=generator).long()
            idx = torch.arange(batch, device=self.device)
            selected_probs = token_logp.exp()[idx, latent]  # [B,L-2,A]
            draw_idx = _sample_probs(selected_probs, generator=generator).long()
            aa = self._aa_ids(self.device)
            x_t = x_s.clone()
            x_t[:, 1:-1] = aa[draw_idx]
        return (x_t, latent) if return_latent else x_t

    @torch.no_grad()
    def sample_trajectory(
        self,
        *,
        batch_size: int,
        seq_len: int,
        nfe: int,
        generator: Optional[torch.Generator] = None,
    ) -> UniformLKFTrajectory:
        if nfe < 1:
            raise ValueError("nfe must be >=1")
        current = self.sample_source(batch_size, seq_len, generator=generator)
        times = torch.linspace(0.0, 1.0, int(nfe) + 1, device=self.device)
        states = [current]
        latents = []
        for j in range(int(nfe)):
            current, latent = self.sample_transition(
                current, times[j], times[j + 1], generator=generator, return_latent=True
            )
            states.append(current)
            latents.append(latent)
        return UniformLKFTrajectory(
            states=torch.stack(states, dim=1),
            times=times,
            latent_ids=torch.stack(latents, dim=1),
        )

    @torch.no_grad()
    def sample_terminal(
        self,
        *,
        batch_size: int,
        seq_len: int,
        nfe: int = 1,
        generator: Optional[torch.Generator] = None,
        return_trajectory: bool = False,
    ):
        traj = self.sample_trajectory(
            batch_size=batch_size, seq_len=seq_len, nfe=nfe, generator=generator
        )
        return traj if return_trajectory else traj.terminal

    @torch.no_grad()
    def continue_to_terminal(
        self,
        x_s: Tensor,
        s: TimeLike,
        *,
        generator: Optional[torch.Generator] = None,
        return_latent: bool = False,
    ):
        return self.sample_transition(
            x_s, s, 1.0, generator=generator, return_latent=return_latent
        )


def load_uniform_lkf_checkpoint(
    checkpoint_path: CheckpointPath,
    map_location: Any = "cpu",
    eval_mode: bool = True,
) -> UniformLKF:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Uniform-LKF checkpoint must be a mapping")
    if str(checkpoint.get("model_type", "")).lower() != "uniform_lkf":
        raise ValueError("Checkpoint is not a scratch uniform_lkf checkpoint")
    if "model_config" not in checkpoint:
        raise KeyError("Uniform-LKF checkpoint is missing model_config")
    config = UniformLKFConfig.from_mapping(checkpoint["model_config"])
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        full = checkpoint["state_dict"]
        prefix = "lkf."
        state = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
        if not state:
            raise KeyError("Lightning checkpoint contains no lkf.* weights")
    else:
        raise KeyError("Expected model_state_dict or Lightning state_dict")
    model = UniformLKF(config)
    model.load_state_dict(state, strict=True)
    # Attach non-architectural provenance for audits.
    for key in (
        "training_source",
        "process_type",
        "initialization",
        "teacher_checkpoint",
        "teacher_trajectory_supervision",
        "dataset_root",
        "source_anchor_prob",
        "checkpoint_format_version",
    ):
        if key in checkpoint:
            setattr(model, key, checkpoint[key])
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "UniformLKF",
    "UniformLKFConfig",
    "UniformLKFOutput",
    "UniformLKFProcess",
    "UniformLKFTrajectory",
    "load_uniform_lkf_checkpoint",
]
