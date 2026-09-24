"""Explicit Gaussian base-noise parameterization for Uniform-LKF sampling.

This module does *not* change the Uniform-LKF transition law.  It reparameterizes
its categorical draws using independent Gaussian variables transformed through
Normal CDF -> Gumbel and Gumbel-max.  Consequently the same native finite-time
kernel is sampled, while every generated sequence comes with an explicit
standard-normal innovation vector xi that can be recorded or mean-shifted.

The diagnostic stochastic-Koopman hypothesis is then testable as

    z(X_t) ~= A_{s,t} z(X_s) + B_{s,t} xi.

Mean-shifting xi therefore provides a causal intervention on the *actual* random
variables that generate the discrete peptide, rather than on a detached latent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from .lkf.uniform_lkf import UniformLKFProcess, _temporary_eval

Tensor = torch.Tensor


@dataclass(frozen=True)
class InnovationLayout:
    router_dim: int
    residue_positions: int
    amino_acids: int

    @property
    def token_dim(self) -> int:
        return int(self.residue_positions * self.amino_acids)

    @property
    def total_dim(self) -> int:
        return int(self.router_dim + self.token_dim)


def transition_innovation_layout(process: UniformLKFProcess, token_length: int) -> InnovationLayout:
    if int(token_length) < 3:
        raise ValueError("token_length must include <cls>, residues, <eos>")
    return InnovationLayout(
        router_dim=int(process.model.latent_components),
        residue_positions=int(token_length) - 2,
        amino_acids=len(process.model.config.aa_token_ids),
    )


def normal_to_gumbel(xi: Tensor, eps: float = 1e-7) -> Tensor:
    """Monotone transform N(0,1) -> standard Gumbel."""
    x = torch.as_tensor(xi)
    # Standard normal CDF, written through erf for broad PyTorch compatibility.
    u = (0.5 * (1.0 + torch.erf(x.float() / math.sqrt(2.0)))).clamp(float(eps), 1.0 - float(eps))
    return -torch.log(-torch.log(u))


def sample_transition_with_gaussian_innovation(
    process: UniformLKFProcess,
    x_s: Tensor,
    s: float | Tensor,
    t: float | Tensor,
    *,
    xi: Optional[Tensor] = None,
    mean_shift: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_latent: bool = False,
):
    """Sample the exact native LKF transition via explicit Gaussian innovations.

    Parameters
    ----------
    xi:
        Optional [B,D_xi] IID standard-normal base noise.  If omitted, it is
        sampled with ``generator``.
    mean_shift:
        Optional [D_xi] or [B,D_xi] additive shift applied to xi before the
        Gaussian->Gumbel transform.  Thus ``mean_shift=v`` realizes the
        intervention xi ~ N(v,I) while keeping the native categorical map.

    Returns
    -------
    x_t, xi_used[, latent]
        ``xi_used`` is the *shifted* Gaussian innovation that physically produced
        x_t.  For paired intervention tests, pass the same unshifted xi to a
        baseline call and a controlled call with mean_shift.
    """
    process.model.validate_peptide_tokens(x_s, name="x_s")
    if x_s.device != process.device:
        raise ValueError("x_s must be on the model device")
    batch, length = x_s.shape
    s_batch = process._times(s, batch, name="s")
    t_batch = process._times(t, batch, name="t")
    if torch.any(s_batch < 0) or torch.any(t_batch > 1) or torch.any(t_batch < s_batch):
        raise ValueError("requires 0<=s<=t<=1")
    equal = torch.isclose(s_batch, t_batch, atol=1e-7, rtol=0)
    if bool(equal.all()):
        layout = transition_innovation_layout(process, length)
        base = torch.zeros(batch, layout.total_dim, device=process.device)
        latent = torch.full((batch,), -1, device=process.device, dtype=torch.long)
        result = (x_s.clone(), base, latent) if return_latent else (x_s.clone(), base)
        return result
    if bool(equal.any()):
        raise ValueError("batched transition cannot mix identity and non-identity intervals")

    layout = transition_innovation_layout(process, length)
    if xi is None:
        xi_base = torch.randn(
            batch, layout.total_dim, device=process.device, dtype=torch.float32,
            generator=generator,
        )
    else:
        xi_base = torch.as_tensor(xi, device=process.device, dtype=torch.float32)
        if xi_base.shape != (batch, layout.total_dim):
            raise ValueError(
                f"xi must have shape {(batch, layout.total_dim)}, got {tuple(xi_base.shape)}"
            )

    xi_used = xi_base
    if mean_shift is not None:
        shift = torch.as_tensor(mean_shift, device=process.device, dtype=torch.float32)
        if shift.ndim == 1:
            if shift.shape[0] != layout.total_dim:
                raise ValueError("1-D mean_shift has wrong innovation dimension")
            shift = shift.unsqueeze(0)
        if shift.ndim != 2 or shift.shape[1] != layout.total_dim or shift.shape[0] not in (1, batch):
            raise ValueError("mean_shift must be [D_xi], [1,D_xi], or [B,D_xi]")
        xi_used = xi_base + shift

    router_xi = xi_used[:, : layout.router_dim]
    token_xi = xi_used[:, layout.router_dim :].reshape(
        batch, layout.residue_positions, layout.amino_acids
    )

    with torch.no_grad(), _temporary_eval(process.model):
        log_router, token_logp = process._component_transition_log_probs(
            x_s, s_batch, t_batch
        )
        router_g = normal_to_gumbel(router_xi)
        latent = torch.argmax(log_router.float() + router_g, dim=-1).long()
        idx = torch.arange(batch, device=process.device)
        selected_logp = token_logp.float()[idx, latent]  # [B,L-2,A]
        token_g = normal_to_gumbel(token_xi)
        draw_idx = torch.argmax(selected_logp + token_g, dim=-1).long()
        aa = process._aa_ids(process.device)
        x_t = x_s.clone()
        x_t[:, 1:-1] = aa[draw_idx]

    if return_latent:
        return x_t, xi_used, latent
    return x_t, xi_used


__all__ = [
    "InnovationLayout",
    "normal_to_gumbel",
    "sample_transition_with_gaussian_innovation",
    "transition_innovation_layout",
]
