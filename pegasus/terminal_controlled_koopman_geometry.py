"""Hard/soft geometry estimators shared by calibration, training and diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .explicit_innovation import transition_innovation_layout
from .lkf.uniform_lkf import UniformLKFProcess
from .terminal_controlled_koopman_model import TerminalControlledKoopmanFlowMap

Tensor = torch.Tensor


def parse_chains(text: str) -> tuple[tuple[float, ...], ...]:
    out: list[tuple[float,...]] = []
    for piece in str(text).split(";"):
        piece = piece.strip()
        if not piece:
            continue
        c = tuple(float(x.strip()) for x in piece.split(",") if x.strip())
        if len(c) < 2 or not math.isclose(c[-1],1.0,abs_tol=1e-8) or any(b<=a for a,b in zip(c,c[1:])):
            raise ValueError(f"invalid terminal chain {piece!r}")
        out.append(c)
    if not out:
        raise ValueError("no terminal chains")
    if len(set(out)) != len(out):
        raise ValueError("duplicate terminal chains")
    return tuple(out)


def chain_name(chain: Sequence[float]) -> str:
    return "-".join(f"{float(x):g}" for x in chain)


@torch.no_grad()
def prepare_chain_start(
    model: TerminalControlledKoopmanFlowMap,
    x_clean: Tensor,
    chain: Sequence[float],
    *,
    generator: torch.Generator,
) -> Tensor:
    s = float(chain[0])
    if s >= 1.0:
        return x_clean.clone()
    return UniformLKFProcess(model.lkf).corrupt_clean(x_clean, s, generator=generator)


@torch.no_grad()
def hard_chain_samples(
    model: TerminalControlledKoopmanFlowMap,
    x_start: Tensor,
    chain: Sequence[float],
    *,
    continuations: int,
    generator: torch.Generator,
) -> tuple[Tensor, list[Tensor], Tensor]:
    """Vectorized independent hard chain continuations.

    Returns terminal anchors [B,M,d], innovations list of [B,M,D], terminal
    tokens [B,M,L].
    """
    b, length = x_start.shape
    m = int(continuations)
    if m < 2:
        raise ValueError("continuations must be >=2 for centered Stein statistics")
    x_rep = x_start[:,None,:].expand(b,m,length).reshape(b*m,length).contiguous()
    terminal, xis = model.sample_chain(
        x_rep, chain, generator=generator, return_innovations=True
    )
    r = model.anchor_features(terminal, 1.0).reshape(b,m,-1)
    xi_out = [x.reshape(b,m,-1) for x in xis]
    return r, xi_out, terminal.reshape(b,m,length)


def chain_sufficient_statistics(
    model: TerminalControlledKoopmanFlowMap,
    x_start: Tensor,
    chain: Sequence[float],
    terminal_anchor: Tensor,
    xi_blocks: Sequence[Tensor],
) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor,Tensor]]]:
    """Return direct-mean and direct-terminal-Stein sufficient statistics.

    A stats: XTX, YTX, scalar state count.
    M stats per pulse: cross-sum [d,D], per-column effective centered count [D].
    """
    b,m,d = terminal_anchor.shape
    r0 = model.anchor_features(x_start, float(chain[0])).double()
    y = terminal_anchor.double().mean(dim=1)
    xtx = r0.T @ r0
    ytx = y.T @ r0
    n = torch.tensor(float(b), device=r0.device, dtype=torch.float64)
    rc = terminal_anchor.double() - y[:,None,:]
    blocks: list[tuple[Tensor,Tensor]] = []
    for xi in xi_blocks:
        x = xi.double()
        xc = x - x.mean(dim=1, keepdim=True)
        cross = torch.einsum("bmd,bmq->dq", rc, xc)
        # Centering within every source state loses one degree of freedom.
        count = torch.full((x.shape[-1],), float(b*(m-1)), device=x.device, dtype=torch.float64)
        blocks.append((cross,count))
    return xtx, ytx, n, blocks


def fit_direct_operator(xtx: Tensor, ytx: Tensor, *, ridge: float) -> Tensor:
    d = xtx.shape[0]
    scale = torch.trace(xtx).clamp_min(1e-12) / float(d)
    reg = float(ridge) * scale * torch.eye(d,device=xtx.device,dtype=xtx.dtype)
    # A = Y^T X (X^T X + lambda I)^-1 under row-state convention.
    return torch.linalg.solve((xtx+reg).T, ytx.T).T.float()


def fit_terminal_response(cross: Tensor, counts: Tensor) -> Tensor:
    den = counts.clamp_min(1.0).unsqueeze(0)
    out = cross / den
    out[:, counts <= 0] = 0
    return out.float()


def soft_terminal_stein_response(
    model: TerminalControlledKoopmanFlowMap,
    x_start: Tensor,
    chain: Sequence[float],
    *,
    continuations: int,
    generator: torch.Generator,
    temperature: float | None = None,
) -> tuple[list[Tensor], Tensor]:
    """Differentiable direct terminal M_C,k from the ST chain surrogate."""
    b,length=x_start.shape
    m=int(continuations)
    if m<2: raise ValueError("continuations must be >=2")
    layout=transition_innovation_layout(model.process,length)
    x_rep=x_start[:,None,:].expand(b,m,length).reshape(b*m,length).contiguous()
    xis=[]
    for _ in range(len(chain)-1):
        xis.append(torch.randn(b*m,layout.total_dim,device=x_start.device,generator=generator))
    r=model.soft_chain_terminal_anchor_samples(
        x_rep,chain,xi_blocks=xis,temperature=temperature
    ).reshape(b,m,-1)
    rc=r-r.mean(dim=1,keepdim=True)
    blocks=[]
    for xi in xis:
        xx=xi.reshape(b,m,-1)
        xc=xx-xx.mean(dim=1,keepdim=True)
        blocks.append(torch.einsum("bmd,bmq->dq",rc,xc)/float(b*(m-1)))
    return blocks,r


def normalized_gramian(
    model: TerminalControlledKoopmanFlowMap,
    blocks: Sequence[Tensor],
) -> Tensor:
    M=torch.cat([b.float() for b in blocks],dim=1)
    G=M@M.T
    T=model.information_transform.float()
    G=T@G@T.T
    return 0.5*(G+G.T)


def spectral_summary(G: Tensor) -> dict[str,float]:
    eig=torch.linalg.eigvalsh(0.5*(G.double()+G.double().T)).clamp_min(0)
    total=eig.sum().clamp_min(1e-30)
    p=eig/total
    eff=float(torch.exp(-(p[p>0]*torch.log(p[p>0])).sum()).item()) if bool((p>0).any()) else 0.0
    return {
        "eigen_min":float(eig.min().item()),
        "eigen_q10":float(torch.quantile(eig,0.10).item()),
        "eigen_median":float(eig.median().item()),
        "eigen_max":float(eig.max().item()),
        "trace":float(eig.sum().item()),
        "effective_rank":eff,
    }


__all__=[
    "parse_chains","chain_name","prepare_chain_start","hard_chain_samples",
    "chain_sufficient_statistics","fit_direct_operator","fit_terminal_response",
    "soft_terminal_stein_response","normalized_gramian","spectral_summary",
]
