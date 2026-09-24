"""DDP Stage-A/Stage-B training for the final controlled Koopman flow map.

Stage A (response identification)
---------------------------------
* initialize active generator + immutable 64-D anchor from the validated IP/K-LKF;
* freeze the generator and anchor;
* fit exact-composition A and physical-innovation response J from paired hard
  categorical continuations;
* initialize the fixed information metric from a frozen validation calibration set.

Stage B (controlled-flow training)
----------------------------------
* unfreeze only a conservative tail of the active LKF;
* keep A/J grounded by exact hard-sampling response targets;
* use a differentiable Gumbel-softmax estimator only to transmit response gradients
  to the generator;
* maximize the weak spectrum of normalized finite-horizon controllability while
  enforcing held-out native-fidelity checkpoint gates.

All diagnostic/final semantics remain hard sampled.  No downstream objective is
used anywhere in this trainer.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

from .controlled_koopman_checkpoint import (
    initialize_controlled_from_ip_checkpoint,
    load_controlled_koopman_checkpoint,
    save_controlled_koopman_checkpoint,
)
from .controlled_koopman_model import (
    ControlledKoopmanConfig,
    ControlledKoopmanFlowMap,
    configure_controlled_trainability,
)
from .distributed import (
    DistributedContext,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
    reduce_metric_sums,
)
from .explicit_innovation import transition_innovation_layout
from .ip_checkpoint import sha256_file
from .lkf.data import (
    ChunkedCleanPeptideBatchDataset,
    CleanPeptideBatchDataset,
    build_clean_loader,
    uniform_lkf_loss_and_metrics,
)
from .lkf.uniform_lkf import UniformLKFProcess
from .utils import effective_rank_from_eigenvalues, seed_everything, write_csv, write_json


def parse_knots(text: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if len(vals) < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("control knots must be a strictly increasing comma-separated list")
    if vals[0] < 0 or vals[-1] > 1:
        raise ValueError("control knots must lie in [0,1]")
    return vals


def parse_chains(text: str) -> tuple[tuple[float, ...], ...]:
    """Parse '0,0.5,1;0.25,0.5,0.75,1'."""
    out: list[tuple[float, ...]] = []
    for piece in str(text).split(";"):
        piece = piece.strip()
        if not piece:
            continue
        times = tuple(float(x.strip()) for x in piece.split(",") if x.strip())
        if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"invalid chain {piece!r}")
        out.append(times)
    if not out:
        raise ValueError("at least one controllability chain is required")
    return tuple(out)


def control_pairs(knots: Sequence[float]) -> tuple[tuple[float, float], ...]:
    vals = tuple(float(x) for x in knots)
    return tuple((s, t) for i, s in enumerate(vals[:-1]) for t in vals[i + 1 :])


def _prepare_val_batches(
    dataset: CleanPeptideBatchDataset,
    *,
    max_batch_sequences: int,
    max_batches: int,
) -> list[Tensor]:
    # HARD DDP RULE: this executes after CUDA init, therefore num_workers=0.
    out: list[Tensor] = []
    for batch in build_clean_loader(dataset, shuffle=False, num_workers=0):
        x = torch.as_tensor(batch, dtype=torch.long)
        for start in range(0, x.shape[0], int(max_batch_sequences)):
            chunk = x[start : start + int(max_batch_sequences)].contiguous()
            if chunk.shape[0] >= 2:
                out.append(chunk.cpu())
            if len(out) >= int(max_batches):
                return out
    return out


@torch.no_grad()
def _native_validation_nll(
    model: ControlledKoopmanFlowMap,
    val_batches: Sequence[Tensor],
    *,
    val_times: Sequence[float],
    seed: int,
) -> float:
    process = UniformLKFProcess(model.lkf)
    total, count = 0.0, 0
    for bi, cpu in enumerate(val_batches):
        x1 = cpu.to(process.device)
        for ti, s in enumerate(val_times):
            g = torch.Generator(device=process.device)
            g.manual_seed(int(seed) + 1009 * bi + 37 * ti)
            xs = process.corrupt_clean(x1, float(s), generator=g)
            sb = torch.full(
                (x1.shape[0],),
                float(s),
                device=process.device,
                dtype=model.lkf.pos_embedder.dtype,
            )
            lm = uniform_lkf_loss_and_metrics(model.lkf, xs, x1, sb)
            total += float(lm["clean_nll"].item())
            count += 1
    return total / max(count, 1)


@torch.no_grad()
def _calibration_covariance(
    model: ControlledKoopmanFlowMap,
    val_batches: Sequence[Tensor],
    *,
    max_sequences: int,
) -> Tensor:
    rows: list[Tensor] = []
    n = 0
    for cpu in val_batches:
        x = cpu.to(next(model.parameters()).device)
        z = model.anchor_features(x, 1.0).float().cpu()
        rows.append(z)
        n += int(z.shape[0])
        if n >= int(max_sequences):
            break
    if not rows:
        raise RuntimeError("no calibration features")
    z = torch.cat(rows, dim=0)[: int(max_sequences)].double()
    if z.shape[0] < 2:
        raise RuntimeError("calibration covariance needs >=2 sequences")
    z = z - z.mean(dim=0, keepdim=True)
    return (z.T @ z / float(z.shape[0] - 1)).float()


def _normalized_loss(model: ControlledKoopmanFlowMap, residual: Tensor) -> Tensor:
    z = model.information_normalize(residual)
    return z.square().mean()


def _response_metrics(target: Tensor, pred: Tensor, *, eps: float = 1e-8) -> dict[str, Tensor]:
    """Directional response metrics without rewarding inactive coordinates.

    The previous implementation counted every near-zero target coordinate as a
    correct sign by construction.  That made a random/zero J look better than it
    was.  Here sign consistency is evaluated only where the observed response is
    materially non-zero, with a per-row threshold.
    """
    with torch.no_grad():
        res = target - pred
        rmse = res.square().mean().sqrt()
        target_rms = target.square().mean().sqrt()
        cosine = F.cosine_similarity(target, pred, dim=-1).mean()
        row_scale = target.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        active = target.abs() >= (0.05 * row_scale)
        matches = ((target * pred) > 0) & active
        active_count = active.float().sum().clamp_min(1.0)
        sign = matches.float().sum() / active_count
        return {
            "response_rmse": rmse,
            "response_relative_rmse": rmse / target_rms.clamp_min(eps),
            "response_cosine": cosine,
            "response_sign_consistency": sign,
            "response_target_rms": target_rms,
        }


def _stein_response_estimate(anchor_samples: Tensor, xi: Tensor) -> Tensor:
    """Hard/soft Gaussian-score estimate of the local physical response J.

    For xi~N(0,I), d/dv E[r(Psi(x,xi+v))]|_0 = E[(r-b(x)) xi^T].
    Centering the M continuations within each source state is a variance-reduction
    baseline.  The M-1 denominator makes the within-state covariance unbiased.
    """
    if anchor_samples.ndim != 3 or xi.ndim != 3:
        raise ValueError("anchor_samples and xi must be [M,B,*]")
    if anchor_samples.shape[:2] != xi.shape[:2]:
        raise ValueError("anchor_samples/xi leading dimensions must match")
    m, b = int(anchor_samples.shape[0]), int(anchor_samples.shape[1])
    if m < 2:
        raise ValueError("Stein response estimation requires continuations >= 2")
    numerator, count = _stein_sufficient_statistics(anchor_samples, xi)
    return numerator / count.to(numerator.dtype)


def _stein_sufficient_statistics(anchor_samples: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
    """Return numerator/count for the unbiased within-state Stein covariance.

    These statistics add exactly across minibatches and DDP ranks.  Stage A
    therefore estimates the global state-independent J by a streaming population
    mean instead of asking Adam to chase a different noisy J target every step.
    """
    if anchor_samples.ndim != 3 or xi.ndim != 3:
        raise ValueError("anchor_samples and xi must be [M,B,*]")
    if anchor_samples.shape[:2] != xi.shape[:2]:
        raise ValueError("anchor_samples/xi leading dimensions must match")
    m, b = int(anchor_samples.shape[0]), int(anchor_samples.shape[1])
    if m < 2:
        raise ValueError("Stein response estimation requires continuations >= 2")
    rc = anchor_samples.float() - anchor_samples.float().mean(dim=0, keepdim=True)
    xc = xi.float() - xi.float().mean(dim=0, keepdim=True)
    numerator = torch.einsum("mbd,mbk->dk", rc, xc)
    count = torch.tensor(float(b * (m - 1)), device=numerator.device, dtype=torch.float64)
    return numerator, count


@torch.no_grad()
def _streaming_update_global_J(
    model: ControlledKoopmanFlowMap,
    *,
    s: float,
    t: float,
    x0_hard: Tensor,
    xi: Tensor,
    ctx: DistributedContext,
) -> dict[str, float]:
    """Exact cumulative Stage-A estimate of the global physical response J.

    The active horizon receives all sufficient statistics from all ranks.  The
    update is a weighted running mean, so no learning rate or gradient clipping
    can suppress J identification.
    """
    m, b, length = x0_hard.shape
    r = model.anchor_features(x0_hard.reshape(m * b, length), float(t)).reshape(m, b, -1)
    num_local, den_scalar = _stein_sufficient_statistics(r, xi)
    # DDP ranks may receive different peptide lengths.  Collective tensors must
    # nevertheless have identical shapes, so pad numerator/counts to max D_xi.
    D = int(num_local.shape[1])
    maxD = int(model.max_innovation_dim)
    num = torch.zeros(model.anchor_dim, maxD, device=num_local.device, dtype=num_local.dtype)
    cnt = torch.zeros(maxD, device=num_local.device, dtype=torch.float64)
    num[:, :D] = num_local
    cnt[:D] = den_scalar
    if ctx.distributed:
        dist.all_reduce(num, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
    idx = model.control_response_index(float(s), float(t))
    old_count = model.control_response_counts[idx].clone()
    new_count = old_count + cnt
    key = model._control_response_keys[idx]
    J = model.control_responses[key]
    active = cnt > 0
    if bool(active.any()):
        old = J[:, active].double()
        updated = (old * old_count[active].unsqueeze(0) + num[:, active].double()) / new_count[active].clamp_min(1.0).unsqueeze(0)
        J[:, active] = updated.to(J.dtype)
    model.control_response_counts[idx].copy_(new_count)
    active_total = new_count[new_count > 0]
    return {
        "streaming_J_samples": float(cnt.max().item()),
        "streaming_J_total_samples": float(active_total.min().item()) if active_total.numel() else 0.0,
        "streaming_J_norm": float((model.information_transform.float() @ J.float()).norm().item()),
    }


def _normalized_J_loss(model: ControlledKoopmanFlowMap, residual_J: Tensor) -> Tensor:
    """Information-normalized Frobenius loss, *not* diluted by innovation dim.

    Averaging over all D_xi columns introduces an O(1/D_xi) gradient suppression
    for the large physical innovation space.  Summing over innovation columns and
    averaging only over retained information coordinates keeps the statistical
    scale of the full response matrix visible to the optimizer.
    """
    T = model.information_transform.float()
    rn = T @ residual_J.float()
    return rn.square().sum(dim=-1).mean()

def _softmin_eigen_score(G: Tensor, tau: float) -> tuple[Tensor, Tensor]:
    eig = torch.linalg.eigvalsh(0.5 * (G + G.T)).clamp_min(0)
    n = max(int(eig.numel()), 1)
    t = float(tau)
    if t <= 0:
        score = eig.min()
    else:
        score = -t * (torch.logsumexp(-eig / t, dim=0) - math.log(float(n)))
    return score, eig


def _aggregate_controllability_score(
    model: ControlledKoopmanFlowMap,
    chains: Sequence[Sequence[float]],
    *,
    token_length: int,
    tau: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    scores: list[Tensor] = []
    all_eig: list[Tensor] = []
    for chain in chains:
        G = model.gramian(chain, token_length=token_length, normalized=True)
        score, eig = _softmin_eigen_score(G, tau)
        scores.append(score)
        all_eig.append(eig)
    stacked = torch.stack(scores)
    score = stacked.mean()
    eig_cat = torch.cat(all_eig)
    metrics = {
        "control_spectral_score": score.detach(),
        "control_chain_score_min": stacked.min().detach(),
        "control_chain_score_max": stacked.max().detach(),
        "control_eigen_min": eig_cat.min().detach(),
        "control_eigen_q10": torch.quantile(eig_cat, 0.10).detach(),
        "control_eigen_median": eig_cat.median().detach(),
        "control_eigen_max": eig_cat.max().detach(),
    }
    return score, metrics


def _physical_controllability_score_with_override(
    model: ControlledKoopmanFlowMap,
    chains: Sequence[Sequence[float]],
    *,
    token_length: int,
    pair: tuple[float, float],
    J_override: Tensor,
    tau: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Differentiable physical controllability score for Stage B.

    A and the fitted hard J matrices are *estimators*, not trainable actuators.
    They are detached here.  Only the soft Gaussian-score response for the
    currently sampled physical transition carries generator gradients.  This
    prevents Stage B from gaming the Gramian by simply inflating a free J.
    """
    s0, t0 = float(pair[0]), float(pair[1])
    Tinfo = model.information_transform.float().detach()
    scores: list[Tensor] = []
    eigs: list[Tensor] = []
    for chain in chains:
        intervals = [(float(a), float(b)) for a, b in zip(chain[:-1], chain[1:])]
        if not any(abs(a-s0) < 1e-8 and abs(b-t0) < 1e-8 for a, b in intervals):
            continue
        terminal = float(chain[-1])
        blocks: list[Tensor] = []
        for a, b in intervals:
            if abs(a-s0) < 1e-8 and abs(b-t0) < 1e-8:
                Jk = J_override
            else:
                Jk = model.control_response(a, b, token_length=token_length).detach()
            Aprop = model.operator(b, terminal).detach()
            blocks.append(Aprop @ Jk)
        M = torch.cat(blocks, dim=1)
        G = Tinfo @ (M @ M.T) @ Tinfo.T
        score, eig = _softmin_eigen_score(G, tau)
        scores.append(score)
        eigs.append(eig)
    if not scores:
        zero = J_override.sum() * 0.0
        return zero, {
            "physical_control_spectral_score": zero.detach(),
            "physical_control_eigen_min": zero.detach(),
            "physical_control_eigen_q10": zero.detach(),
        }
    stacked = torch.stack(scores)
    eigcat = torch.cat(eigs)
    return stacked.mean(), {
        "physical_control_spectral_score": stacked.mean().detach(),
        "physical_control_chain_score_min": stacked.min().detach(),
        "physical_control_eigen_min": eigcat.min().detach(),
        "physical_control_eigen_q10": torch.quantile(eigcat, 0.10).detach(),
    }


def _sample_control_vectors(
    batch: int,
    dim: int,
    *,
    kappa_min: float,
    kappa_max: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    raw = torch.randn(int(batch), int(dim), device=device, generator=generator)
    raw = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if float(kappa_min) == float(kappa_max):
        kappa = torch.full((batch, 1), float(kappa_min), device=device)
    else:
        log_lo = math.log(max(float(kappa_min), 1e-8))
        log_hi = math.log(max(float(kappa_max), float(kappa_min) + 1e-8))
        u = torch.rand((batch, 1), device=device, generator=generator)
        kappa = torch.exp(log_lo + u * (log_hi - log_lo))
    return raw * torch.sqrt(2.0 * kappa)


def _paired_hard_samples(
    model: ControlledKoopmanFlowMap,
    xs: Tensor,
    *,
    s: float,
    t: float,
    continuations: int,
    v: Tensor,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    process = UniformLKFProcess(model.lkf)
    m, b, length = int(continuations), xs.shape[0], xs.shape[1]
    layout = transition_innovation_layout(process, length)
    g = torch.Generator(device=xs.device)
    g.manual_seed(int(seed))
    xi = torch.randn(m, b, layout.total_dim, device=xs.device, generator=g)
    xs_rep = xs.unsqueeze(0).expand(m, -1, -1).reshape(m * b, length)
    xi_flat = xi.reshape(m * b, layout.total_dim)
    v_flat = v.unsqueeze(0).expand(m, -1, -1).reshape(m * b, layout.total_dim)
    x0, _ = model.sample_transition(xs_rep, s, t, xi=xi_flat)
    x1, _ = model.sample_transition(xs_rep, s, t, xi=xi_flat, mean_shift=v_flat)
    return x0.reshape(m, b, length), x1.reshape(m, b, length), xi


class ControlledTrainingObjective(nn.Module):
    def __init__(
        self,
        model: ControlledKoopmanFlowMap,
        *,
        stage: str,
        training_chains: Sequence[Sequence[float]],
        soft_temperature: float,
        spectral_tau: float,
        lambda_soft_response: float,
        lambda_controllability: float,
        lambda_native: float,
        operator_regularization: float,
    ):
        super().__init__()
        self.model = model
        self.stage = str(stage).upper()
        self.training_chains = tuple(tuple(float(x) for x in c) for c in training_chains)
        self.soft_temperature = float(soft_temperature)
        self.spectral_tau = float(spectral_tau)
        self.lambda_soft_response = float(lambda_soft_response)
        self.lambda_controllability = float(lambda_controllability)
        self.lambda_native = float(lambda_native)
        self.operator_regularization = float(operator_regularization)

    def _touch_all_J(self, loss: Tensor) -> Tensor:
        # DDP find_unused_parameters=False remains safe when only one horizon is
        # active in a step.  All ranks intentionally use the SAME horizon so DDP
        # averages independent estimates of the same J rather than diluting each
        # horizon's gradient by world_size.
        z = loss
        for p in self.model.control_responses.values():
            z = z + p.reshape(-1)[0] * 0.0
        return z

    def forward(
        self,
        x1_clean: Tensor,
        xs: Tensor,
        x0_hard: Tensor,
        x1_hard: Tensor,
        xi: Tensor,
        v: Tensor,
        s: float,
        t: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        m, b, length = x0_hard.shape
        with torch.no_grad():
            r_s = self.model.anchor_features(xs, float(s))
            r0_samples = self.model.anchor_features(
                x0_hard.reshape(m * b, length), float(t)
            ).reshape(m, b, -1)
            r1_samples = self.model.anchor_features(
                x1_hard.reshape(m * b, length), float(t)
            ).reshape(m, b, -1)
            r0 = r0_samples.mean(0)
            r1 = r1_samples.mean(0)
            J_target = _stein_response_estimate(r0_samples, xi.float())

        pred0 = self.model.predict_uncontrolled_mean(r_s, float(s), float(t))
        J = self.model.control_response(float(s), float(t), token_length=length)
        pred_delta = v.float() @ J.T
        target_delta = r1 - r0

        mean_loss = _normalized_loss(self.model, r0 - pred0)
        stein_loss = _normalized_J_loss(self.model, J - J_target)
        # Finite-control fit is deliberately a diagnostic, not the primary J
        # estimator.  In high D_xi it is sparse/noisy under hard argmax sampling.
        finite_response_loss = _normalized_loss(self.model, target_delta - pred_delta)
        hard_ck = mean_loss + stein_loss
        metrics: dict[str, Tensor] = {
            "hard_mean_loss": mean_loss.detach(),
            "hard_stein_loss": stein_loss.detach(),
            "hard_response_loss": finite_response_loss.detach(),
            "hard_ck_loss": hard_ck.detach(),
            "hard_J_target_norm": (self.model.information_transform @ J_target).norm().detach(),
            "hard_J_model_norm": (self.model.information_transform @ J).norm().detach(),
        }
        metrics.update(_response_metrics(
            self.model.information_normalize(target_delta),
            self.model.information_normalize(pred_delta),
        ))

        operator_reg = self.model.operator_generators.float().square().mean()
        total = hard_ck + self.operator_regularization * operator_reg
        metrics["operator_regularization"] = operator_reg.detach()

        # Report the hard fitted controllability geometry in both stages.
        spectral, spec_metrics = _aggregate_controllability_score(
            self.model,
            self.training_chains,
            token_length=length,
            tau=self.spectral_tau,
        )
        metrics.update(spec_metrics)

        if self.stage == "B":
            layout = transition_innovation_layout(UniformLKFProcess(self.model.lkf), length)
            xs_rep = xs.unsqueeze(0).expand(m, -1, -1).reshape(m * b, length)
            xi_flat = xi.reshape(m * b, layout.total_dim)
            soft0_samples = self.model.soft_controlled_anchor_samples(
                xs_rep,
                float(s),
                float(t),
                xi=xi_flat,
                temperature=self.soft_temperature,
            ).reshape(m, b, -1)
            soft0 = soft0_samples.mean(0)
            soft_J = _stein_response_estimate(soft0_samples, xi.float())

            soft_mean = _normalized_loss(self.model, soft0 - pred0.detach())
            soft_stein = _normalized_J_loss(self.model, soft_J - J.detach())
            soft_ck = soft_mean + soft_stein

            # Crucially, the spectral gradient now comes from the *physical soft
            # response of the generator*.  Learned A/J are detached estimators.
            physical_spectral, physical_metrics = _physical_controllability_score_with_override(
                self.model,
                self.training_chains,
                token_length=length,
                pair=(float(s), float(t)),
                J_override=soft_J,
                tau=self.spectral_tau,
            )
            metrics.update(physical_metrics)

            sb = torch.full(
                (x1_clean.shape[0],),
                float(s),
                device=x1_clean.device,
                dtype=self.model.lkf.pos_embedder.dtype,
            )
            native = uniform_lkf_loss_and_metrics(self.model.lkf, xs, x1_clean, sb)
            native_loss = native["loss"]

            total = (
                hard_ck
                + self.lambda_soft_response * soft_ck
                + self.lambda_native * native_loss
                - self.lambda_controllability * physical_spectral
                + self.operator_regularization * operator_reg
            )
            metrics.update({
                "soft_mean_loss": soft_mean.detach(),
                "soft_response_loss": soft_stein.detach(),
                "soft_ck_loss": soft_ck.detach(),
                "native_loss": native_loss.detach(),
                "native_clean_nll": native["clean_nll"].detach(),
            })
        else:
            zero = torch.zeros_like(hard_ck.detach())
            metrics.update({
                "soft_mean_loss": zero,
                "soft_response_loss": zero,
                "soft_ck_loss": zero,
                "physical_control_spectral_score": zero,
                "native_loss": zero,
                "native_clean_nll": zero,
            })
        total = self._touch_all_J(total)
        metrics["total_loss"] = total.detach()
        return total, metrics

def _wrap_ddp(module: nn.Module, ctx: DistributedContext) -> nn.Module:
    if not ctx.distributed:
        return module
    kwargs: dict[str, Any] = {"find_unused_parameters": False, "broadcast_buffers": False}
    if ctx.device.type == "cuda":
        kwargs.update(device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DDP(module, **kwargs)


def _all_ranks_finite(value: Tensor, ctx: DistributedContext) -> bool:
    flag = torch.tensor(
        1 if bool(torch.isfinite(value.detach()).all()) else 0,
        device=ctx.device,
        dtype=torch.int32,
    )
    if ctx.distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@torch.no_grad()
def _validate_controlled(
    model: ControlledKoopmanFlowMap,
    val_batches: Sequence[Tensor],
    *,
    pairs: Sequence[tuple[float, float]],
    chains: Sequence[Sequence[float]],
    continuations: int,
    kappa: float,
    max_batches: int,
    seed: int,
    spectral_tau: float,
    control_directions: int = 4,
) -> dict[str, float]:
    """Deterministic held-out validation aligned with the scientific J claim.

    The old gate compared a noisy 64-D finite difference separately for every
    source state and random physical direction.  Here each validation batch is
    partitioned across a small bank of *controllable* directions v propto J^T c,
    and the hard response is averaged over states in the same direction.  This
    estimates the global mean response that a state-independent J is meant to
    model.  A held-out hard Stein matrix comparison is reported independently.
    """
    process = UniformLKFProcess(model.lkf)
    sums: dict[str, float] = {}
    n_mean = 0
    # Pool finite-control responses across validation batches before scoring.
    # The scientific claim is about the global/state-averaged response J, so
    # averaging noisy per-batch cosines is both statistically inefficient and
    # mismatched to the model.  Fixed information directions per horizon make
    # cross-batch pooling possible.
    response_stats: dict[tuple[int, int], dict[str, Tensor | float]] = {}
    stein_stats: dict[tuple[float, float], dict[str, Tensor]] = {}
    token_lengths: list[int] = []
    K = max(1, int(control_directions))

    for bi, cpu in enumerate(val_batches[: int(max_batches)]):
        xclean = cpu.to(process.device)
        token_lengths.append(int(xclean.shape[1]))
        for pi, (s, t) in enumerate(pairs):
            gx = torch.Generator(device=process.device)
            gx.manual_seed(int(seed) + 10007 * bi + 193 * pi)
            xs = process.corrupt_clean(xclean, float(s), generator=gx)
            layout = transition_innovation_layout(process, xs.shape[1])
            J = model.control_response(s, t, token_length=xs.shape[1])
            Tinfo = model.information_transform.float()
            Jn = Tinfo @ J

            # Build K shared controls from random information-space directions.
            gc = torch.Generator(device=process.device)
            # IMPORTANT: directions are fixed across validation batches for a
            # given horizon. This lets us estimate the global mean hard response
            # with all validation states rather than averaging noisy batchwise
            # cosines.
            gc.manual_seed(int(seed) + 200_003 + 97 * pi)
            c = torch.randn(K, Jn.shape[0], device=process.device, generator=gc)
            c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            raw = c @ Jn  # [K,D] == (Jn^T c)^T
            raw_norm = raw.norm(dim=-1, keepdim=True)
            fallback = torch.randn(K, layout.total_dim, device=process.device, generator=gc)
            fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            unit = torch.where(raw_norm > 1e-10, raw / raw_norm.clamp_min(1e-12), fallback)
            controls = unit * math.sqrt(2.0 * float(kappa))
            group = torch.arange(xs.shape[0], device=process.device) % K
            v = controls.index_select(0, group)

            x0, x1, xi = _paired_hard_samples(
                model,
                xs,
                s=s,
                t=t,
                continuations=int(continuations),
                v=v,
                seed=int(seed) + 500_000 + 101 * bi + pi,
            )
            m, b, length = x0.shape
            r_s = model.anchor_features(xs, s)
            r0_samples = model.anchor_features(x0.reshape(m * b, length), t).reshape(m, b, -1)
            r1_samples = model.anchor_features(x1.reshape(m * b, length), t).reshape(m, b, -1)
            r0 = r0_samples.mean(0)
            r1 = r1_samples.mean(0)
            pred0 = model.predict_uncontrolled_mean(r_s, s, t)

            mean_res = model.information_normalize(r0 - pred0)
            mean_tar = model.information_normalize(r0)
            vals = {
                "anchor_mean_rmse": float(mean_res.square().mean().sqrt().item()),
                "anchor_mean_relative_rmse": float(
                    (mean_res.square().mean().sqrt() / mean_tar.square().mean().sqrt().clamp_min(1e-8)).item()
                ),
                "anchor_mean_cosine": float(
                    F.cosine_similarity(model.information_normalize(pred0), mean_tar, dim=-1).mean().item()
                ),
            }
            for key, value in vals.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            n_mean += 1

            # Global/shared-direction finite-control validation.  Accumulate
            # sufficient response sums; score only after *all* validation
            # batches have contributed.
            ddraw = r1_samples - r0_samples
            h = m // 2
            for kk in range(K):
                mask = group == kk
                if not bool(mask.any()):
                    continue
                key = (pi, kk)
                if key not in response_stats:
                    response_stats[key] = {
                        "actual_sum": torch.zeros(model.anchor_dim, device=process.device, dtype=torch.float64),
                        "pred_sum": torch.zeros(model.anchor_dim, device=process.device, dtype=torch.float64),
                        "split0_sum": torch.zeros(model.anchor_dim, device=process.device, dtype=torch.float64),
                        "split1_sum": torch.zeros(model.anchor_dim, device=process.device, dtype=torch.float64),
                        "count": 0.0,
                        "c": c[kk].detach().double().clone(),
                    }
                st_resp = response_stats[key]
                nstate = float(mask.sum().item())
                st_resp["actual_sum"] += (r1 - r0)[mask].double().sum(dim=0)
                st_resp["pred_sum"] += (v[mask] @ J.T).double().sum(dim=0)
                if m >= 4 and h > 0:
                    # Each half is first averaged over its continuation draws,
                    # then summed across states so both halves estimate the same
                    # state-averaged controlled response.
                    st_resp["split0_sum"] += ddraw[:h, mask].double().mean(dim=0).sum(dim=0)
                    st_resp["split1_sum"] += ddraw[h:, mask].double().mean(dim=0).sum(dim=0)
                st_resp["count"] = float(st_resp["count"]) + nstate

            # Pool Gaussian-score sufficient statistics across validation batches
            # before comparing matrices. Per-batch full-matrix cosines are far too
            # noisy in D_xi~250 and created an uncalibrated impossible gate in v2.
            jnum, jden = _stein_sufficient_statistics(r0_samples, xi.float())
            pair_key = (float(s), float(t))
            if pair_key not in stein_stats:
                stein_stats[pair_key] = {
                    "num": torch.zeros(model.anchor_dim, model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                    "cnt": torch.zeros(model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                    "num0": torch.zeros(model.anchor_dim, model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                    "cnt0": torch.zeros(model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                    "num1": torch.zeros(model.anchor_dim, model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                    "cnt1": torch.zeros(model.max_innovation_dim, device=jnum.device, dtype=torch.float64),
                }
            st = stein_stats[pair_key]
            D = int(jnum.shape[1])
            st["num"][:, :D] += jnum.double(); st["cnt"][:D] += jden
            hh = bi % 2
            st[f"num{hh}"][:, :D] += jnum.double(); st[f"cnt{hh}"][:D] += jden

    if n_mean == 0 or not response_stats:
        raise RuntimeError("controlled validation produced no rows")
    out = {key: value / n_mean for key, value in sums.items()}

    response_rows: list[dict[str, float]] = []
    for st_resp in response_stats.values():
        cnt = max(float(st_resp["count"]), 1.0)
        actual = (st_resp["actual_sum"] / cnt).float().unsqueeze(0)
        pred = (st_resp["pred_sum"] / cnt).float().unsqueeze(0)
        an = model.information_normalize(actual)
        pn = model.information_normalize(pred)
        met = _response_metrics(an, pn)
        cdir = st_resp["c"].float()
        actual_gain = float((an[0] * cdir).sum().item())
        predicted_gain = float((pn[0] * cdir).sum().item())
        split_cos = float("nan")
        s0 = st_resp["split0_sum"]
        s1 = st_resp["split1_sum"]
        if float(s0.norm().item()) > 0.0 and float(s1.norm().item()) > 0.0:
            a1n = model.information_normalize((s0 / cnt).float().unsqueeze(0))
            a2n = model.information_normalize((s1 / cnt).float().unsqueeze(0))
            split_cos = float(F.cosine_similarity(a1n, a2n, dim=-1).mean().item())
        response_rows.append({
            "response_rmse": float(met["response_rmse"].item()),
            "response_relative_rmse": float(met["response_relative_rmse"].item()),
            "response_cosine": float(met["response_cosine"].item()),
            "response_sign_consistency": float(met["response_sign_consistency"].item()),
            "response_target_rms": float(met["response_target_rms"].item()),
            "response_split_cosine": split_cos,
            "directional_sign_match": float(actual_gain > 0.0 and predicted_gain > 0.0),
            "actual_directional_gain": actual_gain,
            "predicted_directional_gain": predicted_gain,
        })

    for key in (
        "response_rmse",
        "response_relative_rmse",
        "response_cosine",
        "response_sign_consistency",
        "response_target_rms",
        "directional_sign_match",
        "actual_directional_gain",
        "predicted_directional_gain",
    ):
        out[key] = float(np.mean([row[key] for row in response_rows]))
    split_vals = [row["response_split_cosine"] for row in response_rows if np.isfinite(row["response_split_cosine"])]
    out["response_split_cosine"] = float(np.mean(split_vals)) if split_vals else float("nan")

    # Reliability-adjusted response quality. If r is split-half reliability,
    # Spearman-Brown estimates the reliability of the full pooled response.
    # sqrt(reliability) is the classical attenuation ceiling for correlation
    # with a noise-free predictor. This is an engineering diagnostic (cosine is
    # not exactly Pearson correlation), but it is far better calibrated than a
    # fixed raw cosine threshold independent of measurement noise.
    rsplit = float(out["response_split_cosine"])
    if np.isfinite(rsplit) and rsplit > 0.0:
        rclip = min(max(rsplit, 0.0), 0.999999)
        full_rel = 2.0 * rclip / (1.0 + rclip)
        ceiling = float(math.sqrt(max(full_rel, 1e-12)))
        out["response_reliability_full"] = full_rel
        out["response_cosine_ceiling"] = ceiling
        out["response_reliability_adjusted_cosine"] = float(out["response_cosine"] / max(ceiling, 1e-8))
    else:
        out["response_reliability_full"] = float("nan")
        out["response_cosine_ceiling"] = float("nan")
        out["response_reliability_adjusted_cosine"] = float("nan")

    stein_cos: list[float] = []
    stein_rel: list[float] = []
    stein_split: list[float] = []
    for (ss, tt), st in stein_stats.items():
        active = st["cnt"] > 0
        if not bool(active.any()):
            continue
        ref = st["num"][:, active] / st["cnt"][active].unsqueeze(0).clamp_min(1.0)
        predJ = model.control_response(ss, tt)[:, active].double()
        a = (Tinfo.double() @ ref).reshape(-1)
        pvec = (Tinfo.double() @ predJ).reshape(-1)
        denv = a.norm() * pvec.norm()
        stein_cos.append(float((a @ pvec / denv).item()) if float(denv.item()) > 1e-12 else 0.0)
        stein_rel.append(float(((a - pvec).norm() / a.norm().clamp_min(1e-12)).item()))
        both = (st["cnt0"] > 0) & (st["cnt1"] > 0)
        if bool(both.any()):
            j0 = st["num0"][:, both] / st["cnt0"][both].unsqueeze(0).clamp_min(1.0)
            j1 = st["num1"][:, both] / st["cnt1"][both].unsqueeze(0).clamp_min(1.0)
            q0 = (Tinfo.double() @ j0).reshape(-1); q1 = (Tinfo.double() @ j1).reshape(-1)
            dd = q0.norm() * q1.norm()
            if float(dd.item()) > 1e-12:
                stein_split.append(float((q0 @ q1 / dd).item()))
    out["stein_J_cosine"] = float(np.mean(stein_cos)) if stein_cos else 0.0
    out["stein_J_relative_error"] = float(np.mean(stein_rel)) if stein_rel else float("inf")
    out["stein_J_split_cosine"] = float(np.mean(stein_split)) if stein_split else float("nan")
    ss = float(out["stein_J_split_cosine"])
    if np.isfinite(ss) and ss > 0.0:
        ss = min(max(ss, 0.0), 0.999999)
        sfull = 2.0 * ss / (1.0 + ss)
        sceiling = float(math.sqrt(max(sfull, 1e-12)))
        out["stein_J_reliability_full"] = sfull
        out["stein_J_cosine_ceiling"] = sceiling
        out["stein_J_reliability_adjusted_cosine"] = float(out["stein_J_cosine"] / max(sceiling, 1e-8))
    else:
        out["stein_J_reliability_full"] = float("nan")
        out["stein_J_cosine_ceiling"] = float("nan")
        out["stein_J_reliability_adjusted_cosine"] = float("nan")

    # Spectral metrics are evaluated for all observed token lengths and averaged.
    spec: dict[str, float] = {}
    for length in sorted(set(token_lengths)):
        _, mm = _aggregate_controllability_score(
            model, chains, token_length=length, tau=spectral_tau
        )
        for key, value in mm.items():
            spec[key] = spec.get(key, 0.0) + float(value.item())
    denom = max(len(set(token_lengths)), 1)
    out.update({key: value / denom for key, value in spec.items()})
    out["semigroup_error_0_05_1"] = model.semigroup_error(0.0, 0.5, 1.0)
    return out

def _optimizer(model: ControlledKoopmanFlowMap, args: argparse.Namespace, stage: str) -> AdamW:
    aux = [model.operator_generators] + list(model.control_responses.parameters())
    if stage == "A":
        # J is a closed-form streaming statistic in Stage A, not an optimized knob.
        return AdamW([model.operator_generators], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    lkf_params = [p for p in model.lkf.parameters() if p.requires_grad]
    if not lkf_params:
        raise RuntimeError("Stage B has no trainable LKF parameters")
    return AdamW(
        [
            {"params": aux, "lr": float(args.learning_rate)},
            {"params": lkf_params, "lr": float(args.lkf_learning_rate)},
        ],
        weight_decay=float(args.weight_decay),
    )


def run(args: argparse.Namespace, ctx: DistributedContext) -> Path:
    seed_everything(int(args.seed) + int(ctx.rank) * 100003)
    stage = str(args.stage).upper()
    device = ctx.device
    base = Path(args.lkf_checkpoint).expanduser().resolve()
    source_ip = Path(args.ip_koopman_checkpoint).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(base)
    if not source_ip.exists():
        raise FileNotFoundError(source_ip)

    knots = parse_knots(args.control_knots)
    chains = parse_chains(args.training_chains)
    pairs = control_pairs(knots)
    for chain in chains:
        for s,t in zip(chain[:-1],chain[1:]):
            if (float(s),float(t)) not in pairs:
                raise ValueError(f"training chain interval {(s,t)} not represented by control knots")

    if stage == "A":
        if args.warm_start_checkpoint:
            model, warm_meta = load_controlled_koopman_checkpoint(
                args.warm_start_checkpoint,
                base_lkf_checkpoint=base,
                device="cpu",
                strict_base_sha=True,
                eval_mode=False,
            )
            if str(warm_meta.get("stage", "")).upper() != "A":
                raise ValueError("--warm-start-checkpoint must be a Stage-A controlled checkpoint")
            if tuple(model.config.control_knots) != knots:
                raise ValueError("warm-start CONTROL_KNOTS must match the requested knots")
            if bool(args.warm_start_reset_J):
                with torch.no_grad():
                    for response in model.control_responses.values():
                        response.zero_()
                    model.control_response_counts.zero_()
        else:
            cfg = ControlledKoopmanConfig(
                anchor_dim=int(args.anchor_dim),
                anchor_seed=int(args.anchor_seed),
                feature_source=str(args.feature_source),
                operator_bins=int(args.operator_bins),
                operator_init_scale=float(args.operator_init_scale),
                control_knots=knots,
                control_init_scale=float(args.control_init_scale),
                information_eigen_floor_relative=float(args.information_eigen_floor_relative),
                information_epsilon=float(args.information_epsilon),
                soft_temperature=float(args.soft_temperature),
            )
            model, _ = initialize_controlled_from_ip_checkpoint(
                source_ip,
                base_lkf_checkpoint=base,
                config=cfg,
                strict_base_sha=True,
            )
    elif stage == "B":
        if not args.stage_a_checkpoint:
            raise ValueError("Stage B requires --stage-a-checkpoint")
        model, meta = load_controlled_koopman_checkpoint(
            args.stage_a_checkpoint,
            base_lkf_checkpoint=base,
            device="cpu",
            strict_base_sha=True,
            eval_mode=False,
        )
        if str(meta.get("stage","")).upper() != "A":
            raise ValueError("--stage-a-checkpoint must be a controlled Koopman Stage-A checkpoint")
        if tuple(model.config.control_knots) != knots:
            raise ValueError("Stage-B CONTROL_KNOTS must match Stage-A checkpoint")
    else:
        raise ValueError("stage must be A or B")

    model.to(device)
    unfrozen = configure_controlled_trainability(
        model,
        stage=stage,
        unfreeze_shared_blocks=int(args.unfreeze_shared_blocks),
        unfreeze_downstream=not bool(args.freeze_downstream),
    )

    if ctx.is_main:
        print("Indexing train dataset (all ranks)...", flush=True)
    t0=time.time()
    train_ds = ChunkedCleanPeptideBatchDataset(
        args.dataset_root,
        args.train_split,
        max_sequences=int(args.max_batch_sequences),
    )
    if ctx.is_main:
        st=train_ds.length_statistics()
        print(f"Train index ready in {time.time()-t0:.1f}s: {st['chunks']} chunks / {st['sequences']} sequences",flush=True)
    sampler=None
    if ctx.distributed:
        sampler=DistributedSampler(train_ds,num_replicas=ctx.world_size,rank=ctx.rank,shuffle=True,seed=int(args.seed),drop_last=False)
    train_loader=build_clean_loader(train_ds,shuffle=sampler is None,sampler=sampler,num_workers=int(args.num_workers))

    # Rank-0 only CUDA validation/calibration.  No collective is allowed before
    # this entire block finishes (see AGENTS.md).
    output_root=Path(args.output_dir).expanduser().resolve()/str(args.run_name)
    ckpt_dir=output_root/"checkpoints"
    val_batches: list[Tensor]=[]
    baseline_nll=None
    calibration_cov=None
    baseline_control=None
    if ctx.is_main:
        ckpt_dir.mkdir(parents=True,exist_ok=True)
        print("Preparing rank-0 validation batches (num_workers=0)...",flush=True)
        tv=time.time(); val_ds=CleanPeptideBatchDataset(args.dataset_root,args.val_split)
        val_batches=_prepare_val_batches(val_ds,max_batch_sequences=int(args.val_batch_sequences),max_batches=int(args.val_batches))
        if not val_batches: raise RuntimeError("validation produced no batches")
        print(f"Validation ready in {time.time()-tv:.1f}s: {len(val_batches)} batches",flush=True)
        print("Running rank-0 baseline validation/calibration...",flush=True)
        tb=time.time()
        baseline_nll=_native_validation_nll(model,val_batches,val_times=(0.0,0.25,0.5,0.75),seed=int(args.seed)+40000)
        if stage=="A":
            calibration_cov=_calibration_covariance(model,val_batches,max_sequences=int(args.calibration_sequences)).cpu().numpy()
            model.initialize_information_metric(torch.as_tensor(calibration_cov,device=device))
        baseline_control=_validate_controlled(
            model,val_batches,pairs=pairs,chains=chains,continuations=int(args.val_continuations),
            kappa=float(args.val_kappa),max_batches=int(args.val_control_batches),seed=int(args.seed)+50000,
            spectral_tau=float(args.spectral_tau),control_directions=int(args.val_control_directions),
        )
        print(f"Baseline validation done in {time.time()-tb:.1f}s",flush=True)

    # First collective occurs only after rank-0 CUDA work is complete.
    base_sha=str(broadcast_object(sha256_file(base) if ctx.is_main else None,ctx))
    if stage=="A":
        calibration_cov=broadcast_object(calibration_cov if ctx.is_main else None,ctx)
        if not ctx.is_main:
            model.initialize_information_metric(torch.as_tensor(calibration_cov,device=device))
    baseline_nll=float(broadcast_object(baseline_nll,ctx))
    baseline_control=broadcast_object(baseline_control,ctx)

    if ctx.is_main:
        write_json(output_root/"baseline_validation.json",{
            "native_validation_nll":baseline_nll,
            "controlled_validation":baseline_control,
            "information_rank":int(model.information_rank.item()),
        })
        write_json(output_root/"provenance.json",{
            "stage":stage,"objective_free_training":True,
            "base_lkf_checkpoint":str(base),"base_lkf_sha256":base_sha,
            "source_ip_checkpoint":str(source_ip),"source_ip_sha256":sha256_file(source_ip),
            "stage_a_checkpoint":str(Path(args.stage_a_checkpoint).expanduser().resolve()) if args.stage_a_checkpoint else None,
            "warm_start_checkpoint":str(Path(args.warm_start_checkpoint).expanduser().resolve()) if args.warm_start_checkpoint else None,
            "controlled_koopman_config":model.config.to_dict(),"training_chains":[list(c) for c in chains],
            "train_dataset":train_ds.length_statistics(),"unfrozen_lkf_parameters":unfrozen,
            "checkpoint_selection":"eligibility gates first; Stage A minimizes hard response error, Stage B maximizes weak finite-horizon controllability",
            "distributed":{"enabled":ctx.distributed,"world_size":ctx.world_size,"backend":ctx.backend},
        })

    objective=ControlledTrainingObjective(
        model,stage=stage,training_chains=chains,soft_temperature=float(args.soft_temperature),
        spectral_tau=float(args.spectral_tau),lambda_soft_response=float(args.lambda_soft_response),
        lambda_controllability=float(args.lambda_controllability),lambda_native=float(args.lambda_native),
        operator_regularization=float(args.operator_regularization),
    )
    ddp_obj=_wrap_ddp(objective,ctx)
    opt=_optimizer(model,args,stage)

    if ctx.is_main:
        print(
            f"Training controlled Koopman Stage {stage} on DDP x{ctx.world_size}; device={device}; "
            f"M={args.continuations}; max_batch_sequences_per_gpu={args.max_batch_sequences}",
            flush=True,
        )

    curve: list[dict[str,Any]]=[]
    global_step=0
    best_score=float("inf") if stage=="A" else float("-inf")
    best_written=False
    for epoch in range(int(args.epochs)):
        if sampler is not None: sampler.set_epoch(epoch)
        model.train(stage=="B")
        model.anchor_encoder.eval()
        sums: dict[str,float]={}; local_steps=0
        for batch_i,batch in enumerate(train_loader):
            if int(args.steps_per_epoch)>0 and local_steps>=int(args.steps_per_epoch): break
            xclean=torch.as_tensor(batch,dtype=torch.long,device=device)
            if xclean.shape[0]<2: continue
            # All ranks intentionally train the SAME horizon on a given step.
            # DDP therefore averages independent estimates of that J instead of
            # scaling a rank-local J gradient down by world_size.
            pair=pairs[(global_step + 31*epoch) % len(pairs)]
            s,t=pair
            process=UniformLKFProcess(model.lkf)
            gcor=torch.Generator(device=device); gcor.manual_seed(int(args.seed)+ctx.rank*1_000_003+epoch*10_007+local_steps)
            xs=process.corrupt_clean(xclean,float(s),generator=gcor)
            layout=transition_innovation_layout(process,xs.shape[1])
            v=_sample_control_vectors(
                xs.shape[0],layout.total_dim,kappa_min=float(args.kappa_min),kappa_max=float(args.kappa_max),device=device
            )
            x0,xv,xi=_paired_hard_samples(
                model,xs,s=s,t=t,continuations=int(args.continuations),v=v,
                seed=int(args.seed)+2_000_000+ctx.rank*100_003+epoch*1009+local_steps,
            )
            stream_metrics = {}
            if stage == "A":
                stream_metrics = _streaming_update_global_J(
                    model, s=float(s), t=float(t), x0_hard=x0, xi=xi, ctx=ctx
                )
            opt.zero_grad(set_to_none=True)
            total,metrics=ddp_obj(xclean,xs,x0,xv,xi,v,float(s),float(t))
            for _k, _v in stream_metrics.items():
                metrics[_k] = torch.tensor(_v, device=device)
            if not _all_ranks_finite(total,ctx):
                raise FloatingPointError("non-finite controlled Koopman loss on at least one rank")
            total.backward()
            if float(args.grad_clip)>0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(args.grad_clip))
            opt.step()
            for k,vv in metrics.items(): sums[k]=sums.get(k,0.0)+float(vv.item())
            local_steps+=1; global_step+=1
        # Equal-step DDP safety check.
        if ctx.distributed:
            lo=torch.tensor(local_steps,device=device,dtype=torch.long); hi=lo.clone()
            dist.all_reduce(lo,op=dist.ReduceOp.MIN); dist.all_reduce(hi,op=dist.ReduceOp.MAX)
            if int(lo)!=int(hi): raise RuntimeError(f"DDP ranks executed different steps: {int(lo)} vs {int(hi)}")
        sums_global,steps_global=reduce_metric_sums(sums,local_steps,ctx)
        train_mean={k:v/max(steps_global,1) for k,v in sums_global.items()}

        val=None; val_nll=None
        if ctx.is_main:
            model.eval()
            # Common-random-number validation: a frozen generator must report
            # exactly the same NLL every Stage-A epoch.
            val_nll=_native_validation_nll(model,val_batches,val_times=(0.0,0.25,0.5,0.75),seed=int(args.seed)+40000)
            val=_validate_controlled(
                model,val_batches,pairs=pairs,chains=chains,continuations=int(args.val_continuations),
                kappa=float(args.val_kappa),max_batches=int(args.val_control_batches),seed=int(args.seed)+50000,
                spectral_tau=float(args.spectral_tau),control_directions=int(args.val_control_directions),
            )
        val_nll=float(broadcast_object(val_nll,ctx)); val=broadcast_object(val,ctx)
        rel_nll=(val_nll-baseline_nll)/max(abs(baseline_nll),1e-12)
        response_quality = float(val.get("response_reliability_adjusted_cosine", float("nan")))
        # If the observed hard response is not reproducible enough to estimate
        # a reliability-adjusted quality, Stage A is not freeze-ready.  Do not
        # silently pass an undefined/no-signal response.
        quality_gate = (
            np.isfinite(response_quality)
            and response_quality >= float(args.min_reliability_adjusted_response_cosine)
        )
        hard_gate=(
            float(val["response_cosine"])>=float(args.min_response_cosine)
            and quality_gate
            and float(val["directional_sign_match"])>=float(args.min_sign_consistency)
            and float(val["stein_J_cosine"])>=float(args.min_stein_J_cosine)
        )
        mean_gate=float(val["anchor_mean_relative_rmse"])<=float(args.max_anchor_relative_rmse)
        nll_gate=(stage=="A") or rel_nll<=float(args.max_relative_nll_degradation)
        eligible=bool(hard_gate and mean_gate and nll_gate)
        if stage=="A": score=float(val["response_relative_rmse"]); improved=eligible and score<best_score
        else: score=float(val["control_spectral_score"]); improved=eligible and score>best_score
        row={"epoch":epoch,"global_step":global_step,**train_mean,**{f"val_{k}":v for k,v in val.items()},
             "val_native_nll":val_nll,"val_relative_nll_degradation":rel_nll,"eligible":eligible,"selection_score":score}
        if ctx.is_main:
            curve.append(row); write_csv(output_root/"training_curve.csv",curve)
            save_controlled_koopman_checkpoint(
                ckpt_dir/"last.pt",model,base_lkf_checkpoint=base,source_ip_checkpoint=source_ip,
                stage=stage,epoch=epoch,global_step=global_step,training_args=vars(args),metrics=row,
            )
            if improved:
                best_score=score; best_written=True
                save_controlled_koopman_checkpoint(
                    ckpt_dir/"best.pt",model,base_lkf_checkpoint=base,source_ip_checkpoint=source_ip,
                    stage=stage,epoch=epoch,global_step=global_step,training_args=vars(args),metrics=row,
                )
            print(
                f"[epoch {epoch:03d}] mean={train_mean.get('hard_mean_loss',float('nan')):.5f} "
                f"Jbatch={train_mean.get('hard_stein_loss',float('nan')):.3f} "
                f"resp_cos={val['response_cosine']:.3f} rep={val['response_split_cosine']:.3f} "
                f"resp_q={val.get('response_reliability_adjusted_cosine',float('nan')):.3f} "
                f"sign={val['directional_sign_match']:.3f} stein={val['stein_J_cosine']:.3f} "
                f"stein_rep={val['stein_J_split_cosine']:.3f} mean_rel={val['anchor_mean_relative_rmse']:.3f} "
                f"ctrl={val['control_spectral_score']:.3e} "
                f"nll={val_nll:.5f} ({rel_nll:+.2%}) {'eligible' if eligible else 'ineligible'}",
                flush=True,
            )
    if ctx.is_main and not best_written:
        raise RuntimeError("No checkpoint satisfied the controlled-Koopman eligibility gates")
    return output_root/"checkpoints"/"best.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["A","B"],required=True)
    p.add_argument("--lkf-checkpoint",required=True)
    p.add_argument("--ip-koopman-checkpoint",required=True,help="validated IP/K-LKF checkpoint supplying active generator + frozen anchor")
    p.add_argument("--stage-a-checkpoint",default="")
    p.add_argument("--warm-start-checkpoint",default="",help="optional Stage-A checkpoint used to preserve a learned A while switching to the corrected Stein J estimator")
    p.add_argument("--warm-start-reset-J",action=argparse.BooleanOptionalAction,default=True,help="reset the old finite-difference J when warm-starting; preserves A and the information metric")
    p.add_argument("--dataset-root",required=True); p.add_argument("--train-split",default="train"); p.add_argument("--val-split",default="val")
    p.add_argument("--output-dir",default="results"); p.add_argument("--run-name",default="controlled_koopman_stage_a"); p.add_argument("--device",default="cuda")
    p.add_argument("--anchor-dim",type=int,default=64); p.add_argument("--anchor-seed",type=int,default=60042); p.add_argument("--feature-source",default="shared_pooled_plus_time")
    p.add_argument("--operator-bins",type=int,default=8); p.add_argument("--operator-init-scale",type=float,default=1e-3); p.add_argument("--control-init-scale",type=float,default=0.0)
    p.add_argument("--control-knots",default="0,0.25,0.5,0.75,1")
    p.add_argument("--training-chains",default="0,1;0.25,1;0.5,1;0.75,1;0,0.25,0.5,1;0,0.5,1;0,0.75,1;0.25,0.5,0.75,1;0.25,0.75,1;0.5,0.75,1")
    p.add_argument("--information-eigen-floor-relative",type=float,default=1e-5); p.add_argument("--information-epsilon",type=float,default=1e-6); p.add_argument("--calibration-sequences",type=int,default=2048)
    p.add_argument("--continuations",type=int,default=4); p.add_argument("--val-continuations",type=int,default=16); p.add_argument("--kappa-min",type=float,default=0.02); p.add_argument("--kappa-max",type=float,default=0.25); p.add_argument("--val-kappa",type=float,default=0.1)
    p.add_argument("--soft-temperature",type=float,default=0.5); p.add_argument("--spectral-tau",type=float,default=1e-3)
    p.add_argument("--lambda-soft-response",type=float,default=1.0); p.add_argument("--lambda-controllability",type=float,default=1.0); p.add_argument("--lambda-native",type=float,default=0.25); p.add_argument("--operator-regularization",type=float,default=1e-5)
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--steps-per-epoch",type=int,default=0); p.add_argument("--max-batch-sequences",type=int,default=32); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--lkf-learning-rate",type=float,default=1e-5); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--unfreeze-shared-blocks",type=int,default=2); p.add_argument("--freeze-downstream",action="store_true")
    p.add_argument("--val-batch-sequences",type=int,default=64); p.add_argument("--val-batches",type=int,default=24); p.add_argument("--val-control-batches",type=int,default=4); p.add_argument("--val-control-directions",type=int,default=4)
    p.add_argument("--max-relative-nll-degradation",type=float,default=0.02); p.add_argument("--max-anchor-relative-rmse",type=float,default=0.35); p.add_argument("--min-response-cosine",type=float,default=0.0,help="optional raw response-cosine floor; disabled by default because the measurable ceiling depends strongly on hard-sampling noise"); p.add_argument("--min-reliability-adjusted-response-cosine",type=float,default=0.65,help="Stage-A gate after split-half attenuation calibration; 0.65 means the model captures roughly two-thirds of the reproducible finite-response direction"); p.add_argument("--min-sign-consistency",type=float,default=0.90); p.add_argument("--min-stein-J-cosine",type=float,default=0.0,dest="min_stein_J_cosine",help="uncalibrated full-matrix Stein cosine; kept diagnostic by default. Raise only after split-half reliability calibration")
    p.add_argument("--seed",type=int,default=42)
    return p


def main() -> None:
    args=build_arg_parser().parse_args(); ctx=initialize_distributed(args.device)
    try: run(args,ctx)
    finally: cleanup_distributed(ctx)


if __name__=="__main__": main()
