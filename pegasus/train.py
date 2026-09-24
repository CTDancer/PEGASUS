"""Train the Phase-I Koopman-Regularized Uniform-LKF with optional native DDP.

Stage A (default): freeze the pretrained M8 Uniform-LKF and learn only the
objective-free Koopman observable head + continuous generator.

Stage B: initialize from a passed Stage-A Koopman checkpoint, conservatively
unfreeze the LKF tail, and optimize

    L = L_LKF + lambda_K * L_pegasus.

Multi-GPU training is launched with ``torchrun``.  The implementation uses a
``DistributedSampler`` over pre-chunked same-length peptide batches, DDP for all
trainable gradients, and a fixed-coordinate Koopman representation with per-sample L2
normalization. Only rank zero performs validation, logging, and checkpoint writes.

No downstream objective labels are imported or accessed by this program.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import load_koopman_checkpoint, save_koopman_checkpoint, sha256_file
from .distributed import (
    DistributedContext,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
    max_float,
    reduce_metric_sums,
)
from .lkf.data import (
    ChunkedCleanPeptideBatchDataset,
    CleanPeptideBatchDataset,
    build_clean_loader,
    uniform_lkf_loss_and_metrics,
)
from .lkf.uniform_lkf import UniformLKFProcess, load_uniform_lkf_checkpoint
from .model import KoopmanConfig, KoopmanRegularizedLKF, configure_lkf_trainability
from .utils import (
    effective_rank_from_eigenvalues,
    iter_chunks,
    parse_float_list,
    seed_everything,
    write_csv,
    write_json,
)


def _sample_interval(
    *,
    device: torch.device,
    max_s: float,
    source_anchor_prob: float,
    terminal_anchor_prob: float,
    min_delta: float,
) -> tuple[float, float]:
    if torch.rand((), device=device).item() < source_anchor_prob:
        s = 0.0
    else:
        s = float(torch.rand((), device=device).item()) * float(max_s)
    if torch.rand((), device=device).item() < terminal_anchor_prob:
        t = 1.0
    else:
        room = 1.0 - s
        if room <= min_delta:
            t = 1.0
        else:
            t = s + min_delta + float(torch.rand((), device=device).item()) * (room - min_delta)
    t = min(max(t, s + 1e-5), 1.0)
    return s, t


def _continuations(
    process: UniformLKFProcess,
    x_s: Tensor,
    s: float,
    t: float,
    m: int,
) -> Tensor:
    samples = [process.sample_transition(x_s, s, t) for _ in range(int(m))]
    return torch.stack(samples, dim=0)


def _prepare_val_batches(
    dataset: CleanPeptideBatchDataset,
    *,
    num_workers: int,
    max_batch_sequences: int,
    max_batches: int,
) -> list[Tensor]:
    # Always use 0 workers here: this runs after CUDA init / model.to(device).
    # Forking DataLoader workers post-CUDA routinely deadlocks rank-0 while the
    # other ranks busy-wait on the next broadcast (silent multi-minute hang).
    del num_workers
    out: list[Tensor] = []
    loader = build_clean_loader(dataset, shuffle=False, num_workers=0)
    for batch in loader:
        for chunk in iter_chunks(torch.as_tensor(batch, dtype=torch.long), max_batch_sequences):
            out.append(chunk.cpu().contiguous())
            if len(out) >= max_batches:
                return out
    return out


@torch.no_grad()
def _validation_lkf_nll(
    model: KoopmanRegularizedLKF,
    val_batches: list[Tensor],
    *,
    val_times: tuple[float, ...],
    seed: int,
) -> float:
    lkf = model.lkf
    process = UniformLKFProcess(lkf)
    was_training = lkf.training
    lkf.eval()
    values: list[float] = []
    for bi, batch_cpu in enumerate(val_batches):
        x1 = batch_cpu.to(next(lkf.parameters()).device)
        for ti, s in enumerate(val_times):
            gen = torch.Generator(device=x1.device)
            gen.manual_seed(int(seed) + 100003 * bi + 9176 * ti)
            s_batch = torch.full(
                (x1.shape[0],), float(s), device=x1.device, dtype=lkf.pos_embedder.dtype
            )
            xs = process.corrupt_clean(x1, s_batch, generator=gen)
            metrics = uniform_lkf_loss_and_metrics(
                lkf,
                xs,
                x1,
                s_batch,
                router_balance_coef=0.0,
                router_prior_entropy_coef=0.0,
            )
            values.append(float(metrics["clean_nll"].item()))
    if was_training:
        lkf.train()
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def _validation_closure(
    model: KoopmanRegularizedLKF,
    val_batches: list[Tensor],
    *,
    intervals: tuple[tuple[float, float], ...],
    continuations: int,
    seed: int,
    rank_target: float,
) -> dict[str, float]:
    process = UniformLKFProcess(model.lkf)
    was_training = model.training
    model.eval()
    metric_keys = (
        "koopman_loss",
        "closure_relative_rmse",
        "closure_cosine",
        "source_feature_rms",
        "target_individual_feature_rms",
        "target_mean_feature_rms",
        "source_effective_rank",
        "target_effective_rank",
        "rank_hinge_loss",
    )
    rows: list[dict[str, float]] = []
    for ii, (s, t) in enumerate(intervals):
        sums = {key: 0.0 for key in metric_keys}
        count = 0
        for bi, batch_cpu in enumerate(val_batches):
            x1 = batch_cpu.to(next(model.parameters()).device)
            gen = torch.Generator(device=x1.device)
            gen.manual_seed(int(seed) + 1000003 * ii + 8191 * bi)
            xs = process.corrupt_clean(x1, s, generator=gen)
            xt = torch.stack(
                [process.sample_transition(xs, s, t, generator=gen) for _ in range(continuations)],
                dim=0,
            )
            _, metrics = model.closure_loss(
                xs, xt, s, t, rank_regularization_weight=0.0, rank_target=rank_target
            )
            n = int(x1.shape[0])
            count += n
            for key in metric_keys:
                sums[key] += float(metrics[key].item()) * n
        rows.append({key: value / max(count, 1) for key, value in sums.items()})
    if was_training:
        model.train()
    return {
        "val_koopman_loss": float(sum(r["koopman_loss"] for r in rows) / max(len(rows), 1)),
        "val_closure_relative_rmse": float(
            sum(r["closure_relative_rmse"] for r in rows) / max(len(rows), 1)
        ),
        "val_closure_cosine": float(sum(r["closure_cosine"] for r in rows) / max(len(rows), 1)),
        "val_source_feature_rms": float(sum(r["source_feature_rms"] for r in rows) / max(len(rows), 1)),
        "val_target_individual_feature_rms": float(
            sum(r["target_individual_feature_rms"] for r in rows) / max(len(rows), 1)
        ),
        "val_target_mean_feature_rms": float(
            sum(r["target_mean_feature_rms"] for r in rows) / max(len(rows), 1)
        ),
        "val_source_effective_rank": float(sum(r["source_effective_rank"] for r in rows) / max(len(rows), 1)),
        "val_target_effective_rank": float(sum(r["target_effective_rank"] for r in rows) / max(len(rows), 1)),
        "val_rank_hinge_loss": float(sum(r["rank_hinge_loss"] for r in rows) / max(len(rows), 1)),
        "val_semigroup_max_error": float(
            max(model.semigroup_error(s, 0.5 * (s + t), t) for s, t in intervals)
        ),
    }


@torch.no_grad()
def _validation_feature_covariance(
    model: KoopmanRegularizedLKF,
    val_batches: list[Tensor],
    *,
    time_value: float = 1.0,
) -> dict[str, float]:
    """Diagnose raw-head and actual fixed-norm Koopman feature covariance."""
    was_training = model.training
    model.eval()
    raw_chunks: list[Tensor] = []
    feature_chunks: list[Tensor] = []
    device = next(model.parameters()).device
    for batch_cpu in val_batches:
        x = batch_cpu.to(device)
        raw = model.raw_features(x, time_value).float()
        z = model.features(x, time_value).float()
        raw_chunks.append(raw.cpu())
        feature_chunks.append(z.cpu())
    if was_training:
        model.train()
    if not raw_chunks:
        return {}

    def summarize(prefix: str, values: Tensor) -> dict[str, float]:
        centered = values - values.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / float(max(values.shape[0], 1))
        eig = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0)
        positive = eig[eig > 1e-12]
        cond = float(positive.max() / positive.min()) if positive.numel() else float("inf")
        norms = torch.linalg.vector_norm(values, dim=-1)
        return {
            f"{prefix}_feature_rms": float(values.square().mean().sqrt().item()),
            f"{prefix}_sample_norm_mean": float(norms.mean().item()),
            f"{prefix}_sample_norm_min": float(norms.min().item()),
            f"{prefix}_sample_norm_max": float(norms.max().item()),
            f"{prefix}_cov_trace": float(eig.sum().item()),
            f"{prefix}_cov_min_eigenvalue": float(eig.min().item()),
            f"{prefix}_cov_max_eigenvalue": float(eig.max().item()),
            f"{prefix}_cov_effective_rank": effective_rank_from_eigenvalues(eig),
            f"{prefix}_cov_condition_number": cond,
        }

    out: dict[str, float] = {}
    out.update(summarize("val_raw", torch.cat(raw_chunks, dim=0)))
    out.update(summarize("val_feature", torch.cat(feature_chunks, dim=0)))
    return out


def _parse_intervals(value: str) -> tuple[tuple[float, float], ...]:
    out = []
    for item in value.split(","):
        if not item.strip():
            continue
        s, t = (float(x) for x in item.split(":"))
        if not (0 <= s < t <= 1):
            raise ValueError(f"invalid validation interval {item!r}")
        out.append((s, t))
    if not out:
        raise ValueError("at least one validation interval is required")
    return tuple(out)


def _optimizer_for_stage(
    model: KoopmanRegularizedLKF,
    *,
    stage: str,
    learning_rate: float,
    lkf_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    koopman_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("lkf.") and p.requires_grad
    ]
    if stage == "A":
        return AdamW(koopman_params, lr=learning_rate, weight_decay=weight_decay)
    lkf_params = [p for p in model.lkf.parameters() if p.requires_grad]
    if not lkf_params:
        raise RuntimeError("Stage B requested but no LKF parameters were unfrozen")
    return AdamW(
        [
            {"params": koopman_params, "lr": learning_rate},
            {"params": lkf_params, "lr": lkf_learning_rate},
        ],
        weight_decay=weight_decay,
    )


class _TrainingObjective(nn.Module):
    """One DDP-visible forward containing every trainable Stage-A/B loss path."""

    def __init__(
        self,
        model: KoopmanRegularizedLKF,
        *,
        stage: str,
        lambda_koopman: float,
        rank_regularization_weight: float,
        rank_target: float,
        router_balance_coef: float,
        router_prior_entropy_coef: float,
    ):
        super().__init__()
        self.model = model
        self.stage = str(stage).upper()
        self.lambda_koopman = float(lambda_koopman)
        self.rank_regularization_weight = float(rank_regularization_weight)
        self.rank_target = float(rank_target)
        if self.rank_regularization_weight < 0:
            raise ValueError("rank_regularization_weight must be nonnegative")
        if self.rank_target < 1.0:
            raise ValueError("rank_target must be >=1")
        self.router_balance_coef = float(router_balance_coef)
        self.router_prior_entropy_coef = float(router_prior_entropy_coef)

    def forward(
        self,
        x1: Tensor,
        xs: Tensor,
        xt: Tensor,
        s: float,
        t: float,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        koop_loss, koop_metrics = self.model.closure_loss(
            xs,
            xt,
            s,
            t,
            rank_regularization_weight=self.rank_regularization_weight,
            rank_target=self.rank_target,
        )
        if self.stage == "B":
            s_batch = torch.full(
                (x1.shape[0],),
                float(s),
                device=x1.device,
                dtype=self.model.lkf.pos_embedder.dtype,
            )
            lkf_metrics = uniform_lkf_loss_and_metrics(
                self.model.lkf,
                xs,
                x1,
                s_batch,
                router_balance_coef=self.router_balance_coef,
                router_prior_entropy_coef=self.router_prior_entropy_coef,
            )
            lkf_loss = lkf_metrics["loss"]
            total_loss = lkf_loss + self.lambda_koopman * koop_loss
        else:
            lkf_loss = torch.zeros((), device=x1.device, dtype=koop_loss.dtype)
            total_loss = koop_loss
        return total_loss, koop_loss, lkf_loss, koop_metrics


def _wrap_ddp(module: nn.Module, ctx: DistributedContext) -> nn.Module:
    if not ctx.distributed:
        return module
    kwargs: dict[str, Any] = {
        "find_unused_parameters": False,
        # The fixed-norm representation has no mutable normalization buffers.
        "broadcast_buffers": False,
    }
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


def _assert_equal_local_steps(local_steps: int, ctx: DistributedContext) -> None:
    if not ctx.distributed:
        return
    t_min = torch.tensor(local_steps, device=ctx.device, dtype=torch.long)
    t_max = t_min.clone()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    if int(t_min.item()) != int(t_max.item()):
        raise RuntimeError(
            "DDP ranks executed different optimizer-step counts: "
            f"min={int(t_min.item())}, max={int(t_max.item())}. "
            "This should be impossible with ChunkedCleanPeptideBatchDataset + DistributedSampler."
        )


def run(args: argparse.Namespace, ctx: DistributedContext) -> Path:
    # Use identical initialization on all ranks.  After DDP construction, switch
    # to rank-offset RNG streams for independent corruption/continuation draws.
    seed_everything(args.seed)
    device = ctx.device
    stage = args.stage.upper()
    base_path = Path(args.lkf_checkpoint).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    if stage == "A":
        lkf = load_uniform_lkf_checkpoint(base_path, map_location="cpu", eval_mode=False)
        config = KoopmanConfig(
            feature_dim=args.feature_dim,
            head_hidden_dim=args.head_hidden_dim,
            whitening_momentum=args.whitening_momentum,
            whitening_eps=args.whitening_eps,
            whitening_eig_floor=args.whitening_eig_floor,
            clock="linear",
            operator_init_scale=args.operator_init_scale,
            feature_source=args.feature_source,
        )
        model = KoopmanRegularizedLKF(lkf, config)
    else:
        if not args.stage_a_checkpoint:
            raise ValueError("Stage B requires --stage-a-checkpoint")
        model, stage_a_meta = load_koopman_checkpoint(
            args.stage_a_checkpoint,
            base_lkf_checkpoint=base_path,
            device="cpu",
            strict_base_sha=True,
            eval_mode=False,
        )
        if str(stage_a_meta.get("stage", "")).upper() != "A":
            raise ValueError("--stage-a-checkpoint must be a Stage A checkpoint")

    model.to(device)
    unfrozen = configure_lkf_trainability(
        model.lkf,
        stage=stage,
        unfreeze_shared_blocks=args.unfreeze_shared_blocks,
        unfreeze_downstream=not args.freeze_downstream,
    )

    # Pre-chunk before DistributedSampler so every item is exactly one optimizer
    # step.  This avoids unequal nested chunk counts across ranks.
    train_ds = ChunkedCleanPeptideBatchDataset(
        args.dataset_root,
        args.train_split,
        max_sequences=args.max_batch_sequences,
    )
    train_sampler: Optional[DistributedSampler] = None
    if ctx.distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
    train_loader = build_clean_loader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    # Validation + baseline are rank-0 only and must finish before collectives.
    val_ds: Optional[CleanPeptideBatchDataset] = None
    val_batches: list[Tensor] = []
    output_root = Path(args.output_dir).expanduser().resolve() / args.run_name
    checkpoint_dir = output_root / "checkpoints"
    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print("Preparing rank-0 validation batches (num_workers=0)...", flush=True)
        val_ds = CleanPeptideBatchDataset(args.dataset_root, args.val_split)
        val_batches = _prepare_val_batches(
            val_ds,
            num_workers=0,
            max_batch_sequences=args.val_batch_sequences,
            max_batches=args.val_batches,
        )
        if not val_batches:
            raise RuntimeError("validation dataset produced no batches")

    base_sha256: Optional[str] = sha256_file(base_path) if ctx.is_main else None
    base_sha256 = str(broadcast_object(base_sha256, ctx))

    if ctx.is_main:
        assert val_ds is not None
        provenance = {
            "stage": stage,
            "objective_free_training": True,
            "base_lkf_checkpoint": str(base_path),
            "base_lkf_sha256": base_sha256,
            "stage_a_checkpoint": str(Path(args.stage_a_checkpoint).expanduser().resolve())
            if args.stage_a_checkpoint
            else None,
            "koopman_config": model.config.to_dict(),
            "train_dataset": train_ds.length_statistics(),
            "val_dataset": val_ds.length_statistics(),
            "unfrozen_lkf_parameters": unfrozen,
            "loss": (
                "L_closure + beta_rank * L_effective_rank_hinge"
                if stage == "A"
                else "L_LKF + lambda_K * (L_closure + beta_rank * L_effective_rank_hinge)"
            ),
            "rank_regularization_weight": float(args.rank_regularization_weight),
            "min_effective_rank_fraction": float(args.min_effective_rank_fraction),
            "min_effective_rank_absolute": float(args.min_effective_rank_absolute),
            "distributed": {
                "enabled": ctx.distributed,
                "backend": ctx.backend,
                "world_size": ctx.world_size,
                "launcher": "torchrun" if ctx.distributed else "python",
                "max_batch_sequences_per_gpu": int(args.max_batch_sequences),
                "representation_normalization": "fixed_per_sample_l2",
                "validation": "rank0_only",
            },
        }
        write_json(output_root / "provenance.json", provenance)

    val_times = parse_float_list(args.val_times)
    val_intervals = _parse_intervals(args.val_intervals)
    base_nll: Optional[float] = None
    rank_reference: Optional[float] = None
    rank_gate_threshold: Optional[float] = None
    rank_training_target: Optional[float] = None
    initial_feature_cov: Optional[dict[str, float]] = None
    if ctx.is_main:
        base_nll = _validation_lkf_nll(
            model, val_batches, val_times=val_times, seed=args.seed + 50000
        )
        # Define collapse relative to the *held-out endpoint representation at
        # initialization*, not the EMA buffers (which begin as identity).  This
        # makes the gate architecture/data aware and preserves whatever rich
        # representation Stage A/B started from.
        initial_feature_cov = _validation_feature_covariance(model, val_batches, time_value=1.0)
        rank_reference = float(initial_feature_cov["val_feature_cov_effective_rank"])
        rank_gate_threshold = max(
            float(args.min_effective_rank_absolute),
            float(args.min_effective_rank_fraction) * rank_reference,
        )
        rank_training_target = max(
            float(args.rank_target_absolute),
            float(args.rank_target_fraction) * rank_reference,
        )
        write_json(
            output_root / "baseline_validation.json",
            {
                "base_val_clean_nll": base_nll,
                "initial_endpoint_feature_covariance": initial_feature_cov,
                "rank_reference_effective_rank": rank_reference,
                "rank_gate_threshold": rank_gate_threshold,
                "rank_gate_fraction": float(args.min_effective_rank_fraction),
                "rank_gate_absolute_floor": float(args.min_effective_rank_absolute),
                "rank_training_target_fraction": float(args.rank_target_fraction),
                "rank_training_target_absolute": float(args.rank_target_absolute),
                "rank_training_target": rank_training_target,
            },
        )
    base_nll = float(broadcast_object(base_nll, ctx))
    rank_reference = float(broadcast_object(rank_reference, ctx))
    rank_gate_threshold = float(broadcast_object(rank_gate_threshold, ctx))
    rank_training_target = float(broadcast_object(rank_training_target, ctx))

    training_objective = _TrainingObjective(
        model,
        stage=stage,
        lambda_koopman=args.lambda_koopman,
        rank_regularization_weight=args.rank_regularization_weight,
        rank_target=rank_training_target,
        router_balance_coef=args.router_balance_coef,
        router_prior_entropy_coef=args.router_prior_entropy_coef,
    ).to(device)
    ddp_objective = _wrap_ddp(training_objective, ctx)

    optimizer = _optimizer_for_stage(
        model,
        stage=stage,
        learning_rate=args.learning_rate,
        lkf_learning_rate=args.lkf_learning_rate,
        weight_decay=args.weight_decay,
    )

    def _lr_scale(epoch_index: int) -> float:
        if args.epochs <= 1:
            return 1.0
        progress = min(max(epoch_index / float(args.epochs - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_scale)
    process = UniformLKFProcess(model.lkf)
    curve: list[dict[str, Any]] = []
    best_relative_closure = float("inf")
    best_closure_cosine = -float("inf")
    best_path: Optional[Path] = None
    global_step = 0

    # Independent stochastic trajectories per rank, after synchronized model
    # initialization.  DDP still averages their gradients every step.
    seed_everything(args.seed + 100003 * ctx.rank)

    if ctx.is_main:
        mode = f"DDP x{ctx.world_size}" if ctx.distributed else "single device"
        print(
            f"Training Stage {stage} on {mode}; device={device}; "
            f"M={args.continuations}; max_batch_sequences_per_gpu={args.max_batch_sequences}",
            flush=True,
        )

    for epoch in range(args.epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if stage == "A":
            model.observable_head.train()
            model.lkf.eval()
        else:
            model.train()

        sums: dict[str, float] = {}
        train_steps = 0
        for batch in train_loader:
            if args.steps_per_epoch > 0 and train_steps >= args.steps_per_epoch:
                break
            x1 = torch.as_tensor(batch, dtype=torch.long).to(
                device=device, dtype=torch.long, non_blocking=True
            )
            s, t = _sample_interval(
                device=device,
                max_s=args.max_s,
                source_anchor_prob=args.source_anchor_prob,
                terminal_anchor_prob=args.terminal_anchor_prob,
                min_delta=args.min_interval,
            )
            with torch.no_grad():
                xs = process.corrupt_clean(x1, s)
                xt = _continuations(process, xs, s, t, args.continuations)

            optimizer.zero_grad(set_to_none=True)
            total_loss, koop_loss, lkf_loss, koop_metrics = ddp_objective(x1, xs, xt, s, t)
            if not _all_ranks_finite(total_loss, ctx):
                raise FloatingPointError(
                    f"non-finite training loss detected on at least one rank at epoch={epoch} "
                    f"step={global_step}"
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.gradient_clip
            )
            optimizer.step()

            metrics_now = {
                "train_total_loss": float(total_loss.detach().item()),
                "train_koopman_regularizer_loss": float(koop_loss.detach().item()),
                "train_koopman_loss": float(koop_metrics["koopman_loss"].item()),
                "train_rank_hinge_loss": float(koop_metrics["rank_hinge_loss"].item()),
                "train_source_effective_rank": float(koop_metrics["source_effective_rank"].item()),
                "train_target_effective_rank": float(koop_metrics["target_effective_rank"].item()),
                "train_lkf_loss": float(lkf_loss.detach().item()),
                "train_closure_relative_rmse": float(
                    koop_metrics["closure_relative_rmse"].item()
                ),
                "train_closure_cosine": float(koop_metrics["closure_cosine"].item()),
                "train_source_feature_rms": float(koop_metrics["source_feature_rms"].item()),
                "train_target_individual_feature_rms": float(
                    koop_metrics["target_individual_feature_rms"].item()
                ),
                "train_target_mean_feature_rms": float(
                    koop_metrics["target_mean_feature_rms"].item()
                ),
                "train_s": s,
                "train_t": t,
                "train_interval": t - s,
                "train_local_batch_sequences": float(x1.shape[0]),
            }
            for key, value in metrics_now.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            train_steps += 1
            global_step += 1

        if train_steps == 0:
            raise RuntimeError("training loader produced no optimization steps")
        _assert_equal_local_steps(train_steps, ctx)
        scheduler.step()

        global_sums, global_rank_steps = reduce_metric_sums(sums, train_steps, ctx)
        epoch_seconds = max_float(time.time() - epoch_start, ctx)

        # Rank zero validates the shared synchronized model.  Other ranks block
        # on the object broadcast until validation is complete.
        validation_payload: Optional[dict[str, Any]] = None
        if ctx.is_main:
            val_closure = _validation_closure(
                model,
                val_batches,
                intervals=val_intervals,
                continuations=args.val_continuations,
                seed=args.seed + 100000 + epoch * 1000,
                rank_target=rank_training_target,
            )
            active_nll = _validation_lkf_nll(
                model, val_batches, val_times=val_times, seed=args.seed + 50000
            )
            if stage == "A":
                model.lkf.eval()
            nll_rel = (active_nll - base_nll) / max(abs(base_nll), 1e-12)
            feature_cov = _validation_feature_covariance(model, val_batches, time_value=1.0)
            validation_payload = {
                "val_closure": val_closure,
                "active_nll": active_nll,
                "nll_rel": nll_rel,
                "feature_cov": feature_cov,
            }
        validation_payload = broadcast_object(validation_payload, ctx)
        assert isinstance(validation_payload, dict)
        val_closure = dict(validation_payload["val_closure"])
        active_nll = float(validation_payload["active_nll"])
        nll_rel = float(validation_payload["nll_rel"])
        feature_cov = dict(validation_payload.get("feature_cov", {}))

        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "world_size": ctx.world_size,
            "rank_regularization_weight": float(args.rank_regularization_weight),
            "rank_reference_effective_rank": float(rank_reference),
            "rank_gate_threshold": float(rank_gate_threshold),
            "rank_training_target": float(rank_training_target),
            **{k: v / max(global_rank_steps, 1) for k, v in global_sums.items()},
            **val_closure,
            "val_clean_nll": active_nll,
            "val_clean_nll_relative_change": nll_rel,
            **feature_cov,
            "epoch_seconds": epoch_seconds,
            "koopman_lr": optimizer.param_groups[0]["lr"],
            "lkf_lr": optimizer.param_groups[-1]["lr"] if stage == "B" else 0.0,
        }

        val_feature_rank = float(feature_cov.get("val_feature_cov_effective_rank", 0.0))
        rank_gate_passed = val_feature_rank + 1e-8 >= float(rank_gate_threshold)
        generation_gate_passed = stage == "A" or nll_rel <= args.max_relative_nll_degradation
        eligible = rank_gate_passed and generation_gate_passed
        row["rank_gate_passed"] = bool(rank_gate_passed)
        row["generation_gate_passed"] = bool(generation_gate_passed)
        row["checkpoint_eligible"] = bool(eligible)
        if ctx.is_main:
            curve.append(row)
            write_csv(output_root / "training_curve.csv", curve)
            rel_score = float(val_closure["val_closure_relative_rmse"])
            cos_score = float(val_closure["val_closure_cosine"])
            better = (
                rel_score < best_relative_closure - 1e-8
                or (abs(rel_score - best_relative_closure) <= 1e-8 and cos_score > best_closure_cosine)
            )
            if eligible and better:
                best_relative_closure = rel_score
                best_closure_cosine = cos_score
                best_path = save_koopman_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    base_lkf_checkpoint=base_path,
                    stage=stage,
                    epoch=epoch,
                    global_step=global_step,
                    training_args=vars(args),
                    metrics=row,
                    include_lkf_state=(stage == "B"),
                    base_lkf_sha256=base_sha256,
                )
            save_koopman_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                base_lkf_checkpoint=base_path,
                stage=stage,
                epoch=epoch,
                global_step=global_step,
                training_args=vars(args),
                metrics=row,
                include_lkf_state=(stage == "B"),
                base_lkf_sha256=base_sha256,
            )
            failed = []
            if not rank_gate_passed:
                failed.append("rank-gate-failed")
            if not generation_gate_passed:
                failed.append("generation-gate-failed")
            status = "eligible" if not failed else "+".join(failed)
            feat_min = float(feature_cov.get("val_feature_cov_min_eigenvalue", float("nan")))
            feat_max = float(feature_cov.get("val_feature_cov_max_eigenvalue", float("nan")))
            norm_min = float(feature_cov.get("val_feature_sample_norm_min", float("nan")))
            norm_max = float(feature_cov.get("val_feature_sample_norm_max", float("nan")))
            print(
                f"[epoch {epoch:03d}] "
                f"closure={row['train_koopman_loss']:.5f} "
                f"rank_pen={row['train_rank_hinge_loss']:.4f} "
                f"reg={row['train_koopman_regularizer_loss']:.5f} "
                f"val_koop={row['val_koopman_loss']:.5f} "
                f"rel={row['val_closure_relative_rmse']:.4f} "
                f"cos={row['val_closure_cosine']:.3f} "
                f"target_rms={row['val_target_mean_feature_rms']:.3f} "
                f"rank={val_feature_rank:.1f}/{rank_reference:.1f} "
                f"(target>={rank_training_target:.1f}, gate>={rank_gate_threshold:.1f}) "
                f"norm=[{norm_min:.3f},{norm_max:.3f}] "
                f"eig=[{feat_min:.2e},{feat_max:.2e}] "
                f"val_nll={active_nll:.5f} ({nll_rel:+.2%}) {status}",
                flush=True,
            )

        # All ranks wait for rank-0 validation/checkpoint I/O before entering
        # the next DDP forward.  The broadcast is also the synchronization point.
        _ = broadcast_object(True if ctx.is_main else None, ctx)

    best_path_text = str(best_path) if ctx.is_main and best_path is not None else None
    best_path_text = broadcast_object(best_path_text, ctx)
    if best_path_text is None:
        raise RuntimeError(
            "No eligible checkpoint was saved. The representation-rank gate rejected "
            "collapsed checkpoints and/or the Stage-B generation gate failed. Inspect "
            "training_curve.csv before relaxing either gate."
        )
    if ctx.is_main:
        print(f"Best checkpoint: {best_path_text}", flush=True)
    return Path(best_path_text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train objective-free Koopman-Regularized Uniform-LKF (single GPU or torchrun DDP)"
    )
    p.add_argument("--lkf-checkpoint", required=True, help="Pretrained scratch Uniform-LKF M8 checkpoint")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--stage", choices=("A", "B", "a", "b"), default="A")
    p.add_argument("--stage-a-checkpoint", default=None)
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--run-name", default="koopman_lkf_stage_a")
    p.add_argument(
        "--device",
        default="cuda",
        help="Single-process device. Under torchrun, LOCAL_RANK selects cuda:<rank> automatically.",
    )

    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument(
        "--feature-source",
        choices=("shared_pooled_plus_time", "shared_pooled"),
        default="shared_pooled_plus_time",
    )
    # Deprecated no-op compatibility flags from the whitening implementation.
    p.add_argument("--whitening-momentum", type=float, default=0.99, help=argparse.SUPPRESS)
    p.add_argument("--whitening-eps", type=float, default=1e-8, help=argparse.SUPPRESS)
    p.add_argument("--whitening-eig-floor", type=float, default=1e-3, help=argparse.SUPPRESS)
    p.add_argument("--operator-init-scale", type=float, default=1e-3)
    p.add_argument(
        "--rank-regularization-weight",
        type=float,
        default=1.0,
        help="beta_rank for the effective-rank hinge; zero whenever rank is above target",
    )
    p.add_argument(
        "--min-effective-rank-fraction",
        type=float,
        default=0.90,
        help="checkpoint rank gate as a fraction of the initial held-out normalized endpoint effective rank",
    )
    p.add_argument(
        "--min-effective-rank-absolute",
        type=float,
        default=0.0,
        help="optional absolute effective-rank floor combined with the relative gate",
    )


    p.add_argument(
        "--rank-target-fraction",
        type=float,
        default=0.95,
        help="training rank-hinge target as a fraction of the initial held-out normalized endpoint rank",
    )
    p.add_argument(
        "--rank-target-absolute",
        type=float,
        default=0.0,
        help="optional absolute lower bound for the training rank-hinge target",
    )
    p.add_argument(
        "--continuations",
        type=int,
        default=4,
        help="M Monte-Carlo continuations per local source batch for the training target",
    )
    p.add_argument("--max-s", type=float, default=0.95)
    p.add_argument("--source-anchor-prob", type=float, default=0.20)
    p.add_argument("--terminal-anchor-prob", type=float, default=0.30)
    p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="0 means the full distributed shard; otherwise this many synchronized optimizer steps per rank",
    )
    p.add_argument(
        "--max-batch-sequences",
        type=int,
        default=64,
        help="Maximum peptide sequences per GPU/rank in one optimizer step",
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip", type=float, default=1.0)

    p.add_argument("--lambda-koopman", type=float, default=0.05, help="Stage B only")
    p.add_argument("--lkf-learning-rate", type=float, default=1e-5, help="Stage B only")
    p.add_argument("--unfreeze-shared-blocks", type=int, default=2, help="Stage B only")
    p.add_argument(
        "--freeze-downstream",
        action="store_true",
        help="Stage B: do not unfreeze latent/router/output tail",
    )
    p.add_argument("--router-balance-coef", type=float, default=0.01)
    p.add_argument("--router-prior-entropy-coef", type=float, default=0.01)
    p.add_argument(
        "--max-relative-nll-degradation",
        type=float,
        default=0.02,
        help="Stage B checkpoint gate",
    )

    p.add_argument("--val-times", default="0,0.25,0.5,0.75,0.875")
    p.add_argument(
        "--val-intervals",
        default="0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1",
    )
    p.add_argument("--val-continuations", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--val-batch-sequences", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.continuations < 1 or args.val_continuations < 1:
        raise ValueError("continuation counts must be >=1")
    if args.max_batch_sequences < 1:
        raise ValueError("--max-batch-sequences must be >=1")
    if args.rank_regularization_weight < 0:
        raise ValueError("--rank-regularization-weight must be nonnegative")
    if not (0 < args.min_effective_rank_fraction <= 1):
        raise ValueError("--min-effective-rank-fraction must lie in (0,1]")
    if args.min_effective_rank_absolute < 0:
        raise ValueError("--min-effective-rank-absolute must be nonnegative")
    if not (0 < args.rank_target_fraction <= 1):
        raise ValueError("--rank-target-fraction must lie in (0,1]")
    if args.rank_target_absolute < 0:
        raise ValueError("--rank-target-absolute must be nonnegative")
    if not (0 <= args.max_s < 1):
        raise ValueError("max_s must lie in [0,1)")
    if not (0 <= args.source_anchor_prob <= 1 and 0 <= args.terminal_anchor_prob <= 1):
        raise ValueError("anchor probabilities must lie in [0,1]")

    ctx = initialize_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
