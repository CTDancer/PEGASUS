"""DDP training for the information-preserving Koopman lift.

Stage A: freeze the active LKF generator; train only the learned lift + shared
continuous Koopman generator.  The 64-D anchor is immutable.

Stage B: optional, only after Stage-A evaluation passes.  Load Stage A, keep the
anchor encoder/projection frozen, conservatively fine-tune the active LKF tail,
and optimize L_LKF + lambda_K * R_pegasus.
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

from .distributed import (
    DistributedContext,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
    max_float,
    reduce_metric_sums,
)
from .ip_checkpoint import (
    load_generator_from_koopman_checkpoint,
    load_ip_koopman_checkpoint,
    save_ip_koopman_checkpoint,
    sha256_file,
)
from .ip_model import (
    InformationPreservingKoopmanConfig,
    InformationPreservingKoopmanLKF,
    configure_lkf_trainability,
)
from .lkf.data import (
    ChunkedCleanPeptideBatchDataset,
    CleanPeptideBatchDataset,
    build_clean_loader,
    uniform_lkf_loss_and_metrics,
)
from .lkf.uniform_lkf import UniformLKFProcess, load_uniform_lkf_checkpoint
from .model import covariance_spectrum
from .utils import (
    effective_rank_from_eigenvalues,
    iter_chunks,
    parse_float_list,
    seed_everything,
    write_csv,
    write_json,
)


def _parse_intervals(text: str) -> tuple[tuple[float, float], ...]:
    out = []
    for item in str(text).split(","):
        a, b = item.strip().split(":")
        s, t = float(a), float(b)
        if not (0 <= s < t <= 1):
            raise ValueError(f"invalid interval {item!r}")
        out.append((s, t))
    return tuple(out)


def _sample_interval(
    *, device: torch.device, max_s: float, source_anchor_prob: float,
    terminal_anchor_prob: float, min_delta: float,
) -> tuple[float, float]:
    s = 0.0 if torch.rand((), device=device).item() < source_anchor_prob else float(torch.rand((), device=device).item()) * max_s
    if torch.rand((), device=device).item() < terminal_anchor_prob:
        t = 1.0
    else:
        room = 1.0 - s
        t = 1.0 if room <= min_delta else s + min_delta + float(torch.rand((), device=device).item()) * (room - min_delta)
    return s, min(max(t, s + 1e-5), 1.0)


def _continuations(process: UniformLKFProcess, xs: Tensor, s: float, t: float, m: int) -> Tensor:
    return torch.stack([process.sample_transition(xs, s, t) for _ in range(int(m))], dim=0)


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
    for batch in build_clean_loader(dataset, shuffle=False, num_workers=0):
        for chunk in iter_chunks(torch.as_tensor(batch, dtype=torch.long), max_batch_sequences):
            out.append(chunk.cpu().contiguous())
            if len(out) >= max_batches:
                return out
    return out


@torch.no_grad()
def _validation_lkf_nll(model: InformationPreservingKoopmanLKF, val_batches: list[Tensor], *, val_times: tuple[float, ...], seed: int) -> float:
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
            sb = torch.full((x1.shape[0],), float(s), device=x1.device, dtype=lkf.pos_embedder.dtype)
            xs = process.corrupt_clean(x1, sb, generator=gen)
            metrics = uniform_lkf_loss_and_metrics(lkf, xs, x1, sb, router_balance_coef=0.0, router_prior_entropy_coef=0.0)
            values.append(float(metrics["clean_nll"].item()))
    if was_training:
        lkf.train()
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def _validation_closure(
    model: InformationPreservingKoopmanLKF,
    val_batches: list[Tensor], *, intervals: tuple[tuple[float, float], ...],
    continuations: int, seed: int, rank_target: float,
) -> dict[str, float]:
    process = UniformLKFProcess(model.lkf)
    was_training = model.training
    model.eval()
    keys = (
        "koopman_loss", "closure_relative_rmse", "closure_cosine",
        "anchor_closure_relative_rmse", "anchor_closure_cosine",
        "lift_closure_relative_rmse", "lift_closure_cosine",
        "anchor_target_rms", "lift_target_rms", "lift_individual_rms",
        "source_lift_effective_rank", "target_lift_effective_rank", "rank_hinge_loss",
    )
    rows = []
    for ii, (s, t) in enumerate(intervals):
        sums = {k: 0.0 for k in keys}
        count = 0
        for bi, batch_cpu in enumerate(val_batches):
            x1 = batch_cpu.to(next(model.parameters()).device)
            gen = torch.Generator(device=x1.device)
            gen.manual_seed(int(seed) + 1000003 * ii + 8191 * bi)
            xs = process.corrupt_clean(x1, s, generator=gen)
            xt = torch.stack([process.sample_transition(xs, s, t, generator=gen) for _ in range(continuations)], dim=0)
            _, metrics = model.closure_loss(xs, xt, s, t, rank_regularization_weight=0.0, rank_target=rank_target)
            n = int(x1.shape[0]); count += n
            for k in keys:
                sums[k] += float(metrics[k].item()) * n
        rows.append({k: v / max(count, 1) for k, v in sums.items()})
    if was_training:
        model.train()
    avg = lambda k: float(sum(r[k] for r in rows) / max(len(rows), 1))
    return {
        "val_koopman_loss": avg("koopman_loss"),
        "val_closure_relative_rmse": avg("closure_relative_rmse"),
        "val_closure_cosine": avg("closure_cosine"),
        "val_anchor_relative_rmse": avg("anchor_closure_relative_rmse"),
        "val_anchor_cosine": avg("anchor_closure_cosine"),
        "val_lift_relative_rmse": avg("lift_closure_relative_rmse"),
        "val_lift_cosine": avg("lift_closure_cosine"),
        "val_anchor_target_rms": avg("anchor_target_rms"),
        "val_lift_target_rms": avg("lift_target_rms"),
        "val_lift_individual_rms": avg("lift_individual_rms"),
        "val_source_lift_effective_rank": avg("source_lift_effective_rank"),
        "val_target_lift_effective_rank": avg("target_lift_effective_rank"),
        "val_rank_hinge_loss": avg("rank_hinge_loss"),
        "val_semigroup_max_error": float(max(model.semigroup_error(s, 0.5 * (s + t), t) for s, t in intervals)),
    }


@torch.no_grad()
def _feature_covariance_metrics(model: InformationPreservingKoopmanLKF, val_batches: list[Tensor], *, time_value: float = 1.0) -> dict[str, float]:
    was_training = model.training
    model.eval()
    anchors, lifts, fulls = [], [], []
    device = next(model.parameters()).device
    for batch_cpu in val_batches:
        x = batch_cpu.to(device)
        a = model.anchor_features(x, time_value).cpu()
        l = model.lift_features(x, time_value).cpu()
        anchors.append(a); lifts.append(l); fulls.append(torch.cat([a, l], dim=-1))
    if was_training:
        model.train()
    out: dict[str, float] = {}
    for name, chunks in (("anchor", anchors), ("lift", lifts), ("full", fulls)):
        x = torch.cat(chunks, dim=0).float()
        _, eig = covariance_spectrum(x)
        out[f"val_{name}_cov_effective_rank"] = float(effective_rank_from_eigenvalues(eig))
        out[f"val_{name}_cov_min_eigenvalue"] = float(eig.min().item())
        out[f"val_{name}_cov_max_eigenvalue"] = float(eig.max().item())
        out[f"val_{name}_sample_norm_mean"] = float(torch.linalg.vector_norm(x, dim=-1).mean().item())
    return out


def _optimizer(model: InformationPreservingKoopmanLKF, *, stage: str, learning_rate: float, lkf_learning_rate: float, weight_decay: float) -> AdamW:
    aux = [p for name, p in model.named_parameters() if not name.startswith("lkf.") and p.requires_grad]
    if stage == "A":
        return AdamW(aux, lr=learning_rate, weight_decay=weight_decay)
    lkf_params = [p for p in model.lkf.parameters() if p.requires_grad]
    if not lkf_params:
        raise RuntimeError("Stage B requested but no LKF parameters are unfrozen")
    return AdamW([
        {"params": aux, "lr": learning_rate},
        {"params": lkf_params, "lr": lkf_learning_rate},
    ], weight_decay=weight_decay)


class _TrainingObjective(nn.Module):
    def __init__(self, model: InformationPreservingKoopmanLKF, *, stage: str, lambda_koopman: float,
                 rank_regularization_weight: float, rank_target: float,
                 router_balance_coef: float, router_prior_entropy_coef: float):
        super().__init__(); self.model = model; self.stage = stage
        self.lambda_koopman = float(lambda_koopman); self.rank_regularization_weight = float(rank_regularization_weight)
        self.rank_target = float(rank_target); self.router_balance_coef = float(router_balance_coef)
        self.router_prior_entropy_coef = float(router_prior_entropy_coef)

    def forward(self, x1: Tensor, xs: Tensor, xt: Tensor, s: float, t: float):
        koop, metrics = self.model.closure_loss(xs, xt, s, t,
            rank_regularization_weight=self.rank_regularization_weight, rank_target=self.rank_target)
        if self.stage == "B":
            sb = torch.full((x1.shape[0],), float(s), device=x1.device, dtype=self.model.lkf.pos_embedder.dtype)
            lm = uniform_lkf_loss_and_metrics(self.model.lkf, xs, x1, sb,
                router_balance_coef=self.router_balance_coef,
                router_prior_entropy_coef=self.router_prior_entropy_coef)
            lkf_loss = lm["loss"]
            total = lkf_loss + self.lambda_koopman * koop
        else:
            lkf_loss = torch.zeros((), device=x1.device, dtype=koop.dtype)
            total = koop
        return total, koop, lkf_loss, metrics


def _wrap_ddp(module: nn.Module, ctx: DistributedContext) -> nn.Module:
    if not ctx.distributed:
        return module
    kwargs: dict[str, Any] = {"find_unused_parameters": False, "broadcast_buffers": False}
    if ctx.device.type == "cuda":
        kwargs.update(device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DDP(module, **kwargs)


def _all_ranks_finite(value: Tensor, ctx: DistributedContext) -> bool:
    flag = torch.tensor(1 if bool(torch.isfinite(value.detach()).all()) else 0, device=ctx.device, dtype=torch.int32)
    if ctx.distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _assert_equal_local_steps(local_steps: int, ctx: DistributedContext) -> None:
    if not ctx.distributed: return
    lo = torch.tensor(local_steps, device=ctx.device, dtype=torch.long); hi = lo.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN); dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if int(lo.item()) != int(hi.item()):
        raise RuntimeError(f"DDP ranks executed different step counts: {int(lo.item())} vs {int(hi.item())}")


def run(args: argparse.Namespace, ctx: DistributedContext) -> Path:
    seed_everything(args.seed)
    stage = str(args.stage).upper(); device = ctx.device
    base_path = Path(args.lkf_checkpoint).expanduser().resolve()
    if not base_path.exists(): raise FileNotFoundError(base_path)

    generator_init_path: Optional[Path] = None
    if stage == "A":
        if args.generator_init_koopman_checkpoint:
            generator_init_path = Path(args.generator_init_koopman_checkpoint).expanduser().resolve()
            lkf, _ = load_generator_from_koopman_checkpoint(generator_init_path, base_lkf_checkpoint=base_path, strict_base_sha=True)
        else:
            lkf = load_uniform_lkf_checkpoint(base_path, map_location="cpu", eval_mode=False)
        config = InformationPreservingKoopmanConfig(
            anchor_dim=args.anchor_dim, lift_dim=args.lift_dim, lift_hidden_dim=args.lift_hidden_dim,
            anchor_seed=args.anchor_seed, operator_init_scale=args.operator_init_scale,
            feature_source=args.feature_source, anchor_closure_weight=args.anchor_closure_weight,
            lift_closure_weight=args.lift_closure_weight,
        )
        model = InformationPreservingKoopmanLKF(lkf, config)
    else:
        if not args.stage_a_checkpoint:
            raise ValueError("Stage B requires --stage-a-checkpoint")
        model, meta = load_ip_koopman_checkpoint(args.stage_a_checkpoint, base_lkf_checkpoint=base_path,
                                                 device="cpu", strict_base_sha=True, eval_mode=False)
        if str(meta.get("stage", "")).upper() != "A":
            raise ValueError("--stage-a-checkpoint must be an IP-Koopman Stage-A checkpoint")
        generator_init_path = Path(meta["generator_init_koopman_checkpoint"]).resolve() if meta.get("generator_init_koopman_checkpoint") else None

    model.to(device)
    unfrozen = configure_lkf_trainability(model.lkf, stage=stage,
        unfreeze_shared_blocks=args.unfreeze_shared_blocks,
        unfreeze_downstream=not args.freeze_downstream)
    model.anchor_encoder.eval()
    for p in model.anchor_encoder.parameters():
        if p.requires_grad:
            raise RuntimeError("anchor encoder must remain frozen")

    if ctx.is_main:
        print("Indexing train dataset (all ranks)...", flush=True)
    t_index = time.time()
    train_ds = ChunkedCleanPeptideBatchDataset(
        args.dataset_root,
        args.train_split,
        max_sequences=args.max_batch_sequences,
    )
    if ctx.is_main:
        stats = train_ds.length_statistics()
        print(
            f"Train index ready in {time.time() - t_index:.1f}s: "
            f"{stats['chunks']} chunks / {stats['sequences']} sequences",
            flush=True,
        )
    sampler: Optional[DistributedSampler] = None
    if ctx.distributed:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
    # Prefer spawn-safe worker count after CUDA init; 0 avoids fork deadlocks.
    train_loader = build_clean_loader(
        train_ds,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
    )

    # Validation + baseline are rank-0 only and must finish *before* any
    # collective.  Interleaving barrier/broadcast around CUDA validation is what
    # produced the "device currently unknown" NCCL hang.
    val_ds: Optional[CleanPeptideBatchDataset] = None
    val_batches: list[Tensor] = []
    val_times = parse_float_list(args.val_times)
    intervals = _parse_intervals(args.val_intervals)
    baseline_nll = None
    initial_cov = None
    lift_rank_ref = None
    output_root = Path(args.output_dir).expanduser().resolve() / args.run_name
    ckpt_dir = output_root / "checkpoints"
    if ctx.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print("Preparing rank-0 validation batches (num_workers=0)...", flush=True)
        t_val = time.time()
        val_ds = CleanPeptideBatchDataset(args.dataset_root, args.val_split)
        val_batches = _prepare_val_batches(
            val_ds,
            num_workers=0,
            max_batch_sequences=args.val_batch_sequences,
            max_batches=args.val_batches,
        )
        if not val_batches:
            raise RuntimeError("validation dataset produced no batches")
        print(
            f"Validation ready in {time.time() - t_val:.1f}s: {len(val_batches)} batches",
            flush=True,
        )
        print("Running rank-0 baseline validation...", flush=True)
        t_base = time.time()
        baseline_nll = _validation_lkf_nll(
            model, val_batches, val_times=val_times, seed=args.seed + 50000
        )
        initial_cov = _feature_covariance_metrics(model, val_batches, time_value=1.0)
        lift_rank_ref = float(initial_cov["val_lift_cov_effective_rank"])
        print(f"Baseline validation done in {time.time() - t_base:.1f}s", flush=True)

    base_sha = str(broadcast_object(sha256_file(base_path) if ctx.is_main else None, ctx))
    if ctx.is_main:
        assert val_ds is not None
        write_json(
            output_root / "baseline_validation.json",
            {
                "initial_val_clean_nll": baseline_nll,
                "initial_endpoint_covariance": initial_cov,
                "lift_rank_reference": lift_rank_ref,
                "anchor_is_structurally_frozen": True,
            },
        )
        write_json(
            output_root / "provenance.json",
            {
                "stage": stage,
                "objective_free_training": True,
                "base_lkf_checkpoint": str(base_path),
                "base_lkf_sha256": base_sha,
                "generator_init_koopman_checkpoint": str(generator_init_path)
                if generator_init_path
                else None,
                "stage_a_checkpoint": str(Path(args.stage_a_checkpoint).expanduser().resolve())
                if args.stage_a_checkpoint
                else None,
                "ip_koopman_config": model.config.to_dict(),
                "train_dataset": train_ds.length_statistics(),
                "val_dataset": val_ds.length_statistics(),
                "unfrozen_lkf_parameters": unfrozen,
                "representation": "fixed objective-rich anchor + learned nonlinear lift",
                "checkpoint_selection": "rank/generation gates, then minimum anchor relative closure",
                "distributed": {
                    "enabled": ctx.distributed,
                    "world_size": ctx.world_size,
                    "backend": ctx.backend,
                },
            },
        )
    baseline_nll = float(broadcast_object(baseline_nll, ctx))
    lift_rank_ref = float(broadcast_object(lift_rank_ref, ctx))
    rank_target = max(float(args.rank_target_absolute), float(args.rank_target_fraction) * lift_rank_ref)
    rank_gate = max(float(args.min_effective_rank_absolute), float(args.min_effective_rank_fraction) * lift_rank_ref)

    objective = _TrainingObjective(model, stage=stage, lambda_koopman=args.lambda_koopman,
        rank_regularization_weight=args.rank_regularization_weight, rank_target=rank_target,
        router_balance_coef=args.router_balance_coef, router_prior_entropy_coef=args.router_prior_entropy_coef).to(device)
    ddp = _wrap_ddp(objective, ctx)
    optimizer = _optimizer(model, stage=stage, learning_rate=args.learning_rate,
                           lkf_learning_rate=args.lkf_learning_rate, weight_decay=args.weight_decay)
    def lr_scale(e: int) -> float:
        if args.epochs <= 1: return 1.0
        p = min(max(e / float(args.epochs - 1), 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
    process = UniformLKFProcess(model.lkf)
    curve: list[dict[str, Any]] = []; global_step = 0; best_path: Optional[Path] = None
    best_anchor_rel = float("inf"); best_anchor_cos = -float("inf")
    seed_everything(args.seed + 100003 * ctx.rank)
    if ctx.is_main:
        mode = f"DDP x{ctx.world_size}" if ctx.distributed else "single device"
        print(f"Training IP-Koopman Stage {stage} on {mode}; device={device}; M={args.continuations}; "
              f"anchor={model.anchor_dim}, lift={model.lift_dim}, total={model.feature_dim}", flush=True)

    for epoch in range(args.epochs):
        start_time = time.time()
        if sampler is not None: sampler.set_epoch(epoch)
        if stage == "A":
            model.lift_head.train(); model.lkf.eval(); model.anchor_encoder.eval()
        else:
            model.train(); model.anchor_encoder.eval()
        sums: dict[str, float] = {}; steps = 0
        for batch in train_loader:
            if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch: break
            x1 = torch.as_tensor(batch, dtype=torch.long).to(device, non_blocking=True)
            s, t = _sample_interval(device=device, max_s=args.max_s,
                source_anchor_prob=args.source_anchor_prob, terminal_anchor_prob=args.terminal_anchor_prob,
                min_delta=args.min_interval)
            with torch.no_grad():
                xs = process.corrupt_clean(x1, s)
                xt = _continuations(process, xs, s, t, args.continuations)
            optimizer.zero_grad(set_to_none=True)
            total, koop, lkf_loss, km = ddp(x1, xs, xt, s, t)
            if not _all_ranks_finite(total, ctx):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch} step={global_step}")
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.gradient_clip)
            optimizer.step()
            now = {
                "train_total_loss": float(total.detach()), "train_koopman_regularizer_loss": float(koop.detach()),
                "train_koopman_loss": float(km["koopman_loss"]), "train_rank_hinge_loss": float(km["rank_hinge_loss"]),
                "train_anchor_relative_rmse": float(km["anchor_closure_relative_rmse"]),
                "train_anchor_cosine": float(km["anchor_closure_cosine"]),
                "train_lift_relative_rmse": float(km["lift_closure_relative_rmse"]),
                "train_lift_cosine": float(km["lift_closure_cosine"]),
                "train_source_lift_effective_rank": float(km["source_lift_effective_rank"]),
                "train_target_lift_effective_rank": float(km["target_lift_effective_rank"]),
                "train_lkf_loss": float(lkf_loss.detach()), "train_s": s, "train_t": t,
            }
            for k, v in now.items(): sums[k] = sums.get(k, 0.0) + v
            steps += 1; global_step += 1
        if steps == 0: raise RuntimeError("training loader produced no steps")
        _assert_equal_local_steps(steps, ctx); scheduler.step()
        global_sums, global_steps = reduce_metric_sums(sums, steps, ctx)
        epoch_seconds = max_float(time.time() - start_time, ctx)

        payload = None
        if ctx.is_main:
            vc = _validation_closure(model, val_batches, intervals=intervals,
                continuations=args.val_continuations, seed=args.seed + 100000 + epoch * 1000, rank_target=rank_target)
            nll = _validation_lkf_nll(model, val_batches, val_times=val_times, seed=args.seed + 50000)
            cov = _feature_covariance_metrics(model, val_batches, time_value=1.0)
            payload = {"vc": vc, "nll": nll, "cov": cov}
        payload = broadcast_object(payload, ctx); assert isinstance(payload, dict)
        vc, nll, cov = dict(payload["vc"]), float(payload["nll"]), dict(payload["cov"])
        nll_rel = (nll - baseline_nll) / max(abs(baseline_nll), 1e-12)
        row: dict[str, Any] = {
            "epoch": epoch, "global_step": global_step, "world_size": ctx.world_size,
            "rank_reference": lift_rank_ref, "rank_target": rank_target, "rank_gate": rank_gate,
            **{k: v / max(global_steps, 1) for k, v in global_sums.items()}, **vc, **cov,
            "val_clean_nll": nll, "val_clean_nll_relative_change": nll_rel,
            "epoch_seconds": epoch_seconds, "koopman_lr": optimizer.param_groups[0]["lr"],
            "lkf_lr": optimizer.param_groups[-1]["lr"] if stage == "B" else 0.0,
        }
        lift_rank = float(cov["val_lift_cov_effective_rank"])
        rank_ok = lift_rank + 1e-8 >= rank_gate
        gen_ok = stage == "A" or nll_rel <= args.max_relative_nll_degradation
        eligible = rank_ok and gen_ok
        row["rank_gate_passed"] = rank_ok; row["generation_gate_passed"] = gen_ok; row["checkpoint_eligible"] = eligible
        if ctx.is_main:
            curve.append(row); write_csv(output_root / "training_curve.csv", curve)
            a_rel = float(vc["val_anchor_relative_rmse"]); a_cos = float(vc["val_anchor_cosine"])
            better = a_rel < best_anchor_rel - 1e-8 or (abs(a_rel - best_anchor_rel) <= 1e-8 and a_cos > best_anchor_cos)
            if eligible and better:
                best_anchor_rel, best_anchor_cos = a_rel, a_cos
                best_path = save_ip_koopman_checkpoint(ckpt_dir / "best.pt", model,
                    base_lkf_checkpoint=base_path, stage=stage, epoch=epoch, global_step=global_step,
                    training_args=vars(args), metrics=row, base_lkf_sha256=base_sha,
                    generator_init_checkpoint=generator_init_path)
            save_ip_koopman_checkpoint(ckpt_dir / "last.pt", model,
                base_lkf_checkpoint=base_path, stage=stage, epoch=epoch, global_step=global_step,
                training_args=vars(args), metrics=row, base_lkf_sha256=base_sha,
                generator_init_checkpoint=generator_init_path)
            status = "eligible" if eligible else "+".join((["rank-gate-failed"] if not rank_ok else []) + (["generation-gate-failed"] if not gen_ok else []))
            print(
                f"[epoch {epoch:03d}] reg={row['train_koopman_regularizer_loss']:.5f} "
                f"anchor_rel={vc['val_anchor_relative_rmse']:.4f} anchor_cos={vc['val_anchor_cosine']:.3f} "
                f"lift_rel={vc['val_lift_relative_rmse']:.4f} lift_cos={vc['val_lift_cosine']:.3f} "
                f"lift_rank={lift_rank:.1f}/{lift_rank_ref:.1f} (target>={rank_target:.1f}, gate>={rank_gate:.1f}) "
                f"lift_rms={vc['val_lift_individual_rms']:.3f} val_nll={nll:.5f} ({nll_rel:+.2%}) {status}",
                flush=True,
            )
        _ = broadcast_object(True if ctx.is_main else None, ctx)

    text = str(best_path) if ctx.is_main and best_path is not None else None
    text = broadcast_object(text, ctx)
    if text is None:
        raise RuntimeError("No eligible IP-Koopman checkpoint saved; inspect training_curve.csv")
    return Path(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train information-preserving Koopman lift on Uniform-LKF")
    p.add_argument("--lkf-checkpoint", required=True, help="immutable base Uniform-LKF architecture/provenance checkpoint")
    p.add_argument("--generator-init-koopman-checkpoint", default=None,
                   help="Stage A: optional previous Stage-B K-LKF checkpoint whose lkf_state_dict initializes the generator; old Koopman features are discarded")
    p.add_argument("--dataset-root", required=True); p.add_argument("--train-split", default="train"); p.add_argument("--val-split", default="val")
    p.add_argument("--stage", choices=("A","B","a","b"), default="A"); p.add_argument("--stage-a-checkpoint", default=None)
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results")); p.add_argument("--run-name", default="ip_koopman_stage_a")
    p.add_argument("--device", default="cuda")
    p.add_argument("--anchor-dim", type=int, default=64); p.add_argument("--lift-dim", type=int, default=64); p.add_argument("--lift-hidden-dim", type=int, default=256)
    p.add_argument("--anchor-seed", type=int, default=60042); p.add_argument("--operator-init-scale", type=float, default=1e-3)
    p.add_argument("--feature-source", choices=("shared_pooled_plus_time","shared_pooled"), default="shared_pooled_plus_time")
    p.add_argument("--anchor-closure-weight", type=float, default=1.0); p.add_argument("--lift-closure-weight", type=float, default=1.0)
    p.add_argument("--rank-regularization-weight", type=float, default=1.0); p.add_argument("--rank-target-fraction", type=float, default=0.95); p.add_argument("--rank-target-absolute", type=float, default=0.0)
    p.add_argument("--min-effective-rank-fraction", type=float, default=0.90); p.add_argument("--min-effective-rank-absolute", type=float, default=0.0)
    p.add_argument("--continuations", type=int, default=4); p.add_argument("--max-s", type=float, default=0.95); p.add_argument("--source-anchor-prob", type=float, default=0.20); p.add_argument("--terminal-anchor-prob", type=float, default=0.30); p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--steps-per-epoch", type=int, default=0); p.add_argument("--max-batch-sequences", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-5); p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--lambda-koopman", type=float, default=0.05); p.add_argument("--lkf-learning-rate", type=float, default=1e-5); p.add_argument("--unfreeze-shared-blocks", type=int, default=2); p.add_argument("--freeze-downstream", action="store_true")
    p.add_argument("--router-balance-coef", type=float, default=0.01); p.add_argument("--router-prior-entropy-coef", type=float, default=0.01); p.add_argument("--max-relative-nll-degradation", type=float, default=0.02)
    p.add_argument("--val-times", default="0,0.25,0.5,0.75,0.875"); p.add_argument("--val-intervals", default="0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1")
    p.add_argument("--val-continuations", type=int, default=8); p.add_argument("--val-batches", type=int, default=4); p.add_argument("--val-batch-sequences", type=int, default=32); p.add_argument("--num-workers", type=int, default=2); p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.continuations < 1 or args.val_continuations < 1: raise ValueError("continuation counts must be >=1")
    if args.max_batch_sequences < 2: raise ValueError("max_batch_sequences must be >=2")
    if args.rank_regularization_weight < 0: raise ValueError("rank regularization must be nonnegative")
    if not (0 < args.rank_target_fraction <= 1 and 0 < args.min_effective_rank_fraction <= 1): raise ValueError("rank fractions must lie in (0,1]")
    if args.anchor_closure_weight <= 0 or args.lift_closure_weight <= 0: raise ValueError("closure weights must be positive")
    ctx = initialize_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
