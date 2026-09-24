"""Train scratch peptide Uniform-LKF directly from clean peptide data.

No PepDFM checkpoint and no PepDFM trajectory cache is read by this file.
The objective is the exact normalized sequence-level latent clean-posterior NLL
under analytic uniform-AA corruption.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy

try:
    from .uniform_lkf import UniformLKF, UniformLKFConfig, UniformLKFProcess
except ImportError:  # pragma: no cover
    from pegasus.lkf.uniform_lkf import UniformLKF, UniformLKFConfig, UniformLKFProcess


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_token_batch(value: Any) -> Tensor:
    x = torch.as_tensor(value, dtype=torch.long)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"input_ids must be 1-D or 2-D, got {tuple(x.shape)}")
    return x


def peptide_length_groups(
    input_ids: Any,
    attention_mask: Any = None,
    *,
    pad_token_id: int = 1,
) -> dict[int, Tensor]:
    """Strip trailing pads and split a prebatched HF row by true token length.

    Some tokenized peptide shards pack nearby lengths into one row and pad with
    <pad>. Uniform-LKF requires packed <cls> ... AA ... <eos> tensors with no
    padding, because interior positions x[:,1:-1] are treated as residues.
    """
    x = _as_token_batch(input_ids)
    if attention_mask is None:
        lengths = (x != int(pad_token_id)).sum(dim=1)
    else:
        mask = torch.as_tensor(attention_mask, dtype=torch.long)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != x.shape:
            raise ValueError(
                f"attention_mask shape {tuple(mask.shape)} does not match input_ids {tuple(x.shape)}"
            )
        lengths = mask.sum(dim=1)
    groups: dict[int, Tensor] = {}
    for length in lengths.unique(sorted=True).tolist():
        true_len = int(length)
        if true_len < 3:
            raise ValueError(f"peptide token length {true_len} is shorter than <cls> AA <eos>")
        groups[true_len] = x[lengths == length, :true_len].contiguous()
    return groups


def _resolve_hf_split(root: str, split: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The datasets package is required for scratch Uniform-LKF training") from exc
    root_path = Path(root).expanduser()
    child = root_path / split
    if child.exists():
        return load_from_disk(str(child)), str(child.resolve())
    obj = load_from_disk(str(root_path))
    try:
        return obj[split], f"{root_path.resolve()}[{split}]"
    except Exception as exc:
        raise ValueError(f"Could not resolve split {split!r} from {root!r}") from exc


class CleanPeptideBatchDataset(Dataset):
    """Map-style wrapper around the existing prebatched HF peptide dataset."""

    def __init__(self, root: str, split: str, *, pad_token_id: int = 1):
        self.dataset, self.resolved_source = _resolve_hf_split(root, split)
        self.pad_token_id = int(pad_token_id)
        self._index: list[tuple[int, int]] = []
        self._length_counts: dict[int, int] = {}
        self._source_rows = 0
        self._sequences = 0
        for row_index in range(len(self.dataset)):
            record = self.dataset[int(row_index)]
            if "input_ids" not in record:
                raise KeyError(f"Dataset row {row_index} has no input_ids field")
            groups = peptide_length_groups(
                record["input_ids"],
                record.get("attention_mask"),
                pad_token_id=self.pad_token_id,
            )
            self._source_rows += 1
            for length, batch in groups.items():
                self._index.append((int(row_index), int(length)))
                n = int(batch.shape[0])
                self._length_counts[int(length)] = self._length_counts.get(int(length), 0) + n
                self._sequences += n
        if not self._index:
            raise ValueError(f"No peptide sequences found in {self.resolved_source}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Tensor:
        row_index, length = self._index[int(index)]
        record = self.dataset[int(row_index)]
        groups = peptide_length_groups(
            record["input_ids"],
            record.get("attention_mask"),
            pad_token_id=self.pad_token_id,
        )
        return groups[int(length)]

    def length_statistics(self) -> dict[str, Any]:
        counts = dict(sorted(self._length_counts.items()))
        return {
            "source": self.resolved_source,
            "records": len(self._index),
            "source_rows": self._source_rows,
            "sequences": self._sequences,
            "min_token_length": min(counts),
            "max_token_length": max(counts),
            "length_counts": counts,
        }


def build_clean_loader(
    dataset: CleanPeptideBatchDataset,
    *,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    # Each HF row is already a same-length mini-batch. batch_size=None preserves it.
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def _entropy(probs: Tensor, dim: int = -1) -> Tensor:
    return -(probs * probs.clamp_min(1e-30).log()).sum(dim=dim)


def uniform_lkf_loss_and_metrics(
    lkf: UniformLKF,
    x_s: Tensor,
    x_1: Tensor,
    s: Tensor,
    *,
    router_balance_coef: float = 0.01,
    router_prior_entropy_coef: float = 0.01,
) -> dict[str, Tensor]:
    """Exact residue-normalized clean posterior NLL plus mild anti-collapse terms.

    Both router regularizers *maximize* entropy during minimization:

      -H(E_b w_b) encourages global component usage;
      -E_b H(w_b) keeps the prior over latent completions sufficiently broad.

    Component specialization is measured through the posterior entropy reduction
    after observing the clean target; it is not forced by a low-entropy router.
    """
    if router_balance_coef < 0 or router_prior_entropy_coef < 0:
        raise ValueError("router coefficients must be nonnegative")
    log_mix, log_router, per_latent = lkf.clean_posterior_terms(x_s, x_1, s)
    residue_tokens = int(x_1.shape[0]) * int(x_1.shape[1] - 2)
    if residue_tokens <= 0:
        raise ValueError("No peptide residue tokens in batch")
    sequence_nll = -log_mix.mean()
    nll_per_residue = -log_mix.sum() / float(residue_tokens)

    w = log_router.exp()
    marginal = w.mean(dim=0)
    h_marginal = _entropy(marginal)
    h_prior = _entropy(w, dim=-1).mean()
    if lkf.latent_components == 1:
        router_reg = torch.zeros((), device=x_s.device, dtype=nll_per_residue.dtype)
    else:
        router_reg = (
            -float(router_balance_coef) * h_marginal
            -float(router_prior_entropy_coef) * h_prior
        )

    with torch.no_grad():
        posterior = torch.softmax(log_router + per_latent, dim=-1)
        h_post = _entropy(posterior, dim=-1).mean()
        information_gain = h_prior - h_post
        effective_components = h_marginal.exp()
        posterior_effective_components = h_post.exp()
        posterior_max = posterior.max(dim=-1).values.mean()
        marginal_min = marginal.min()
        marginal_max = marginal.max()

    return {
        "loss": nll_per_residue + router_reg,
        "clean_nll": nll_per_residue,
        "sequence_nll": sequence_nll.detach(),
        "router_reg": router_reg.detach(),
        "router_h_marginal": h_marginal.detach(),
        "router_h_prior": h_prior.detach(),
        "router_h_posterior": h_post.detach(),
        "router_information_gain": information_gain.detach(),
        "router_effective_components": effective_components.detach(),
        "posterior_effective_components": posterior_effective_components.detach(),
        "posterior_max_probability": posterior_max.detach(),
        "router_min_usage": marginal_min.detach(),
        "router_max_usage": marginal_max.detach(),
    }


def sample_training_times(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    source_anchor_prob: float,
    max_s: float,
) -> Tensor:
    """Stratified uniform interior times plus explicit s=0 source anchors."""
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    if not (0.0 <= source_anchor_prob <= 1.0):
        raise ValueError("source_anchor_prob must lie in [0,1]")
    if not (0.0 < max_s < 1.0):
        raise ValueError("max_s must lie in (0,1)")
    jitter = torch.rand(batch_size, device=device, dtype=dtype)
    strata = (torch.arange(batch_size, device=device, dtype=dtype) + jitter) / float(batch_size)
    perm = torch.randperm(batch_size, device=device)
    s = strata[perm] * float(max_s)
    if source_anchor_prob > 0:
        anchor = torch.rand(batch_size, device=device) < float(source_anchor_prob)
        s = torch.where(anchor, torch.zeros_like(s), s)
    return s


class UniformLKFLightningModule(pl.LightningModule):
    def __init__(
        self,
        lkf: UniformLKF,
        *,
        dataset_root: str,
        train_split: str,
        val_split: str,
        source_anchor_prob: float,
        max_s: float,
        val_times: Sequence[float],
        router_balance_coef: float,
        router_prior_entropy_coef: float,
        learning_rate: float,
        weight_decay: float,
        warmup_fraction: float,
        min_lr_ratio: float,
        validation_seed: int,
        dataset_statistics: Mapping[str, Any],
    ):
        super().__init__()
        self.lkf = lkf
        self.process = UniformLKFProcess(lkf)
        self.dataset_root = str(Path(dataset_root).expanduser().resolve())
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.source_anchor_prob = float(source_anchor_prob)
        self.max_s = float(max_s)
        self.val_times = tuple(float(x) for x in val_times)
        if not self.val_times or any(x < 0 or x >= 1 for x in self.val_times):
            raise ValueError("val_times must lie in [0,1)")
        self.router_balance_coef = float(router_balance_coef)
        self.router_prior_entropy_coef = float(router_prior_entropy_coef)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_fraction = float(warmup_fraction)
        self.min_lr_ratio = float(min_lr_ratio)
        self.validation_seed = int(validation_seed)
        self.dataset_statistics = dict(dataset_statistics)
        self.save_hyperparameters(ignore=["lkf", "dataset_statistics"])

    def _validate_batch(self, batch: Tensor) -> Tensor:
        x = torch.as_tensor(batch, device=self.device, dtype=torch.long)
        self.lkf.validate_peptide_tokens(x, name="clean_dataset_batch")
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x_1 = self._validate_batch(batch)
        s = sample_training_times(
            x_1.shape[0],
            device=x_1.device,
            dtype=self.lkf.pos_embedder.dtype,
            source_anchor_prob=self.source_anchor_prob,
            max_s=self.max_s,
        )
        with torch.no_grad():
            x_s = self.process.corrupt_clean(x_1, s)
        metrics = uniform_lkf_loss_and_metrics(
            self.lkf,
            x_s,
            x_1,
            s,
            router_balance_coef=self.router_balance_coef,
            router_prior_entropy_coef=self.router_prior_entropy_coef,
        )
        batch_size = int(x_1.shape[0])
        for key, value in metrics.items():
            self.log(
                f"train_{key}",
                value,
                on_step=(key in {"loss", "clean_nll"}),
                on_epoch=True,
                prog_bar=(key in {"loss", "clean_nll", "router_information_gain"}),
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log("train_s_mean", s.mean(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train_source_anchor_fraction", (s == 0).float().mean(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return metrics["loss"]

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x_1 = self._validate_batch(batch)
        batch_size = int(x_1.shape[0])
        nlls: list[Tensor] = []
        for j, s_value in enumerate(self.val_times):
            gen = torch.Generator(device=x_1.device)
            # Stable across runs and independent of global training RNG.
            gen.manual_seed(self.validation_seed + 1000003 * int(batch_idx) + 9176 * int(j))
            s = torch.full(
                (batch_size,),
                float(s_value),
                device=x_1.device,
                dtype=self.lkf.pos_embedder.dtype,
            )
            x_s = self.process.corrupt_clean(x_1, s, generator=gen)
            metrics = uniform_lkf_loss_and_metrics(
                self.lkf,
                x_s,
                x_1,
                s,
                router_balance_coef=0.0,
                router_prior_entropy_coef=0.0,
            )
            nlls.append(metrics["clean_nll"])
            tag = str(s_value).replace(".", "p")
            for key in (
                "clean_nll",
                "router_h_marginal",
                "router_h_prior",
                "router_h_posterior",
                "router_information_gain",
                "router_effective_components",
                "posterior_effective_components",
                "posterior_max_probability",
            ):
                self.log(
                    f"val_s{tag}_{key}",
                    metrics[key],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        mean_nll = torch.stack(nlls).mean()
        self.log(
            "val_clean_nll",
            mean_nll,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = int(total_steps * self.warmup_fraction)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                frac = step / max(warmup_steps, 1)
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * frac
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda),
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint.update(
            {
                "model_type": "uniform_lkf",
                "model_config": self.lkf.model_config,
                "checkpoint_format_version": 1,
                "training_source": "clean_peptide_data",
                "process_type": "uniform_aa_clean_posterior",
                "initialization": "random_from_scratch",
                "teacher_checkpoint": None,
                "teacher_trajectory_supervision": False,
                "dataset_root": self.dataset_root,
                "train_split": self.train_split,
                "val_split": self.val_split,
                "source_anchor_prob": self.source_anchor_prob,
                "max_s": self.max_s,
                "val_times": list(self.val_times),
                "router_balance_coef": self.router_balance_coef,
                "router_prior_entropy_coef": self.router_prior_entropy_coef,
                "dataset_statistics": self.dataset_statistics,
            }
        )


def _parse_devices(value: str) -> Any:
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return int(value)


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, Mapping):
        raise TypeError(f"Checkpoint {path} is not a mapping")
    return obj


def _validate_resume(path: Path, module: UniformLKFLightningModule) -> None:
    ckpt = _torch_load_checkpoint(path)
    if str(ckpt.get("model_type", "")).lower() != "uniform_lkf":
        raise ValueError("Refusing to resume scratch Uniform-LKF from a non-uniform_lkf checkpoint")
    if dict(ckpt.get("model_config", {})) != module.lkf.model_config:
        raise ValueError("Resume checkpoint architecture/token support differs")
    required = {
        "training_source": "clean_peptide_data",
        "process_type": "uniform_aa_clean_posterior",
        "initialization": "random_from_scratch",
        "teacher_trajectory_supervision": False,
    }
    for key, expected in required.items():
        if ckpt.get(key) != expected:
            raise ValueError(f"Resume provenance mismatch for {key}: {ckpt.get(key)!r}")
    if ckpt.get("teacher_checkpoint", None) is not None:
        raise ValueError("Scratch checkpoint unexpectedly records a teacher_checkpoint")


def _build_logger(kind: str, save_dir: str, name: str, wandb_project: Optional[str]):
    if kind == "none":
        return False
    if kind == "csv":
        return CSVLogger(save_dir=save_dir, name=name)
    if kind == "wandb":
        if not wandb_project:
            raise ValueError("--wandb-project is required with --logger wandb")
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:  # pragma: no cover
            from pytorch_lightning.loggers import WandbLogger
        return WandbLogger(project=wandb_project, name=name, save_dir=save_dir)
    raise ValueError("logger must be csv, wandb, or none")


def _make_trainer(args: argparse.Namespace, checkpoint_dir: Path) -> pl.Trainer:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="uniform-lkf-{epoch:03d}-{val_clean_nll:.4f}",
            monitor="val_clean_nll",
            mode="min",
            save_top_k=int(args.save_top_k),
            save_last=True,
            auto_insert_metric_name=False,
        )
    ]
    logger = _build_logger(args.logger, str(checkpoint_dir.parent), args.run_name, args.wandb_project)
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    devices = _parse_devices(args.devices)
    strategy: Any = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif isinstance(devices, list) and len(devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    return pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=int(args.epochs),
        accumulate_grad_batches=int(args.accumulate_grad_batches),
        gradient_clip_val=float(args.gradient_clip_val),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=int(args.log_every_n_steps),
        deterministic=bool(args.deterministic),
        enable_progress_bar=True,
        use_distributed_sampler=True,
    )


def run_training(args: argparse.Namespace) -> None:
    train_ds = CleanPeptideBatchDataset(
        args.dataset_root, args.train_split, pad_token_id=int(args.pad_token_id)
    )
    val_ds = CleanPeptideBatchDataset(
        args.dataset_root, args.val_split, pad_token_id=int(args.pad_token_id)
    )
    train_stats = train_ds.length_statistics()
    val_stats = val_ds.length_statistics()
    inferred_max = max(int(train_stats["max_token_length"]), int(val_stats["max_token_length"]))
    max_seq_len = inferred_max if args.max_seq_len is None else int(args.max_seq_len)
    if max_seq_len < inferred_max:
        raise ValueError(
            f"--max-seq-len={max_seq_len} is smaller than observed token length {inferred_max}"
        )

    if args.run_name is None:
        args.run_name = f"peptide_uniform_lkf_M{int(args.latent_components)}"
    config = UniformLKFConfig(
        vocab_size=int(args.vocab_size),
        seq_len=max_seq_len,
        model_dim=int(args.model_dim),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        latent_components=int(args.latent_components),
        latent_layers=int(args.latent_layers),
        cls_token_id=int(args.cls_token_id),
        pad_token_id=int(args.pad_token_id),
        eos_token_id=int(args.eos_token_id),
        unk_token_id=int(args.unk_token_id),
        aa_token_ids=tuple(int(x) for x in args.aa_token_ids),
    )
    lkf = UniformLKF(config)
    module = UniformLKFLightningModule(
        lkf,
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        val_split=args.val_split,
        source_anchor_prob=args.source_anchor_prob,
        max_s=args.max_s,
        val_times=args.val_times,
        router_balance_coef=args.router_balance_coef,
        router_prior_entropy_coef=args.router_prior_entropy_coef,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
        validation_seed=args.seed + 10000,
        dataset_statistics={"train": train_stats, "val": val_stats},
    )

    # Fail before launching GPUs if tokenization does not match the declared ESM support.
    for name, ds in (("train", train_ds), ("val", val_ds)):
        for i in range(min(len(ds), 16)):
            module.lkf.validate_peptide_tokens(ds[i], name=f"{name}[{i}]")

    output_root = Path(args.output_dir).expanduser()
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_root / "checkpoints"
    with (run_root / "dataset_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": train_stats, "val": val_stats}, handle, indent=2)

    resume = args.resume
    resume_path: Optional[Path] = None
    if resume:
        if str(resume).lower() == "last":
            candidate = checkpoint_dir / "last.ckpt"
            if candidate.exists():
                resume_path = candidate
        else:
            candidate = Path(str(resume)).expanduser()
            if candidate.exists():
                resume_path = candidate.resolve()
        if resume_path is not None:
            _validate_resume(resume_path, module)

    trainer = _make_trainer(args, checkpoint_dir)
    trainer.fit(
        module,
        train_dataloaders=build_clean_loader(train_ds, shuffle=True, num_workers=args.num_workers),
        val_dataloaders=build_clean_loader(val_ds, shuffle=False, num_workers=args.num_workers),
        ckpt_path=(str(resume_path) if resume_path is not None else resume),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train scratch Uniform-AA peptide LKF from clean data")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--vocab-size", type=int, default=24)
    p.add_argument("--cls-token-id", type=int, default=0)
    p.add_argument("--pad-token-id", type=int, default=1)
    p.add_argument("--eos-token-id", type=int, default=2)
    p.add_argument("--unk-token-id", type=int, default=3)
    p.add_argument("--aa-token-ids", type=int, nargs="+", default=list(range(4, 24)))
    p.add_argument("--max-seq-len", type=int, default=None, help="Defaults to maximum observed train/val token length")
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--latent-components", type=int, default=8)
    p.add_argument("--latent-layers", type=int, default=4)
    p.add_argument("--source-anchor-prob", type=float, default=0.25)
    p.add_argument("--max-s", type=float, default=0.95, help="Upper endpoint for non-anchor corruption times; must be <1")
    p.add_argument("--val-times", type=_parse_floats, default=_parse_floats("0,0.25,0.5,0.75,0.875"))
    p.add_argument("--router-balance-coef", type=float, default=0.01)
    p.add_argument("--router-prior-entropy-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="1")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--logger", choices=("csv", "wandb", "none"), default="csv")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--save-top-k", type=int, default=3)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _seed_everything(int(args.seed))
    pl.seed_everything(int(args.seed), workers=True)
    run_training(args)


if __name__ == "__main__":
    main()


__all__ = [
    "CleanPeptideBatchDataset",
    "peptide_length_groups",
    "UniformLKFLightningModule",
    "build_clean_loader",
    "sample_training_times",
    "uniform_lkf_loss_and_metrics",
    "run_training",
]
