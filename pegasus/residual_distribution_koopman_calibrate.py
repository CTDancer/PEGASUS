"""Objective-free calibration of a shared residual law around the Koopman mean.

No neural model is trained.  For each finite-time chain C we sample many hard
terminal continuations from training states and decompose

    z_T = mu_hat_C(x_s) + e_predictive
        = E_emp[z_T|x_s] + e_intrinsic.

The checkpoint stores state-balanced empirical residual banks.  The intrinsic
bank describes stochastic branching after removing each state's empirical mean;
the predictive bank also includes direct-mean model error and therefore gives an
end-to-end predictive law for new states.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .lkf.data import CleanPeptideBatchDataset, build_clean_loader
from .residual_distribution_koopman_model import (
    ResidualDistributionConfig,
    save_residual_distribution_checkpoint,
)
from .residual_distribution_koopman_stats import (
    covariance,
    directional_profiles,
    fixed_orthonormal_directions,
    safe_cosine,
)
from .terminal_controlled_koopman_checkpoint import load_terminal_controlled_koopman_checkpoint
from .terminal_controlled_koopman_geometry import chain_name, parse_chains, prepare_chain_start
from .utils import write_json


def _collect_clean_states(root: str | Path, split: str, *, token_length: int, max_states: int) -> torch.Tensor:
    # CUDA is already initialized by checkpoint loading in the normal call path;
    # project DDP rules therefore require num_workers=0.
    ds = CleanPeptideBatchDataset(root, split)
    chunks: list[torch.Tensor] = []
    total = 0
    for batch in build_clean_loader(ds, shuffle=False, num_workers=0):
        x = torch.as_tensor(batch, dtype=torch.long)
        if int(x.shape[1]) != int(token_length):
            continue
        take = min(int(max_states) - total, int(x.shape[0]))
        if take > 0:
            chunks.append(x[:take].cpu().contiguous())
            total += take
        if total >= int(max_states):
            break
    if not chunks:
        raise ValueError(f"no states of token_length={token_length} in split={split!r}")
    out = torch.cat(chunks, dim=0)
    if out.shape[0] < 16:
        raise ValueError(f"only {out.shape[0]} usable states; need at least 16")
    return out


@torch.no_grad()
def _sample_terminal_z(model, x_start: torch.Tensor, chain: Sequence[float], *, continuations: int, chunk: int, seed: int) -> np.ndarray:
    device = next(model.parameters()).device
    xs = x_start.to(device)
    b, length = xs.shape
    pieces: list[torch.Tensor] = []
    done = 0
    while done < int(continuations):
        m = min(int(chunk), int(continuations) - done)
        xr = xs[:, None, :].expand(b, m, length).reshape(b * m, length).contiguous()
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + 104729 * done)
        terminal = model.sample_chain(xr, chain, generator=g)
        r = model.anchor_features(terminal, 1.0)
        z = model.information_normalize(r).reshape(b, m, -1)
        pieces.append(z.cpu())
        done += m
    return torch.cat(pieces, dim=1).numpy().astype(np.float64)


@torch.no_grad()
def _source_and_pred_mean(model, x_start: torch.Tensor, chain: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    xs = x_start.to(device)
    r0 = model.anchor_features(xs, float(chain[0]))
    z0 = model.information_normalize(r0)
    pred = model.information_normalize(model.predict_uncontrolled_mean(r0, chain))
    return z0.cpu().numpy().astype(np.float64), pred.cpu().numpy().astype(np.float64)


def _balanced_bank(residuals: np.ndarray, max_bank_size: int, seed: int) -> np.ndarray:
    x = np.asarray(residuals, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("residuals must be [states,continuations,dim]")
    s, m, d = x.shape
    flat = x.reshape(s * m, d)
    n = min(int(max_bank_size), flat.shape[0])
    if n == flat.shape[0]:
        return flat.astype(np.float32)
    # Each state has equal continuation count, so uniform flatten sampling is
    # state-balanced in expectation.  Seed is fixed for exact reproducibility.
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(flat.shape[0], size=n, replace=False)
    return flat[idx].astype(np.float32)


def _global_summary(bank: np.ndarray, directions: np.ndarray, quantiles, betas) -> dict[str, Any]:
    prof = directional_profiles(bank, directions, quantiles=quantiles, betas=betas)
    return {
        "mean": np.mean(bank, axis=0).astype(np.float32),
        "covariance": covariance(bank).astype(np.float32),
        **{k: np.asarray(v, dtype=np.float32) for k, v in prof.items()},
    }


def run(args) -> Path:
    device = torch.device(args.device)
    model, _ = load_terminal_controlled_koopman_checkpoint(
        args.koopman_checkpoint,
        base_lkf_checkpoint=args.lkf_checkpoint,
        device=device,
        strict_base_sha=not bool(args.allow_base_sha_mismatch),
        eval_mode=True,
    )
    chains = parse_chains(args.chains) if args.chains else model.chains
    for c in chains:
        model.match_chain(c)
    clean = _collect_clean_states(
        args.dataset_root, args.train_split, token_length=int(args.token_length), max_states=int(args.states)
    )
    cfg = ResidualDistributionConfig(
        chains=tuple(tuple(float(x) for x in c) for c in chains),
        direction_count=int(args.direction_count),
        direction_seed=int(args.direction_seed),
        quantiles=tuple(float(x) for x in args.quantiles.split(",") if x.strip()),
        betas=tuple(float(x) for x in args.betas.split(",") if x.strip()),
        max_bank_size=int(args.bank_size),
    )
    directions = fixed_orthonormal_directions(model.anchor_dim, cfg.direction_count, cfg.direction_seed)
    laws: dict[str, dict[str, Any]] = {}
    rows = []
    print(
        f"Calibrating shared residual laws: states={clean.shape[0]}, continuations={args.continuations}, "
        f"chains={len(chains)}, device={device}", flush=True,
    )

    for ci, chain in enumerate(chains):
        c = model.match_chain(chain)
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + 100003 * ci)
        x_start = prepare_chain_start(model, clean.to(device), c, generator=g)
        z0, pred_mean = _source_and_pred_mean(model, x_start, c)
        z = _sample_terminal_z(
            model, x_start, c,
            continuations=int(args.continuations), chunk=int(args.continuation_chunk),
            seed=int(args.seed) + 200003 * ci,
        )
        empirical_mean = z.mean(axis=1)
        intrinsic = z - empirical_mean[:, None, :]
        predictive = z - pred_mean[:, None, :]
        mean_error = empirical_mean - pred_mean

        intrinsic_bank = _balanced_bank(intrinsic, int(args.bank_size), int(args.seed) + 3001 * ci)
        predictive_bank = _balanced_bank(predictive, int(args.bank_size), int(args.seed) + 4001 * ci)
        state_prof = directional_profiles(intrinsic, directions, quantiles=cfg.quantiles, betas=cfg.betas)
        global_intrinsic = _global_summary(intrinsic_bank, directions, cfg.quantiles, cfg.betas)
        global_predictive = _global_summary(predictive_bank, directions, cfg.quantiles, cfg.betas)

        # Training split-half sanity check.  This does not replace validation but
        # catches a catastrophically undersampled bank immediately.
        half = z.shape[1] // 2
        ia = z[:, :half] - z[:, :half].mean(axis=1, keepdims=True)
        ib = z[:, half : 2 * half] - z[:, half : 2 * half].mean(axis=1, keepdims=True)
        ca = covariance(ia.reshape(-1, ia.shape[-1]))
        cb = covariance(ib.reshape(-1, ib.shape[-1]))
        split_cov_cos = safe_cosine(ca, cb)

        laws[chain_name(c)] = {
            "intrinsic_bank": intrinsic_bank,
            "predictive_bank": predictive_bank,
            "directions": directions.astype(np.float32),
            "global_intrinsic": global_intrinsic,
            "global_predictive": global_predictive,
            "state_summaries": {
                "source_z": z0.astype(np.float32),
                "mean_error": mean_error.astype(np.float32),
                **{k: np.asarray(v, dtype=np.float32) for k, v in state_prof.items()},
            },
            "calibration_states": int(z.shape[0]),
            "calibration_continuations": int(z.shape[1]),
            "train_intrinsic_covariance_split_cosine": float(split_cov_cos),
        }
        row = {
            "chain": chain_name(c),
            "mean_model_cosine": safe_cosine(pred_mean, empirical_mean),
            "mean_model_relative_rmse": float(np.linalg.norm(pred_mean-empirical_mean)/max(np.linalg.norm(empirical_mean),1e-12)),
            "mean_error_rms": float(np.sqrt(np.mean(mean_error**2))),
            "intrinsic_bank_size": int(intrinsic_bank.shape[0]),
            "predictive_bank_size": int(predictive_bank.shape[0]),
            "train_intrinsic_covariance_split_cosine": float(split_cov_cos),
        }
        rows.append(row)
        print(
            f"[chain {chain_name(c)}] mean_rel={row['mean_model_relative_rmse']:.4f} "
            f"cov_split={split_cov_cos:.3f} bank={intrinsic_bank.shape[0]}", flush=True,
        )

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "chains": rows,
        "mean_relative_rmse_mean": float(np.mean([r["mean_model_relative_rmse"] for r in rows])),
        "train_intrinsic_covariance_split_cosine_mean": float(np.mean([r["train_intrinsic_covariance_split_cosine"] for r in rows])),
        "states": int(clean.shape[0]),
        "continuations": int(args.continuations),
        "objective_free": True,
    }
    ckpt = save_residual_distribution_checkpoint(
        out / "checkpoints" / "calibrated.pt",
        terminal_checkpoint=args.koopman_checkpoint,
        base_lkf_checkpoint=args.lkf_checkpoint,
        config=cfg,
        laws=laws,
        calibration_args=vars(args),
        calibration_summary=summary,
    )
    write_json(out / "calibration_summary.json", {**summary, "checkpoint": str(ckpt)})
    print({**summary, "checkpoint": str(ckpt)}, flush=True)
    return ckpt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--koopman-checkpoint", required=True)
    p.add_argument("--lkf-checkpoint", required=True)
    p.add_argument("--allow-base-sha-mismatch", action="store_true")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--token-length", type=int, default=14)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chains", default="0,1;0.25,1;0.5,1;0.75,1;0,0.5,1;0,0.25,0.5,1")
    p.add_argument("--states", type=int, default=128)
    p.add_argument("--continuations", type=int, default=128)
    p.add_argument("--continuation-chunk", type=int, default=32)
    p.add_argument("--bank-size", type=int, default=16384)
    p.add_argument("--direction-count", type=int, default=32)
    p.add_argument("--direction-seed", type=int, default=84217)
    p.add_argument("--quantiles", default="0.05,0.1,0.5,0.8,0.9,0.95")
    p.add_argument("--betas", default="0.5,1,2,4")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
