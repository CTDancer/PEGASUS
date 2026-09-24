"""Shared utilities for K-LKF training/evaluation."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def iter_chunks(batch: Tensor, max_sequences: int) -> Iterator[Tensor]:
    if max_sequences <= 0 or batch.shape[0] <= max_sequences:
        yield batch
        return
    for start in range(0, batch.shape[0], max_sequences):
        yield batch[start : start + max_sequences]


def parse_float_list(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        vals = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    else:
        vals = tuple(float(x) for x in value)
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def parse_int_list(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        vals = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    else:
        vals = tuple(int(x) for x in value)
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def parse_interval_list(value: str) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"interval {piece!r} must be s:t")
        s, t = (float(x.strip()) for x in piece.split(":", 1))
        if not (0 <= s < t <= 1):
            raise ValueError(f"invalid interval {piece!r}")
        out.append((s, t))
    if not out:
        raise ValueError("no intervals parsed")
    return tuple(out)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def effective_rank_from_eigenvalues(values: Tensor) -> float:
    eig = torch.as_tensor(values, dtype=torch.float64).clamp_min(0)
    total = eig.sum()
    if float(total) <= 0:
        return 0.0
    p = (eig / total).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()).item())


def normalized_hamming_metrics(tokens: Tensor, *, max_samples: int = 512) -> dict[str, float]:
    x = torch.as_tensor(tokens, dtype=torch.long).cpu()
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 3:
        raise ValueError("tokens must be [N,L] with peptide boundaries")
    interior = x[:, 1:-1]
    n = interior.shape[0]
    unique = torch.unique(interior, dim=0).shape[0]
    use = interior[: min(n, int(max_samples))]
    if use.shape[0] >= 2:
        dist = (use[:, None, :] != use[None, :, :]).float().mean(dim=-1)
        mask = ~torch.eye(use.shape[0], dtype=torch.bool)
        pairwise = float(dist[mask].mean().item())
        nn = dist.masked_fill(~mask, float("inf")).min(dim=1).values
        nearest = float(nn.mean().item())
    else:
        pairwise = float("nan")
        nearest = float("nan")
    runs = []
    has_triple = []
    for row in interior.tolist():
        best = cur = 1
        for a, b in zip(row, row[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        runs.append(best)
        has_triple.append(best >= 3)
    counts = torch.bincount(interior.reshape(-1), minlength=24).float()[4:24]
    probs = counts / counts.sum().clamp_min(1)
    residue_entropy = float((-(probs * probs.clamp_min(1e-30).log()).sum() / math.log(20)).item())
    return {
        "unique_fraction": float(unique / n),
        "pairwise_hamming": pairwise,
        "nearest_neighbor_hamming": nearest,
        "mean_longest_identical_run": float(np.mean(runs)),
        "triple_repeat_fraction": float(np.mean(has_triple)),
        "normalized_residue_entropy": residue_entropy,
    }
