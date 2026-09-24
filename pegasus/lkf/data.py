"""Minimal clean-peptide dataset and Uniform-LKF loss utilities.

This is the only dataset code Koopman training needs.  It deliberately omits the
original Lightning trainer so the standalone project does not depend on the TCFM
package layout or PyTorch Lightning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .uniform_lkf import UniformLKF


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


# HuggingFace DatasetDict on this host uses short names: train / val / test.
# Do not invent a "validation" split.  Alias common synonyms so trainers that
# default to HuggingFace's longer names still load the existing held-out set.
_SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "evaluate": "test",
    "eval": "test",
}


def _split_candidates(split: str) -> list[str]:
    name = str(split).strip()
    if not name:
        raise ValueError("split name must be non-empty")
    names = [name]
    alias = _SPLIT_ALIASES.get(name.lower())
    if alias and alias not in names:
        names.append(alias)
    return names


def _resolve_hf_split(root: str, split: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The datasets package is required for Koopman-LKF training") from exc
    root_path = Path(root).expanduser()
    candidates = _split_candidates(split)
    for name in candidates:
        child = root_path / name
        if child.exists():
            return load_from_disk(str(child)), str(child.resolve())
    obj = load_from_disk(str(root_path))
    last_exc: Optional[BaseException] = None
    for name in candidates:
        try:
            return obj[name], f"{root_path.resolve()}[{name}]"
        except Exception as exc:
            last_exc = exc
    available = None
    if hasattr(obj, "keys"):
        try:
            available = list(obj.keys())
        except Exception:
            available = None
    extra = f" (available splits: {available})" if available is not None else ""
    raise ValueError(
        f"Could not resolve split {split!r} from {root!r}{extra}. "
        "This peptide DatasetDict uses train/val/test, not validation."
    ) from last_exc


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


def _chunk_ranges(n: int, max_sequences: int, *, min_sequences: int = 2) -> list[tuple[int, int]]:
    """Split ``n`` items into contiguous ranges of size at most ``max_sequences``.

    Remainders smaller than ``min_sequences`` are merged into the previous chunk
    when possible (slightly exceeding ``max_sequences``), otherwise dropped.
    Centered covariance / effective-rank terms need at least two samples, so
    singleton optimizer steps must not enter DDP training.
    """
    if n <= 0:
        return []
    if int(min_sequences) < 1:
        raise ValueError(f"min_sequences must be >=1, got {min_sequences}")
    if int(max_sequences) < int(min_sequences):
        raise ValueError(
            f"max_sequences ({max_sequences}) must be >= min_sequences ({min_sequences})"
        )
    ranges: list[tuple[int, int]] = []
    for start in range(0, n, int(max_sequences)):
        stop = min(start + int(max_sequences), n)
        ranges.append((start, stop))
    if len(ranges) >= 2:
        last_start, last_stop = ranges[-1]
        if last_stop - last_start < int(min_sequences):
            prev_start, _ = ranges[-2]
            ranges[-2] = (prev_start, last_stop)
            ranges.pop()
    elif ranges and ranges[0][1] - ranges[0][0] < int(min_sequences):
        ranges = []
    return ranges


class ChunkedCleanPeptideBatchDataset(Dataset):
    """Pre-chunked peptide batches so each item is one optimizer step.

    Same-length HF rows can contain more sequences than a GPU should process in
    one step.  Splitting those rows *before* ``DistributedSampler`` makes the
    dataset length equal the number of optimizer steps, so every DDP rank sees
    the same step count (no nested per-rank chunking imbalance).

    Chunks always contain at least two sequences so effective-rank regularization
    (which needs a centered covariance) never crashes a single DDP rank.
    """

    def __init__(
        self,
        root: str,
        split: str,
        *,
        max_sequences: int,
        min_sequences: int = 2,
        pad_token_id: int = 1,
    ):
        if int(max_sequences) <= 0:
            raise ValueError(f"max_sequences must be positive, got {max_sequences}")
        if int(min_sequences) < 1:
            raise ValueError(f"min_sequences must be >=1, got {min_sequences}")
        self.max_sequences = int(max_sequences)
        self.min_sequences = int(min_sequences)
        self.pad_token_id = int(pad_token_id)
        self.dataset, self.resolved_source = _resolve_hf_split(root, split)
        # (source_row, token_length, start, stop) within that length group
        self._index: list[tuple[int, int, int, int]] = []
        self._length_counts: dict[int, int] = {}
        self._source_rows = 0
        self._sequences = 0
        self._dropped_sequences = 0
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
                n = int(batch.shape[0])
                self._length_counts[int(length)] = self._length_counts.get(int(length), 0) + n
                self._sequences += n
                kept = 0
                for start, stop in _chunk_ranges(
                    n, self.max_sequences, min_sequences=self.min_sequences
                ):
                    self._index.append((int(row_index), int(length), int(start), int(stop)))
                    kept += stop - start
                self._dropped_sequences += n - kept
        if not self._index:
            raise ValueError(f"No peptide sequences found in {self.resolved_source}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Tensor:
        row_index, length, start, stop = self._index[int(index)]
        record = self.dataset[int(row_index)]
        groups = peptide_length_groups(
            record["input_ids"],
            record.get("attention_mask"),
            pad_token_id=self.pad_token_id,
        )
        chunk = groups[int(length)][start:stop].contiguous()
        if int(chunk.shape[0]) < self.min_sequences:
            raise RuntimeError(
                f"chunk size {int(chunk.shape[0])} < min_sequences={self.min_sequences}"
            )
        return chunk

    def length_statistics(self) -> dict[str, Any]:
        counts = dict(sorted(self._length_counts.items()))
        return {
            "source": self.resolved_source,
            "records": len(self._index),
            "chunks": len(self._index),
            "max_sequences": self.max_sequences,
            "min_sequences": self.min_sequences,
            "source_rows": self._source_rows,
            "sequences": self._sequences,
            "dropped_sequences": self._dropped_sequences,
            "min_token_length": min(counts),
            "max_token_length": max(counts),
            "length_counts": counts,
        }


def build_clean_loader(
    dataset: Dataset,
    *,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    if sampler is not None and shuffle:
        raise ValueError("DataLoader cannot use both sampler and shuffle=True")
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
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


__all__ = [
    "ChunkedCleanPeptideBatchDataset",
    "CleanPeptideBatchDataset",
    "build_clean_loader",
    "peptide_length_groups",
    "uniform_lkf_loss_and_metrics",
]
