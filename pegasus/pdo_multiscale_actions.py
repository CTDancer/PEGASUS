"""Concrete exact-Hamming action exposure for scale-factored PDO.

This module is deliberately independent of the experimental multiscale diagnostics.
It extends only the original PEGASUS v3 fine-action idea: actions are actual discrete
sequences around the current incumbent, and no high-order objective additivity is assumed.

For small H1 neighborhoods the complete action family is enumerated exactly.  For larger
neighborhoods (or long sequences) a fixed-size set of exact-Hamming-k actions is sampled
uniformly from compatible position/substitution choices.  Every returned action is a
fully realized canonical sequence that can be scored by the frozen KFM terminal readout
and, if selected, queried by the exact oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch

from . import koopman_sparse_verify_optimizer_v1_1_batched_diversity as base
from .pdo_features import AA_ALPHABET, FineAction, encode_sequence, enumerate_hamming1_actions
from .probes import decode_esm_tokens


@dataclass(frozen=True)
class MultiscaleAction:
    """One concrete non-noop intervention action at an exact Hamming radius."""

    index: int
    k: int
    action_id: str
    target_sequence: str
    tokens: torch.Tensor
    edited_positions_0based: tuple[int, ...]

    @property
    def is_noop(self) -> bool:
        return False

    def as_fine_action(self) -> FineAction:
        """Compatibility adapter for the original v3 exact-acceptance helper."""
        return FineAction(
            index=int(self.index),
            action_id=str(self.action_id),
            position_0based=(
                int(self.edited_positions_0based[0])
                if len(self.edited_positions_0based) == 1
                else None
            ),
            source_aa=None,
            target_aa=None,
            target_sequence=str(self.target_sequence),
            tokens=self.tokens.detach().cpu().clone(),
            is_noop=False,
        )


def parse_radius_spec(spec: str, sequence_length: int) -> tuple[int, ...]:
    """Resolve a comma-separated radius specification to unique integer Hamming radii.

    Accepted entries:
    * ``1`` / ``2`` / ``4``: exact Hamming radius.
    * ``2%``: percentage of sequence length (rounded to nearest integer, minimum one).
    * ``0.05L``: fraction of sequence length (same semantics as 5%).

    This keeps the current peptide default explicit while allowing length-normalized
    configurations for proteins without changing the optimizer implementation.
    """
    L = int(sequence_length)
    if L <= 0:
        raise ValueError("sequence_length must be positive")
    text = str(spec).strip()
    if not text:
        raise ValueError("radius specification cannot be empty")
    out: list[int] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.endswith("%"):
            frac = float(token[:-1]) / 100.0
            if not math.isfinite(frac) or frac <= 0:
                raise ValueError(f"invalid percentage radius {token!r}")
            k = max(1, int(round(frac * L)))
        elif token.lower().endswith("l"):
            frac = float(token[:-1])
            if not math.isfinite(frac) or frac <= 0:
                raise ValueError(f"invalid fractional radius {token!r}")
            k = max(1, int(round(frac * L)))
        else:
            value = float(token)
            if not math.isfinite(value) or value <= 0 or abs(value - round(value)) > 1e-12:
                raise ValueError(f"exact Hamming radius must be a positive integer: {token!r}")
            k = int(round(value))
        if k > L:
            raise ValueError(f"requested Hamming radius {k} exceeds sequence length {L}")
        if k not in out:
            out.append(k)
    if not out:
        raise ValueError("radius specification produced no radii")
    return tuple(out)


def exact_hamming_distance(a: str, b: str) -> int:
    aa = str(a)
    bb = str(b)
    if len(aa) != len(bb):
        raise ValueError("sequences must have equal length")
    return int(sum(x != y for x, y in zip(aa, bb)))


def _anchor_admissible(
    tokens: torch.Tensor,
    anchors: Sequence[torch.Tensor],
    min_lineage_hamming: float,
) -> bool:
    floor = float(min_lineage_hamming)
    return all(base._hamming(tokens, other) >= floor - 1e-12 for other in anchors)


def _sample_target_sequence(
    incumbent_sequence: str,
    *,
    k: int,
    rng: np.random.Generator,
) -> tuple[str, tuple[int, ...]]:
    seq = str(incumbent_sequence)
    L = len(seq)
    kk = int(k)
    positions = tuple(sorted(int(x) for x in rng.choice(L, size=kk, replace=False).tolist()))
    chars = list(seq)
    for pos in positions:
        src = chars[pos]
        choices = [aa for aa in AA_ALPHABET if aa != src]
        chars[pos] = str(choices[int(rng.integers(0, len(choices)))])
    target = "".join(chars)
    if exact_hamming_distance(seq, target) != kk:
        raise RuntimeError("exact-Hamming sampler invariant failed")
    return target, positions


def expose_exact_hamming_actions(
    lkf: object,
    incumbent_tokens: torch.Tensor,
    *,
    k: int,
    candidate_cap: int,
    rng: np.random.Generator,
    anchors: Sequence[torch.Tensor] = (),
    min_lineage_hamming: float = 0.0,
    exhaustive_h1_max_actions: int = 2048,
    max_sampling_attempts: int = 100000,
) -> list[MultiscaleAction]:
    """Expose a finite concrete action family at exact Hamming radius ``k``.

    H1 is enumerated exactly whenever its complete canonical neighborhood is no larger
    than ``exhaustive_h1_max_actions``.  Otherwise actions are sampled uniformly from
    exact-k compatible edits until ``candidate_cap`` unique diversity-admissible targets
    have been collected or the attempt budget is exhausted.
    """
    x = torch.as_tensor(incumbent_tokens, dtype=torch.long).detach().cpu().reshape(-1)
    if x.numel() < 3:
        raise ValueError("incumbent tokens must include boundary tokens")
    seq = decode_esm_tokens(x.reshape(1, -1))[0]
    L = len(seq)
    kk = int(k)
    if kk <= 0 or kk > L:
        raise ValueError("k must lie in [1, sequence_length]")
    cap = int(candidate_cap)
    if cap <= 0:
        raise ValueError("candidate_cap must be positive")
    if int(exhaustive_h1_max_actions) < 0:
        raise ValueError("exhaustive_h1_max_actions cannot be negative")
    if int(max_sampling_attempts) <= 0:
        raise ValueError("max_sampling_attempts must be positive")

    # Exact v3 H1 enumeration when tractable.  This preserves the original physical
    # action family exactly rather than approximating H1 with random samples.
    h1_total = L * (len(AA_ALPHABET) - 1)
    if kk == 1 and h1_total <= int(exhaustive_h1_max_actions):
        fine = enumerate_hamming1_actions(lkf, x, include_noop=False)
        out: list[MultiscaleAction] = []
        for a in fine:
            if not _anchor_admissible(a.tokens, anchors, min_lineage_hamming):
                continue
            out.append(
                MultiscaleAction(
                    index=len(out),
                    k=1,
                    action_id=f"h1::{a.action_id}",
                    target_sequence=str(a.target_sequence),
                    tokens=a.tokens.detach().cpu().clone(),
                    edited_positions_0based=(int(a.position_0based),),
                )
            )
        # ``candidate_cap`` does not truncate exact H1: when enumeration is explicitly
        # chosen, preserving the complete v3 H1 action set is more important than a
        # uniform cap.  Long H1 neighborhoods fall back to sampled exposure instead.
        return out

    total_possible = math.comb(L, kk) * (len(AA_ALPHABET) - 1) ** kk
    target_count = min(cap, int(total_possible))
    seen: set[str] = set()
    out: list[MultiscaleAction] = []
    attempts = 0
    while len(out) < target_count and attempts < int(max_sampling_attempts):
        attempts += 1
        target, positions = _sample_target_sequence(seq, k=kk, rng=rng)
        if target in seen:
            continue
        seen.add(target)
        tok = encode_sequence(lkf, target)
        if not _anchor_admissible(tok, anchors, min_lineage_hamming):
            continue
        out.append(
            MultiscaleAction(
                index=len(out),
                k=kk,
                action_id=(
                    f"h{kk}::" + ";".join(f"p{p:04d}_{seq[p]}>{target[p]}" for p in positions)
                ),
                target_sequence=target,
                tokens=tok.detach().cpu().clone(),
                edited_positions_0based=positions,
            )
        )
    return out


def verify_action_family(
    actions: Iterable[MultiscaleAction],
    incumbent_sequence: str,
    *,
    k: int,
) -> None:
    """Hard correctness check used by production before any KFM/oracle call."""
    rows = list(actions)
    seqs = [str(a.target_sequence) for a in rows]
    if len(seqs) != len(set(seqs)):
        raise RuntimeError("multiscale action family contains duplicate target sequences")
    for i, a in enumerate(rows):
        if int(a.index) != i:
            raise RuntimeError("multiscale action indices must be contiguous")
        if int(a.k) != int(k):
            raise RuntimeError("multiscale action has incorrect scale label")
        if exact_hamming_distance(incumbent_sequence, a.target_sequence) != int(k):
            raise RuntimeError("multiscale action does not realize the requested exact Hamming radius")


__all__ = [
    "MultiscaleAction",
    "parse_radius_spec",
    "exact_hamming_distance",
    "expose_exact_hamming_actions",
    "verify_action_family",
]
