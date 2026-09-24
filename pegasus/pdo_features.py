"""Physical fine-action exposure and frozen-LKF features for PEGASUS v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .probes import decode_esm_tokens

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class FineAction:
    index: int
    action_id: str
    position_0based: int | None
    source_aa: str | None
    target_aa: str | None
    target_sequence: str
    tokens: torch.Tensor
    is_noop: bool = False
    # Composite-edit metadata.  These appended fields preserve the positional
    # constructor used by the existing H1 runtime while allowing H2 diagnostics to
    # represent the *actual* compatible edit set explicitly.
    edit_positions_0based: tuple[int, ...] = ()
    source_aas: tuple[str, ...] = ()
    target_aas: tuple[str, ...] = ()
    hamming_distance: int = -1


def find_lkf(obj: Any) -> Any | None:
    """Locate the frozen Uniform-LKF under KFM/residual wrappers."""
    seen: set[int] = set()
    frontier = [obj]
    for _ in range(8):
        nxt = []
        for cur in frontier:
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            if all(
                hasattr(cur, name)
                for name in ("token_embedder", "time_embedder", "shared_blocks", "pos_embedder")
            ):
                return cur
            for attr in ("lkf", "base_model", "model", "backbone", "terminal_model"):
                if hasattr(cur, attr):
                    child = getattr(cur, attr)
                    if child is not None:
                        nxt.append(child)
        frontier = nxt
    return None


@torch.no_grad()
def lkf_hidden(
    lkf: Any,
    tokens: torch.Tensor,
    *,
    time_value: float = 1.0,
    batch_size: int = 256,
) -> np.ndarray:
    """Matched LKF representation used by the historical KFM/LKF diagnostics.

    This is the residue-pooled output after the frozen LKF shared blocks at t=1;
    it intentionally contains no KFM projection or information normalization.
    """
    toks = torch.as_tensor(tokens, dtype=torch.long)
    if toks.ndim == 1:
        toks = toks.reshape(1, -1)
    dev = next(lkf.parameters()).device
    outs: list[np.ndarray] = []
    for lo in range(0, int(toks.shape[0]), int(batch_size)):
        x = toks[lo : lo + int(batch_size)].to(dev)
        if hasattr(lkf, "validate_peptide_tokens"):
            lkf.validate_peptide_tokens(x, name="pdo_lkf_x")
        if hasattr(lkf, "_batch_time"):
            t = lkf._batch_time(
                float(time_value),
                batch_size=x.shape[0],
                device=x.device,
                dtype=lkf.pos_embedder.dtype,
                name="pdo_lkf_time",
            )
        else:
            t = torch.full(
                (x.shape[0],),
                float(time_value),
                device=x.device,
                dtype=lkf.pos_embedder.dtype,
            )
        cond = lkf.time_embedder(t)
        h = lkf.token_embedder(x) + lkf.pos_embedder[:, : x.shape[1]]
        for block in lkf.shared_blocks:
            h = block(h, cond)
        pooled = h[:, 1:-1].mean(dim=1).float().detach().cpu().numpy()
        outs.append(np.asarray(pooled, dtype=np.float64))
    return np.concatenate(outs, axis=0)


def _aa_token_map(lkf: Any) -> tuple[dict[str, int], dict[int, str]]:
    """Recover canonical ESM token IDs without depending on hard-coded ordering."""
    # Historical project convention is canonical ESM IDs 4..23.  decode_esm_tokens is
    # authoritative for the actual mapping in this codebase, so recover it once.
    cfg = getattr(lkf, "config", None)
    ids = tuple(int(x) for x in getattr(cfg, "aa_token_ids", tuple(range(4, 24))))
    cls_id = int(getattr(cfg, "cls_token_id", 0))
    eos_id = int(getattr(cfg, "eos_token_id", 2))
    id_to_aa: dict[int, str] = {}
    for tid in ids:
        probe = torch.tensor([[cls_id, int(tid), eos_id]], dtype=torch.long)
        aa = decode_esm_tokens(probe)[0]
        if len(aa) != 1 or aa not in AA_ALPHABET:
            raise RuntimeError(f"cannot map canonical token ID {tid} to amino acid")
        id_to_aa[int(tid)] = aa
    aa_to_id = {aa: tid for tid, aa in id_to_aa.items()}
    if set(aa_to_id) != set(AA_ALPHABET):
        raise RuntimeError("LKF canonical token set does not match the 20-amino-acid alphabet")
    return aa_to_id, id_to_aa



def encode_sequence(lkf: Any, sequence: str) -> torch.Tensor:
    """Encode a canonical peptide using the frozen LKF token convention."""
    seq = str(sequence).strip().upper()
    aa_to_id, _ = _aa_token_map(lkf)
    try:
        ids = [int(aa_to_id[a]) for a in seq]
    except KeyError as exc:
        raise ValueError(f"noncanonical residue {exc.args[0]!r} in {seq!r}") from exc
    cfg = getattr(lkf, "config", None)
    cls_id = int(getattr(cfg, "cls_token_id", 0))
    eos_id = int(getattr(cfg, "eos_token_id", 2))
    return torch.tensor([cls_id, *ids, eos_id], dtype=torch.long)


def enumerate_hamming1_actions(
    lkf: Any,
    incumbent_tokens: torch.Tensor,
    *,
    include_noop: bool = True,
) -> list[FineAction]:
    """Enumerate every concrete canonical one-substitution action exactly once."""
    x = torch.as_tensor(incumbent_tokens, dtype=torch.long).detach().cpu().reshape(-1)
    if x.numel() < 3:
        raise ValueError("peptide token sequence must include boundary tokens")
    if hasattr(lkf, "validate_peptide_tokens"):
        lkf.validate_peptide_tokens(x.reshape(1, -1), name="pdo_incumbent")
    seq = decode_esm_tokens(x.reshape(1, -1))[0]
    if len(seq) != x.numel() - 2:
        raise RuntimeError("decoded peptide length does not match token interior")
    aa_to_id, id_to_aa = _aa_token_map(lkf)

    out: list[FineAction] = []
    if include_noop:
        out.append(
            FineAction(
                index=0,
                action_id="noop",
                position_0based=None,
                source_aa=None,
                target_aa=None,
                target_sequence=seq,
                tokens=x.clone(),
                is_noop=True,
                edit_positions_0based=(),
                source_aas=(),
                target_aas=(),
                hamming_distance=0,
            )
        )
    for pos, src in enumerate(seq):
        if src not in aa_to_id:
            raise ValueError(f"noncanonical incumbent residue {src!r}")
        for aa in AA_ALPHABET:
            if aa == src:
                continue
            tok = x.clone()
            tok[pos + 1] = int(aa_to_id[aa])
            tgt = seq[:pos] + aa + seq[pos + 1 :]
            out.append(
                FineAction(
                    index=len(out),
                    action_id=f"p{pos:02d}_to_{aa}",
                    position_0based=int(pos),
                    source_aa=src,
                    target_aa=aa,
                    target_sequence=tgt,
                    tokens=tok,
                    is_noop=False,
                    edit_positions_0based=(int(pos),),
                    source_aas=(src,),
                    target_aas=(aa,),
                    hamming_distance=1,
                )
            )
    expected = len(seq) * (len(AA_ALPHABET) - 1) + int(include_noop)
    if len(out) != expected:
        raise RuntimeError("Hamming-1 enumeration invariant failed")
    if len({a.target_sequence for a in out if not a.is_noop}) != expected - int(include_noop):
        raise RuntimeError("Hamming-1 enumeration produced duplicate target sequences")
    return out



def enumerate_hamming2_actions(
    lkf: Any,
    incumbent_tokens: torch.Tensor,
) -> list[FineAction]:
    """Enumerate every compatible exact-distance-2 canonical substitution.

    Each returned action edits two *distinct* positions, so conflicting assignments are
    impossible by construction.  Objective effects are never inferred by additivity: the
    returned tokens encode the concrete two-edit sequence that must be represented and
    evaluated directly.
    """
    x = torch.as_tensor(incumbent_tokens, dtype=torch.long).detach().cpu().reshape(-1)
    if x.numel() < 4:
        raise ValueError("Hamming-2 requires at least two peptide residues")
    if hasattr(lkf, "validate_peptide_tokens"):
        lkf.validate_peptide_tokens(x.reshape(1, -1), name="pdo_h2_incumbent")
    seq = decode_esm_tokens(x.reshape(1, -1))[0]
    if len(seq) != x.numel() - 2:
        raise RuntimeError("decoded peptide length does not match token interior")
    aa_to_id, _ = _aa_token_map(lkf)

    out: list[FineAction] = []
    for i in range(len(seq)):
        src_i = seq[i]
        for j in range(i + 1, len(seq)):
            src_j = seq[j]
            for aa_i in AA_ALPHABET:
                if aa_i == src_i:
                    continue
                for aa_j in AA_ALPHABET:
                    if aa_j == src_j:
                        continue
                    tok = x.clone()
                    tok[i + 1] = int(aa_to_id[aa_i])
                    tok[j + 1] = int(aa_to_id[aa_j])
                    chars = list(seq)
                    chars[i] = aa_i
                    chars[j] = aa_j
                    tgt = "".join(chars)
                    out.append(
                        FineAction(
                            index=len(out),
                            action_id=f"p{i:02d}_to_{aa_i}__p{j:02d}_to_{aa_j}",
                            position_0based=None,
                            source_aa=None,
                            target_aa=None,
                            target_sequence=tgt,
                            tokens=tok,
                            is_noop=False,
                            edit_positions_0based=(int(i), int(j)),
                            source_aas=(src_i, src_j),
                            target_aas=(aa_i, aa_j),
                            hamming_distance=2,
                        )
                    )
    expected = (len(seq) * (len(seq) - 1) // 2) * (len(AA_ALPHABET) - 1) ** 2
    if len(out) != expected:
        raise RuntimeError("Hamming-2 enumeration invariant failed")
    if len({a.target_sequence for a in out}) != expected:
        raise RuntimeError("Hamming-2 enumeration produced duplicate target sequences")
    return out


def enumerate_hamming_leq2_actions(
    lkf: Any,
    incumbent_tokens: torch.Tensor,
    *,
    include_noop: bool = True,
) -> list[FineAction]:
    """Return no-op + all H1 + all exact H2 actions with contiguous indices."""
    h1 = enumerate_hamming1_actions(lkf, incumbent_tokens, include_noop=include_noop)
    h2 = enumerate_hamming2_actions(lkf, incumbent_tokens)
    merged = list(h1) + list(h2)
    return [
        FineAction(
            index=i,
            action_id=a.action_id,
            position_0based=a.position_0based,
            source_aa=a.source_aa,
            target_aa=a.target_aa,
            target_sequence=a.target_sequence,
            tokens=a.tokens,
            is_noop=a.is_noop,
            edit_positions_0based=a.edit_positions_0based,
            source_aas=a.source_aas,
            target_aas=a.target_aas,
            hamming_distance=(
                int(a.hamming_distance)
                if int(a.hamming_distance) >= 0
                else (0 if a.is_noop else 1)
            ),
        )
        for i, a in enumerate(merged)
    ]

def action_feature_matrix_gram(
    lkf: Any,
    actions: Sequence[FineAction],
    incumbent_tokens: torch.Tensor,
    *,
    batch_size: int = 256,
    svd_rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Large-action equivalent of :func:`action_feature_matrix` using V^T V.

    For H<=2 the action matrix is very tall.  Computing its right singular subspace via
    the feature-dimensional Gram matrix is substantially cheaper.  The resulting basis
    can differ from direct SVD by signs/rotations inside degenerate eigenspaces, but Q Q^T
    and all isotropic-ridge / information-geometry calculations are unchanged.  Production
    H1 continues to use the original direct-SVD function.
    """
    if not actions:
        raise ValueError("empty action family")
    toks=torch.stack([a.tokens for a in actions],dim=0)
    z=lkf_hidden(lkf,toks,batch_size=batch_size)
    xz=lkf_hidden(lkf,torch.as_tensor(incumbent_tokens).reshape(1,-1),batch_size=batch_size)[0]
    V=np.asarray(z-xz.reshape(1,-1),dtype=np.float64)
    for i,a in enumerate(actions):
        if a.is_noop: V[i]=0.0
    norms=np.linalg.norm(V,axis=1); positive=norms[norms>1e-12]
    scale=float(np.median(positive)) if positive.size else 1.0; scale=max(scale,1e-12)
    Vs=V/scale
    if not np.any(np.abs(Vs)>0):
        return V,np.zeros((len(actions),0),dtype=np.float64),np.zeros((V.shape[1],0)),scale
    G=Vs.T@Vs
    ev,evec=np.linalg.eigh(0.5*(G+G.T))
    order=np.argsort(ev)[::-1]; ev=np.maximum(ev[order],0.0); evec=evec[:,order]
    s=np.sqrt(ev)
    tol=max(float(svd_rtol)*float(s[0]),1e-12)
    rank=int(np.sum(s>tol))
    basis=evec[:,:rank].copy()
    Q=Vs@basis
    return V,Q,basis,scale


def action_feature_matrix(
    lkf: Any,
    actions: Sequence[FineAction],
    incumbent_tokens: torch.Tensor,
    *,
    batch_size: int = 256,
    svd_rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return LKF displacements and numerically stable decision-span coordinates.

    Returns ``(V, Q, basis, global_scale)`` where V contains raw LKF intervention
    displacements and Q is V projected onto its empirical decision span.  Only a single
    scalar normalization is applied for conditioning; no direction-wise whitening or
    learned decision balancing is introduced in v1.
    """
    if not actions:
        raise ValueError("empty action family")
    toks = torch.stack([a.tokens for a in actions], dim=0)
    z = lkf_hidden(lkf, toks, batch_size=batch_size)
    xz = lkf_hidden(
        lkf,
        torch.as_tensor(incumbent_tokens).reshape(1, -1),
        batch_size=batch_size,
    )[0]
    V = np.asarray(z - xz.reshape(1, -1), dtype=np.float64)
    # Force the no-op displacement to exact zero rather than relying on floating point.
    for i, a in enumerate(actions):
        if a.is_noop:
            V[i] = 0.0

    norms = np.linalg.norm(V, axis=1)
    positive = norms[norms > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)
    Vs = V / scale
    if not np.any(np.abs(Vs) > 0):
        return V, np.zeros((len(actions), 0), dtype=np.float64), np.zeros((V.shape[1], 0)), scale
    _u, s, vt = np.linalg.svd(Vs, full_matrices=False)
    tol = max(float(svd_rtol) * float(s[0]), 1e-12)
    rank = int(np.sum(s > tol))
    basis = vt[:rank].T.copy()
    Q = Vs @ basis
    return V, Q, basis, scale


__all__ = [
    "AA_ALPHABET",
    "FineAction",
    "find_lkf",
    "lkf_hidden",
    "encode_sequence",
    "enumerate_hamming1_actions",
    "enumerate_hamming2_actions",
    "enumerate_hamming_leq2_actions",
    "action_feature_matrix",
    "action_feature_matrix_gram",
]
