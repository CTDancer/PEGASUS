#!/usr/bin/env python3
"""Simple unconditional peptide generation with Uniform-LKF.

Example:
  python Koopman/scripts/lkf_unconditional_generate.py --length 12 --num-sequences 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # pegasus_release/
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "M8.ckpt"

# ESM amino-acid token IDs used by Uniform-LKF (ids 4..23).
ESM_ID_TO_AA = {
    4: "L",
    5: "A",
    6: "G",
    7: "V",
    8: "S",
    9: "E",
    10: "R",
    11: "T",
    12: "I",
    13: "D",
    14: "P",
    15: "K",
    16: "Q",
    17: "N",
    18: "F",
    19: "Y",
    20: "M",
    21: "H",
    22: "W",
    23: "C",
}


def decode_tokens(token_ids: torch.Tensor) -> list[str]:
    sequences: list[str] = []
    for row in token_ids:
        ids = [int(x) for x in row.tolist()]
        # Drop fixed <cls>/<eos> boundary tokens.
        residues = ids[1:-1]
        aa = []
        for token_id in residues:
            letter = ESM_ID_TO_AA.get(token_id)
            if letter is None:
                raise ValueError(f"Non-amino-acid token id {token_id} in sample {ids}")
            aa.append(letter)
        sequences.append("".join(aa))
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, required=True, help="Peptide length (residues)")
    parser.add_argument("--num-sequences", type=int, required=True, help="Number of sequences to generate")
    parser.add_argument("--nfe", type=int, default=8, help="Native LKF steps (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Sampling batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"LKF checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output fasta/text path; prints to stdout if omitted",
    )
    args = parser.parse_args()

    if args.length < 1:
        raise SystemExit("--length must be >= 1")
    if args.num_sequences < 1:
        raise SystemExit("--num-sequences must be >= 1")
    if args.nfe < 1:
        raise SystemExit("--nfe must be >= 1")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pegasus.lkf.uniform_lkf import UniformLKFProcess, load_uniform_lkf_checkpoint

    device = torch.device(args.device)
    model = load_uniform_lkf_checkpoint(args.checkpoint, map_location=device, eval_mode=True).to(device)
    process = UniformLKFProcess(model)

    # Model seq_len includes <cls> and <eos>.
    token_len = int(args.length) + 2
    max_bio = int(model.seq_len) - 2
    if args.length > max_bio:
        raise SystemExit(f"--length {args.length} exceeds model max biological length {max_bio}")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    all_tokens = []
    remaining = int(args.num_sequences)
    with torch.no_grad():
        while remaining > 0:
            batch = min(int(args.batch_size), remaining)
            tokens = process.sample_terminal(
                batch_size=batch,
                seq_len=token_len,
                nfe=int(args.nfe),
                generator=generator,
            )
            all_tokens.append(tokens.detach().cpu())
            remaining -= batch

    sequences = decode_tokens(torch.cat(all_tokens, dim=0))
    lines = [f"{seq}" for i, seq in enumerate(sequences)]
    text = "\n".join(lines) + "\n"

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {len(sequences)} sequences to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
