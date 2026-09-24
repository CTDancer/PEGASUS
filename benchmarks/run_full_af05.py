#!/usr/bin/env python3
"""Full-K PEGASUS v3 with KOOPMAN_ADAPTIVE_FRACTION=0.5.

Matched panel to the E4 AF=0.5 ablations:
  8 targets × 3 seeds × 2 lengths = 48 jobs
  radius ladder 1,2,4,8
  same pegasus_joint_population_pdo.sh launcher (default Full v3 runner)

Outputs (does not overwrite E5 R1248 AF=0.25):
  E4/full_af05/L{12,40}/{target}_seed{seed}/

Job order: L40 first (fills the initial 24 slots), then L12 from the queue.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
QUEUE = BENCH / "E4_FULL_AF05_JOB_MANIFEST.csv"
LOCK = BENCH / "full_af05_queue.lock"
LOG = BENCH / "logs/full_af05"
PYTHON = os.environ.get("PEGASUS_PYTHON", "python")
LAUNCH = ROOT / "scripts/pegasus_joint_population_pdo.sh"
TARGET_MANIFEST = ROOT / "benchmarks/target_manifest.json"
# E3 Panel-B targets with known binders (from PEGASUS/benchmark_sequences.xlsx).
EXTRA_TARGET_SEQ_DIR = ROOT / "benchmarks/targets"

TARGETS = [
    "AMHR2", "CLK1", "DUSP12", "EWS_FLI1", "MYC", "PPP5", "TM3", "UCHL5",
    # Missing Full AF=0.5 joint-v3 coverage for E3 Panel B:
    "3IDJ", "5AZ8", "7JVS",
]
SEEDS = [42, 73, 101]
LENGTHS = [40, 12]  # L40 claimed first
RADIUS = "1,2,4,8"
ADAPTIVE_FRACTION = "0.5"
SUBDIR = "full_af05"
# Default Full joint-v3 runner (not an ablation module).
RUNNER_MODULE = "pegasus.pdo_pegasus_joint_population_pdo_v3"

FIELDS = [
    "experiment", "variant", "target", "seed", "length",
    "script", "config", "output_dir", "device", "status", "notes",
]


def target_sequences() -> dict[str, str]:
    payload = json.loads(TARGET_MANIFEST.read_text())
    seqs: dict[str, str] = {}
    for e in payload["targets"]:
        path = Path(e["seq_file"])
        if not path.is_absolute():
            path = ROOT / path
        seqs[e["run_tag"]] = path.read_text().strip()
    for path in sorted(EXTRA_TARGET_SEQ_DIR.glob("*.seq")):
        seqs[path.stem] = path.read_text().strip()
    missing = [t for t in TARGETS if t not in seqs]
    if missing:
        raise SystemExit(f"Missing target sequences for: {missing}")
    return seqs


def build_manifest(rewrite: bool = False) -> None:
    existing: dict[str, dict] = {}
    if QUEUE.exists() and not rewrite:
        with QUEUE.open() as handle:
            for row in csv.DictReader(handle):
                existing[row["output_dir"]] = row

    rows: list[dict] = []
    for length in LENGTHS:
        for target in TARGETS:
            for seed in SEEDS:
                out = ROOT / "results" / SUBDIR / f"L{length}" / f"{target}_seed{seed}"
                prior = existing.get(str(out))
                if prior:
                    rows.append(prior)
                    continue
                done = (out / "summary.json").is_file()
                rows.append({
                    "experiment": "E4",
                    "variant": "full_af05",
                    "target": target,
                    "seed": seed,
                    "length": length,
                    "script": str(LAUNCH),
                    "config": RADIUS,
                    "output_dir": str(out),
                    "device": "",
                    "status": "done" if done else "pending",
                    "notes": (
                        f"Full-K AF={ADAPTIVE_FRACTION}; "
                        f"launcher={LAUNCH.name}; runner={RUNNER_MODULE}"
                    ),
                })

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows() -> list[dict]:
    with QUEUE.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict]) -> None:
    tmp = QUEUE.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(QUEUE)


def claim(gpu: int) -> dict | None:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_rows()
        pending = [r for r in rows if r["status"] == "pending"]
        if not pending:
            return None
        # Longest first so the initial 24 slots are all L40.
        pending.sort(key=lambda r: (-int(r["length"]), r["target"], int(r["seed"])))
        chosen = pending[0]
        for row in rows:
            if row["output_dir"] == chosen["output_dir"] and row["status"] == "pending":
                row["status"] = "running"
                row["device"] = f"cuda:{gpu}"
                break
        write_rows(rows)
        return chosen


def finish(output_dir: str, status: str, note: str = "") -> None:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_rows()
        for row in rows:
            if row["output_dir"] == output_dir and row["status"] == "running":
                row["status"] = status
                if note:
                    row["notes"] = note
        write_rows(rows)


def run_job(row: dict, gpu: int, sequences: dict[str, str]) -> int:
    out = Path(row["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "DEVICE": "cuda:0",
        "PEPTIVERSE_DEVICE": "cuda",
        "RUNNER_MODULE": RUNNER_MODULE,
        "PEPTIDE_LENGTH": str(row["length"]),
        "SEED": str(row["seed"]),
        "TARGET": sequences[row["target"]],
        "PDO_MULTISCALE_RADIUS_SPEC": row["config"],
        "OUTPUT_DIR": str(out),
        "STAGE": "run",
        "POPULATION_MODE": "contraction",
        "PDO_JOINT_ACQUISITION_MODE": "shared_rank",
        "KOOPMAN_ADAPTIVE_FRACTION": ADAPTIVE_FRACTION,
        "PYTHONPATH": str(ROOT),
        "PATH": f"{Path(PYTHON).parent}:{env.get('PATH', '')}",
    })
    LOG.mkdir(parents=True, exist_ok=True)
    log = LOG / f"full_af05_L{row['length']}_{row['target']}_seed{row['seed']}.log"
    with log.open("w") as handle:
        handle.write(
            f"# Full-K AF={ADAPTIVE_FRACTION} L{row['length']} {row['target']} "
            f"seed{row['seed']} gpu={gpu} runner={RUNNER_MODULE}\n"
        )
        handle.flush()
        proc = subprocess.run(
            ["bash", str(LAUNCH)], cwd=str(ROOT), env=env,
            stdout=handle, stderr=subprocess.STDOUT,
        )
    return int(proc.returncode)


def worker(gpu: int) -> None:
    sequences = target_sequences()
    while True:
        row = claim(gpu)
        if row is None:
            if not any(r["status"] == "running" for r in read_rows()):
                return
            time.sleep(20)
            continue
        code = run_job(row, gpu, sequences)
        ok = code == 0 and (Path(row["output_dir"]) / "summary.json").is_file()
        # Soft AF check
        note = ""
        if ok:
            try:
                af = float(
                    (json.loads((Path(row["output_dir"]) / "summary.json").read_text())
                     .get("search") or {}).get("koopman_adaptive_fraction")
                )
                if abs(af - float(ADAPTIVE_FRACTION)) > 1e-9:
                    ok = False
                    note = f"AF mismatch: got {af}, expected {ADAPTIVE_FRACTION}"
            except Exception as exc:
                ok = False
                note = f"AF audit failed: {exc}"
        finish(row["output_dir"], "done" if ok else "failed", note=note)
        print(
            f"[gpu {gpu}] full_af05 L{row['length']} {row['target']} "
            f"seed{row['seed']} -> rc={code} {'ok' if ok else 'FAILED'}"
            + (f" ({note})" if note else ""),
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the Full AF=0.5 PEGASUS joint-population v3 panel."
    )
    parser.add_argument(
        "--build-manifest-only",
        action="store_true",
        help="Only (re)write the job manifest CSV and exit.",
    )
    parser.add_argument(
        "--rewrite-manifest",
        action="store_true",
        help="Rebuild the manifest from scratch instead of merging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build/read the manifest, print status, and exit without launching workers.",
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # internal worker entry
    )
    args = parser.parse_args(argv)

    if args.worker is not None:
        worker(int(args.worker))
        return

    rewrite = bool(args.rewrite_manifest) or os.environ.get("FULL_AF05_REWRITE_MANIFEST") == "1"
    build_manifest(rewrite=rewrite)
    sequences = target_sequences()
    print(f"[full_af05] targets loaded: {len(sequences)}", flush=True)

    if args.build_manifest_only:
        print(f"[full_af05] wrote manifest: {QUEUE}", flush=True)
        return

    gpus: list[int] = []
    for token in os.environ.get("FULL_AF05_GPUS", "1,3,4,5,6,7").split(","):
        if token == "":
            continue
        gpus.append(int(token))
    slots = int(os.environ.get("FULL_AF05_SLOTS_PER_GPU", "4"))
    worker_gpus = [g for g in gpus for _ in range(slots)]

    rows = read_rows()
    pending = sum(1 for r in rows if r["status"] == "pending")
    done = sum(1 for r in rows if r["status"] == "done")
    print(
        f"[full_af05] KOOPMAN_ADAPTIVE_FRACTION={ADAPTIVE_FRACTION} "
        f"gpus={gpus} slots_per_gpu={slots} workers={len(worker_gpus)}",
        flush=True,
    )
    print(f"[full_af05] launcher={LAUNCH}", flush=True)
    print(f"[full_af05] runner={RUNNER_MODULE}", flush=True)
    print(f"[full_af05] output=results/{SUBDIR}/", flush=True)
    print(f"[full_af05] pending={pending} done={done} queue={QUEUE}", flush=True)

    if args.dry_run:
        print("[full_af05] dry-run complete; not launching workers", flush=True)
        return

    print(
        f"[full_af05] first wave: up to {len(worker_gpus)} L40 jobs start now; "
        f"remaining jobs stay queued until a slot frees",
        flush=True,
    )
    children = [subprocess.Popen([PYTHON, __file__, "--worker", str(g)]) for g in worker_gpus]
    codes = [proc.wait() for proc in children]
    raise SystemExit(max(codes) if codes else 0)


if __name__ == "__main__":
    main()
