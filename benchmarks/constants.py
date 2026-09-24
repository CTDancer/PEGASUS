"""Frozen / confirmed E3 classical-benchmark constants.

Owner decisions (2026-09-13) froze classical pop size, LKF init, query budget,
checkpoints, NSGA-III partitions, and SMS-EMOA offspring count. Strong-hit
remains intentionally null for cross-method tables (see FROZEN_DECISIONS.md).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
E3_ROOT = BENCH / "E3"
NEUTRAL_POOL_ROOT = BENCH / "neutral_pools"
TARGET_SEQ_DIR = ROOT / "benchmarks/targets"
LEGACY_TARGET_SEQ_DIR = (
    ROOT / "results/full_koopman_ablation_wo_known_binders_seed42/targets"
)
BENCHMARK_XLSX = ROOT / "benchmark_sequences.xlsx"
PEPTIVERSE_ROOT = Path(__import__("os").environ.get("PEPTIVERSE_ROOT", "PeptiVerse"))

# FROZEN: production Uniform-LKF used by PEGASUS joint-v3.
LKF_CHECKPOINT = ROOT / "checkpoints" / "M8.ckpt"
# FROZEN: match PEGASUS launcher default NATIVE_NFE=1 (unguided prior draws).
LKF_NFE = 1
NEUTRAL_GENERATOR = "unguided_lkf_m8"

# E3 Panel B (spec §1.2). OX1R == TM3 sequence in historical PEGASUS labels.
E3_TARGETS = (
    "3IDJ",
    "5AZ8",
    "7JVS",
    "AMHR2",
    "OX1R",
    "DUSP12",
    "EWS_FLI1",
    "MYC",
)
TARGET_ALIASES = {
    "OX1R": "TM3",  # same 425-aa sequence; historical .seq filename
    "EWS_FLI1": "EWS_FLI1",
    "EWS::FLI1": "EWS_FLI1",
}

SEEDS = (42, 73, 101)
LENGTHS = (12, 40)

# Maximize-oriented optimization names (PEGASUS / kfocus convention).
OPT_NAMES = (
    "Non-Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
)
RAW_NAMES = (
    "Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
)

# FROZEN: TCFM/benchmark/constants.py HV_REFERENCE and e456 hypervolume_ref0.
HV_REFERENCE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(CANONICAL_AA)}
IDX_TO_AA = {i: a for a, i in AA_TO_IDX.items()}

FINAL_LIBRARY_SIZE = 100

# FROZEN classical defaults (owner confirmation 2026-09-13).
DEFAULT_POP_SIZE = 100
DEFAULT_MAX_UNIQUE_QUERIES = 2000
DEFAULT_QUERY_CHECKPOINTS = (100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)

# FROZEN: Das–Dennis n_partitions=3 → 56 reference directions; pop_size=100.
NSGA3_N_PARTITIONS = 3

# FROZEN: steady-state SMS-EMOA.
SMS_EMOA_N_OFFSPRINGS = 1

CLASSICAL_METHODS = ("nsga3", "sms_emoa", "spea2", "mopso")

# External learned/generative baselines (E3 Stage B; native init — not LKF pools).
LEARNED_METHODS = ("mog_dfm", "areuredi")

# Official repos / checkpoints used by TCFM adapters (provenance frozen below).
MOG_DFM_ROOT = Path("./MOG-DFM")
AREUREDI_ROOT = Path("./AReUReDi")
DEFAULT_MOG_CHECKPOINT = Path(
    "./checkpoints/MOG-DFM/ckpt/peptide/"
    "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
)
DEFAULT_AREUREDI_CHECKPOINT = Path(
    "./AReUReDi/peptides/ckpt/best.pt"
)

# Native knobs from TCFM/benchmark (do not retune on E3 test).
MOG_STEP_SIZE = 1.0 / 100.0
MOG_N_SAMPLES = 1
AREUREDI_OPTIMIZATION_STEPS = 100
AREUREDI_TOP_P = 1.0
AREUREDI_NUM_SAMPLES = 1
# Balanced six-objective preference (revision instructions §2).
EQUAL_OBJECTIVE_WEIGHTS = tuple(1.0 / 6.0 for _ in OPT_NAMES)

# Production stop: unique native terminal outputs (not a 2000-query cap).
N_TERMINAL_UNIQUE = 100
# No request-count failsafe — MOG/AReUReDi make many oracle asks by design.
# Optional manual override via --safety-max-total-oracle-requests if ever needed.
LEARNED_SAFETY_MAX_TOTAL_ORACLE_REQUESTS = None

# CLI names accepted by MOG/AReUReDi score-model wiring; Hemolysis → Non-Hemolysis.
OBJECTIVE_CLI_NAMES = (
    "Hemolysis",
    "Non-Fouling",
    "Solubility",
    "Permeability",
    "Half-Life",
    "Affinity",
)
