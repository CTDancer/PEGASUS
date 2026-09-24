#!/usr/bin/env bash
# Shared environment for LKF–KFM calibration (PEGASUS release).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PEGASUS_ROOT="${PEGASUS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PEGASUS_ROOT}"
export PYTHONPATH="${PEGASUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PY="${PEGASUS_PYTHON:-python}"
export TORCHRUN="${TORCHRUN:-torchrun}"

export LKF_CHECKPOINT="${LKF_CHECKPOINT:-${PEGASUS_ROOT}/checkpoints/M8.ckpt}"
export DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the tokenized peptide dataset root}"

export KFM_RESULT_ROOT="${KFM_RESULT_ROOT:-${PEGASUS_ROOT}/results/kfm_calibration}"
export KFM_LOG_DIR="${KFM_LOG_DIR:-${PEGASUS_ROOT}/results/kfm_calibration/logs}"

export TERMINAL_INSTALL="${TERMINAL_INSTALL:-${PEGASUS_ROOT}/results/terminal_controlled_koopman_calibration}"
export RESIDUAL_INSTALL="${RESIDUAL_INSTALL:-${PEGASUS_ROOT}/results/residual_distribution_koopman_calibration}"

export KFM_GPUS="${KFM_GPUS:-0}"
export KFM_NPROC="${KFM_NPROC:-1}"

export ANCHOR_DIM=64
export ANCHOR_SEED=60042
export FEATURE_SOURCE=shared_pooled_plus_time
export SEED=42
export DIRECT_OPERATOR_RIDGE=0.0001
export CONTINUATIONS_CALIB=8
export CHAINS_TERMINAL="0,1;0.25,1;0.5,1;0.75,1;0,0.25,0.5,1;0,0.5,1;0,0.75,1;0.25,0.5,0.75,1;0.25,0.75,1;0.5,0.75,1"
export TRAINING_CHAINS="${CHAINS_TERMINAL}"
export CHAINS_RESIDUAL="0,1;0.25,1;0.5,1;0.75,1;0,0.5,1;0,0.25,0.5,1"

mkdir -p "${KFM_RESULT_ROOT}" "${KFM_LOG_DIR}"

"${PY}" - <<'PY'
import pathlib
import pegasus

p = pathlib.Path(pegasus.__file__).resolve().parent
assert (p / "train.py").exists(), f"pegasus.train missing next to {p}"
print(f"[kfm] pegasus package OK: {p}")
PY
