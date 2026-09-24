#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PEGASUS_PYTHON:-python}"

"${PY}" -u -m pegasus.lkf_unconditional_generate \
  --length "${LENGTH:-12}" \
  --num-sequences "${NUM_SEQUENCES:-100}" \
  --nfe "${NFE:-1}" \
  --seed "${SEED:-42}" \
  --checkpoint "${LKF_CHECKPOINT:-checkpoints/M8.ckpt}" \
  --output "${OUTPUT:-results/lkf_unconditional_nfe1_len12_seed42.txt}"
