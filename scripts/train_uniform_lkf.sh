#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# Train teacher-free Uniform-AA peptide LKF directly from clean peptide data.
# No PepDFM checkpoint and no PepDFM trajectory cache is used.
#
# Examples:
#   M=1 ./scripts/train_uniform_lkf.sh
#   M=8 ./scripts/train_uniform_lkf.sh
#   M=8 DEVICES=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_uniform_lkf.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,6,7}"

M=${M:-8}
LATENT_LAYERS=${LATENT_LAYERS:-4}
DEVICES=${DEVICES:-4}
DATASET_ROOT=${DATASET_ROOT:-./data/peptide/tokenized_peptide_batched}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/LKF_uniform_scratch}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
PRECISION=${PRECISION:-bf16-mixed}
MODEL_DIM=${MODEL_DIM:-512}
N_HEADS=${N_HEADS:-8}
N_LAYERS=${N_LAYERS:-12}
SOURCE_ANCHOR_PROB=${SOURCE_ANCHOR_PROB:-0.25}
MAX_S=${MAX_S:-0.95}
ROUTER_BALANCE_COEF=${ROUTER_BALANCE_COEF:-0.01}
ROUTER_PRIOR_ENTROPY_COEF=${ROUTER_PRIOR_ENTROPY_COEF:-0.01}
NUM_WORKERS=${NUM_WORKERS:-2}
RESUME=${RESUME:-}

ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --train-split train
  --val-split val
  --model-dim "${MODEL_DIM}"
  --n-heads "${N_HEADS}"
  --n-layers "${N_LAYERS}"
  --latent-components "${M}"
  --latent-layers "${LATENT_LAYERS}"
  --source-anchor-prob "${SOURCE_ANCHOR_PROB}"
  --max-s "${MAX_S}"
  --val-times "0,0.25,0.5,0.75,0.875"
  --router-balance-coef "${ROUTER_BALANCE_COEF}"
  --router-prior-entropy-coef "${ROUTER_PRIOR_ENTROPY_COEF}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay 1e-5
  --warmup-fraction 0.1
  --min-lr-ratio 0.1
  --devices "${DEVICES}"
  --precision "${PRECISION}"
  --gradient-clip-val 1.0
  --num-workers "${NUM_WORKERS}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "peptide_uniform_lkf_M${M}"
  --save-top-k 3
  --seed 42
)

if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

"${PEGASUS_PYTHON:-python}" -u -m pegasus.lkf.train_uniform "${ARGS[@]}"
