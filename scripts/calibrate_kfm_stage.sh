#!/usr/bin/env bash
# Run one KFM calibration stage (matched production hyperparameters).
#
# Usage:
#   bash scripts/calibrate_kfm_stage.sh <stage>
#
# Stages (in order):
#   legacy_a | legacy_b | ip_a | ip_b | controlled_a | terminal | residual | install
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PEGASUS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PEGASUS_ROOT="${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/kfm_env.sh"

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: $0 <legacy_a|legacy_b|ip_a|ip_b|controlled_a|terminal|residual|install>" >&2
  exit 2
fi

if [[ ! -f "${LKF_CHECKPOINT}" ]]; then
  echo "Missing LKF checkpoint: ${LKF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Missing dataset root: ${DATASET_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${KFM_GPUS}"
NPROC="${KFM_NPROC}"
if [[ "${#GPU_ARR[@]}" -lt "${NPROC}" ]]; then
  NPROC="${#GPU_ARR[@]}"
fi
export CUDA_VISIBLE_DEVICES="${KFM_GPUS}"

run_ddp () {
  local module="$1"; shift
  local log="$1"; shift
  echo "[kfm] torchrun nproc=${NPROC} gpus=${KFM_GPUS} module=${module}" | tee -a "${log}"
  echo "[kfm] PYTHONPATH=${PYTHONPATH}" | tee -a "${log}"
  # Explicitly pass env so torchrun children resolve the configured Koopman trainers.
  PYTHONPATH="${PEGASUS_ROOT}" \
  stdbuf -oL -eL "${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" \
    -m "${module}" "$@" 2>&1 | tee -a "${log}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -ne 0 ]]; then
    echo "[kfm] torchrun failed rc=${rc}" | tee -a "${log}"
    exit "${rc}"
  fi
}

run_single () {
  local module="$1"; shift
  local log="$1"; shift
  # Prefer first listed GPU for single-process stages.
  local first_gpu
  first_gpu="$(echo "${KFM_GPUS}" | cut -d, -f1)"
  echo "[kfm] single-process gpu=${first_gpu} module=${module}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${first_gpu}" PYTHONPATH="${PEGASUS_ROOT}" \
    stdbuf -oL -eL "${PY}" -m "${module}" "$@" 2>&1 | tee -a "${log}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -ne 0 ]]; then
    echo "[kfm] single-process failed rc=${rc}" | tee -a "${log}"
    exit "${rc}"
  fi
}

require_file () {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required artifact missing: ${path}" >&2
    exit 1
  fi
}

case "${STAGE}" in
  legacy_a)
    LOG="${KFM_LOG_DIR}/01_legacy_a.log"
    run_ddp pegasus.train "${LOG}" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train --val-split val \
      --stage A \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name koopman_lkf_stage_a \
      --device cuda \
      --feature-dim "${ANCHOR_DIM}" \
      --head-hidden-dim 256 \
      --feature-source "${FEATURE_SOURCE}" \
      --operator-init-scale 0.001 \
      --rank-regularization-weight 1.0 \
      --min-effective-rank-fraction 0.9 \
      --rank-target-fraction 0.95 \
      --continuations 4 \
      --max-s 0.95 \
      --source-anchor-prob 0.2 \
      --terminal-anchor-prob 0.3 \
      --min-interval 0.05 \
      --epochs 30 \
      --steps-per-epoch 0 \
      --max-batch-sequences 64 \
      --learning-rate 0.0001 \
      --weight-decay 1e-05 \
      --gradient-clip 1.0 \
      --lambda-koopman 0.05 \
      --lkf-learning-rate 1e-05 \
      --unfreeze-shared-blocks 2 \
      --router-balance-coef 0.01 \
      --router-prior-entropy-coef 0.01 \
      --max-relative-nll-degradation 0.02 \
      --val-times "0,0.25,0.5,0.75,0.875" \
      --val-intervals "0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1" \
      --val-continuations 8 \
      --val-batches 4 \
      --val-batch-sequences 32 \
      --num-workers 2 \
      --seed "${SEED}"
    require_file "${KFM_RESULT_ROOT}/koopman_lkf_stage_a/checkpoints/best.pt"
    ;;

  legacy_b)
    require_file "${KFM_RESULT_ROOT}/koopman_lkf_stage_a/checkpoints/best.pt"
    LOG="${KFM_LOG_DIR}/02_legacy_b.log"
    run_ddp pegasus.train "${LOG}" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train --val-split val \
      --stage B \
      --stage-a-checkpoint "${KFM_RESULT_ROOT}/koopman_lkf_stage_a/checkpoints/best.pt" \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name koopman_lkf_stage_b \
      --device cuda \
      --feature-dim "${ANCHOR_DIM}" \
      --head-hidden-dim 256 \
      --feature-source "${FEATURE_SOURCE}" \
      --operator-init-scale 0.001 \
      --rank-regularization-weight 1.0 \
      --min-effective-rank-fraction 0.9 \
      --rank-target-fraction 0.95 \
      --continuations 4 \
      --max-s 0.95 \
      --source-anchor-prob 0.2 \
      --terminal-anchor-prob 0.3 \
      --min-interval 0.05 \
      --epochs 20 \
      --steps-per-epoch 0 \
      --max-batch-sequences 16 \
      --learning-rate 0.0001 \
      --weight-decay 1e-05 \
      --gradient-clip 1.0 \
      --lambda-koopman 0.05 \
      --lkf-learning-rate 1e-05 \
      --unfreeze-shared-blocks 2 \
      --router-balance-coef 0.01 \
      --router-prior-entropy-coef 0.01 \
      --max-relative-nll-degradation 0.02 \
      --val-times "0,0.25,0.5,0.75,0.875" \
      --val-intervals "0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1" \
      --val-continuations 8 \
      --val-batches 4 \
      --val-batch-sequences 32 \
      --num-workers 2 \
      --seed "${SEED}"
    require_file "${KFM_RESULT_ROOT}/koopman_lkf_stage_b/checkpoints/best.pt"
    ;;

  ip_a)
    require_file "${KFM_RESULT_ROOT}/koopman_lkf_stage_b/checkpoints/best.pt"
    LOG="${KFM_LOG_DIR}/03_ip_a.log"
    run_ddp pegasus.ip_train "${LOG}" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --generator-init-koopman-checkpoint "${KFM_RESULT_ROOT}/koopman_lkf_stage_b/checkpoints/best.pt" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train --val-split val \
      --stage A \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name ip_koopman_stage_a \
      --device cuda \
      --anchor-dim "${ANCHOR_DIM}" \
      --lift-dim "${ANCHOR_DIM}" \
      --lift-hidden-dim 256 \
      --anchor-seed "${ANCHOR_SEED}" \
      --operator-init-scale 0.001 \
      --feature-source "${FEATURE_SOURCE}" \
      --anchor-closure-weight 1.0 \
      --lift-closure-weight 1.0 \
      --rank-regularization-weight 1.0 \
      --rank-target-fraction 0.95 \
      --min-effective-rank-fraction 0.9 \
      --continuations 4 \
      --max-s 0.95 \
      --source-anchor-prob 0.2 \
      --terminal-anchor-prob 0.3 \
      --min-interval 0.05 \
      --epochs 30 \
      --steps-per-epoch 0 \
      --max-batch-sequences 64 \
      --learning-rate 0.0001 \
      --weight-decay 1e-05 \
      --gradient-clip 1.0 \
      --lambda-koopman 0.05 \
      --lkf-learning-rate 1e-05 \
      --unfreeze-shared-blocks 2 \
      --router-balance-coef 0.01 \
      --router-prior-entropy-coef 0.01 \
      --max-relative-nll-degradation 0.02 \
      --val-times "0,0.25,0.5,0.75,0.875" \
      --val-intervals "0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1" \
      --val-continuations 8 \
      --val-batches 4 \
      --val-batch-sequences 32 \
      --num-workers 2 \
      --seed "${SEED}"
    require_file "${KFM_RESULT_ROOT}/ip_koopman_stage_a/checkpoints/best.pt"
    ;;

  ip_b)
    require_file "${KFM_RESULT_ROOT}/ip_koopman_stage_a/checkpoints/best.pt"
    LOG="${KFM_LOG_DIR}/04_ip_b.log"
    run_ddp pegasus.ip_train "${LOG}" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train --val-split val \
      --stage B \
      --stage-a-checkpoint "${KFM_RESULT_ROOT}/ip_koopman_stage_a/checkpoints/best.pt" \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name ip_koopman_stage_b \
      --device cuda \
      --anchor-dim "${ANCHOR_DIM}" \
      --lift-dim "${ANCHOR_DIM}" \
      --lift-hidden-dim 256 \
      --anchor-seed "${ANCHOR_SEED}" \
      --operator-init-scale 0.001 \
      --feature-source "${FEATURE_SOURCE}" \
      --anchor-closure-weight 1.0 \
      --lift-closure-weight 1.0 \
      --rank-regularization-weight 1.0 \
      --rank-target-fraction 0.95 \
      --min-effective-rank-fraction 0.9 \
      --continuations 4 \
      --max-s 0.95 \
      --source-anchor-prob 0.2 \
      --terminal-anchor-prob 0.3 \
      --min-interval 0.05 \
      --epochs 20 \
      --steps-per-epoch 0 \
      --max-batch-sequences 16 \
      --learning-rate 0.0001 \
      --weight-decay 1e-05 \
      --gradient-clip 1.0 \
      --lambda-koopman 0.05 \
      --lkf-learning-rate 1e-05 \
      --unfreeze-shared-blocks 2 \
      --router-balance-coef 0.01 \
      --router-prior-entropy-coef 0.01 \
      --max-relative-nll-degradation 0.02 \
      --val-times "0,0.25,0.5,0.75,0.875" \
      --val-intervals "0:0.25,0:0.5,0:1,0.25:0.75,0.5:1,0.75:1" \
      --val-continuations 8 \
      --val-batches 4 \
      --val-batch-sequences 32 \
      --num-workers 2 \
      --seed "${SEED}"
    require_file "${KFM_RESULT_ROOT}/ip_koopman_stage_b/checkpoints/best.pt"
    ;;

  controlled_a)
    require_file "${KFM_RESULT_ROOT}/ip_koopman_stage_b/checkpoints/best.pt"
    LOG="${KFM_LOG_DIR}/05_controlled_a.log"
    run_ddp pegasus.controlled_koopman_train "${LOG}" \
      --stage A \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --ip-koopman-checkpoint "${KFM_RESULT_ROOT}/ip_koopman_stage_b/checkpoints/best.pt" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train --val-split val \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name controlled_koopman_stage_a \
      --device cuda \
      --anchor-dim "${ANCHOR_DIM}" \
      --anchor-seed "${ANCHOR_SEED}" \
      --feature-source "${FEATURE_SOURCE}" \
      --operator-bins 8 \
      --operator-init-scale 0.001 \
      --control-init-scale 0.0 \
      --control-knots "0,0.25,0.5,0.75,1" \
      --training-chains "${TRAINING_CHAINS}" \
      --information-eigen-floor-relative 1e-05 \
      --information-epsilon 1e-06 \
      --calibration-sequences 2048 \
      --continuations "${CONTINUATIONS_CALIB}" \
      --val-continuations 16 \
      --kappa-min 0.02 \
      --kappa-max 0.25 \
      --val-kappa 0.1 \
      --soft-temperature 0.5 \
      --spectral-tau 0.001 \
      --lambda-soft-response 1.0 \
      --lambda-controllability 1.0 \
      --lambda-native 0.25 \
      --operator-regularization 1e-05 \
      --epochs 30 \
      --steps-per-epoch 0 \
      --max-batch-sequences 32 \
      --num-workers 2 \
      --learning-rate 0.0001 \
      --lkf-learning-rate 1e-05 \
      --weight-decay 0.0001 \
      --grad-clip 1.0 \
      --unfreeze-shared-blocks 2 \
      --val-batch-sequences 64 \
      --val-batches 24 \
      --val-control-batches 4 \
      --val-control-directions 4 \
      --max-relative-nll-degradation 0.02 \
      --max-anchor-relative-rmse 0.35 \
      --min-response-cosine 0.0 \
      --min-reliability-adjusted-response-cosine 0.65 \
      --min-sign-consistency 0.9 \
      --min-stein-J-cosine 0.0 \
      --seed "${SEED}"
    require_file "${KFM_RESULT_ROOT}/controlled_koopman_stage_a/checkpoints/best.pt"
    ;;

  terminal)
    require_file "${KFM_RESULT_ROOT}/controlled_koopman_stage_a/checkpoints/best.pt"
    LOG="${KFM_LOG_DIR}/06_terminal.log"
    run_ddp pegasus.terminal_controlled_koopman_calibrate "${LOG}" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --controlled-stage-a-checkpoint "${KFM_RESULT_ROOT}/controlled_koopman_stage_a/checkpoints/best.pt" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train \
      --output-dir "${KFM_RESULT_ROOT}" \
      --run-name terminal_controlled_koopman_calibration \
      --device cuda \
      --seed "${SEED}" \
      --chains "${CHAINS_TERMINAL}" \
      --continuations "${CONTINUATIONS_CALIB}" \
      --steps-per-chain 32 \
      --max-batch-sequences 32 \
      --num-workers 2 \
      --log-every 10 \
      --direct-operator-ridge "${DIRECT_OPERATOR_RIDGE}" \
      --soft-temperature 0.5 \
      --diagnostic-token-length 14
    require_file "${KFM_RESULT_ROOT}/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt"
    ;;

  residual)
    require_file "${KFM_RESULT_ROOT}/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt"
    LOG="${KFM_LOG_DIR}/07_residual.log"
    # Residual calib is typically single-process (large MC banks).
    run_single pegasus.residual_distribution_koopman_calibrate "${LOG}" \
      --koopman-checkpoint "${KFM_RESULT_ROOT}/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt" \
      --lkf-checkpoint "${LKF_CHECKPOINT}" \
      --dataset-root "${DATASET_ROOT}" \
      --train-split train \
      --token-length 14 \
      --output-dir "${KFM_RESULT_ROOT}/residual_distribution_koopman_calibration" \
      --device cuda:0 \
      --seed "${SEED}" \
      --chains "${CHAINS_RESIDUAL}" \
      --states 128 \
      --continuations 128 \
      --continuation-chunk 32 \
      --bank-size 16384 \
      --direction-count 32 \
      --direction-seed 84217 \
      --quantiles "0.05,0.1,0.5,0.8,0.9,0.95" \
      --betas "0.5,1,2,4"
    require_file "${KFM_RESULT_ROOT}/residual_distribution_koopman_calibration/checkpoints/calibrated.pt"
    ;;

  install)
    require_file "${KFM_RESULT_ROOT}/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt"
    require_file "${KFM_RESULT_ROOT}/residual_distribution_koopman_calibration/checkpoints/calibrated.pt"
    LOG="${KFM_LOG_DIR}/08_install.log"
    "${PY}" "${ROOT}/scripts/install_kfm.py" 2>&1 | tee -a "${LOG}"
    ;;

  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

echo "[kfm] STAGE ${STAGE} DONE"
