#!/usr/bin/env bash
# Sequential KFM calibration pipeline (matched production stages).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PEGASUS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PEGASUS_ROOT="${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/kfm_env.sh"

STAGE_SCRIPT="${ROOT}/scripts/calibrate_kfm_stage.sh"
chmod +x "${STAGE_SCRIPT}" "${ROOT}/scripts/kfm_env.sh" || true


STAGES=(legacy_a legacy_b ip_a ip_b controlled_a terminal residual install)
echo "[kfm-pipeline] starting stages: ${STAGES[*]}"
echo "[kfm-pipeline] LKF=${LKF_CHECKPOINT}"
echo "[kfm-pipeline] RESULT_ROOT=${KFM_RESULT_ROOT}"
echo "[kfm-pipeline] GPUS=${KFM_GPUS} NPROC=${KFM_NPROC}"

for stage in "${STAGES[@]}"; do
  marker="${KFM_RESULT_ROOT}/STAGE_${stage}.DONE"
  if [[ -f "${marker}" ]]; then
    echo "[kfm-pipeline] skip ${stage} (marker exists: ${marker})"
    continue
  fi
  echo "[kfm-pipeline] ==== BEGIN ${stage} $(date -Is) ===="
  if ! bash "${STAGE_SCRIPT}" "${stage}"; then
    echo "[kfm-pipeline] STAGE ${stage} FAILED — stopping pipeline" >&2
    exit 1
  fi
  date -Is > "${marker}"
  echo "[kfm-pipeline] ==== END ${stage} $(date -Is) ===="
done

echo "[kfm-pipeline] ALL STAGES COMPLETE"
echo "  terminal: ${TERMINAL_INSTALL}/checkpoints/calibrated.pt"
echo "  residual: ${RESIDUAL_INSTALL}/checkpoints/calibrated.pt"
