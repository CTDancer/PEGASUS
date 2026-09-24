#!/usr/bin/env bash
set -euo pipefail

# Respect an already-set pin (schedulers pass CUDA_VISIBLE_DEVICES=<physical GPU>).
# Default remains 2 only for interactive one-off launches that do not set it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# PEGASUS joint Population PDO v3.  All paths are absolute so the script can be launched
# from any shell after the PEGASUS folder is uploaded to the cluster.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESIDUAL_CHECKPOINT="${RESIDUAL_CHECKPOINT:-${PROJECT_ROOT}/results/residual_distribution_koopman_calibration/checkpoints/calibrated.pt}"
TERMINAL_CHECKPOINT="${TERMINAL_CHECKPOINT:-${PROJECT_ROOT}/results/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt}"
LKF_CHECKPOINT="${LKF_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/M8.ckpt}"
PEPTIVERSE_ROOT="${PEPTIVERSE_ROOT:?Set PEPTIVERSE_ROOT to your PeptiVerse install}"
DEVICE="${DEVICE:-cuda:0}"
PEPTIVERSE_DEVICE="${PEPTIVERSE_DEVICE:-cuda}"
PEPTIDE_LENGTH="${PEPTIDE_LENGTH:-12}"
TARGET="${TARGET:-RITLKESGPPLVKPTQTLTLTCSFSGFSLSDFGVGVGWIRQPPGKALEWLAIIYSDDDKRYSPSLNTRLTITKDTSKNQVVLVMTRVSPVDTATYFCAHRRGPTTLFGVPIARGPVNAMDVWGQGITVTISSTSTKGPSVFPLAPSGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYTCNVNHKPSNTKVDKRVEPKSC}"
STAGE="${STAGE:-run}"
SEED="${SEED:-101}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
RADIUS_SPEC="${PDO_MULTISCALE_RADIUS_SPEC:-1,2,4}"
RADIUS_TAG="$(printf '%s' "${RADIUS_SPEC}" | tr -d ',')"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_ROOT}/pegasus_joint_population_v3_len${PEPTIDE_LENGTH}_h${RADIUS_TAG}_seed${SEED}}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PEGASUS_PYTHON:-python}"

preflight() {
"${PY}" - <<'PY'
from pathlib import Path
from pegasus.lkf.uniform_lkf import load_uniform_lkf_checkpoint
from pegasus.residual_distribution_koopman_model import load_residual_distribution_checkpoint
import os

pep_len=int(os.environ.get('PEPTIDE_LENGTH','40'))
need=pep_len+2
lkf_path=os.environ.get('LKF_CHECKPOINT','./checkpoints/M8.ckpt')
res=os.environ.get('RESIDUAL_CHECKPOINT','./results/residual_distribution_koopman_calibration/checkpoints/calibrated.pt')
term=os.environ.get('TERMINAL_CHECKPOINT','./results/terminal_controlled_koopman_calibration/checkpoints/calibrated.pt')
for p in (lkf_path,res,term):
    if not Path(p).exists():
        raise SystemExit(f'MISSING REQUIRED CHECKPOINT: {p}')
lkf=load_uniform_lkf_checkpoint(lkf_path,map_location='cpu',eval_mode=True)
print('[PEGASUS joint Population-PDO preflight]')
print('  requested peptide residues:', pep_len)
print('  requested token length:', need)
print('  LKF max token length:', int(lkf.seq_len), '(max residues:', int(lkf.seq_len)-2, ')')
if need > int(lkf.seq_len):
    raise SystemExit(f'UNSUPPORTED LENGTH: need {need} tokens but LKF supports {lkf.seq_len}. Use length-capable LKF/KFM checkpoints; do not truncate.')
model,_=load_residual_distribution_checkpoint(res,terminal_checkpoint=term,base_lkf_checkpoint=lkf_path,device='cpu',strict_sha=True)
kfm_len=int(model.base_model.lkf.seq_len)
print('  KFM embedded LKF max token length:', kfm_len, '(max residues:', kfm_len-2, ')')
if need > kfm_len:
    raise SystemExit(f'UNSUPPORTED LENGTH: KFM supports only {kfm_len-2} residues.')
print('  length support: OK')
print('  protected rule: >=1 exact non-noop fine exposure per active lineage before empirical stopping')
print('  joint rule: protected PEGASUS freezes one query per unresolved lineage; joint rank only orders that batch')
print('  lazy rule: after each exact label, ordinary protected PDO may provisionally cancel pending frozen queries; later same-batch labels may reactivate them')
print('  scale rule: all configured Hamming radii remain ordinary protected action families; no scale selector or per-scale query floor')
print('  diversity rule: signed reachable-contraction coordination only inside exact queried near-optimal sets')
print('  safety rule: unseen actions are NEVER selectable; exact oracle alone authorizes acceptance')
PY
}

run_one() {
  "${PY}" -u -m "${RUNNER_MODULE:-pegasus.pdo_pegasus_joint_population_pdo_v3}" \
    --residual-checkpoint "${RESIDUAL_CHECKPOINT}" \
    --terminal-checkpoint "${TERMINAL_CHECKPOINT}" \
    --lkf-checkpoint "${LKF_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --peptiverse-root "${PEPTIVERSE_ROOT}" \
    --peptiverse-device "${PEPTIVERSE_DEVICE}" \
    --target "${TARGET}" \
    --preferences "${PREFERENCES:-1,1,1,1,1,1}" \
    --rho "${RHO:-0.01}" \
    --peptide-length "${PEPTIDE_LENGTH}" \
    --native-nfe "${NATIVE_NFE:-1}" \
    --chains "${CHAINS:-0.5,1;0.75,1}" \
    --initial-readout-queries "${INITIAL_READOUT_QUERIES:-100}" \
    --discovery-pool-multiplier "${DISCOVERY_POOL_MULTIPLIER:-16}" \
    --discovery-max-pool-expansions "${DISCOVERY_MAX_POOL_EXPANSIONS:-3}" \
    --ridge-alphas "${RIDGE_ALPHAS:-0.01,0.1,1,10,100}" \
    --cv-folds "${CV_FOLDS:-4}" \
    --lineages "${LINEAGES:-12}" \
    --slates-per-lineage "${SLATES_PER_LINEAGE:-64}" \
    --slate-per-chain "${SLATE_PER_CHAIN:-288}" \
    --verification-k "${VERIFICATION_K:-3}" \
    --readout-mode "${READOUT_MODE:-anchor_absolute}" \
    --accept-epsilon "${ACCEPT_EPSILON:-1e-4}" \
    --strong-gain "${STRONG_GAIN:-0.02}" \
    --min-lineage-hamming "${MIN_LINEAGE_HAMMING:-0.20}" \
    --output-min-hamming "${OUTPUT_MIN_HAMMING:-0.20}" \
    --num-output-sequences "${NUM_OUTPUT_SEQUENCES:-100}" \
    --cheap-generation-batch "${CHEAP_GENERATION_BATCH:-256}" \
    --max-cheap-draws-per-candidate "${MAX_CHEAP_DRAWS_PER_CANDIDATE:-4096}" \
    --max-root-attempts-per-chain "${MAX_ROOT_ATTEMPTS_PER_CHAIN:-8192}" \
    --feature-batch-size "${FEATURE_BATCH_SIZE:-256}" \
    --log-top-predictions "${LOG_TOP_PREDICTIONS:-16}" \
    --save-every-slates "${SAVE_EVERY_SLATES:-10}" \
    --max-unique-oracle-queries "${MAX_UNIQUE_ORACLE_QUERIES:-100000}" \
    --seed "${SEED}" \
    --pdo-budget-mode "${PDO_BUDGET_MODE:-protected_baseline}" \
    --pdo-mode-policy "${PDO_MODE_POLICY:-always_pdo}" \
    --pdo-fine-prior-labels "${PDO_FINE_PRIOR_LABELS:-all_paid}" \
    --koopman-branch-allocation-mode "${KOOPMAN_BRANCH_ALLOCATION_MODE:-ac_opportunity}" \
    --koopman-pilot-starts-per-chain "${KOOPMAN_PILOT_STARTS_PER_CHAIN:-6}" \
    --koopman-adaptive-fraction "${KOOPMAN_ADAPTIVE_FRACTION:-0.5}" \
    --koopman-min-candidates-per-branch "${KOOPMAN_MIN_CANDIDATES_PER_BRANCH:-1}" \
    --v3-coarse-verification-mode "${V3_COARSE_VERIFICATION_MODE:-exploit_first}" \
    --pdo-coarse-v3-contender-cap "${PDO_COARSE_V3_CONTENDER_CAP:-32}" \
    --pdo-coarse-v3-global-jackknife-folds "${PDO_COARSE_V3_GLOBAL_JACKKNIFE_FOLDS:-4}" \
    --pdo-coarse-v3-min-global-train-labels "${PDO_COARSE_V3_MIN_GLOBAL_TRAIN_LABELS:-8}" \
    --pdo-coarse-epsilon-dec "${PDO_COARSE_EPSILON_DEC:-0.002}" \
    --pdo-coarse-query-cap-per-slate "${PDO_COARSE_QUERY_CAP_PER_SLATE:--1}" \
    --pdo-multiscale-radius-spec "${RADIUS_SPEC}" \
    --pdo-multiscale-candidates-per-scale "${PDO_MULTISCALE_CANDIDATES_PER_SCALE:-256}" \
    --pdo-multiscale-exhaustive-h1-max-actions "${PDO_MULTISCALE_EXHAUSTIVE_H1_MAX_ACTIONS:-2048}" \
    --pdo-multiscale-contender-cap-per-scale "${PDO_MULTISCALE_CONTENDER_CAP_PER_SCALE:-32}" \
    --pdo-multiscale-epsilon-dec "${PDO_MULTISCALE_EPSILON_DEC:-0.002}" \
    --pdo-multiscale-query-cap-per-turn "${PDO_MULTISCALE_QUERY_CAP_PER_TURN:-8}" \
    --pdo-multiscale-max-sampling-attempts "${PDO_MULTISCALE_MAX_SAMPLING_ATTEMPTS:-100000}" \
    --pdo-population-mode "${POPULATION_MODE:-contraction}" \
    --pdo-population-exact-tie-epsilon "${PDO_POPULATION_EXACT_TIE_EPSILON:--1}" \
    --pdo-population-coordinate-passes "${PDO_POPULATION_COORDINATE_PASSES:-3}" \
    --pdo-joint-rank-rtol "${PDO_JOINT_RANK_RTOL:-1e-7}" \
    --pdo-joint-acquisition-mode "${PDO_JOINT_ACQUISITION_MODE:-shared_rank}" \
    "$@"
}

case "${STAGE}" in
  preflight)
    export PEPTIDE_LENGTH LKF_CHECKPOINT RESIDUAL_CHECKPOINT TERMINAL_CHECKPOINT
    preflight
    ;;
  run)
    export PEPTIDE_LENGTH LKF_CHECKPOINT RESIDUAL_CHECKPOINT TERMINAL_CHECKPOINT
    preflight
    run_one "$@"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; use preflight or run" >&2
    exit 2
    ;;
esac
