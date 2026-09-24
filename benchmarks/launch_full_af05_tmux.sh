#!/usr/bin/env bash
# Launch Full-K AF=0.5 panel in a detached tmux session.
#
#   bash benchmarks/scripts/launch_full_af05_tmux.sh
#
# Defaults: GPUs 1,3,4,5,6,7 × 4 slots = 24 parallel jobs.
# L40 fills the first wave; L12 remains queued until slots free.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PEGASUS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BENCH="${ROOT}/benchmarks"
PYBIN=""
SESSION="${SESSION:-full-af05}"
GPUS="${FULL_AF05_GPUS:-1,3,4,5,6,7}"
SLOTS="${FULL_AF05_SLOTS_PER_GPU:-4}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists; attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

python_bin="python"
"${python_bin}" - "${GPUS}" <<'PY'
import subprocess, sys
requested = {int(x) for x in sys.argv[1].split(",") if x != ""}
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
idx = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
bus_to_idx = {}
for line in idx.splitlines():
    i, bus = [p.strip() for p in line.split(",")]
    bus_to_idx[bus] = int(i)
busy = {}
for line in out.splitlines():
    if not line.strip():
        continue
    bus, pid, mem = [p.strip() for p in line.split(",")]
    gpu = bus_to_idx.get(bus)
    if gpu is not None:
        busy.setdefault(gpu, []).append((pid, mem))
clash = sorted(set(busy) & requested)
if clash:
    print(f"WARNING: requested GPUs already have compute processes: "
          f"{ {g: busy[g] for g in clash} }")
else:
    print(f"GPU check OK: requested {sorted(requested)} are free.")
PY

mkdir -p "${BENCH}/logs/full_af05"
SESSION_LOG="${BENCH}/logs/full_af05/scheduler.log"

tmux new-session -d -s "${SESSION}" -c "${ROOT}"
tmux send-keys -t "${SESSION}" "cd ${ROOT}" C-m
tmux send-keys -t "${SESSION}" "export PATH=${PYBIN}:\$PATH PYTHONPATH=${ROOT}" C-m
tmux send-keys -t "${SESSION}" \
  "export FULL_AF05_GPUS=${GPUS} FULL_AF05_SLOTS_PER_GPU=${SLOTS} KOOPMAN_ADAPTIVE_FRACTION=0.5" C-m
tmux send-keys -t "${SESSION}" \
  "stdbuf -oL -eL ${python_bin} ${BENCH}/scripts/run_full_af05.py 2>&1 | tee -a ${SESSION_LOG}" C-m

echo "started tmux session '${SESSION}'"
echo "  gpus:            ${GPUS}"
echo "  slots_per_gpu:   ${SLOTS}"
echo "  parallel slots:  $(( $(echo ${GPUS} | tr ',' '\n' | grep -c .) * SLOTS ))"
echo "  AF:              0.5 via pegasus_joint_population_pdo.sh (Full v3)"
echo "  outputs:         ${BENCH}/E4/full_af05/"
echo "  queue:           ${BENCH}/E4_FULL_AF05_JOB_MANIFEST.csv"
echo "  log:             ${SESSION_LOG}"
echo "attach with: tmux attach -t ${SESSION}"
