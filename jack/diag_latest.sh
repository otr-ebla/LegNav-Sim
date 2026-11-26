#!/bin/bash
# Deterministic (deploy-mode) check of the latest CALF snapshot of every s18_run3 run, at ghost 0 (GUI-like)
# and 0.2 (training). Shares the GPU with running trainings (no memory preallocation).
#   [GPU=<index>] ./jack/diag_latest.sh [extra_ckpt ...]      full output in logs/diag_<stamp>.log
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${VIRTUAL_ENV:-}" ] || source 2drlenv/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU:-0}" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.
VARIANT=base; RUN_TAG=diag
source g5k/s18_env.sh > /dev/null
LOG="logs/diag_$(date +%Y%m%d_%H%M).log"; mkdir -p logs

ckpts=("$@")
for d in checkpoints/s18_run3_*/; do
  [[ "$d" == *smoke* ]] && continue
  c=$(find "$d" -name 'u*_calf.msgpack' | sort | tail -1)
  [ -n "$c" ] && ckpts+=("$c")
done

for c in "${ckpts[@]}"; do
  for g in 0.0 0.2; do
    echo "######## $c  ghost=$g" | tee -a "$LOG"
    python -m legnav.evaluation.diag_s18 --ckpt "$c" --mode det --ghost "$g" --envs 256 --steps 1000 2>&1 \
      | tee -a "$LOG" | grep -E "sigma|active collisions|^\[|^ +(success|active_col) "
  done
done
echo "full output -> $LOG"
