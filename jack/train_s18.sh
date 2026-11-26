#!/bin/bash
# Scenario-18 asymmetric PPO on the UniSi lab machines (jack: 1x RTX 3080 10 GB, fermi: 2x RTX A4000 16 GB;
# shared home). Same recipe as Grid'5000 (g5k/s18_env.sh),
# 1024-env paper batch (131,072 steps/update) so it fits in 10 GB.
# Runs detached (survives the ssh logout); log in logs/s18_<variant>_<stamp>.log.
#   [GPU=<index>] ./jack/train_s18.sh <variant>        variants: smoke legacy base clip10 yield6 cmdobs cmdnokl cmdsmoke brake brakesmoke
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${VIRTUAL_ENV:-}" ] || source 2drlenv/bin/activate

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
STAMP="$(hostname -s)_gpu${GPU}_$(date +%Y%m%d_%H%M)"
VARIANT="${1:-base}"; RUN_TAG="$STAMP"
source g5k/s18_env.sh
export LEGNAV_NUM_ENVS="${LEGNAV_NUM_ENVS:-1024}" LEGNAV_N_MINIBATCHES="${LEGNAV_N_MINIBATCHES:-8}" \
       LEGNAV_FT_SNAP_EVERY="${LEGNAV_FT_SNAP_EVERY:-50}"
mkdir -p logs
LOG="logs/s18_${VARIANT}_${STAMP}.log"
nohup python -m legnav.algorithms.jax_ppo_asym > "$LOG" 2>&1 &
echo "started PID $! -> $LOG   (follow: tail -f $LOG)"
