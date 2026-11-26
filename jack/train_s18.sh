#!/bin/bash
# Scenario-18 asymmetric PPO on jack (UniSi, 1x RTX 3080 10 GB). Same recipe as Grid'5000 (g5k/s18_env.sh),
# 1024-env paper batch (131,072 steps/update) so it fits in 10 GB.
# Runs detached (survives the ssh logout); log in logs/jack_s18_<variant>_<stamp>.log.
#   ./jack/train_s18.sh <variant>        variants: smoke legacy base clip10 yield6
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${VIRTUAL_ENV:-}" ] || source 2drlenv/bin/activate

STAMP=$(date +%Y%m%d_%H%M)
VARIANT="${1:-base}"; RUN_TAG="jack_$STAMP"
source g5k/s18_env.sh
export LEGNAV_NUM_ENVS="${LEGNAV_NUM_ENVS:-1024}" LEGNAV_N_MINIBATCHES="${LEGNAV_N_MINIBATCHES:-8}" \
       LEGNAV_FT_SNAP_EVERY="${LEGNAV_FT_SNAP_EVERY:-50}"
mkdir -p logs
LOG="logs/jack_s18_${VARIANT}_${STAMP}.log"
nohup python -m legnav.algorithms.jax_ppo_asym > "$LOG" 2>&1 &
echo "started PID $! -> $LOG   (follow: tail -f $LOG)"
