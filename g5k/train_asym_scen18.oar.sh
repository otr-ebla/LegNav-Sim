#!/bin/bash
#OAR -n legnav-asym-s18
#OAR -q abaca
#OAR -p granite
#OAR -l host=1/gpu=1,walltime=1:45:00
#OAR -O logs/OAR_%jobid%.out
#OAR -E logs/OAR_%jobid%.err
# Asymmetric PPO, scenario 18 on Grid'5000. Flags and variants live in g5k/s18_env.sh (shared with jack).
#   oarsub -S "./g5k/train_asym_scen18.oar.sh <variant>"     variants: smoke legacy base clip10 yield6 cmdobs cmdnokl cmdsmoke
#   Fine-tune a checkpoint: add `export LEGNAV_FT_INIT=<ckpt.msgpack>` below (OAR does NOT forward the submit shell's env)
set -euo pipefail
cd "$HOME/legnav"
module load conda
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q '^legnav '; then bash g5k/setup_env.sh; eval "$(conda shell.bash hook)"; fi
conda activate legnav

VARIANT="${1:-base}"; RUN_TAG="${OAR_JOB_ID:-local}"
source g5k/s18_env.sh
# Blackwell 96 GB: 4x envs per update. Minibatches scale with it, so minibatch size (16384) and the total
# number of gradient steps stay those of the 1024-env recipe; only the data per update grows (4x fewer, less noisy updates).
# For the 1024-env baseline: LEGNAV_NUM_ENVS=1024 LEGNAV_N_MINIBATCHES=8 LEGNAV_FT_SNAP_EVERY=50.
export LEGNAV_NUM_ENVS="${LEGNAV_NUM_ENVS:-4096}" LEGNAV_N_MINIBATCHES="${LEGNAV_N_MINIBATCHES:-32}" LEGNAV_FT_SNAP_EVERY="${LEGNAV_FT_SNAP_EVERY:-20}"
nvidia-smi -L
exec python -m legnav.algorithms.jax_ppo_asym
