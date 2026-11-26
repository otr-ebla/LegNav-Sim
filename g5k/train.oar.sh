#!/bin/bash
#OAR -n legnav-ppo-test
#OAR -q abaca
#OAR -p granite
#OAR -l host=1/gpu=1,walltime=1:00:00
#OAR -O logs/OAR_%jobid%.out
#OAR -E logs/OAR_%jobid%.err
set -euo pipefail
cd "$HOME/legnav"
echo "Job $OAR_JOB_ID on $(hostname) at $(date)"
nvidia-smi
module load conda
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q '^legnav '; then bash g5k/setup_env.sh; eval "$(conda shell.bash hook)"; fi
conda activate legnav
export LEGNAV_TOTAL_STEPS="${LEGNAV_TOTAL_STEPS:-5000000}"
exec python -m legnav.algorithms.jax_ppo
