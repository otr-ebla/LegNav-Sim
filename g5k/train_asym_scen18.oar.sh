#!/bin/bash
#OAR -n legnav-asym-s18
#OAR -q abaca
#OAR -p granite
#OAR -l host=1/gpu=1,walltime=1:45:00
#OAR -O logs/OAR_%jobid%.out
#OAR -E logs/OAR_%jobid%.err
# Asymmetric PPO, scenario 18 (5x9 m room), Shelfy actuation (delay + lag + slew), TIAGo base footprint.
#   From scratch (default):  oarsub -S ./g5k/train_asym_scen18.oar.sh
#   Fine-tune a checkpoint:  add `export LEGNAV_FT_INIT=<ckpt.msgpack>` below (OAR does NOT forward the submit shell's env)
# Scratch = stock PPO curriculum (scenarios 0..max) with a fraction LEGNAV_FT_SCEN18_PROB of the episodes
# forced to scenario 18; snapshots + deployable CALF exports every 50 updates (no resume: the walltime ends the run).
set -euo pipefail
cd "$HOME/legnav"
module load conda
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q '^legnav '; then bash g5k/setup_env.sh; eval "$(conda shell.bash hook)"; fi
conda activate legnav

export LEGNAV_ROBOT_FOOTPRINT="${LEGNAV_ROBOT_FOOTPRINT:-rect:0.72x0.50}"   # Shelfy = TIAGo OMNI base
export LEGNAV_REAL_ENV=1                  # delay + first-order lag + slew (jax_env_real)
export LEGNAV_SOCIAL_FT=1                 # group-space / anticipation / feet / wall shaping
export LEGNAV_FT_PRED=1                   # waypoint-aware space-time conflict replaces constant-velocity CPA
export LEGNAV_GRP_CROSS_TERMINAL=1        # TRAINING ONLY: crossing a group link ends the episode (-60)
export LEGNAV_PRIV_V2=1                   # critic sees waypoints, group flags, conflict weight, actuation
if [ -n "${LEGNAV_FT_INIT:-}" ]; then
  export LEGNAV_FT_INIT LEGNAV_FT_SCEN18_PROB="${LEGNAV_FT_SCEN18_PROB:-1.0}"
else
  export LEGNAV_FT_SCEN18_PROB="${LEGNAV_FT_SCEN18_PROB:-0.8}"
  export LEGNAV_CALF_TEMPLATE="${LEGNAV_CALF_TEMPLATE:-checkpoints/multiseed_tiago_ft2/ppo_asym_ft/ft_final_calf.msgpack}"  # export container only
fi
export LEGNAV_TOTAL_STEPS="${LEGNAV_TOTAL_STEPS:-150000000}"   # run 1 (70M, 27 min): curriculum stuck at scen<=4 / 5.3 m, blended success ~57%

# ── Run 3 fixes (s18_run2 post-mortem, legnav/evaluation/diag_s18.py) ─────────────────────────
# Variant = first script argument:  oarsub -S "./g5k/train_asym_scen18.oar.sh base"
#   smoke   : base, 3M steps, snapshot every 2 updates (checks the new code paths + CALF export; ~10 min)
#   legacy  : run-2 recipe (no fix), as a control
#   base    : all fixes below, stock yield (3)
#   clip10  : base but the old +-10 reward clip           (ablation: is the clip the collision cause?)
#   yield6  : base + yield coefficient 6 (stock paper 10)  (ablation: brake before contact)
# All full variants run the same 150M schedule (entropy and LR anneal over the whole run), so the
# winner IS the deployable policy: no retraining with a different schedule afterwards.
VARIANT="${1:-base}"
if [ "$VARIANT" != "legacy" ]; then
  export LEGNAV_REW_CLIP=30          # -60/std(~2.5) = -24: terminals no longer all saturate at +-10
  export LEGNAV_S18_ENT=0.01,0.001   # entropy annealed, decoupled from the stalled curriculum
  export LEGNAV_S18_GATE=1           # curriculum advances on the non-s18 episodes only
  export LEGNAV_S18_GHOST=0.2        # pinned: 80% of the people react to the robot, 20% ignore it
  export LEGNAV_CMD_PEN=0.05,0.08    # command-vs-actual (v, w) gap: anti-chattering, prices noise
fi
case "$VARIANT" in
  legacy|base) ;;
  smoke)  export LEGNAV_TOTAL_STEPS=3000000 LEGNAV_FT_SNAP_EVERY=2 ;;
  clip10) export LEGNAV_REW_CLIP=10 ;;
  yield6) export LEGNAV_FT_YIELD=6 ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
export LEGNAV_CKPT_ROOT="$HOME/legnav/checkpoints/s18_run3_${VARIANT}_${OAR_JOB_ID:-local}"
echo "variant=$VARIANT  clip=${LEGNAV_REW_CLIP:-10} ent=${LEGNAV_S18_ENT:-curriculum} gate=${LEGNAV_S18_GATE:-0} \
cmd_pen=${LEGNAV_CMD_PEN:-off} yield=${LEGNAV_FT_YIELD:-3} steps=$LEGNAV_TOTAL_STEPS -> $LEGNAV_CKPT_ROOT"
export PYTHONUNBUFFERED=1                                        # live lines in the OAR log
# Blackwell 96 GB: 4x envs per update. Minibatches scale with it, so minibatch size (16384) and the total
# number of gradient steps stay those of the 1024-env recipe; only the data per update grows (4x fewer, less noisy updates).
# For the 1024-env baseline: LEGNAV_NUM_ENVS=1024 LEGNAV_N_MINIBATCHES=8 LEGNAV_FT_SNAP_EVERY=50.
export LEGNAV_NUM_ENVS="${LEGNAV_NUM_ENVS:-4096}" LEGNAV_N_MINIBATCHES="${LEGNAV_N_MINIBATCHES:-32}" LEGNAV_FT_SNAP_EVERY="${LEGNAV_FT_SNAP_EVERY:-20}"
nvidia-smi -L
exec python -m legnav.algorithms.jax_ppo_asym
