# Scenario 18 / Shelfy policy: experiment log

Goal: a PPO policy for Shelfy (TIAGo OMNI base, rect footprint 0.72x0.50 m) that crosses the scenario-18 room
(5x9 m, people + groups) without contacts, trained in the real-robot env (`LEGNAV_REAL_ENV=1`: 1-2 step transport
delay, first-order lag tau 0.4-0.7 s, slew limits, max_v 0.40-0.50 m/s, DT 0.15 s) so it can run on the real robot.

Recipe flags and variants: `g5k/s18_env.sh` (shared by Grid'5000 and the UniSi machines).
Launch: `jack/train_s18.sh` (UniSi) or `g5k/train_asym_scen18.oar.sh` (Grid'5000).
Ranking: `jack/diag_latest.sh` -> `legnav/evaluation/diag_s18.py`, deterministic, 256 envs x 1000 steps,
ghost 0.0 (every person reacts, GUI-like) and 0.2 (training setting). Rank snapshots with this, not `best.msgpack`
(training success is stochastic and blended with the curriculum).
Deploy contract (obs builder, parity check): `legnav/deployment/shelfy/`.

## Results (deterministic diag, ALL episodes)

| Run | Checkpoint | ghost 0: success / active col | ghost 0.2: success / active col | Status |
|---|---|---|---|---|
| run2 | `checkpoints/s18_run2/final/` (local Mac only) | 78% / - | 63-67% / - | **current deploy fallback** |
| run3 base | u700 | 67.0% / 21.5% | 55.2% / 28.8% | worse than run2, kept u700 + final as reference |
| run3 clip10 | u1100 | 65.4% / 25.3% | 53.8% / 32.7% | deleted |
| run3 yield6 | u700 | 52.6% / 15.1% (static col 30%) | 44.8% / 21.1% | deleted |

Training-log plateau shared by every run3/run4 variant: curriculum stuck at scen<=4, 5.1 m, training success
~57-62%, Acol ~25-30%, KL ~0.12-0.16, clip fraction ~0.4.

## What each attempt tested and what we learned

### run2 (baseline recipe)
Post-mortem (`diag_s18.py`, per-term reward sums):
- Reward clip +-10 after per-step std normalisation (std ~2.5) saturates 86-93% of the terminals:
  a collision and a timeout look the same to the critic.
- Entropy coefficient tied to blended success stayed at 0.012; sigma_v grew to 0.97. The actuator lag
  low-passes the noise, so the policy is never punished for it.
- Smoothness / rotation penalties were computed on lag-filtered velocities -> commanded w flips sign on ~36% of
  deterministic steps (chattering invisible to the reward).
- Active collisions happen at ~0.34 m/s, 25-37% of them with standing people (ghost 0): the robot does not brake.

### run3 (2026-11-26): fixes from the post-mortem
Common flags: `LEGNAV_REW_CLIP=30`, `LEGNAV_S18_ENT=0.01,0.001` (annealed entropy), `LEGNAV_S18_GATE=1`
(curriculum advances on non-s18 episodes only), `LEGNAV_S18_GHOST=0.2`, `LEGNAV_CMD_PEN=0.05,0.08`
(command-vs-actual gap penalty).
- `base`: all fixes. **Did not beat run2** (67%/55% vs 78%/63-67%). Same plateau.
- `clip10`: base with the old +-10 clip. Same as base -> the clip is not the cause of the collisions.
- `yield6`: yield coefficient 6. Fewer active collisions (15%) but static collisions jump to 30% and success
  drops: a stronger yield term makes the robot hesitate and drift into walls, it does not teach braking.
- Conclusion: the reward fixes are harmless but do not move the plateau.

### run4: real-robot observation v672
Hypothesis: the actor cannot see the 1-2 commands still in the transport-delay buffer, so it cannot predict its
own motion. `LEGNAV_OBS_CMD=1` appends the pending commands (obs 668 -> 672; deploy must send them, see
`legnav/deployment/shelfy/`). `LEGNAV_TARGET_KL=0.02` stops the PPO epoch early when KL explodes.
- `cmdobs` (obs + target-KL) and `cmdnokl` (obs only): at equal updates both stay inside the base band
  (Acol 23-29%, same scen<=4 / 5.1 m plateau). The KL stop only started firing around u380.
- Seed noise between two practically identical runs is ~4-5 Acol points: differences smaller than that are noise.
- Conclusion: seeing the pending commands alone does not make the robot brake.

### run5: stopping-distance speed barrier (`LEGNAV_BRAKE_PEN`, in progress)
`brake` = cmdobs + a penalty in `jax_env_multi.brake_barrier`: for every person in the forward corridor
(|lateral| < half width + 0.35 m), the highest speed that still stops before contact is
`v_safe = relu(gap - 0.10) / (1.2*DT + tau)` (gap = surface distance minus 0.25 m reach); the penalty is
`-coef * max_i relu(max(v_actual, v_cmd) - v_safe_i) / max_v`, coef 1.0. Zero when stopped, zero for people
behind or beside the robot. At 0.45 m/s: 0.09 at 0.4 m, 0.70 at 0.2 m, 1.0 below 0.1 m.
Success criterion at u200-300: Acol below 20% (base was 25-28% at the same updates) without a success drop.

## Housekeeping
- Smoke runs (`smoke`, `cmdsmoke`, `brakesmoke`) only validate code paths: checkpoints deleted after passing.
- Failed ablations: checkpoints deleted, results kept in this file and in `logs/` on jack/fermi (shared home).
