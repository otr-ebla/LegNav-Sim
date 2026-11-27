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
| run2 | `checkpoints/s18_run2/final/` (Mac + shared home) | 76.4% / 12.7% | 64.2% / 20.1% | superseded by ft2 |
| run3 base | u700 | 67.0% / 21.5% | 55.2% / 28.8% | worse than run2, kept u700 + final as reference |
| run3 clip10 | u1100 | 65.4% / 25.3% | 53.8% / 32.7% | deleted |
| ft2 (no real-env training) | `checkpoints/multiseed_tiago_ft2/ppo_asym_ft/ft_final_calf` | 85.2% / 5.0% | 73.2% / 11.9% | fine-tune source |
| **ftreal seed 2** | `s18_run3_ftreal_fermi_gpu0_20261126_1845/ppo_asym_ft/ft_final_calf` | **90.6% / 3.9%** | **77.5% / 9.6%** | **deploy candidate** |
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

### run5: stopping-distance speed barrier (`LEGNAV_BRAKE_PEN`, abandoned)
`brake` = cmdobs + a penalty in `jax_env_multi.brake_barrier`: for every person in the forward corridor
(|lateral| < half width + 0.35 m), the highest speed that still stops before contact is
`v_safe = relu(gap - 0.10) / (1.2*DT + tau)` (gap = surface distance minus 0.25 m reach); the penalty is
`-coef * max_i relu(max(v_actual, v_cmd) - v_safe_i) / max_v`, coef 1.0. Zero when stopped, zero for people
behind or beside the robot. At 0.45 m/s: 0.09 at 0.4 m, 0.70 at 0.2 m, 1.0 below 0.1 m.
Success criterion at u200-300: Acol below 20% (base was 25-28% at the same updates) without a success drop.
Result: seed 42 u220 Acol 21.7%, seed 1 u260 Acol 27.4% (cmdobs 26.5% at u220): inside the seed noise, and the
curriculum still stalls at scen<=3-4. Both runs stopped, checkpoints deleted. The flag stays in the code (off).

### run6: fine-tune the ideal-env v668 policy in the real env (`ftreal`, done: deploy candidate)
Key measurement (deterministic diag of existing policies, ALL episodes, success / active col):

| Policy | ideal env ghost 0 | ideal env ghost 0.2 | real env ghost 0 | real env ghost 0.2 |
|---|---|---|---|---|
| ft2 `multiseed_tiago_ft2/ppo_asym_ft/ft_final_calf` (v668, trained without delay/lag) | 93.4% / 1.8% | 84.4% / 3.8% | **85.2% / 5.0%** | **73.2% / 11.9%** |
| run2 (scratch in the real env) | - | - | 76.4% / 12.7% | 64.2% / 20.1% |

The ideal-env social policy, unchanged, already beats every policy trained from scratch in the real env: the
scratch runs fail on the curriculum (never past scen<=4 / 5.1 m), not on the actuation. The real-env gap of ft2
is ~8 points of success. `ftreal` = `LEGNAV_FT_INIT=ft2` in the real env: actor copied, new privileged critic
(30 warm-up updates with the actor frozen), actor LR 5e-5 -> 1e-5, entropy 0.005, ghost 0.2, scenario 18 only,
target-KL 0.02, 40M steps (~305 updates), snapshots every 25. Obs stays v668: deployable with the existing bridge,
as long as kin v,w is the measured odometry velocity (see `legnav/deployment/shelfy/`).
Seeds: 42 (fermi GPU1), 2 (fermi GPU0), 1 (jack). Goal: real env >= 90% / <= 3% at ghost 0.

Results (real env, deterministic, `ft_final_calf`, success / active col; ALL = 80% scen18 + 20% curriculum):

| Policy | ghost 0 ALL | ghost 0 scen18 | ghost 0.2 ALL | ghost 0.2 scen18 |
|---|---|---|---|---|
| ft2 (start) | 85.2% / 5.0% | 86.1% / 4.4% | 73.2% / 11.9% | 74.4% / 11.2% |
| ftreal seed 2 | **90.6% / 3.9%** | **93.7% / 2.2%** | **77.5% / 9.6%** | 80.2% / 8.5% |
| ftreal seed 1 | 89.7% / 4.8% | 92.1% / 3.2% | 76.9% / 11.4% | 79.4% / 9.9% |
| ftreal seed 42 | 86.2% / 3.9% | 93.2% / 2.6% | 74.8% / 10.1% | 80.8% / 8.9% |

- All three seeds gain ~6-7 points on scenario 18 (86% -> 92-94% at ghost 0) and roughly halve the active
  collisions: the fine-tune recovers almost all of the ideal-env performance (93%).
- Seed 42 forgets the other scenarios (curriculum episodes: static collisions 23-29%), because ftreal trains on
  scenario 18 only. Seed 2 keeps them (static col ~5%). A later run could use `LEGNAV_FT_SCEN18_PROB=0.8`.
- Ghost 0.2 (20% of the people ignore the robot) stays the weak point: ~9% active + ~9% passive collisions,
  mostly people walking into the robot.
- **Deploy candidate: seed 2**, `checkpoints/s18_run3_ftreal_fermi_gpu0_20261126_1845/ppo_asym_ft/ft_final_calf.msgpack`
  (copy on the Mac: `checkpoints/s18_ftreal_seed2/`). Seeds 1 and 42: final only, snapshots deleted.

## Housekeeping
- Smoke runs (`smoke`, `cmdsmoke`, `brakesmoke`) only validate code paths: checkpoints deleted after passing.
- Failed ablations: checkpoints deleted, results kept in this file and in `logs/` on jack/fermi (shared home).
