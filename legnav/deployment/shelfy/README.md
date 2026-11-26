# Shelfy real-robot policy: deployment contract

This folder is the robot side of the Shelfy track: what a ROS node must feed the policy, and a NumPy
reference that is checked against the training env. Robot integration code (ROS nodes, launch files)
lives in the robot workspace and imports or copies `obs.py`; it does not re-derive the observation.

## Who owns what

| Piece | Where | Owner / role |
|---|---|---|
| Shared simulator (scenarios, humans, LiDAR, rewards) | `legnav/core/jax_env*.py`, `jax_humans.py`, `jax_legs.py` | sim, used by every policy |
| Real-robot actuation (delay, lag, slew, max_v randomisation), pending-command features | `legnav/core/jax_env_real.py` (`LEGNAV_REAL_ENV=1`) | Shelfy track, training side |
| Observation layouts, single source of `OBS_SIZE` | `legnav/core/obs_layouts.py` (`LEGNAV_OBS_CMD=1` selects v672) | contract between training, eval, deployment |
| Stacking wrapper (frames, ego deltas, extra kin) | `legnav/core/jax_wrappers.py` | sim |
| Asymmetric PPO trainer, CALF export | `legnav/algorithms/jax_ppo_asym.py` | training |
| Diagnostics in the training env | `legnav/evaluation/diag_s18.py` | evaluation |
| Run recipe (flags, variants) | `g5k/s18_env.sh` | shared by all machines |
| Machine launchers | `jack/` (UniSi jack, fermi), `g5k/` (Grid'5000) | infrastructure |
| Robot-side observation + parity test | `legnav/deployment/shelfy/` (this folder) | deployment |
| TurtleBot / simulation ROS bridge (older v668 CALF policy, *last command* as v, w) | external repo HRI_Sim, `cohuro_bridge` | not this repo |

## Observation (layout v672, policies trained with `cmdobs`)

`[kin(5) | cmd(4) | goal(2x3) | ego_deltas(3x3) | lidar(216x3)]` = 672 floats, built every DT = 0.15 s.

- `kin = [v/max_v, w, (max_v-0.2)/1.8, goal_dist/MAX_GOAL, goal_align/pi]`, where **v and w are the
  measured base velocity** (odometry twist), not the last command. The HRI_Sim bridge passes the
  last command here, which is correct only for the ideal-actuation v668 policies.
- `cmd = [v_k/max_v, w_k, v_{k-1}/max_v, w_{k-1}]`: the two last commands **published on cmd_vel**
  (after any safety clamp), newest first, zeros at episode start. Call
  `ShelfyObsBuilder.record_command(v, w)` right after each publish.
- `max_v` must be the one the policy is told (same value in kin and in the action scaling), trained
  in [0.40, 0.50] m/s.
- LiDAR: 216 rays at `-pi + k*2pi/215` in the robot frame (CCW), ranges in metres, normalised
  `(12 - r)/11.8` clipped to [0, 1]. Frames `o_{t-4}, o_{t-2}, o_t` (stride 2).

Action (deterministic mean of the CALF actor): `v = clip(a0, 0, 1) * max_v`, `w = clip(a1, -1, 1)` rad/s.

Layout of a checkpoint: `params['Dense_0']['kernel'].shape[0]` is 203 for v668 and 207 for v672
(`obs.layout_of`). A v672 policy fed a v668 observation fails loudly in `obs_layouts.fit_obs`.

## Checks

```bash
LEGNAV_REAL_ENV=1 LEGNAV_OBS_CMD=1 LEGNAV_ROBOT_FOOTPRINT=rect:0.72x0.50 PYTHONPATH=. \
  python -m legnav.deployment.shelfy.check_parity        # robot builder == training obs (float32)
```
