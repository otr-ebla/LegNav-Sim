"""
jax_env_real.py — Real-robot variant of the training env (actuation realism only)
==================================================================================
Wraps jax_env_multi.reset_env / step_env (the env the PPO/SAC/TQC trainers use)
WITHOUT touching them. The simulated robot is
ideal (commanded v, w == actual v, w in the same step); a real base is not. This
module inserts, between the policy action and the unicycle integration:

    command ─► transport delay (whole policy steps)
            ─► first-order lag (tau)
            ─► acceleration slew limit (a_max, alpha_max)
            ─► actual (v, w), integrated by the unchanged jax_env_multi.step_env

Everything else (scenarios, TIAGo/circle footprint, humans, obstacles, rewards, LiDAR, obs layout/size) is the stock
env, so checkpoints stay loadable for fine-tuning: obs layout is identical, with
kin_vec's v, w being the ACTUAL velocities and max_v keeping the same
(max_v - 0.2) / 1.8 normalisation.

Per-episode domain randomisation of tau / accel / delay / max_v is drawn at reset
inside RealRobotConfig's ranges, so the policy does not overfit one measurement.

Usage (training): swap the env functions before building the stacked env, e.g.
    import legnav.algorithms.jax_train as jt
    from legnav.core.jax_env_real import reset_env_real, step_env_real
    jt.reset_env, jt.step_env = reset_env_real, step_env_real
(init_env_state / rebuild_vmap_step look these names up at call time.)
Robot footprint is the upstream one: LEGNAV_ROBOT_FOOTPRINT (circle | rect:LxW).
"""

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct

from legnav.config import RobotConfig
from legnav.core import jax_env_multi
from legnav.core.jax_env import EnvState, GOAL_VEC_SIZE

DT = RobotConfig.DT


class RealRobotConfig:
    """Actuation parameters. Defaults = measurements on the Jetson (Shelfy).
    Replace with the TIAGo base values before training for TIAGo."""
    # Speed limits (max_v is sampled per episode and fed to the policy as usual)
    V_MAX_RANGE   = (0.40, 0.50)    # m/s   measured 0.45
    W_MAX         = 1.0             # rad/s measured 1.0
    # Lag time constant (s)
    TAU_RANGE     = (0.40, 0.70)    # measured 0.55
    # Acceleration limits (m/s^2, rad/s^2)
    A_RANGE       = (0.50, 0.80)    # measured 0.65
    ALPHA_RANGE   = (1.10, 1.70)    # measured 1.4
    # Transport delay in policy steps (DT=0.15 s): measured 0.15 s == 1 step
    DELAY_2_PROB  = 0.2             # chance of 2 steps (jitter / slow inference)
    SUBSTEPS      = 5               # actuator integration sub-steps per policy step


@struct.dataclass
class RealEnvState(EnvState):
    cmd_buf:   jnp.ndarray   # (2, 2) last two issued commands [[v,w] older, [v,w] newest]
    tau:       jnp.float32
    a_max:     jnp.float32
    alpha_max: jnp.float32
    delay2:    jnp.ndarray   # bool: use the 2-step delay this episode


_BASE_FIELDS = [f.name for f in dataclasses.fields(EnvState)]


def reset_env_real(key, max_goal_dist: float = 3.0, **kwargs):
    """kwargs: scenario_idx, ghost_prob, max_scenario, min_goal_dist (as jax_env_multi)."""
    k_base, k_v, k_t, k_a, k_al, k_d = jax.random.split(key, 6)
    obs, s = jax_env_multi.reset_env(k_base, max_goal_dist=max_goal_dist, **kwargs)
    C = RealRobotConfig

    max_v = jax.random.uniform(k_v, minval=C.V_MAX_RANGE[0], maxval=C.V_MAX_RANGE[1])
    base = {n: getattr(s, n) for n in _BASE_FIELDS}
    base["max_v"] = max_v
    state = RealEnvState(
        **base,
        cmd_buf=jnp.zeros((2, 2), jnp.float32),
        tau=jax.random.uniform(k_t, minval=C.TAU_RANGE[0], maxval=C.TAU_RANGE[1]),
        a_max=jax.random.uniform(k_a, minval=C.A_RANGE[0], maxval=C.A_RANGE[1]),
        alpha_max=jax.random.uniform(k_al, minval=C.ALPHA_RANGE[0], maxval=C.ALPHA_RANGE[1]),
        delay2=jax.random.uniform(k_d) < C.DELAY_2_PROB,
    )
    # kin_vec = obs[2:7] = [v/max_v, w, (max_v-0.2)/1.8, ...]; only max_v changed.
    obs = obs.at[GOAL_VEC_SIZE + 2].set((max_v - 0.2) / 1.8)
    return obs, state


def _actuate(state: RealEnvState, cmd):
    """Advance (v, w) one policy step under delay + lag + slew. Returns
    (v_end, w_end, v_mean, w_mean). All velocities are non-negative in v."""
    C = RealRobotConfig
    applied = jnp.where(state.delay2, state.cmd_buf[0], state.cmd_buf[1])
    v_cmd = jnp.clip(applied[0], 0.0, state.max_v)
    w_cmd = jnp.clip(applied[1], -C.W_MAX, C.W_MAX)

    h = DT / C.SUBSTEPS
    v, w = state.v, state.w
    v_sum = w_sum = 0.0
    for _ in range(C.SUBSTEPS):                     # small static unroll
        dv = jnp.clip((v_cmd - v) / state.tau * h, -state.a_max * h, state.a_max * h)
        dw = jnp.clip((w_cmd - w) / state.tau * h, -state.alpha_max * h, state.alpha_max * h)
        v_new, w_new = v + dv, w + dw
        v_sum += 0.5 * (v + v_new)                  # trapezoid mean over the sub-step
        w_sum += 0.5 * (w + w_new)
        v, w = v_new, w_new
    return v, w, v_sum / C.SUBSTEPS, w_sum / C.SUBSTEPS


# ── Command-space smoothness (LEGNAV_CMD_PEN="cv,cw", off when unset) ─────────────────────────
# The stock smooth / rotation penalties are computed on the velocities fed to step_env, i.e. AFTER the
# lag filter: with tau ~0.55 s >> DT a bang-bang command sequence barely shows up there, so the policy
# chatters for free (commanded w changed sign on ~36% of steps, deterministic mean, s18_run2) and the
# entropy bonus pushes sigma to its cap (noise is low-passed too). This term penalises the gap between
# the NEW command and the actual (v, w) the policy observes in kin_vec, so it is observable (no hidden
# delay-buffer state) and also prices exploration noise: E[(mu + s*eps - v)^2] = (mu - v)^2 + s^2.
import os as _os
_CMD_PEN = tuple(float(c) for c in _os.environ.get("LEGNAV_CMD_PEN", "").split(",")) \
    if _os.environ.get("LEGNAV_CMD_PEN") else None


def step_env_real(key, state: RealEnvState, action, **kwargs):
    v_end, w_end, v_mean, w_mean = _actuate(state, action)

    # Stock step integrates a constant-velocity arc: feed it the step-mean (v, w)
    # so the pose displacement matches the lagged profile.
    cmd = jnp.stack([jnp.clip(action[0], 0.0, state.max_v),
                     jnp.clip(action[1], -RealRobotConfig.W_MAX, RealRobotConfig.W_MAX)])
    # For the predicted-trajectory conflict term: the robot relaxes from its true end velocity to
    # the NEWEST command with the episode's lag (so this step's action shows up immediately).
    obs, new_state, reward, done, info = jax_env_multi.step_env(
        key, state, jnp.stack([v_mean, w_mean]),
        robot_pred=(v_end, w_end, cmd[0], cmd[1], state.tau), **kwargs
    )

    new_state = new_state.replace(
        v=v_end, w=w_end,                           # END-of-step velocity is the true state
        cmd_buf=jnp.stack([state.cmd_buf[1], cmd]),
    )
    # Observation must show the actual velocity (kin_vec[0:2]), not the step mean.
    obs = obs.at[GOAL_VEC_SIZE + 0].set(v_end / jnp.maximum(state.max_v, 1e-3))
    obs = obs.at[GOAL_VEC_SIZE + 1].set(w_end)
    if _CMD_PEN is not None:
        cv, cw = _CMD_PEN
        cmd_pen = -(cv * jnp.square((cmd[0] - state.v) / jnp.maximum(state.max_v, 1e-3))
                    + cw * jnp.square(cmd[1] - state.w))
        cmd_pen = jnp.where(done, 0.0, cmd_pen)      # terminal rewards stay exact
        reward = reward + cmd_pen
        info = {**info, "rew_cmd": cmd_pen,
                "rew_terms": info["rew_terms"].at[2].add(cmd_pen)}   # slot 2 = smooth
    return obs, new_state, reward, done, info
