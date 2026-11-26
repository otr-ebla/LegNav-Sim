"""Parity check: robot-side ShelfyObsBuilder (NumPy) vs the training env's stacked obs (JAX).

Replays the real-robot training env with random actions and rebuilds every observation from the raw
quantities a robot node has (ranges in metres, odometry pose, goal, measured v/w, published commands).
  LEGNAV_REAL_ENV=1 LEGNAV_OBS_CMD=1 PYTHONPATH=. python -m legnav.deployment.shelfy.check_parity
(omit LEGNAV_OBS_CMD for the v668 layout). Exit code 1 on mismatch.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from legnav.core.jax_env_real import reset_env_real, step_env_real
from legnav.core.jax_wrappers import make_stacked_env
from legnav.core.obs_layouts import OBS_LAYOUT
from legnav.deployment.shelfy.obs import ShelfyObsBuilder, MAX_LIDAR, LIDAR_NORM_R


def _raw(st):
    es = st.env_state
    ranges = MAX_LIDAR - np.asarray(st.lidar_stack[-1], np.float64) * (MAX_LIDAR - LIDAR_NORM_R)
    pose = (float(es.x), float(es.y), float(es.theta))
    return ranges, pose, (float(es.goal_x), float(es.goal_y)), float(es.v), float(es.w)


def main(episodes=3, steps=60):
    reset_s, step_s = make_stacked_env(reset_env_real, step_env_real, stack_dim=3)
    rs = jax.jit(lambda k: reset_s(k, max_goal_dist=6.0, scenario_idx=-1, ghost_prob=0.2, max_scenario=4))
    ss = jax.jit(step_s)
    worst = 0.0
    for ep in range(episodes):
        key = jax.random.PRNGKey(ep)
        obs, st = rs(key)
        b = ShelfyObsBuilder(float(st.env_state.max_v), OBS_LAYOUT)
        ranges, pose, goal, v, w = _raw(st)
        worst = max(worst, np.abs(b.reset(ranges, pose, goal, v, w) - np.asarray(obs)).max())
        for t in range(steps):
            key, ka, ks = jax.random.split(key, 3)
            act = jax.random.uniform(ka, (2,), minval=jnp.array([0.0, -1.0]),
                                     maxval=jnp.array([float(st.env_state.max_v), 1.0]))
            obs, st, _, done, _ = ss(ks, st, act)
            b.record_command(*np.asarray(st.env_state.cmd_buf[1]))   # what the node published
            ranges, pose, goal, v, w = _raw(st)
            err = np.abs(b.step(ranges, pose, goal, v, w) - np.asarray(obs))
            if err.max() > 1e-4:
                i = int(err.argmax())
                print(f"MISMATCH ep {ep} t {t}: index {i} (of {obs.shape[0]}) err {err[i]:.2e}")
                return 1
            worst = max(worst, err.max())
            if bool(done):
                break
    print(f"layout {OBS_LAYOUT}: parity OK, max abs err {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
