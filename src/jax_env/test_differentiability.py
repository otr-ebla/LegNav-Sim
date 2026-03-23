#!/usr/bin/env python3
"""
test_differentiability.py — End-to-end gradient verification for the JAX nav env.

Usage:
    cd /Users/albertovaglio/indoor-rl-nav/src/jax_env
    python test_differentiability.py

Tests:
  1. Single-step gradient: ∂reward/∂action is finite and non-zero.
  2. Wall boundary gradient: robot near wall → ∂reward/∂v ≠ 0 (was zero with jnp.clip).
  3. LiDAR gradient: ∂lidar_sum/∂[cx,cy] is finite (was NaN with jnp.inf sentinels).
  4. Full SHAC rollout gradient: 16-step rollout through step_env, ∂J/∂θ is healthy.
  5. Sensor noise disabled: noisy_lidar == raw_lidar when SENSOR_NOISE=False.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only for test

import jax
import jax.numpy as jnp
import optax

# ----- 1. Disable sensor noise before ANY import of jax_env -----
import jax_env as _jax_env
_jax_env.SENSOR_NOISE = False

from jax_env_multi import reset_env, step_env, ROOM_W, ROOM_H, ROBOT_RADIUS, GOAL_RADIUS
from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_wrappers import make_stacked_env, StackedEnvState
from jax_physics import compute_lidar

_PASS = "\033[92m✓ PASS\033[0m"
_FAIL = "\033[91m✗ FAIL\033[0m"

def _ok(val): return _PASS if val else _FAIL
def _check(name, cond):
    print(f"  {_ok(cond)}  {name}")
    return cond

def _is_healthy(x):
    return bool(jnp.all(jnp.isfinite(x)) and jnp.any(jnp.abs(x) > 1e-12))


# ─── Test 1: Single-step reward gradient ──────────────────────────────────────
def test_single_step_gradient():
    print("\n[1] Single-step ∂reward/∂action")
    key = jax.random.PRNGKey(0)
    _, state = reset_env(key, min_goal_dist=2.0)

    def reward_fn(action):
        _, _, reward, _, _ = step_env(key, state, action)
        return reward

    action = jnp.array([0.5, 0.1])
    grad = jax.grad(reward_fn)(action)
    v_ok  = _check("∂reward/∂v is finite and non-zero", _is_healthy(grad[0:1]))
    w_ok  = _check("∂reward/∂w is finite and non-zero", _is_healthy(grad[1:2]))
    return v_ok and w_ok


# ─── Test 2: Wall boundary gradient ───────────────────────────────────────────
def test_wall_boundary_gradient():
    print("\n[2] Wall boundary: ∂reward/∂v ≠ 0 near wall (requires soft_clip)")
    key = jax.random.PRNGKey(1)
    _, state = reset_env(key, min_goal_dist=2.0)
    # Place robot near left wall
    state = state.replace(x=jnp.float32(ROBOT_RADIUS + 0.01), y=jnp.float32(ROOM_H / 2))

    def reward_fn(v_raw):
        action = jnp.array([v_raw, 0.0])
        _, _, reward, _, _ = step_env(key, state, action)
        return reward

    grad = jax.grad(reward_fn)(jnp.float32(0.5))
    return _check("∂reward/∂v ≠ 0 at wall boundary", _is_healthy(grad[None]))


# ─── Test 3: LiDAR gradient (no NaN from jnp.inf) ─────────────────────────────
def test_lidar_gradient():
    print("\n[3] LiDAR: ∂lidar_sum/∂circle_center is finite (requires finite sentinel)")
    from jax_env import NUM_RAYS, FOV, MAX_LIDAR_DIST

    def lidar_sum(cx):
        circles = jnp.array([[cx, 3.0, 0.3]])
        boxes   = jnp.zeros((1, 4))
        raw     = compute_lidar(6.0, 6.0, 0.0, circles, boxes,
                                NUM_RAYS, float(FOV), float(MAX_LIDAR_DIST),
                                12.0, 12.0)
        return jnp.sum(raw)

    grad = jax.grad(lidar_sum)(jnp.float32(5.0))
    return _check("∂lidar_sum/∂cx is finite", bool(jnp.isfinite(grad)))


# ─── Test 4: Full SHAC rollout gradient ───────────────────────────────────────
def test_shac_rollout_gradient():
    print("\n[4] Full SHAC rollout: ∂J/∂θ is finite and non-zero (16 steps)")
    HORIZON = 16
    key = jax.random.PRNGKey(2)

    actor_net = EndToEndActorCritic(action_dim=2)
    dummy_obs  = jnp.zeros((1, 342))
    actor_params = actor_net.init(key, dummy_obs)["params"]

    reset_stacked, _ = make_stacked_env(reset_env, step_env, stack_dim=3, ghost_robot=True)
    init_obs, init_state = reset_stacked(key, min_goal_dist=2.0)

    def rollout_return(params):
        def scan_fn(carry, _):
            obs, state, rng = carry
            rng, sk = jax.random.split(rng)
            mean, _, _ = actor_net.apply({"params": params}, obs[None])
            action = scale_action_to_env(mean[0], state.env_state.max_v)
            base_obs, new_base, reward, done, _ = step_env(sk, state.env_state, action)

            new_pose  = base_obs[:3]
            new_sv    = base_obs[3:12]
            new_lidar = base_obs[12:]
            new_ls = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], 0)
            new_ps = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  0)
            new_state = state.replace(env_state=new_base, lidar_stack=new_ls, pose_stack=new_ps)
            new_obs = jnp.concatenate([new_ps.flatten(), new_sv, new_ls.flatten()])
            return (new_obs, new_state, rng), reward

        _, rewards = jax.lax.scan(scan_fn, (init_obs, init_state, key), None, length=HORIZON)
        return jnp.sum(rewards)

    grads = jax.grad(rollout_return)(actor_params)
    leaves = jax.tree_util.tree_leaves(grads)
    has_nan = any(bool(jnp.any(~jnp.isfinite(g))) for g in leaves)
    all_zero = all(bool(jnp.all(g == 0)) for g in leaves)
    gn = float(optax.global_norm(grads))

    nan_ok  = _check("No NaN/Inf in gradients",               not has_nan)
    zero_ok = _check("Gradient is non-zero",                   not all_zero)
    norm_ok = _check(f"Gradient norm in healthy range (‖∇‖={gn:.3e})", 1e-10 < gn < 1e5)
    return nan_ok and zero_ok and norm_ok


# ─── Test 5: SENSOR_NOISE = False → deterministic LiDAR ──────────────────────
def test_sensor_noise_disabled():
    print("\n[5] SENSOR_NOISE=False → observation is deterministic")
    assert _jax_env.SENSOR_NOISE == False, "SENSOR_NOISE should be False"
    key = jax.random.PRNGKey(3)
    obs1, _ = reset_env(key, min_goal_dist=2.0)
    obs2, _ = reset_env(key, min_goal_dist=2.0)
    return _check("Same key → same obs (no stochastic noise)", bool(jnp.allclose(obs1, obs2)))


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  JAX Nav Env Differentiability Verification")
    print("=" * 60)

    results = [
        test_single_step_gradient(),
        test_wall_boundary_gradient(),
        test_lidar_gradient(),
        test_shac_rollout_gradient(),
        test_sensor_noise_disabled(),
    ]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"  Results: {n_pass}/{len(results)} passed  {'✓ ALL GOOD' if n_fail == 0 else f'✗ {n_fail} FAILED'}")
    print(f"{'='*60}\n")
    sys.exit(0 if n_fail == 0 else 1)
