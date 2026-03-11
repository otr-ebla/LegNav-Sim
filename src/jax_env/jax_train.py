"""
jax_train.py — Rollout Collection
===================================
FIXES vs previous version:

  FIX — _vmap_step stale-closure bug (Bug #3):
    The old code stored _vmap_step as a module-level global that was mutated
    inside init_env_state(). collect_rollouts was @jax.jit, so it captured
    _vmap_step at first-trace time and never saw the updated closure after a
    curriculum stage change.
    FIX: init_env_state() now RETURNS the freshly built vmap_step function.
    collect_rollouts receives it as an explicit argument declared in
    static_argnums=(2,3) so that a new vmap_step (new Python object) triggers
    a retrace — meaning the new curriculum distance actually takes effect.

  FIX — @jax.jit removed from reset_env / step_env (Issue #8, coordinated
    with jax_env.py): JIT now lives only at the outermost vmap level here.

  All previous changes (OBS_SIZE=342, ROLLOUT_STEPS, curriculum) are carried.
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import functools
import jax
import jax.numpy as jnp
#from jax_env import reset_env, step_env
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_network import scale_actions_batched


def _verify_gpu():
    try:
        cuda_devices = jax.devices("cuda")
    except RuntimeError:
        cuda_devices = []
    if not cuda_devices:
        raise RuntimeError(
            f"No CUDA GPU found. All devices: {jax.devices()}."
        )
    print(f"GPU verified: {cuda_devices}")
    return cuda_devices[0]

GPU_DEVICE = _verify_gpu()

# Config
NUM_ENVS      = 16384
ROLLOUT_STEPS = 150 #64

# OBS_SIZE: 9 (pose×3) + 9 (state_vec) + 324 (lidar×3) = 342
OBS_SIZE = 3 * 3 + 9 + 108 * 3    # 342

# Build stacked env functions once — these are pure function pairs, no JIT here.
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)


def init_env_state(rng_key, min_goal_dist: float = 3.0):
    """
    Initialise all NUM_ENVS environments and return the vmap_step closure
    bound to the current curriculum distance.

    Returns: (env_obs, env_state, vmap_step)
      vmap_step — the JIT-compiled vmapped autoreset stepper for this stage.
                  Pass it into collect_rollouts as a static arg so that a
                  curriculum change (new Python object) triggers a retrace.

    FIX (Bug #3): vmap_step is returned and threaded explicitly instead of
    being a mutable global, eliminating the stale-closure bug.
    """
    step_auto = make_autoreset_env(reset_stacked, step_stacked,
                                   min_goal_dist=min_goal_dist)
    # JIT lives here — outermost level, no nested JIT inside step_env/reset_env.
    vmap_step = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))

    def _reset_with_dist(key):
        return reset_stacked(key, min_goal_dist=min_goal_dist)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)
    return env_obs, env_state, vmap_step


def batched_sample_action(rng_key, mean, logstd):
    std       = jnp.exp(logstd)
    noise     = jax.random.normal(rng_key, shape=mean.shape)
    actions   = mean + noise * std
    log_probs = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return actions, log_probs


# static_argnums:
#   2 = apply_fn   (unchanged across curriculum)
#   3 = vmap_step  (NEW — changes on curriculum stage, triggers retrace)
@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts(rng_key, params, apply_fn, vmap_step, env_state, env_obs):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.

    vmap_step is a static arg: when init_env_state() returns a new vmap_step
    object (after a curriculum change), passing it here causes JAX to retrace
    and compile a new kernel that uses the updated autoreset closure.
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        mean, logstd, values = apply_fn({"params": params}, current_obs)
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd)

        max_v       = current_state.env_state.max_v
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions
        )

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
            "passive_col":  infos["passive_col"],
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )
    return rollout_history, new_state, new_obs