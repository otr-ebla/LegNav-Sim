"""
jax_train.py — Rollout Collection
===================================
FIXES vs previous version:

  FIX — @jax.jit removed from reset_env / step_env (Issue #8).

  FIX — ghost_prob curriculum + vmap_step cache:
    Compiled vmap_step objects are cached in _VMAP_STEP_CACHE keyed by the
    ghost_robot bool. Curriculum transitions that keep the same ghost value
    reuse the cached kernel — zero retrace cost.

  FIX — Dynamic Curriculum Variables:
    max_goal_dist and scenario_idx are now passed as dynamic runtime tensors
    to the step_autoreset function using in_axes=(0, 0, 0, None, None). This 
    completely eliminates JIT recompilation freezes when the curriculum advances.

  FIX — ROLLOUT_STEPS 150→96: frees ~1.2GB rollout buffer.
"""

import os
import functools
import jax
import jax.numpy as jnp
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_network import scale_actions_batched, squash_corrected_log_prob



def batched_sample_action(rng_key, mean, logstd, max_v):
    std     = jnp.exp(logstd)
    noise   = jax.random.normal(rng_key, shape=mean.shape)
    raw_actions = mean + noise * std
    
    # Unified math logic. Stop duplicating equations.
    log_probs = squash_corrected_log_prob(raw_actions, mean, logstd, max_v)
    return raw_actions, log_probs

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
NUM_ENVS      = 8192
ROLLOUT_STEPS = 64
OBS_SIZE      = 3 * 3 + 9 + 108 * 3    # 342

_VMAP_STEP_CACHE: dict = {}   # {ghost_robot_bool: vmap_step_fn}


def init_env_state(rng_key, max_goal_dist: float = 3.0, ghost_prob: float = 1.0, scenario_idx: int = -1):
    """
    Initialise all NUM_ENVS environments and return the vmap_step closure.
    Only called once at startup, and when ghost_prob changes (see rebuild_vmap_step).
    """
    ghost_robot = (ghost_prob >= 0.5)

    # Build reset/step closures (pure Python, no JAX tracing here)
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob
    )

    # ONLY ghost_robot triggers recompilation now.
    cache_key = ghost_robot
    
    if cache_key not in _VMAP_STEP_CACHE:
        _VMAP_STEP_CACHE.clear()
        step_auto = make_autoreset_env(reset_stacked, step_stacked)
        _VMAP_STEP_CACHE[cache_key] = jax.jit(
            # None = max_goal_dist and scenario_idx are shared across the entire batch
            jax.vmap(step_auto, in_axes=(0, 0, 0, None, None)) 
        )
    vmap_step = _VMAP_STEP_CACHE[cache_key]

    # Initialize the very first environments with the correct starting curriculum
    def _reset_with_dist(key):
        return reset_stacked(key, max_goal_dist=max_goal_dist, scenario_idx=scenario_idx)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))

    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)
    
    return env_obs, env_state, vmap_step


def rebuild_vmap_step(ghost_prob: float):
    """
    FIX 4: Rebuild ONLY the vmap_step closure when ghost_prob changes at a
    curriculum transition — without touching env_state or env_obs.

    The old code called init_env_state on ghost transitions, which wiped all
    8192 live environments and reset the env_state array from the host. This
    destroyed the temporal link needed for GAE: last_val was computed for the
    old episodes, then the state was replaced with freshly spawned episodes,
    injecting a massive discontinuity into value targets.

    The correct behaviour: only the step closure changes (ghost_robot bool baked
    in). Existing episodes continue uninterrupted. The new ghost_robot behaviour
    takes effect from the next step call forward, which is exactly correct.

    ghost_prob >= 0.5  → ghost_robot=True  (humans ignore robot, training mode)
    ghost_prob <  0.5  → ghost_robot=False (humans avoid robot, harder mode)
    """
    ghost_robot = (ghost_prob >= 0.5)
    cache_key   = ghost_robot

    # Build fresh closures with the new ghost_prob baked in
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_prob=ghost_prob
    )

    # Always rebuild: the ghost bool changed, so we must recompile the kernel
    _VMAP_STEP_CACHE.clear()
    step_auto = make_autoreset_env(reset_stacked, step_stacked)
    _VMAP_STEP_CACHE[cache_key] = jax.jit(
        jax.vmap(step_auto, in_axes=(0, 0, 0, None, None))
    )
    return _VMAP_STEP_CACHE[cache_key]


def batched_sample_action(rng_key, mean, logstd, max_v):
    """
    FIX 1: Sample raw actions and compute log_prob corrected for sigmoid/tanh squashing.
    max_v is the per-env max velocity array (shape: NUM_ENVS,) used to scale
    the Jacobian of the v = sigmoid(raw) * max_v transformation.
    """
    std     = jnp.exp(logstd)
    noise   = jax.random.normal(rng_key, shape=mean.shape)
    raw_actions = mean + noise * std
    # Base Gaussian log_prob (before squash correction)
    base_log_probs = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    # Jacobian correction — uses per-env max_v for the sigmoid dimension
    v_squash = jax.nn.sigmoid(raw_actions[:, 0])
    w_squash = jnp.tanh(raw_actions[:, 1])
    log_dv   = jnp.log(v_squash * (1.0 - v_squash) * max_v + 1e-6)
    log_dw   = jnp.log(1.0 - w_squash ** 2 + 1e-6)
    log_probs = base_log_probs - (log_dv + log_dw)
    return raw_actions, log_probs


@functools.partial(jax.jit, static_argnums=(2, 3))
def collect_rollouts(rng_key, params, apply_fn, vmap_step, env_state, env_obs, max_goal_dist, scenario_idx):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        mean, logstd, values = apply_fn({"params": params}, current_obs)
        max_v       = current_state.env_state.max_v
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd, max_v)

        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(
            step_keys, current_state, env_actions, max_goal_dist, scenario_idx
        )

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "max_v":        max_v,          # FIX 1: needed for Jacobian correction in loss
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
            "passive_col":  infos["passive_col"],
            "active_col":   infos["active_col"],
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )
    _, _, last_val = apply_fn({"params": params}, new_obs)
    
    return rollout_history, new_state, new_obs, last_val