"""
jax_train.py — Rollout Collection
===================================
CHANGES vs previous version:
  NUM_ENVS      16384 -> 4096   (same batch size, 4x longer horizon)
  ROLLOUT_STEPS    64 -> 256    (episodes are 15-1000 steps; 64 was too short)

FIX in questa versione:

  BUG 7 — init_env_state ricompila il JIT ad ogni chiamata (PERFORMANCE):
    _vmap_reset era definita a livello modulo ma poi wrappata in jax.jit()
    DENTRO init_env_state ad ogni invocazione. Ogni chiamata a init_env_state
    ricreava un nuovo oggetto JIT compilato (cache miss garantito).
    FIX: il JIT di _vmap_reset è applicato a livello modulo, una volta sola,
    e init_env_state usa direttamente la versione già compilata.

  INVARIATO — collect_rollouts, batched_sample_action, OBS_SIZE, NUM_ENVS,
  ROLLOUT_STEPS erano tutti corretti.
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import functools
import jax
import jax.numpy as jnp
from jax_env import reset_env, step_env
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
NUM_ENVS      = 4096    # was 16384
ROLLOUT_STEPS = 256     # was 64
OBS_SIZE      = 3 * 3 + 6 + 108 * 3    # 339

# Environment setup
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto   = make_autoreset_env(reset_stacked, step_stacked)

# FIX BUG 7: JIT applicato a livello modulo, UNA VOLTA SOLA.
# Prima era: jax.jit(_vmap_reset) chiamato dentro init_env_state ogni volta → cache miss.
_vmap_reset = jax.jit(jax.vmap(reset_stacked))
_vmap_step  = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))


def init_env_state(rng_key):
    """Initialise all NUM_ENVS environments once before the training loop."""
    reset_keys = jax.random.split(rng_key, NUM_ENVS)
    # FIX: usa direttamente _vmap_reset già JIT-compilato (nessuna ricompilazione)
    obs, state = _vmap_reset(reset_keys)
    return obs, state


def batched_sample_action(rng_key, mean, logstd):
    std       = jnp.exp(logstd)
    noise     = jax.random.normal(rng_key, shape=mean.shape)
    actions   = mean + noise * std
    log_probs = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return actions, log_probs


@functools.partial(jax.jit, static_argnums=(2,))
def collect_rollouts(rng_key, params, apply_fn, env_state, env_obs):
    """
    Collect ROLLOUT_STEPS steps across NUM_ENVS environments.
    Persistent env state across calls; autoreset handles episode boundaries.
    """
    def _env_step(carry, _):
        current_state, current_obs, current_rng = carry
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)

        mean, logstd, values = apply_fn({"params": params}, current_obs)
        raw_actions, log_probs = batched_sample_action(action_rng, mean, logstd)

        max_v       = current_state.env_state.max_v
        env_actions = scale_actions_batched(raw_actions, max_v)

        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = _vmap_step(
            step_keys, current_state, env_actions
        )

        transition = {
            "obs":          current_obs,
            "actions":      raw_actions,      # RAW — must match log_prob space
            "log_probs":    log_probs,
            "values":       values,
            "rewards":      rewards,
            "dones":        dones,
            "goal_reached": infos["goal_reached"],
            "collision":    infos["collision"],
        }
        return (next_state, next_obs, current_rng), transition

    (new_state, new_obs, _), rollout_history = jax.lax.scan(
        _env_step, (env_state, env_obs, rng_key), None, length=ROLLOUT_STEPS
    )
    return rollout_history, new_state, new_obs