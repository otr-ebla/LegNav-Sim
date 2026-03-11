"""
fast_ppo_eval.py — Bare-metal, hyper-optimized PPO testing script
=================================================================
Evaluates a single PPO policy across thousands of parallel environments 
in a fraction of a second.
"""

import os
import time
import jax
import jax.numpy as jnp
import flax.serialization

# Enforce GPU acceleration and Triton GEMM for speed
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"

# CRITICAL: Set USE_LEGS before importing environments 
import jax_env
jax_env.USE_LEGS = True  # Set to False if you trained with the cylinder model

from jax_env import MAX_STEPS
from jax_wrappers import make_stacked_env
from jax_env_multi import reset_env, step_env
from jax_network import EndToEndActorCritic

# Number of parallel environments (tune down to 2048 if you hit OOM)
N_ENVS = 4096

def load_checkpoint(path):
    with open(path, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle.get("actor_params", bundle.get("params"))

def _squash_ppo(mean, max_v):
    """Maps unbounded network output to physical velocity limits."""
    v = jax.nn.sigmoid(mean[..., 0]) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)

def main():
    print(f"🚀 Initializing Fast PPO Evaluation (N_ENVS = {N_ENVS})...")
    
    # Load and pin parameters to the GPU
    params = load_checkpoint("checkpoints/ppo_model_best.msgpack")
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]
    params = jax.device_put(params, gpu)

    network = EndToEndActorCritic(action_dim=2)
    reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

    @jax.jit
    def eval_kernel(params, rng_key):
        """Single fused XLA kernel for the entire 400-step simulation."""
        reset_keys = jax.random.split(rng_key, N_ENVS)
        obs, state = jax.vmap(reset_stacked)(reset_keys)

        # State tracking: state, obs, active_mask, hit_goal, hit_col, hit_pcol
        carry = (
            state, 
            obs, 
            jnp.ones(N_ENVS, dtype=jnp.bool_),
            jnp.zeros(N_ENVS, dtype=jnp.bool_),
            jnp.zeros(N_ENVS, dtype=jnp.bool_),
            jnp.zeros(N_ENVS, dtype=jnp.bool_)
        )

        def step_fn(carry, step_idx):
            state, obs, active, goals, cols, pcols = carry
            
            # Forward pass
            mean, _, _ = network.apply({"params": params}, obs)
            actions = jax.vmap(_squash_ppo)(mean, state.env_state.max_v)
            
            # Environment step
            k_step = jax.random.fold_in(rng_key, step_idx)
            step_keys = jax.random.split(k_step, N_ENVS)
            next_obs, next_state, _, done, info = jax.vmap(step_stacked)(step_keys, state, actions)
            
            # Accumulate metrics (only if the environment was still active)
            new_goals = goals | (info["goal_reached"] & active)
            new_cols  = cols  | (info["collision"] & active)
            new_pcols = pcols | (info["passive_col"] & active)
            
            next_active = active & ~done
            
            return (next_state, next_obs, next_active, new_goals, new_cols, new_pcols), None

        # Execute the 400 steps entirely on the accelerator
        final_carry, _ = jax.lax.scan(step_fn, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32))
        _, _, _, final_goals, final_cols, final_pcols = final_carry
        
        return final_goals.mean(), final_cols.mean(), final_pcols.mean()

    rng = jax.random.PRNGKey(42)
    rng, sub_rng = jax.random.split(rng)
    
    # ── JIT Warmup ────────────────────────────────────────────────────────────
    print("Compiling XLA graph (this takes ~15 seconds)...")
    t0 = time.time()
    g, c, pc = eval_kernel(params, sub_rng)
    g.block_until_ready()
    print(f"Compilation finished in {time.time() - t0:.2f}s.\n")
    
    # ── Timed Evaluation ──────────────────────────────────────────────────────
    rng, sub_rng = jax.random.split(rng)
    t0 = time.time()
    
    g_rate, c_rate, pc_rate = eval_kernel(params, sub_rng)
    g_rate.block_until_ready()  # Force GPU synchronization before stopping the clock
    
    t_eval = time.time() - t0
    total_steps = N_ENVS * MAX_STEPS
    
    print("── Results ─────────────────────────")
    print(f"Time:          {t_eval:.3f} seconds")
    print(f"Throughput:    {total_steps / t_eval:,.0f} steps/second")
    print(f"Success Rate:  {g_rate * 100:.1f}%")
    print(f"Active Col:    {c_rate * 100:.1f}%")
    print(f"Passive Col:   {pc_rate * 100:.1f}%")
    print("────────────────────────────────────")

if __name__ == "__main__":
    main()