import jax
import jax.numpy as jnp
from jax_env import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# =============================================================================
# MASSIVE VECTORIZATION SETUP
# =============================================================================

NUM_ENVS = 4096  # Run 4096 parallel simulators on a single GPU
ROLLOUT_STEPS = 128  # Number of steps per PPO iteration

# 1. Apply Wrappers (Functional composition)
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto = make_autoreset_env(reset_stacked, step_stacked)

# 2. Vectorize the Environment (The magic wand)
# vmap automatically transforms a single-env function into a multi-env batch function
vmap_reset = jax.vmap(reset_stacked)
# in_axes=(0, 0, 0) means we map over axis 0 for keys, states, and actions
vmap_step = jax.vmap(step_auto, in_axes=(0, 0, 0)) 

# =============================================================================
# THE GPU ROLLOUT LOOP
# =============================================================================

@jax.jit(static_argnums=2)
def collect_rollouts(rng_key: jnp.ndarray, params: dict, apply_fn):
    """
    Collects a full batch of experience (NUM_ENVS * ROLLOUT_STEPS) entirely on the GPU.
    """
    # 1. Initialization
    rng_key, reset_key = jax.random.split(rng_key)
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    
    # Initialize all 4096 environments at once
    initial_obs, initial_state = vmap_reset(reset_keys)

    # 2. Define the inner step function for lax.scan
    def _env_step_fn(carry, _):
        """
        Executes a single step for all parallel environments and returns transitions.
        """
        current_state, current_obs, current_rng = carry
        
        # Split RNG for the policy and the environment step
        current_rng, action_rng, step_rng = jax.random.split(current_rng, 3)
        
        # --- PLUG YOUR CUSTOM NEURAL NETWORK HERE ---
        # Example: actions, log_probs, values = custom_network_apply(policy_params, current_obs, action_rng)
        # For this template, we generate random dummy actions
        dummy_v = jax.random.uniform(action_rng, shape=(NUM_ENVS,), minval=0.0, maxval=2.0)
        dummy_w = jax.random.uniform(action_rng, shape=(NUM_ENVS,), minval=-1.0, maxval=1.0)
        actions = jnp.stack([dummy_v, dummy_w], axis=-1)
        # --------------------------------------------
        
        # Step all 4096 environments simultaneously
        step_keys = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, infos = vmap_step(step_keys, current_state, actions)
        
        # Package the transition data for RL training (PPO buffer)
        transition = {
            "obs": current_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_obs": next_obs
            # Add log_probs and values from your network here
        }
        
        # Next carry state for the loop
        next_carry = (next_state, next_obs, current_rng)
        
        return next_carry, transition

    # 3. Execute the high-speed loop
    initial_carry = (initial_state, initial_obs, rng_key)
    
    # jax.lax.scan compiles a for-loop directly into XLA, avoiding Python overhead entirely
    # The 'None' means we don't have an input array to iterate over, just length
    final_carry, rollout_history = jax.lax.scan(
        _env_step_fn, 
        initial_carry, 
        None, 
        length=ROLLOUT_STEPS
    )
    
    # rollout_history contains arrays of shape (ROLLOUT_STEPS, NUM_ENVS, ...)
    return rollout_history, final_carry

# Example execution:
# master_rng = jax.random.PRNGKey(42)
# dummy_params = {} 
# history, final_state = collect_rollouts(master_rng, dummy_params)
# print(f"Collected {NUM_ENVS * ROLLOUT_STEPS} steps in an instant.")