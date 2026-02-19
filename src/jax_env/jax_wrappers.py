import jax
import jax.numpy as jnp
from flax import struct
from jax_env import EnvState, get_obs

# =============================================================================
# TEMPORAL STACK WRAPPER
# =============================================================================

@struct.dataclass
class StackedEnvState:
    """
    Augments the base environment state with historical buffers.
    """
    env_state: EnvState
    lidar_stack: jnp.ndarray  # Shape: (stack_dim, NUM_RAYS)
    pose_stack: jnp.ndarray   # Shape: (stack_dim, 3)

def make_stacked_env(base_reset_fn, base_step_fn, stack_dim: int = 3, num_rays: int = 108):
    """
    Wraps the base environment functions to provide temporal stacking.
    Returns: stacked_reset_fn, stacked_step_fn
    """
    
    @jax.jit
    def reset_stacked(key: jnp.ndarray) -> tuple[jnp.ndarray, StackedEnvState]:
        base_obs, base_state = base_reset_fn(key)
        
        # Extract components from the flat base observation
        # Layout: pose(3) + state(5) + lidar(num_rays)
        pose = base_obs[0:3]
        state_vec = base_obs[3:8]
        lidar = base_obs[8:]
        
        # Duplicate the first frame 'stack_dim' times
        lidar_stack = jnp.tile(lidar, (stack_dim, 1))
        pose_stack = jnp.tile(pose, (stack_dim, 1))
        
        stacked_state = StackedEnvState(
            env_state=base_state,
            lidar_stack=lidar_stack,
            pose_stack=pose_stack
        )
        
        # Flatten the stacked observation for the neural network
        # Shape will be: (stack_dim * 3) + 5 + (stack_dim * num_rays)
        flat_obs = jnp.concatenate([pose_stack.flatten(), state_vec, lidar_stack.flatten()])
        return flat_obs, stacked_state

    @jax.jit
    def step_stacked(key: jnp.ndarray, state: StackedEnvState, action: jnp.ndarray) -> tuple[jnp.ndarray, StackedEnvState, jnp.float32, jnp.bool_, dict]:
        # 1. Step the base environment
        base_obs, new_base_state, reward, done, info = base_step_fn(key, state.env_state, action)
        
        # 2. Extract new observation components
        new_pose = base_obs[0:3]
        new_state_vec = base_obs[3:8]
        new_lidar = base_obs[8:]
        
        # 3. Shift buffers and append new frame (Functional approach)
        # jnp.roll shifts the array. We then overwrite the last index (-1).
        new_lidar_stack = jnp.roll(state.lidar_stack, shift=-1, axis=0)
        new_lidar_stack = new_lidar_stack.at[-1, :].set(new_lidar)
        
        new_pose_stack = jnp.roll(state.pose_stack, shift=-1, axis=0)
        new_pose_stack = new_pose_stack.at[-1, :].set(new_pose)
        
        # 4. Create new immutable state
        new_stacked_state = StackedEnvState(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack
        )
        
        flat_obs = jnp.concatenate([new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()])
        return flat_obs, new_stacked_state, reward, done, info

    return reset_stacked, step_stacked

# =============================================================================
# AUTO-RESET WRAPPER (Mandatory for JAX RL)
# =============================================================================

def make_autoreset_env(reset_fn, step_fn):
    """
    Automatically resets the environment if done == True.
    """
    @jax.jit
    def step_autoreset(key: jnp.ndarray, state, action: jnp.ndarray):
        obs, next_state, reward, done, info = step_fn(key, state, action)
        
        # If the episode just finished, we need to reset the state instantly.
        # We split the key to get a fresh random seed for the reset.
        reset_key, _ = jax.random.split(key)
        reset_obs, reset_state = reset_fn(reset_key)
        
        # Branchless selection: if done, use reset_state, else use next_state
        # jax.tree_map applies jnp.where to every leaf in the dataclass
        final_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(done, x, y),
            reset_state,
            next_state
        )
        
        final_obs = jnp.where(done, reset_obs, obs)
        
        return final_obs, final_state, reward, done, info

    return step_autoreset