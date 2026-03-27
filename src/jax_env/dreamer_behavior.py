"""
dreamer_behavior.py — Actor-Critic and Latent Imagination Engine
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable
from dreamer_rssm import DETERMINISTIC_SIZE, symlog, symexp

class DreamerActor(nn.Module):
    """
    Predicts the action distribution from the latent state.
    Outputs mean and standard deviation for continuous control.
    """
    action_dim: int
    min_std: float = 0.1

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        
        mean = nn.Dense(self.action_dim)(x)
        # Softplus ensures standard deviation is strictly positive
        std_raw = nn.Dense(self.action_dim)(x)
        std = jax.nn.softplus(std_raw) + self.min_std
        
        # Tanh squashing will be applied during sampling to bound actions [-1, 1]
        return mean, std

class DreamerCritic(nn.Module):
    """
    Predicts the expected symlog-scaled return from a latent state.
    """
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)

def sample_action(rng_key: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Samples an action using the reparameterization trick and applies Tanh squashing.
    """
    noise = jax.random.normal(rng_key, mean.shape)
    raw_action = mean + noise * std
    action = jnp.tanh(raw_action)
    
    # Calculate log probability of the squashed action
    log_prob = -0.5 * jnp.sum(jnp.square(noise) + jnp.log(2 * jnp.pi) + 2 * jnp.log(std), axis=-1)
    # Correction for Tanh squashing
    log_prob -= jnp.sum(jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1)
    
    return action, log_prob

def compute_lambda_returns(rewards: jnp.ndarray, values: jnp.ndarray, continues: jnp.ndarray, 
                           bootstrap: jnp.ndarray, gamma: float = 0.99, lambda_: float = 0.95) -> jnp.ndarray:
    """
    Computes the exponentially weighted lambda-returns backwards through the imagined trajectory.
    """
    def step(next_return, inputs):
        r, v, c = inputs
        # Lambda return formula: r_t + gamma * c_t * ((1 - lambda) * v_{t+1} + lambda * return_{t+1})
        ret = r + gamma * c * ((1.0 - lambda_) * v + lambda_ * next_return)
        return ret, ret

    # We process the sequence from the end to the beginning
    _, returns = jax.lax.scan(
        step, 
        bootstrap, 
        (rewards, values, continues), 
        reverse=True
    )
    return returns

def unroll_imagination(rng_key: jnp.ndarray, 
                       rssm_step_fn: Callable, rssm_prior_fn: Callable, 
                       actor_apply_fn: Callable, params: dict,
                       start_h: jnp.ndarray, start_z: jnp.ndarray, 
                       horizon: int = 15):
    """
    Dreams a trajectory of length 'horizon' strictly inside the latent space.
    """
    def imagine_step(carry, _):
        h_prev, z_prev, current_key = carry
        current_key, action_key, prior_key = jax.random.split(current_key, 3)
        
        # 1. Actor chooses an action based on the current dream state
        mean, std = actor_apply_fn(params['actor'], h_prev, z_prev)
        action, log_prob = sample_action(action_key, mean, std)
        
        # 2. RSSM predicts the deterministic next step
        h_next = rssm_step_fn(params['rssm'], h_prev, z_prev, action)
        
        # 3. RSSM Prior predicts the stochastic next step
        z_next, _ = rssm_prior_fn(params['rssm'], h_next, prior_key)
        
        # Save the transition data
        transition = {
            'h': h_prev,
            'z': z_prev,
            'action': action,
            'log_prob': log_prob
        }
        
        return (h_next, z_next, current_key), transition

    # Scan the imagination loop for H steps
    final_state, imagined_trajectory = jax.lax.scan(
        imagine_step, 
        (start_h, start_z, rng_key), 
        None, 
        length=horizon
    )
    
    return imagined_trajectory, final_state[0], final_state[1]