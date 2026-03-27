"""
dreamer_rssm.py — Recurrent State Space Model for DreamerV3
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# Core dimensions for DreamerV3 categorical latents
NUM_CATEGORIES = 32
CATEGORY_SIZE = 32
DETERMINISTIC_SIZE = 512

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """
    Symmetric logarithmic scaling to compress extreme reward/value ranges.
    """
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of symlog to recover original scale.
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)

class CategoricalStraightThrough(nn.Module):
    """
    Samples categorical latents and applies the straight-through gradient estimator.
    """
    @nn.compact
    def __call__(self, logits: jnp.ndarray, sample: bool = True) -> jnp.ndarray:
        probs = nn.softmax(logits, axis=-1)
        
        if sample:
            # Gumbel-Softmax trick for differentiable sampling
            key = self.make_rng('gumbel')
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-8) + 1e-8)
            noisy_logits = logits + gumbel_noise
            indices = jnp.argmax(noisy_logits, axis=-1)
        else:
            indices = jnp.argmax(logits, axis=-1)
            
        # Hard one-hot vector for forward pass
        hard_z = jax.nn.one_hot(indices, CATEGORY_SIZE, dtype=jnp.float32)
        
        # Straight-through trick: forward pass uses hard_z, backward pass uses probs
        z = hard_z + probs - jax.lax.stop_gradient(probs)
        
        # Flatten the [32, 32] matrix into a 1024 vector for the GRU
        return z.reshape(z.shape[:-2] + (NUM_CATEGORIES * CATEGORY_SIZE,))

class RSSM(nn.Module):
    """
    The deterministic and stochastic transition model.
    """
    action_dim: int
    
    @nn.compact
    def __call__(self):
        # We will not use the standard __call__ directly.
        pass

    @nn.compact  # <-- ADD THIS
    def step_gru(self, h_prev: jnp.ndarray, z_prev: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        Advances the deterministic hidden state.
        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        """
        x = jnp.concatenate([z_prev, action], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        gru = nn.GRUCell(features=DETERMINISTIC_SIZE)
        new_h, _ = gru(h_prev, x)
        return new_h

    @nn.compact  # <-- ADD THIS
    def prior(self, h_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts the next stochastic state WITHOUT seeing the observation.
        """
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(h_t))
        logits = nn.Dense(NUM_CATEGORIES * CATEGORY_SIZE)(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        
        sampler = CategoricalStraightThrough()
        z_t = sampler(logits, sample=True)
        return z_t, logits

    @nn.compact  # <-- ADD THIS
    def posterior(self, h_t: jnp.ndarray, obs_embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the stochastic state USING the actual observation.
        """
        x = jnp.concatenate([h_t, obs_embed], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        logits = nn.Dense(NUM_CATEGORIES * CATEGORY_SIZE)(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        
        sampler = CategoricalStraightThrough()
        z_t = sampler(logits, sample=True)
        return z_t, logits
        
class DreamerEncoder(nn.Module):
    """
    Re-uses your optimized 1D CNN architecture to encode LiDAR and state vectors.
    """
    stack_dim: int = 3
    num_rays: int = 108

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        pose_size = 3 * self.stack_dim          
        state_size = 9                          

        pose_stack = obs[..., :pose_size]
        state_vec  = obs[..., pose_size : pose_size + state_size]
        lidar_flat = obs[..., pose_size + state_size:]

        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))

        fused = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        return nn.relu(nn.Dense(DETERMINISTIC_SIZE)(fused))