import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class EndToEndActorCritic(nn.Module):
    """
    Pure JAX/Flax implementation of your Custom CNN + Pose/State fusion.
    """
    action_dim: int
    stack_dim: int = 3
    num_rays: int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 1. Unpack the flattened observation
        # Layout: pose_stack (3 * stack_dim) + state_vec (5) + lidar_stack (num_rays * stack_dim)
        pose_size = 3 * self.stack_dim
        state_size = 5
        
        pose_stack = x[..., :pose_size]
        state_vec = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size :]
        
        # 2. Reshape Lidar for 1D CNN
        # From flat to (Batch, Spatial/Rays, Channels/Time)
        lidar_cnn_input = lidar_flat.reshape((-1, self.num_rays, self.stack_dim))
        
        # 3. 1D CNN Spatial/Temporal Feature Extraction
        # In JAX, the last dimension is the feature/channel dimension by default
        cnn = nn.Conv(features=16, kernel_size=(6,), strides=(2,))(lidar_cnn_input)
        cnn = nn.relu(cnn)
        cnn = nn.Conv(features=32, kernel_size=(3,), strides=(2,))(cnn)
        cnn = nn.relu(cnn)
        
        # Flatten CNN output
        cnn_features = cnn.reshape((cnn.shape[0], -1))
        
        # 4. Global State Processing
        global_input = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_features = nn.Dense(64)(global_input)
        global_features = nn.relu(global_features)
        
        # 5. Fusion
        fused = jnp.concatenate([cnn_features, global_features], axis=-1)
        shared = nn.Dense(256)(fused)
        shared = nn.relu(shared)
        
        # 6. Actor Head (Continuous Actions)
        # We output the mean of the action distribution
        actor_mean = nn.Dense(self.action_dim)(shared)
        
        # We learn a separate, state-independent log standard deviation parameter
        actor_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        
        # 7. Critic Head (Value Estimate)
        value = nn.Dense(1)(shared)
        
        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)

def sample_action(rng_key: jnp.ndarray, mean: jnp.ndarray, logstd: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Samples an action from the Gaussian policy and computes its log probability.
    """
    std = jnp.exp(logstd)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    action = mean + noise * std
    
    # Calculate log probability of the sampled action
    var = jnp.exp(2 * logstd)
    log_prob = -0.5 * (((action - mean) ** 2) / var + jnp.log(var) + jnp.log(2 * jnp.pi))
    log_prob = jnp.sum(log_prob, axis=-1) # Sum over action dimensions
    
    return action, log_prob