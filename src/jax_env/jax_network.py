"""
jax_network.py — Actor-Critic Neural Network
=============================================
CHANGES vs previous version:

  Updated stack_dim awareness: OBS_SIZE is now 342 (was 339).
    pose_size  = 3 * stack_dim = 9   (unchanged)
    state_size = 9               (was 6 — rear_prox expanded to 4 scalars)
    lidar_flat = 342 - 9 - 9 = 324  (unchanged: 108 rays × 3 frames)

  IMPROVEMENT — LR warmup note:
    The network itself is unchanged. The optimizer in jax_ppo.py now uses
    a warmup schedule (see that file). No changes needed here.

  UNCHANGED — Architecture, log_std head, critic head, entropy fix all correct.

Architecture:
  LiDAR stack  -> Conv1D(32,7) -> Conv1D(64,5) -> Conv1D(64,3) -> LayerNorm
  Pose+State   -> Dense(128)   -> Dense(64)
  Concat       -> Dense(256)   -> Dense(128)             [shared trunk]
  Actor mean   -> Dense(action_dim)
  Actor logstd -> Dense(action_dim)  [state-dependent, bias init -1.0]
  Critic       -> Dense(128)   -> Dense(64) -> Dense(1)  [separate head]
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0   # FIX: was 0.5 (std up to 1.65). Entropy reached 2.4 — policy
                     # was nearly uniform over action range after 290 updates.
                     # Capped at 0.0 → std ≤ 1.0, still ample exploration.

# state_size must match STATE_VEC_SIZE in jax_env.py
_STATE_VEC_SIZE = 9   # v, w, max_v_norm, goal_dist, goal_align, rear_prox×4


class EndToEndActorCritic(nn.Module):
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pose_size  = 3 * self.stack_dim          # 9
        state_size = _STATE_VEC_SIZE             # 9

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # 1-D CNN on stacked LiDAR
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        # Global state MLP
        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        # Shared trunk
        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        # Actor mean
        actor_mean = nn.Dense(self.action_dim)(shared)

        # Actor log_std — state-dependent Dense head, bias init -1.0
        actor_logstd = nn.Dense(
            self.action_dim,
            bias_init=nn.initializers.constant(-1.0),
        )(shared)
        actor_logstd = jnp.clip(actor_logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Critic head — FIX: branches from `shared`, not `fused`.
        # Old code branched from fused (pre-trunk), making the critic shallower
        # than the actor (fused→128→64→1 vs actor's fused→256→128→head).
        # Shallower critic underfits as policy improves → value loss rises
        # monotonically (50→124 in logs) → noisy advantages → corrupted gradients.
        critic = nn.relu(nn.Dense(128)(fused))
        critic = nn.relu(nn.Dense(64)(critic))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)


def sample_action(
    rng_key: jnp.ndarray,
    mean:    jnp.ndarray,
    logstd:  jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reparameterised Gaussian sample in raw (pre-squash) space."""
    std      = jnp.exp(logstd)
    noise    = jax.random.normal(rng_key, shape=mean.shape)
    action   = mean + noise * std
    log_prob = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return action, log_prob


def scale_action_to_env(raw_action: jnp.ndarray, max_v: float) -> jnp.ndarray:
    """raw -> env action: sigmoid*max_v for v, tanh for w."""
    v = jax.nn.sigmoid(raw_action[..., 0]) * max_v
    w = jnp.tanh(raw_action[..., 1])
    return jnp.stack([v, w], axis=-1)


def scale_actions_batched(raw_actions: jnp.ndarray, max_v: jnp.ndarray) -> jnp.ndarray:
    """Batched: raw_actions (N,2), max_v (N,) -> env_actions (N,2)."""
    v = jax.nn.sigmoid(raw_actions[:, 0]) * max_v
    w = jnp.tanh(raw_actions[:, 1])
    return jnp.stack([v, w], axis=-1)


def get_deterministic_action(mean: jnp.ndarray, max_v: float = 1.5) -> jnp.ndarray:
    """Evaluation-time deterministic action (no noise)."""
    return scale_action_to_env(mean, max_v)