"""
jax_network.py — Actor-Critic Neural Network for PPO (Shared Trunk + GRU Memory)

CHANGES vs previous version:
  - Added GRUCell after the fused CNN+global representation.
  - __call__ now takes `hidden` (GRU carry) as mandatory second argument
    and returns `new_hidden` alongside mean, logstd, value.
  - initialize_carry() returns a zero hidden state of the correct shape.
  - GRU_HIDDEN_SIZE = 128 — chosen to match the existing shared trunk width
    so the downstream actor/critic heads are structurally unchanged.
  - All existing squashing, log-prob, and action-scaling helpers are untouched.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0

_STATE_VEC_SIZE = 9
GRU_HIDDEN_SIZE = 128   # width of GRU memory cell


class EndToEndActorCritic(nn.Module):
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(
        self,
        x:      jnp.ndarray,           # (..., OBS_SIZE)
        hidden: jnp.ndarray,            # (..., GRU_HIDDEN_SIZE)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns (actor_mean, actor_logstd, value, new_hidden).
        hidden must be initialised to zeros at the start of every episode
        (or every rollout when TBPTT is not used).
        """
        pose_size  = 3 * self.stack_dim          # 9
        state_size = _STATE_VEC_SIZE             # 9

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # ── 1-D CNN on stacked LiDAR ──────────────────────────────────────────
        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        # ── Global state MLP ──────────────────────────────────────────────────
        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1)

        # ── Fused trunk (pre-GRU) ─────────────────────────────────────────────
        fused  = jnp.concatenate([cnn_feat, global_in], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(GRU_HIDDEN_SIZE)(shared))

        # ── GRU memory cell ───────────────────────────────────────────────────
        # nn.GRUCell signature: (carry, inputs) → (new_carry, new_carry)
        # We use the new_carry as our recurrent feature vector.
        gru_cell = nn.GRUCell(features=GRU_HIDDEN_SIZE)
        new_hidden, gru_out = gru_cell(hidden, shared)
        # gru_out == new_hidden for a single-step GRUCell call.

        # ── Actor head ────────────────────────────────────────────────────────
        actor_mean = nn.Dense(self.action_dim)(gru_out)

        # FIX: tanh squashing keeps gradients alive at the log_std boundaries
        logstd_param = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        actor_logstd_raw = jnp.broadcast_to(logstd_param, actor_mean.shape)
        actor_logstd = (
            LOG_STD_MIN
            + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (jnp.tanh(actor_logstd_raw) + 1.0)
        )

        # ── Critic head ───────────────────────────────────────────────────────
        critic = nn.relu(nn.Dense(64)(gru_out))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1), new_hidden

    @staticmethod
    def initialize_carry(batch_size: int) -> jnp.ndarray:
        """
        Returns a zero GRU carry of shape (batch_size, GRU_HIDDEN_SIZE).
        Call this at the start of every rollout (or every episode for eval).
        """
        return jnp.zeros((batch_size, GRU_HIDDEN_SIZE))


# ── Action squashing helpers (unchanged) ──────────────────────────────────────

def _squash_log_jacobian(raw_actions: jnp.ndarray, max_v: float = 1.0) -> jnp.ndarray:
    """
    Log-determinant of the Jacobian of the squashing functions.
    v = sigmoid(raw[0]) * max_v  →  d/d(raw[0]) = sigmoid * (1 - sigmoid) * max_v
    w = tanh(raw[1])             →  d/d(raw[1]) = 1 - tanh²
    Returns shape (...,) — to be subtracted from the Gaussian log_prob.
    """
    v_squash = jax.nn.sigmoid(raw_actions[..., 0])
    w_squash = jnp.tanh(raw_actions[..., 1])
    log_dv = jnp.log(v_squash * (1.0 - v_squash) * max_v + 1e-6)
    log_dw = jnp.log(1.0 - w_squash ** 2 + 1e-6)
    return log_dv + log_dw


def squash_corrected_log_prob(
    raw_actions: jnp.ndarray,
    mean:        jnp.ndarray,
    logstd:      jnp.ndarray,
    max_v:       float = 1.0,
) -> jnp.ndarray:
    std = jnp.exp(logstd)
    z   = (raw_actions - mean) / (std + 1e-8)
    base_log_prob = jnp.sum(-0.5 * (z ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return base_log_prob - _squash_log_jacobian(raw_actions, max_v)


def sample_action(
    rng_key: jnp.ndarray, mean: jnp.ndarray, logstd: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    std      = jnp.exp(logstd)
    noise    = jax.random.normal(rng_key, shape=mean.shape)
    action   = mean + noise * std
    log_prob = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)
    return action, log_prob


def scale_action_to_env(raw_action: jnp.ndarray, max_v: float) -> jnp.ndarray:
    v = jax.nn.sigmoid(raw_action[..., 0]) * max_v
    w = jnp.tanh(raw_action[..., 1])
    return jnp.stack([v, w], axis=-1)


def scale_actions_batched(raw_actions: jnp.ndarray, max_v: jnp.ndarray) -> jnp.ndarray:
    v = jax.nn.sigmoid(raw_actions[:, 0]) * max_v
    w = jnp.tanh(raw_actions[:, 1])
    return jnp.stack([v, w], axis=-1)


def get_deterministic_action(mean: jnp.ndarray, max_v: float = 1.5) -> jnp.ndarray:
    return scale_action_to_env(mean, max_v)