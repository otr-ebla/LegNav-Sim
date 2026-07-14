"""
jax_network_asym.py — Asymmetric Actor-Critic (CALF actor + privileged critic)

The actor is byte-for-byte the CALF policy of jax_network.EndToEndActorCritic:
LiDAR stack → SE(2) reprojection → shared 1D-CNN → temporal attention ⊕ g_T
→ MLP → (mean, logstd). It sees ONLY what the real robot can measure, so a
trained checkpoint deploys unchanged.

The critic is train-time only and never runs on the robot. It owns a SECOND,
independent CALF trunk (so value gradients never touch the actor's encoder) and
a parallel MLP branch over the privileged human state from jax_privileged.py —
shoe centres + body velocities of the humans within 5 m, stacked over 3 frames.
The two branches are concatenated into the value head:

    obs  ──► CALF_actor  ──► actor head ──► (mean, logstd)
    obs  ──► CALF_critic ──┐
    priv ──► MLP ──────────┴► value head ──► V(s)
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from typing import Tuple
import numpy as np

from legnav.core.jax_network import SharedEncoder, LOG_STD_MIN, LOG_STD_MAX, USE_TANH_INSIDE


class AsymmetricActorCritic(nn.Module):
    action_dim:  int
    stack_dim:   int  = 3
    num_rays:    int  = 216
    tanh_inside: bool = USE_TANH_INSIDE

    def setup(self):
        self.actor_encoder  = SharedEncoder(stack_dim=self.stack_dim, num_rays=self.num_rays)
        self.critic_encoder = SharedEncoder(stack_dim=self.stack_dim, num_rays=self.num_rays)

        self.actor_out = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.log_std   = self.param('log_std', constant(1.0), (self.action_dim,))

        # Privileged branch: the (PRIV_HUMANS × PRIV_FEAT × PRIV_STACK) vector
        # goes straight into an MLP — no CNN, no attention.
        self.priv_1 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.priv_2 = nn.Dense(64,  kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.value_1   = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.value_2   = nn.Dense(64,  kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.value_out = nn.Dense(1,   kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Deployable path: observation → (mean, logstd). No privileged input."""
        feat     = self.actor_encoder(x)                      # (..., 128)
        raw_mean = self.actor_out(feat)

        if self.tanh_inside:
            v_mean = jnp.tanh(raw_mean[..., 0]) * 0.5 + 0.5
            w_mean = jnp.tanh(raw_mean[..., 1])
            actor_mean = jnp.stack([v_mean, w_mean], axis=-1)
        else:
            actor_mean = raw_mean

        logstd_raw = jnp.broadcast_to(self.log_std, actor_mean.shape)
        actor_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (jnp.tanh(logstd_raw) + 1.0)
        return actor_mean, actor_logstd

    def critic(self, x: jnp.ndarray, priv: jnp.ndarray) -> jnp.ndarray:
        """Train-time path: observation + privileged human state → V(s)."""
        obs_feat  = self.critic_encoder(x)                    # (..., 128)
        priv_feat = nn.relu(self.priv_2(nn.relu(self.priv_1(priv))))   # (..., 64)

        z = jnp.concatenate([obs_feat, priv_feat], axis=-1)   # (..., 192)
        z = nn.relu(self.value_1(z))
        z = nn.relu(self.value_2(z))
        return jnp.squeeze(self.value_out(z), axis=-1)

    def __call__(self, x: jnp.ndarray, priv: jnp.ndarray):
        mean, logstd = self.actor(x)
        value        = self.critic(x, priv)
        return mean, logstd, value
