"""
vanilla_mlp_network.py — VanillaMLPActorCritic network definition (no GPU deps)
===============================================================================

Lightweight module containing ONLY the Flax network class.
Safe to import from CPU-only eval scripts (``jax_eval_multi.py``,
``test_scenarios_eval.py``) because it has **no JAX GPU pinning** and
depends only on ``flax`` and ``jax``.

The full PPO training logic lives in ``ppo_mlp_baseline.py``.
"""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal, constant

# Keep LOG_STD bounds aligned with jax_network.py
LOG_STD_MIN = -4.0
LOG_STD_MAX  =  0.0


class VanillaMLPActorCritic(nn.Module):
    """
    Two-hidden-layer MLP actor-critic for navigation.

    The full 662-dim stacked observation is fed directly into the first
    linear layer — no CNN, no attention, no LiDAR-specific processing.

    Architecture::

        obs (662,)
          Dense(hidden_dim=128) → ReLU
          Dense(hidden_dim=128) → ReLU    ← shared trunk
          ├─ Dense(action_dim=2)                          → mean
          ├─ learnable log_std  (soft-clamped to [LOG_STD_MIN, LOG_STD_MAX])
          └─ Dense(1, bias_init=0)                        → value (scalar)

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.  Default 2 ([v, w]).
    hidden_dim : int
        Width of each hidden layer.  Default 128.
    """

    action_dim: int = 2
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """
        Parameters
        ----------
        obs : (..., 662) float32

        Returns
        -------
        mean   : (..., action_dim)
        logstd : (..., action_dim)
        value  : (...)
        """
        # ── Shared trunk ──────────────────────────────────────────────────
        x = nn.relu(nn.Dense(self.hidden_dim,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(obs))
        x = nn.relu(nn.Dense(self.hidden_dim,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))

        # ── Actor head ────────────────────────────────────────────────────
        mean = nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)
        raw_logstd = self.param("log_std", constant(-1.0), (self.action_dim,))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(raw_logstd) + 1.0
        )

        # ── Critic head ───────────────────────────────────────────────────
        value = nn.Dense(1,
                         kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(x)
        value = jnp.squeeze(value, axis=-1)

        return mean, logstd, value
