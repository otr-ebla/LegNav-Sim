"""
dreamer_behavior.py — Actor-Critic and Latent Imagination Engine

No crash-level bugs in this file. Minor cleanup only:
  - unroll_imagination return value simplified: final_h/final_z were always
    discarded by the caller with `_, _`; the function now returns only `traj`
    so the API matches actual usage and avoids confusion.
  - All other logic (sample_action, compute_lambda_returns, scan) was correct.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable
from dreamer_rssm import DETERMINISTIC_SIZE


# ---------------------------------------------------------------------------
# Actor and Critic
# ---------------------------------------------------------------------------

class DreamerActor(nn.Module):
    action_dim: int
    min_std: float = 0.1

    @nn.compact
    def __call__(self, h_t: jnp.ndarray,
                 z_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        mean    = nn.Dense(self.action_dim)(x)
        std_raw = nn.Dense(self.action_dim)(x)
        std     = jax.nn.softplus(std_raw) + self.min_std
        return mean, std


class DreamerCritic(nn.Module):
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h_t, z_t], axis=-1)
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        x = nn.relu(nn.Dense(DETERMINISTIC_SIZE)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def sample_action(
    rng_key: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    noise      = jax.random.normal(rng_key, mean.shape)
    raw_action = mean + noise * std
    action     = jnp.tanh(raw_action)
    log_prob   = -0.5 * jnp.sum(
        jnp.square(noise) + jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std), axis=-1)
    log_prob  -= jnp.sum(jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1)
    return action, log_prob


# ---------------------------------------------------------------------------
# Lambda returns
# ---------------------------------------------------------------------------

def compute_lambda_returns(
    rewards: jnp.ndarray, values: jnp.ndarray, continues: jnp.ndarray,
    bootstrap: jnp.ndarray, gamma: float = 0.99, lambda_: float = 0.95,
) -> jnp.ndarray:
    def step(next_return, inputs):
        r, v, c = inputs
        ret = r + gamma * c * ((1.0 - lambda_) * v + lambda_ * next_return)
        return ret, ret

    _, returns = jax.lax.scan(
        step, bootstrap, (rewards, values, continues), reverse=True,
    )
    return returns


# ---------------------------------------------------------------------------
# Latent imagination
# ---------------------------------------------------------------------------

def unroll_imagination(
    rng_key:        jnp.ndarray,
    rssm_step_fn:   Callable,   # (rssm_params, h, z, a)          -> h_next
    rssm_prior_fn:  Callable,   # (rssm_params, h, gumbel_key)    -> (z_next, _)
    actor_apply_fn: Callable,   # (actor_params, h, z)             -> (mean, std)
    params:         dict,       # {'rssm': ..., 'actor': ...}
    start_h:        jnp.ndarray,
    start_z:        jnp.ndarray,
    horizon:        int = 15,
) -> dict:
    """
    Dreams a trajectory of length `horizon` inside the latent space.
    Returns only the trajectory dict (final h/z are discarded by all callers).
    All computation runs inside jax.lax.scan — zero Python loops, full GPU.
    """
    def imagine_step(carry, _):
        h_prev, z_prev, current_key = carry
        current_key, action_key, prior_key = jax.random.split(current_key, 3)

        mean, std  = actor_apply_fn(params['actor'], h_prev, z_prev)
        action, log_prob = sample_action(action_key, mean, std)

        h_next     = rssm_step_fn(params['rssm'], h_prev, z_prev, action)
        z_next, _  = rssm_prior_fn(params['rssm'], h_next, prior_key)

        transition = {
            'h': h_prev, 'z': z_prev,
            'action': action, 'log_prob': log_prob,
        }
        return (h_next, z_next, current_key), transition

    _, traj = jax.lax.scan(
        imagine_step, (start_h, start_z, rng_key), None, length=horizon,
    )
    return traj