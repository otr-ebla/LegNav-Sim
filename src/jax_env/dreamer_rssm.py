"""
dreamer_rssm.py — Recurrent State Space Model for DreamerV3

Fixes vs submitted version:
  - BUG 1 FIXED: RSSM.init() was called with method=rssm.step_gru, which only
    initialised the weight paths touched by step_gru (missing prior/post dense
    layers entirely). Fixed by giving RSSM a proper __call__ that touches ALL
    sub-layers, so a single init() call registers every parameter.
  - BUG 2 FIXED: CategoricalStraightThrough had a Python `if sample:` branch
    inside @nn.compact. Since `sample` is never static in the JIT call graph,
    the False path was never traced. Fixed by making `sample` a static Module
    attribute so the branch is resolved at module-definition time (two module
    variants: one for sampling, one for greedy), eliminating the runtime branch.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

NUM_CATEGORIES     = 32
CATEGORY_SIZE      = 32
LATENT_SIZE        = NUM_CATEGORIES * CATEGORY_SIZE   # 1024
DETERMINISTIC_SIZE = 512


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

def symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Categorical straight-through sampler
# BUG 2 FIX: `sample` is now a static Module field, not a runtime argument,
# so there is never a Python branch inside a JIT-traced function.
# Use CategoricalStraightThrough(sample=True)  for training (Gumbel noise).
# Use CategoricalStraightThrough(sample=False) for greedy evaluation.
# ---------------------------------------------------------------------------

class CategoricalStraightThrough(nn.Module):
    """
    Straight-through categorical sampler.
    `sample=True`  → Gumbel-Softmax, requires rngs={'gumbel': key} in .apply()
    `sample=False` → argmax, no RNG needed
    """
    sample: bool = True   # static field — resolved at module instantiation

    @nn.compact
    def __call__(self, logits: jnp.ndarray) -> jnp.ndarray:
        probs = jax.nn.softmax(logits, axis=-1)

        if self.sample:
            # self.sample is a Python bool resolved at trace time — safe.
            key = self.make_rng('gumbel')
            u   = jax.random.uniform(key, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
            gumbel_noise = -jnp.log(-jnp.log(u))
            indices = jnp.argmax(logits + gumbel_noise, axis=-1)
        else:
            indices = jnp.argmax(logits, axis=-1)

        hard_z = jax.nn.one_hot(indices, CATEGORY_SIZE, dtype=jnp.float32)
        # Straight-through: forward=hard_z, backward through soft probs
        z = hard_z + probs - jax.lax.stop_gradient(probs)
        return z.reshape(z.shape[:-2] + (LATENT_SIZE,))


# ---------------------------------------------------------------------------
# RSSM
# BUG 1 FIX: __call__ now touches ALL sub-layers so rssm.init() registers
# every parameter in one shot. sub-methods (step_gru, prior, posterior) are
# still called individually via method= in .apply() — they just share the
# same param tree that __call__ has fully initialised.
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    action_dim: int

    def setup(self):
        self.cell       = nn.GRUCell(features=DETERMINISTIC_SIZE)
        self.step_dense = nn.Dense(DETERMINISTIC_SIZE)

        self.prior_dense1 = nn.Dense(DETERMINISTIC_SIZE)
        self.prior_dense2 = nn.Dense(LATENT_SIZE)

        self.post_dense1  = nn.Dense(DETERMINISTIC_SIZE)
        self.post_dense2  = nn.Dense(LATENT_SIZE)

        # Two sampler instances: one for training (Gumbel), one for eval (greedy)
        self.sampler_train = CategoricalStraightThrough(sample=True)
        self.sampler_eval  = CategoricalStraightThrough(sample=False)

    def __call__(self, h: jnp.ndarray, z: jnp.ndarray,
                 action: jnp.ndarray, obs_embed: jnp.ndarray) -> jnp.ndarray:
        """
        Touches ALL sub-layers so that init() registers every parameter.
        Not used directly at train/inference time — use the sub-methods below.
        """
        # GRU path
        x  = jnp.concatenate([z, action], axis=-1)
        x  = nn.relu(self.step_dense(x))
        h2, _ = self.cell(h, x)

        # Prior path
        xp = nn.relu(self.prior_dense1(h2))
        lp = self.prior_dense2(xp).reshape(xp.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))

        # Posterior path
        xq = jnp.concatenate([h2, obs_embed], axis=-1)
        xq = nn.relu(self.post_dense1(xq))
        lq = self.post_dense2(xq).reshape(xq.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))

        # Touch both samplers (eval sampler needs no RNG — passes silently)
        _ = self.sampler_eval(lp)
        return h2   # unused output; __call__ exists purely for init()

    # ------------------------------------------------------------------
    # Sub-methods used at runtime via rssm.apply(..., method=rssm.X)
    # ------------------------------------------------------------------

    def step_gru(self, h_prev: jnp.ndarray, z_prev: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        """h_t = GRU(h_{t-1}, MLP([z_{t-1}, a_{t-1}]))"""
        x = jnp.concatenate([z_prev, action], axis=-1)
        x = nn.relu(self.step_dense(x))
        new_h, _ = self.cell(h_prev, x)
        return new_h

    def prior(self, h_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict stochastic state WITHOUT observation. Needs rngs={'gumbel': key}."""
        x      = nn.relu(self.prior_dense1(h_t))
        logits = self.prior_dense2(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        z_t    = self.sampler_train(logits)   # uses gumbel RNG
        return z_t, logits

    def posterior(self, h_t: jnp.ndarray,
                  obs_embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict stochastic state USING observation. Needs rngs={'gumbel': key}."""
        x      = jnp.concatenate([h_t, obs_embed], axis=-1)
        x      = nn.relu(self.post_dense1(x))
        logits = self.post_dense2(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        z_t    = self.sampler_train(logits)   # uses gumbel RNG
        return z_t, logits

    def prior_greedy(self, h_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Greedy (no-noise) prior for evaluation. No RNG needed."""
        x      = nn.relu(self.prior_dense1(h_t))
        logits = self.prior_dense2(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        z_t    = self.sampler_eval(logits)
        return z_t, logits

    def posterior_greedy(self, h_t: jnp.ndarray,
                         obs_embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Greedy (no-noise) posterior for evaluation. No RNG needed."""
        x      = jnp.concatenate([h_t, obs_embed], axis=-1)
        x      = nn.relu(self.post_dense1(x))
        logits = self.post_dense2(x)
        logits = logits.reshape(logits.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        z_t    = self.sampler_eval(logits)
        return z_t, logits


# ---------------------------------------------------------------------------
# Encoder — unchanged, was correct
# ---------------------------------------------------------------------------

class DreamerEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        pose_size  = 3 * self.stack_dim
        state_size = 9

        pose_stack = obs[..., :pose_size]
        state_vec  = obs[..., pose_size : pose_size + state_size]
        lidar_flat = obs[..., pose_size + state_size:]

        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))

        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))

        fused = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        return nn.relu(nn.Dense(DETERMINISTIC_SIZE)(fused))