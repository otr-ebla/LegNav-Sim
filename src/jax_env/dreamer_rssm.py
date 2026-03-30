"""
dreamer_rssm.py — Recurrent State Space Model for DreamerV3

Fixes vs submitted version:
  - BUG 1 FIXED: RSSM.init() now called through __call__ so all sub-layer
    weights are registered in one shot.
  - BUG 2 FIXED: CategoricalStraightThrough `sample` is a static Module field,
    not a runtime argument, eliminating Python branches inside JIT-traced code.

Architectural upgrades:
  - BLOCK GRU: nn.GRUCell(512) replaced with BlockDiagonalGRU(num_blocks=8,
    block_size=64). The monolithic GRU scales O(N^2) in recurrent parameters
    and FLOPs. The block-diagonal variant partitions hidden state into 8
    independent blocks of size 64 — same total capacity (512), but each block
    only attends to its own slice. Recurrent parameter count drops from
    512x512 = 262144 to 8x(64x64) = 32768 per gate — an 8x reduction.
    XLA fuses the vmapped block matmuls into a single batched kernel.
  - LAYER NORM: nn.LayerNorm injected after every Dense and before every swish
    activation in the GRU input projection and prior/posterior heads. Prevents
    internal covariate shift under symlog-distorted gradient magnitudes.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CATEGORIES     = 32
CATEGORY_SIZE      = 32
LATENT_SIZE        = NUM_CATEGORIES * CATEGORY_SIZE   # 1024
DETERMINISTIC_SIZE = 512

GRU_NUM_BLOCKS = 8
GRU_BLOCK_SIZE = DETERMINISTIC_SIZE // GRU_NUM_BLOCKS   # 64


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)

def symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)

def apply_unimix(logits: jnp.ndarray, mix_ratio: float = 0.01) -> jnp.ndarray:
    """
    Unimix: mixes the categorical distribution toward uniform by mix_ratio.
    Returns safe log-probabilities that prevent NaN gradients when a category
    collapses to near-zero probability during KL divergence computation.
    """
    probs        = jax.nn.softmax(logits, axis=-1)
    uniform_prob = 1.0 / logits.shape[-1]
    mixed_probs  = (1.0 - mix_ratio) * probs + mix_ratio * uniform_prob
    return jnp.log(mixed_probs)


# ---------------------------------------------------------------------------
# Block-Diagonal GRU
#
# Standard GRU with hidden size H: recurrent weight matrix per gate is (H, H),
# costing O(H^2) parameters and FLOPs.
#
# Block-diagonal GRU: partitions H into `num_blocks` slices of `block_size`.
# Each block's recurrent weight is (block_size, block_size). Total recurrent
# params per gate = num_blocks * block_size^2 = H * block_size = H^2/num_blocks.
# With num_blocks=8, block_size=64: 8 * 64^2 = 32768 vs 512^2 = 262144 — 8x
# fewer parameters and proportionally fewer FLOPs per recurrent step.
#
# The input projection remains full-width (dense over the full input), since
# the input path does not dominate the parameter budget.
#
# Implementation: nn.vmap a single-block Dense pair over the block axis.
# XLA sees `num_blocks` independent (block_size, block_size) matmuls and
# fuses them into a single batched GEMM — faster than sequential Dense calls.
# ---------------------------------------------------------------------------

class _SingleBlockRec(nn.Module):
    """One block's recurrent projection: [S] -> (r_h [S], u_h [S])."""
    block_size: int

    @nn.compact
    def __call__(self, h_b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r_h = nn.Dense(self.block_size, name='r_h')(h_b)
        u_h = nn.Dense(self.block_size, name='u_h')(h_b)
        return r_h, u_h


class BlockDiagonalGRU(nn.Module):
    """
    GRU cell with a block-diagonal recurrent weight matrix.

    Total hidden size = num_blocks * block_size = DETERMINISTIC_SIZE.
    Input projection is a standard full Dense (not block-diagonal).
    Recurrent projection is vmapped over blocks — each block only sees its
    own slice of the previous hidden state.
    """
    num_blocks: int = GRU_NUM_BLOCKS
    block_size:  int = GRU_BLOCK_SIZE

    @property
    def hidden_size(self) -> int:
        return self.num_blocks * self.block_size

    @nn.compact
    def __call__(
        self,
        h_prev: jnp.ndarray,   # [..., hidden_size]
        x:      jnp.ndarray,   # [..., input_dim]  (already LN-swish projected)
    ) -> jnp.ndarray:          # [..., hidden_size]

        B = self.num_blocks
        S = self.block_size
        H = self.hidden_size
        batch_shape = h_prev.shape[:-1]

        # Reshape hidden state into blocks for vmapped recurrent step
        h_blocks = h_prev.reshape(*batch_shape, B, S)   # [..., B, S]

        # Input projections: full dense over x, then split into blocks
        # Three gates: reset (r), update (u), new (n)
        r_x = nn.Dense(H, name='r_x')(x).reshape(*batch_shape, B, S)
        u_x = nn.Dense(H, name='u_x')(x).reshape(*batch_shape, B, S)
        n_x = nn.Dense(H, name='n_x')(x).reshape(*batch_shape, B, S)

        # Vmapped block recurrent projections (reset + update gates only;
        # new gate uses reset gate output, so no separate recurrent path)
        VmappedRec = nn.vmap(
            _SingleBlockRec,
            variable_axes={'params': 0},
            split_rngs={'params': False},
            in_axes=-2,
            out_axes=-2,
        )
        r_h, u_h = VmappedRec(block_size=S, name='block_rec')(h_blocks)
        # r_h, u_h: [..., B, S]

        r      = jax.nn.sigmoid(r_x + r_h)          # reset gate  [..., B, S]
        u      = jax.nn.sigmoid(u_x + u_h)          # update gate [..., B, S]
        n      = jnp.tanh(n_x + r * h_blocks)       # candidate   [..., B, S]

        h_next = (1.0 - u) * n + u * h_blocks       # [..., B, S]
        return h_next.reshape(*batch_shape, H)       # [..., H]


# ---------------------------------------------------------------------------
# Categorical straight-through sampler
# ---------------------------------------------------------------------------

class CategoricalStraightThrough(nn.Module):
    """
    `sample=True`  — Gumbel-Softmax; requires rngs={'gumbel': key} in .apply()
    `sample=False` — argmax; no RNG needed
    """
    sample: bool = True

    @nn.compact
    def __call__(self, logits: jnp.ndarray) -> jnp.ndarray:
        log_probs = apply_unimix(logits)
        probs     = jnp.exp(log_probs)

        if self.sample:
            # Gumbel-max trick requires log_probs, not raw logits.
            # Using raw logits here would bypass the unimix floor: a near-zero
            # category still has a large negative raw logit, so the 1% uniform
            # injection never influences which category gets selected.
            # log_probs already encodes the mixture; argmax(log_probs + Gumbel)
            # is a valid sample from the regularized distribution.
            key          = self.make_rng('gumbel')
            u            = jax.random.uniform(key, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
            gumbel_noise = -jnp.log(-jnp.log(u))
            indices      = jnp.argmax(log_probs + gumbel_noise, axis=-1)
        else:
            # Greedy: argmax over regularized log_probs.
            # log is monotone so argmax(log_probs) == argmax(mixed_probs),
            # selecting the mode of the unimix distribution rather than raw logits.
            indices = jnp.argmax(log_probs, axis=-1)

        hard_z = jax.nn.one_hot(indices, CATEGORY_SIZE, dtype=jnp.float32)
        z      = hard_z + probs - jax.lax.stop_gradient(probs)
        return z.reshape(z.shape[:-2] + (LATENT_SIZE,))


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    action_dim: int

    def setup(self):
        # Block-diagonal GRU — 8x fewer recurrent parameters than nn.GRUCell(512)
        self.cell      = BlockDiagonalGRU(num_blocks=GRU_NUM_BLOCKS, block_size=GRU_BLOCK_SIZE)
        self.step_dense = nn.Dense(DETERMINISTIC_SIZE)
        self.step_norm  = nn.LayerNorm()

        self.prior_dense1 = nn.Dense(DETERMINISTIC_SIZE)
        self.prior_norm1  = nn.LayerNorm()
        self.prior_dense2 = nn.Dense(LATENT_SIZE)

        self.post_dense1  = nn.Dense(DETERMINISTIC_SIZE)
        self.post_norm1   = nn.LayerNorm()
        self.post_dense2  = nn.Dense(LATENT_SIZE)

        self.sampler_train = CategoricalStraightThrough(sample=True)
        self.sampler_eval  = CategoricalStraightThrough(sample=False)

    def __call__(self, h: jnp.ndarray, z: jnp.ndarray,
                 action: jnp.ndarray, obs_embed: jnp.ndarray) -> jnp.ndarray:
        """Touches ALL sub-layers so a single init() call registers every param."""
        # GRU path
        x  = jnp.concatenate([z, action], axis=-1)
        x  = nn.swish(self.step_norm(self.step_dense(x)))
        h2 = self.cell(h, x)

        # Prior path
        xp = nn.swish(self.prior_norm1(self.prior_dense1(h2)))
        lp = self.prior_dense2(xp).reshape(xp.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))

        # Posterior path — touch post_dense2 for init
        xq = jnp.concatenate([h2, obs_embed], axis=-1)
        xq = nn.swish(self.post_norm1(self.post_dense1(xq)))
        self.post_dense2(xq)

        _ = self.sampler_eval(lp)
        return h2   # unused; __call__ exists purely to drive init()

    # ------------------------------------------------------------------
    # Runtime sub-methods (called via method= in .apply())
    # ------------------------------------------------------------------

    def step_gru(self, h_prev: jnp.ndarray, z_prev: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        """h_t = BlockGRU(h_{t-1}, LN_swish([z_{t-1}, a_{t-1}]))"""
        x = jnp.concatenate([z_prev, action], axis=-1)
        x = nn.swish(self.step_norm(self.step_dense(x)))
        return self.cell(h_prev, x)

    def prior(self, h_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prior WITHOUT observation. Needs rngs={'gumbel': key}."""
        x      = nn.swish(self.prior_norm1(self.prior_dense1(h_t)))
        logits = self.prior_dense2(x).reshape(x.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        return self.sampler_train(logits), logits

    def posterior(self, h_t: jnp.ndarray,
                  obs_embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior USING observation. Needs rngs={'gumbel': key}."""
        x      = jnp.concatenate([h_t, obs_embed], axis=-1)
        x      = nn.swish(self.post_norm1(self.post_dense1(x)))
        logits = self.post_dense2(x).reshape(x.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        return self.sampler_train(logits), logits

    def prior_greedy(self, h_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Greedy prior for evaluation. No RNG needed."""
        x      = nn.swish(self.prior_norm1(self.prior_dense1(h_t)))
        logits = self.prior_dense2(x).reshape(x.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        return self.sampler_eval(logits), logits

    def posterior_greedy(self, h_t: jnp.ndarray,
                         obs_embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Greedy posterior for evaluation. No RNG needed."""
        x      = jnp.concatenate([h_t, obs_embed], axis=-1)
        x      = nn.swish(self.post_norm1(self.post_dense1(x)))
        logits = self.post_dense2(x).reshape(x.shape[:-1] + (NUM_CATEGORIES, CATEGORY_SIZE))
        return self.sampler_eval(logits), logits


# ---------------------------------------------------------------------------
# Encoder
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

        cnn = nn.swish(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.swish(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.swish(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.swish(nn.Dense(128)(global_in))

        fused = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        return nn.swish(nn.LayerNorm()(nn.Dense(DETERMINISTIC_SIZE)(fused)))