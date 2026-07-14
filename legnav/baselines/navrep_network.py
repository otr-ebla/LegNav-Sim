"""
navrep_network.py — Faithful NavRep (V + M + C), paper-consistent.

Reference: Dugas, Nieto, Siegwart, Chung, "NavRep: Unsupervised Representations
for Reinforcement Learning of Robot Navigation in Dynamic Human Environments",
ICRA 2021.

Paper-faithful core
-------------------
  V (LiDAR encoding): a variational auto-encoder. 4-layer CNN encoder + 2-layer
     FCN → (z_mean, z_logvar); 2-layer FCN + 4-layer ConvTranspose decoder
     reconstructs the scan. Latent z has dimension 32.
  M (prediction): a sequence-to-sequence predictive model. We use the paper's
     Transformer variant — 8 causal-self-attention blocks, 8 heads each — over
     the z-sequence. M emits two predictions per step: the next latent ẑ_{t+1}
     and the next goal-velocity state ŝ_r,{t+1} (Eq. 2). Its hidden state h has
     dimension 64.
  C (controller): a 2-layer FCN consuming [z_t, h_t, s_r]. V and M are FROZEN;
     only C is trained with PPO (stop_gradient on z_t / h_t in the forward pass).

Training regime: JOINT V+M (the paper's recommended NavRep config). V and M are
trained together to minimise reconstruction + KL + next-state-prediction loss;
see pretrain_navrep.py. Then C is trained with PPO on the frozen features.

Environment adaptations (kept on purpose; see request)
------------------------------------------------------
  * LiDAR is the env's 216-ray 1D scan (not 1080); reconstruction uses MSE, as
    the paper does for its 1D variant.
  * Temporal context is the env's 3-frame stack (stack_dim=3): M attends over
    the 3 encoded frames rather than a long recurrent rollout.
  * s_r (controller state) is the env's full goal_stack(6)+kin_vec(5)=11-D
    vector. M's auxiliary s_r-prediction target is the most-recent 7-D
    [goal(2), kin_vec(5)] slice (velocity + goal in the robot frame).
  * Actions are the env's 2-D differential-drive [v, w].
  * Because the env observation is stateless and carries no past actions, M is
    conditioned on the latent sequence only (no action input). This is the one
    necessary deviation from the paper's action-conditioned M.

Obs layout (659D from make_stacked_env, stack_dim=3):
  obs[:6]    goal_stack   (3 × 2)   frame 0 = oldest, frame 2 = newest
  obs[6:11]  kin_vec      (5)
  obs[11:]   lidar_stack  (3 × 216) frame 0 = oldest, frame 2 = newest
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal, constant

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0

_GOAL_STACK_END = 6
_KIN_END        = 11
NUM_RAYS        = 216
STACK_DIM       = 3

Z_DIM       = 32     # paper: z latent features are of size 32
H_DIM       = 64     # paper: h latent features are of size 64
STATE_R_DIM = 11     # env adaptation: goal_stack(6) + kin_vec(5)
SR_DIM      = 7      # M's goal-velocity target: newest goal(2) + kin_vec(5)

# Dimensionality of the frozen V+M feature vector fed to the controller
FEAT_DIM = Z_DIM + H_DIM + STATE_R_DIM   # 32 + 64 + 11 = 107

# Keys in the flat params dict that belong to V (encoder) or M (Transformer).
# Everything else belongs to controller C.
_VM_KEYS = frozenset({"encoder", "M"})


# ══════════════════════════════════════════════════════════════════════════════
# Module V — VAE (4-layer CNN encoder + 2-layer FCN ; 2-layer FCN + 4-layer
#            ConvTranspose decoder)
# ══════════════════════════════════════════════════════════════════════════════

class LidarEncoder(nn.Module):
    z_dim: int = Z_DIM

    @nn.compact
    def __call__(self, lidar: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_shape = lidar.shape[:-1]
        x = lidar[..., None]                                      # (..., 216, 1)
        # 4-layer CNN encoder (SAME padding → out = ceil(in/stride))
        x = nn.relu(nn.Conv(64, (8,), strides=(2,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 108, 64)
        x = nn.relu(nn.Conv(128, (4,), strides=(2,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 54, 128)
        x = nn.relu(nn.Conv(128, (4,), strides=(2,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 27, 128)
        x = nn.relu(nn.Conv(256, (3,), strides=(1,),
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))          # (..., 27, 256)
        x = x.reshape(*batch_shape, -1)                           # (..., 6912)
        # 2-layer FCN
        x = nn.relu(nn.Dense(512,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(256,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        z_mean = nn.Dense(self.z_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0), name="z_mean")(x)
        z_logvar = nn.Dense(self.z_dim,
                            kernel_init=orthogonal(0.01),
                            bias_init=constant(0.0), name="z_logvar")(x)
        return z_mean, z_logvar


class LidarDecoder(nn.Module):
    num_rays: int = NUM_RAYS

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch_shape = z.shape[:-1]
        # 2-layer FCN
        x = nn.relu(nn.Dense(256,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(z))
        x = nn.relu(nn.Dense(27 * 128,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        x = x.reshape(*batch_shape, 27, 128)                      # (..., 27, 128)
        # 4-layer ConvTranspose decoder (SAME → out = in*stride)
        x = nn.relu(nn.ConvTranspose(128, (3,), strides=(1,), padding='SAME',
                                     kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.0))(x))  # (..., 27, 128)
        x = nn.relu(nn.ConvTranspose(128, (4,), strides=(2,), padding='SAME',
                                     kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.0))(x))  # (..., 54, 128)
        x = nn.relu(nn.ConvTranspose(64, (4,), strides=(2,), padding='SAME',
                                     kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.0))(x))  # (..., 108, 64)
        x = nn.ConvTranspose(1, (4,), strides=(2,), padding='SAME',
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)           # (..., 216, 1)
        x = nn.sigmoid(x)                  # inv_lidar is normalised to [0, 1]
        return x[..., 0]                                           # (..., 216)


# ══════════════════════════════════════════════════════════════════════════════
# Module M — Causal Transformer over z-sequences (paper Transformer variant)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=False,
        )(h, h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        return x + h


class TransformerM(nn.Module):
    """Causal Transformer M (paper: 8 blocks, 8 heads).

    (..., T, Z) → (next_z (..., T, Z), sr_pred (..., T, SR), h (..., T, d_model))

    Two prediction heads implement Eq. 2: the next latent ẑ_{t+1} and the next
    goal-velocity state ŝ_r,{t+1}. The hidden state h feeds the controller.
    """
    d_model:  int = H_DIM
    n_layers: int = 8
    n_heads:  int = 8
    max_len:  int = 64
    z_dim:    int = Z_DIM
    sr_dim:   int = SR_DIM

    @nn.compact
    def __call__(self, z_seq: jnp.ndarray):
        T = z_seq.shape[-2]

        h = nn.Dense(self.d_model,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0), name="in_proj")(z_seq)

        pos = self.param("pos_emb", constant(0.0), (self.max_len, self.d_model))
        h = h + pos[:T]

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]

        for i in range(self.n_layers):
            h = TransformerBlock(d_model=self.d_model, n_heads=self.n_heads,
                                 mlp_dim=4 * self.d_model,
                                 name=f"block_{i}")(h, causal)

        h = nn.LayerNorm(name="out_ln")(h)
        next_z = nn.Dense(self.z_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0),
                          name="next_z_head")(h)
        sr_pred = nn.Dense(self.sr_dim,
                           kernel_init=orthogonal(0.01),
                           bias_init=constant(0.0),
                           name="sr_head")(h)
        return next_z, sr_pred, h          # (..,T,Z), (..,T,SR), (..,T,d_model)


# ══════════════════════════════════════════════════════════════════════════════
# Joint world model (V + M) — used only for pretraining (pretrain_navrep.py)
# ══════════════════════════════════════════════════════════════════════════════

class NavRepWorldModel(nn.Module):
    """Joint V + M for JOINT pretraining.

    Param keys: {"encoder", "decoder", "M"} — train_navrep.py loads "encoder"
    and "M" into the actor-critic (the decoder is only needed at pretrain time).

    forward(lidar_seq, rng):
        lidar_seq : (B, T, NUM_RAYS)  per-env time-ordered LiDAR
      returns
        recon      : (B, T, NUM_RAYS)  reconstruction of the current frames
        next_recon : (B, T, NUM_RAYS)  decoded next-state prediction V(ẑ_{t+1})
        sr_pred    : (B, T, SR_DIM)    predicted next goal-velocity state
        z_mean, z_logvar : (B, T, Z)   for the KL term
    """
    z_dim:  int = Z_DIM
    sr_dim: int = SR_DIM

    @nn.compact
    def __call__(self, lidar_seq: jnp.ndarray, rng: jax.Array):
        encoder = LidarEncoder(z_dim=self.z_dim, name="encoder")
        decoder = LidarDecoder(name="decoder")
        M       = TransformerM(z_dim=self.z_dim, sr_dim=self.sr_dim, name="M")

        z_mean, z_logvar = encoder(lidar_seq)                 # (B, T, Z)
        eps = jax.random.normal(rng, z_mean.shape)
        z   = z_mean + jnp.exp(0.5 * z_logvar) * eps          # sampled latent

        recon = decoder(z)                                    # (B, T, R)

        next_z, sr_pred, _h = M(z)                            # (B,T,Z),(B,T,SR)
        next_recon = decoder(next_z)                          # (B, T, R)

        return recon, next_recon, sr_pred, z_mean, z_logvar


# ══════════════════════════════════════════════════════════════════════════════
# Full NavRep Actor-Critic (V + M + C, stop_gradient on V and M outputs)
# ══════════════════════════════════════════════════════════════════════════════

class NavRepActorCritic(nn.Module):
    action_dim: int = 2
    hidden_dim: int = 64
    z_dim:      int = Z_DIM
    freeze_vm:  bool = True   # canonical NavRep: V+M frozen, only C trained.

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        batch_shape = obs.shape[:-1]

        goal_flat  = obs[..., :_GOAL_STACK_END]
        kin_vec    = obs[..., _GOAL_STACK_END:_KIN_END]
        lidar_flat = obs[..., _KIN_END:]

        lidar_stack = lidar_flat.reshape(*batch_shape, STACK_DIM, NUM_RAYS)

        # V: encode each of 3 frames (deterministic: use z_mean) → (..., 3, Z)
        z_mean, _z_logvar = LidarEncoder(z_dim=self.z_dim, name="encoder")(lidar_stack)

        # M: causal Transformer over 3 frames → hidden h (..., 3, d_model)
        _next_z, _sr_pred, h_seq = TransformerM(z_dim=self.z_dim, name="M")(z_mean)

        # Most recent latent / hidden. Frozen (stop_gradient) when freeze_vm.
        _freeze = jax.lax.stop_gradient if self.freeze_vm else (lambda x: x)
        z_t = _freeze(z_mean[..., -1, :])                          # (..., Z)
        h_t = _freeze(h_seq[..., -1, :])                           # (..., H)

        state_r = jnp.concatenate([goal_flat, kin_vec], axis=-1)   # (..., 11)
        feat    = jnp.concatenate([z_t, h_t, state_r], axis=-1)    # (..., 107)

        # Controller (C): policy head
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(pi))
        mean = nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(pi)

        raw_logstd = self.param("log_std", constant(-1.0), (self.action_dim,))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(raw_logstd) + 1.0
        )
        logstd = jnp.broadcast_to(logstd, mean.shape)

        # Controller (C): value head
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(vf))
        value = nn.Dense(1,
                         kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(vf)
        value = jnp.squeeze(value, axis=-1)

        return mean, logstd, value


# ══════════════════════════════════════════════════════════════════════════════
# Controller-only module (C) — same Dense structure, same Flax param names
# ══════════════════════════════════════════════════════════════════════════════

class NavRepControllerOnly(nn.Module):
    """Controller C operating on pre-extracted V+M features (FEAT_DIM = 107).

    Parameter names match the Dense_0..Dense_5 / log_std entries that Flax
    generates inside NavRepActorCritic, so you can pass the controller subset
    of the full params dict directly:

        ctrl_params = {k: v for k, v in params.items() if k not in _VM_KEYS}
        mean, logstd, value = controller_only.apply({"params": ctrl_params}, feat)
    """
    action_dim: int = 2
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, feat: jnp.ndarray):
        # Policy head — Dense_0, Dense_1, Dense_2
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        pi = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(pi))
        mean = nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(pi)

        raw_logstd = self.param("log_std", constant(-1.0), (self.action_dim,))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(raw_logstd) + 1.0
        )
        logstd = jnp.broadcast_to(logstd, mean.shape)

        # Value head — Dense_3, Dense_4, Dense_5
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(feat))
        vf = nn.tanh(nn.Dense(self.hidden_dim,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(vf))
        value = nn.Dense(1,
                         kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(vf)
        return mean, logstd, jnp.squeeze(value, axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extractor — runs frozen V+M once, returns feat for controller C
# ══════════════════════════════════════════════════════════════════════════════

def navrep_extract_features(params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Run frozen V+M on `obs`, return concatenated [z_t, h_t, state_r].

    Args:
        params: full NavRepActorCritic params (must contain "encoder" and "M").
        obs:    (..., OBS_SIZE) observation array.

    Returns:
        feat: (..., FEAT_DIM=107) — stop_gradient already applied to z_t / h_t.
    """
    batch_shape = obs.shape[:-1]
    goal_flat   = obs[..., :_GOAL_STACK_END]           # (..., 6)
    kin_vec     = obs[..., _GOAL_STACK_END:_KIN_END]   # (..., 5)
    lidar_flat  = obs[..., _KIN_END:]                  # (..., 3*216)
    lidar_stack = lidar_flat.reshape(*batch_shape, STACK_DIM, NUM_RAYS)

    # V — encode each of the 3 stacked frames
    z_mean, _ = LidarEncoder(z_dim=Z_DIM).apply(
        {"params": params["encoder"]}, lidar_stack
    )                                                   # (..., 3, Z_DIM)

    # M — causal Transformer over z sequence
    _next_z, _sr_pred, h_seq = TransformerM(z_dim=Z_DIM).apply(
        {"params": params["M"]}, z_mean
    )                                                   # (..., 3, H_DIM)

    # Most-recent latent / hidden state, frozen
    z_t     = jax.lax.stop_gradient(z_mean[..., -1, :])   # (..., 32)
    h_t     = jax.lax.stop_gradient(h_seq[...,  -1, :])   # (..., 64)
    state_r = jnp.concatenate([goal_flat, kin_vec], axis=-1)  # (..., 11)

    return jnp.concatenate([z_t, h_t, state_r], axis=-1)  # (..., 107)
