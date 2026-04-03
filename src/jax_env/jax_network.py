"""
jax_network.py — Actor-Critic Neural Network for PPO (CNN + Frame-Stack Attention)

"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from typing import Tuple
import numpy as np

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.0

_STATE_VEC_SIZE = 5
ATTN_HEADS      = 4    # numero di teste attention sul frame stack
ATTN_HEAD_DIM   = 16   # dim per testa → QKV dim = ATTN_HEADS * ATTN_HEAD_DIM = 64


class LidarFrameCNN(nn.Module):
    """
    Shared CNN applied on a single LiDAR frame (num_rays,).
    It is instantiated ONCE and called 3 times → real weight sharing.

    Input:  (..., num_rays)
    Output: (..., FRAME_FEAT=64)
    """
    frame_feat: int = 64

    @nn.compact
    def __call__(self, frame: jnp.ndarray) -> jnp.ndarray:
        batch_shape = frame.shape[:-1]
        z = frame[..., None]                                               # (..., num_rays, 1)
        z = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(z))
        z = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(z))
        z = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(z))
        # z: (..., spatial=14, 64)
        z_flat = z.reshape((*batch_shape, -1))                             # (..., 896)
        return nn.relu(
            nn.Dense(self.frame_feat,
                     kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z_flat)
        )                                                                   # (..., 64)


class FrameStackAttention(nn.Module):
    """
    1-layer Multi-Head Self-Attention sul frame stack LiDAR.

    Input:  (batch, stack_dim, feat_dim)   — sequenza di frame CNN
    Output: (batch, stack_dim * feat_dim)  — rappresentazione contestuale flat

    Ogni frame può "attendere" agli altri frame della sequenza, permettendo
    alla rete di confrontare frame consecutivi (es. detectare moto umano).
    """
    num_heads: int = ATTN_HEADS
    head_dim:  int = ATTN_HEAD_DIM

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., S, D) dove S = stack_dim, D = feat_dim per frame
        S, D = x.shape[-2], x.shape[-1]
        qkv_dim = self.num_heads * self.head_dim

        Q = nn.Dense(qkv_dim, use_bias=False)(x)   # (..., S, qkv_dim)
        K = nn.Dense(qkv_dim, use_bias=False)(x)
        V = nn.Dense(qkv_dim, use_bias=False)(x)

        # Reshape per multi-head: (..., S, H, head_dim)
        batch_shape = x.shape[:-2]
        def split_heads(t):
            return t.reshape((*batch_shape, S, self.num_heads, self.head_dim))

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # (..., H, S, head_dim)
        Q = jnp.moveaxis(Q, -2, -3)
        K = jnp.moveaxis(K, -2, -3)
        V = jnp.moveaxis(V, -2, -3)

        scale  = 1.0 / jnp.sqrt(self.head_dim).astype(jnp.float32)
        scores = jnp.einsum('...hqd,...hkd->...hqk', Q, K) * scale   # (..., H, S, S)
        weights = jax.nn.softmax(scores, axis=-1)
        out    = jnp.einsum('...hqk,...hkd->...hqd', weights, V)      # (..., H, S, head_dim)

        # Riassembla: (..., S, qkv_dim)
        out = jnp.moveaxis(out, -3, -2).reshape((*batch_shape, S, qkv_dim))

        # Projection finale + residual + LayerNorm
        proj = nn.Dense(D, use_bias=False)(out)
        return nn.LayerNorm()(x + proj)   # (..., S, D) — residual connection


class SharedEncoder(nn.Module):
    """
    Shared observation encoder — identical trunk to EndToEndActorCritic.
    Enables SAC and TQC to use the exact same feature extractor as PPO
    for fair algorithm comparison.

    Input:  (..., OBS_SIZE)
    Output: (..., 128) feature vector
    """
    stack_dim: int = 3
    num_rays:   int = 216

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pose_size  = 3 * self.stack_dim
        state_size = _STATE_VEC_SIZE

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        batch_shape  = lidar_flat.shape[:-1]
        lidar_frames = lidar_flat.reshape((*batch_shape, self.stack_dim, self.num_rays))

        FRAME_FEAT  = 64
        cnn_encoder = LidarFrameCNN(frame_feat=FRAME_FEAT)
        frame_feats = [cnn_encoder(lidar_frames[..., i, :]) for i in range(self.stack_dim)]
        frame_seq   = jnp.stack(frame_feats, axis=-2)
        frame_seq   = nn.LayerNorm()(frame_seq)

        attn_out  = FrameStackAttention()(frame_seq)
        attn_flat = attn_out.reshape((*batch_shape, self.stack_dim * FRAME_FEAT))  # (..., 192)

        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1)
        fused     = jnp.concatenate([attn_flat, global_in], axis=-1)

        shared = nn.relu(
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(fused)
        )
        return nn.relu(
            nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(shared)
        )  # (..., 128)


class EndToEndActorCritic(nn.Module):
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 216

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,    # (..., OBS_SIZE)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        STATELESS: prende solo l'osservazione, restituisce (mean, logstd, value).
        Nessun hidden state — compatibile con forward pass piatto su (T*N, D).
        """
        pose_size  = 3 * self.stack_dim    # 9
        state_size = _STATE_VEC_SIZE       # 5

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        # ── CNN shared per-frame → sequenza di token temporali ───────────────
        # LidarFrameCNN è istanziata UNA VOLTA: tutte e 3 le call condividono
        # gli stessi parametri → weight sharing reale (non simulato con nomi).
        batch_shape = lidar_flat.shape[:-1]

        # (..., num_rays * stack_dim) → (..., stack_dim, num_rays)
        lidar_frames = lidar_flat.reshape((*batch_shape, self.stack_dim, self.num_rays))

        FRAME_FEAT = 64
        cnn_encoder = LidarFrameCNN(frame_feat=FRAME_FEAT)  # istanza unica → pesi condivisi

        # Applica la stessa CNN ai 3 frame → 3 token da 64-dim
        frame_feats = [cnn_encoder(lidar_frames[..., i, :]) for i in range(self.stack_dim)]

        # (..., stack_dim=3, FRAME_FEAT=64)
        frame_seq = jnp.stack(frame_feats, axis=-2)
        frame_seq = nn.LayerNorm()(frame_seq)

        # ── Spatio-Temporal Self-Attention ────────────────────────────────────
        # Timeline of 3 64-dim tokens: each token is the compressed
        # representation of a single LiDAR frame. Attention learns inter-frame differences
        # (LiDAR optical flow, apparent obstacle velocity).
        attn_out  = FrameStackAttention()(frame_seq)                        # (..., 3, 64)
        attn_flat = attn_out.reshape((*batch_shape, self.stack_dim * FRAME_FEAT))  # (..., 192)

        # ── Global state MLP ──────────────────────────────────────────────────
        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1)  # (..., 14)

        # ── Fused trunk ───────────────────────────────────────────────────────
        fused = jnp.concatenate([attn_flat, global_in], axis=-1)       # (..., 206)
        shared = nn.relu(
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(fused)
        )
        shared = nn.relu(
            nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(shared)
        )

        # ── Actor head ────────────────────────────────────────────────────────
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(shared)

        logstd_param     = self.param('log_std', constant(-1.0), (self.action_dim,)) # state independent learnable log std
        actor_logstd_raw = jnp.broadcast_to(logstd_param, actor_mean.shape)
        actor_logstd     = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            jnp.tanh(actor_logstd_raw) + 1.0
        )

        # ── Critic head ───────────────────────────────────────────────────────
        critic = nn.relu(
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(shared)
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)


# ── Action squashing helpers (invariati) ─────────────────────────────────────

def _squash_log_jacobian(raw_actions: jnp.ndarray, max_v: float = 1.0) -> jnp.ndarray:
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
    return base_log_prob  # Removed _squash_log_jacobian for PPO stability


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