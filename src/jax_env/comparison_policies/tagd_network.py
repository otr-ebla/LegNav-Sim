"""
tagd_network.py — TAGD Policy (JAX/Flax re-implementation)
==========================================================
Faithful re-implementation of:
  de Heuvel et al., "TAGD: Temporal Accumulation of Group Descriptors
  for Robot Navigation among Pedestrians", IEEE RA-L 2024.

Architecture overview
---------------------
1. **TAGD computation** (no ICP — paper reports 4.8% degradation without it):
   - Divide 210 lidar rays into N_c=30 sectors × 7 rays each.
   - For each sector, find the closest lidar return, group points within
     d_thresh=0.25 m, compute the centroid.
   - Do this for both the previous (frame t-1) and current (frame t) scan.
   - TAGD_i = [prev_cx, prev_cy, curr_cx, curr_cy]  (4-dim per sector).

2. **Virtual waypoints**: 5 points at 0.3 m intervals along goal direction
   in robot ego frame (replaces A* path planning from the paper).

3. **Two-stream spatial-temporal attention network**:
   - *Spatial stream*: input = (N_c, 7 × 2 + 10) = (30, 24).
     Current lidar points (Cartesian) + virtual waypoints per sector.
   - *Temporal stream*: input = (N_c, 4 + 10) = (30, 14).
     TAGD descriptor + virtual waypoints per sector.
   - Each stream: embedding MLP (→ 64) + soft attention (score 1 + feature 30)
     → weighted sum → 30-dim output.
   - Fusion: concat(30, 30) = 60-dim.

4. **Actor / Critic heads** (DDPG — deterministic):
   - Shared backbone: 60 → 128 → 64 → 64.
   - Actor:  64 → 2   (tanh-squashed v, w).
   - Critic: takes (obs, action), fused 60+2 → 128 → 64 → 64 → 1.

Observation layout (662-dim stacked obs from make_stacked_env, stack_dim=3):
  obs[0:9]     = pose_stack (3 × 3),  oldest → newest; newest = obs[6:9]
  obs[9:14]    = state_vec  (v/vmax, w, (vmax-0.2)/1.8, dist/D, align/π)
  obs[14:230]  = lidar frame 0 (oldest,  216 rays, inv-normalised)
  obs[230:446] = lidar frame 1 (middle,  216 rays)
  obs[446:662] = lidar frame 2 (newest,  216 rays)

Inverse-normalised lidar:  v = (MAX_LIDAR_DIST - d) / (MAX_LIDAR_DIST - R)
  → raw distance:           d = MAX_LIDAR_DIST - v * (MAX_LIDAR_DIST - R)

LiDAR angles (ego frame, same as DWA planner):
  angles = linspace(-π, π, 216)   →  ray 0 behind, ray 108 ahead.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

# ── Constants (must match jax_env) ─────────────────────────────────────────────
NUM_RAYS       = 216
MAX_LIDAR_DIST = 12.0
ROBOT_RADIUS   = 0.3   # overridden below from jax_env when available
POSE_SIZE      = 3
STATE_VEC_SIZE = 5
STACK_DIM      = 3

try:
    from jax_env import NUM_RAYS as _NR, MAX_LIDAR_DIST as _MLD, ROBOT_RADIUS as _RR
    NUM_RAYS       = _NR
    MAX_LIDAR_DIST = _MLD
    ROBOT_RADIUS   = _RR
except ImportError:
    pass

# ── TAGD hyper-parameters (from paper) ─────────────────────────────────────────
N_SECTORS    = 30
RAYS_PER_SEC = 7          # 30 × 7 = 210 used; remaining 6 rays discarded
D_THRESH     = 0.25       # m — grouping radius for centroid computation
N_WAYPOINTS  = 5
WP_STEP      = 0.3        # m — inter-waypoint spacing along goal direction
COORD_NORM   = MAX_LIDAR_DIST  # normalise descriptor coords to ≈ [-1, 1]

# obs index where max_v is encoded:  state_vec starts at 9, max_v_norm is [2]
MAX_V_OBS_IDX = 11        # obs[11] = (max_v - 0.2) / 1.8

# Lidar ray angles in ego frame (shared constant)
_ANGLES_ALL  = jnp.linspace(-jnp.pi, jnp.pi, NUM_RAYS)
_ANGLES_210  = _ANGLES_ALL[:N_SECTORS * RAYS_PER_SEC]   # first 210 rays


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — TAGD descriptor computation
# ══════════════════════════════════════════════════════════════════════════════

def _inv_norm_to_dist(v: jnp.ndarray) -> jnp.ndarray:
    """Inverse-normalised lidar value(s) → raw metric distance (m)."""
    return MAX_LIDAR_DIST - v * (MAX_LIDAR_DIST - ROBOT_RADIUS)


def _lidar_to_cartesian_210(lidar_vals: jnp.ndarray) -> jnp.ndarray:
    """
    Convert first 210 rays of an inverse-normalised lidar scan to Cartesian.
    Args:
        lidar_vals: (..., NUM_RAYS)   inv-normalised
    Returns:
        pts: (..., 210, 2)   in robot ego frame (m)
    """
    dists  = _inv_norm_to_dist(lidar_vals[..., :N_SECTORS * RAYS_PER_SEC])
    angles = _ANGLES_210
    xs = dists * jnp.cos(angles)
    ys = dists * jnp.sin(angles)
    return jnp.stack([xs, ys], axis=-1)


def _sector_centroid(sector_pts: jnp.ndarray) -> jnp.ndarray:
    """
    TAGD grouping: find closest return to robot, then average all returns
    within D_THRESH of it.

    Args:
        sector_pts: (RAYS_PER_SEC, 2)
    Returns:
        centroid: (2,)
    """
    dists_sq = jnp.sum(sector_pts ** 2, axis=-1)   # (7,)
    closest_idx = jnp.argmin(dists_sq)
    closest_pt  = sector_pts[closest_idx]           # (2,)

    diff     = sector_pts - closest_pt[None, :]     # (7, 2)
    d_to_c   = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-9)  # (7,)
    in_grp   = (d_to_c < D_THRESH).astype(jnp.float32)       # (7,)
    n_in     = jnp.sum(in_grp) + 1e-9
    centroid = jnp.sum(sector_pts * in_grp[:, None], axis=0) / n_in

    # If no point passes the threshold (pathological), return closest
    return jnp.where(n_in > 1.0, centroid, closest_pt)


def compute_tagd(
    lidar_prev: jnp.ndarray,
    lidar_curr: jnp.ndarray,
) -> tuple:
    """
    Compute TAGD descriptors from two consecutive lidar scans (no ICP).

    Args:
        lidar_prev: (NUM_RAYS,)  inv-normalised previous frame
        lidar_curr: (NUM_RAYS,)  inv-normalised current frame

    Returns:
        tagd:     (N_SECTORS, 4)              [px, py, cx, cy] per sector
        curr_sec: (N_SECTORS, RAYS_PER_SEC, 2) current Cartesian sector pts
        prev_sec: (N_SECTORS, RAYS_PER_SEC, 2) previous Cartesian sector pts
    """
    n = N_SECTORS * RAYS_PER_SEC

    prev_cart = _lidar_to_cartesian_210(lidar_prev)             # (210, 2)
    curr_cart = _lidar_to_cartesian_210(lidar_curr)             # (210, 2)

    prev_sec = prev_cart.reshape(N_SECTORS, RAYS_PER_SEC, 2)   # (30, 7, 2)
    curr_sec = curr_cart.reshape(N_SECTORS, RAYS_PER_SEC, 2)   # (30, 7, 2)

    # vmap centroid computation over sectors
    prev_cen = jax.vmap(_sector_centroid)(prev_sec)             # (30, 2)
    curr_cen = jax.vmap(_sector_centroid)(curr_sec)             # (30, 2)

    tagd = jnp.concatenate([prev_cen, curr_cen], axis=-1)      # (30, 4)
    return tagd, curr_sec, prev_sec


def _virtual_waypoints(goal_vec_raw: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 5 virtual waypoints along goal direction in ego frame.

    Args:
        goal_vec_raw: (2,)  (gdx_ego/D, gdy_ego/D) from obs[6:8]
    Returns:
        wps_flat: (N_WAYPOINTS * 2,) = (10,)  normalised by COORD_NORM
    """
    norm     = jnp.sqrt(jnp.sum(goal_vec_raw ** 2) + 1e-9)
    goal_dir = goal_vec_raw / norm                           # unit direction (2,)
    steps    = jnp.arange(1, N_WAYPOINTS + 1, dtype=jnp.float32) * WP_STEP
    wps      = goal_dir[None, :] * steps[:, None]           # (5, 2)
    return (wps / COORD_NORM).reshape(-1)                   # (10,)


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Attention stream
# ══════════════════════════════════════════════════════════════════════════════

class AttentionStream(nn.Module):
    """
    Soft location-based attention over N_SECTORS sectors.

    For each sector (in parallel via weight-shared Dense layers):
      1. Embedding MLP: input_dim → 256 → 128 → 64
      2. Score MLP:     64 → 50 → 1  (logit)
      3. Feature MLP:   64 → 50 → 30

    Softmax over sector scores → weighted sum of features → 30-dim output.
    Dense layers naturally batch over the leading (sector) dimension, giving
    full weight sharing across sectors.
    """

    @nn.compact
    def __call__(self, sector_inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            sector_inputs: (N_SECTORS, input_dim)
        Returns:
            attended: (30,)
        """
        # Embedding (shared weights across sectors via Dense broadcasting)
        x = sector_inputs
        x = nn.relu(nn.Dense(256)(x))   # (N, 256)
        x = nn.relu(nn.Dense(128)(x))   # (N, 128)
        embed = nn.relu(nn.Dense(64)(x))  # (N, 64)

        # Score: (N, 1) → softmax → (N,)
        scores  = nn.relu(nn.Dense(50)(embed))   # (N, 50)
        logits  = nn.Dense(1)(scores)            # (N, 1)
        weights = jax.nn.softmax(logits[:, 0], axis=0)  # (N,)

        # Feature: (N, 30)
        feats = nn.relu(nn.Dense(50)(embed))  # (N, 50)
        feats = nn.Dense(30)(feats)           # (N, 30)

        # Weighted sum
        return jnp.sum(weights[:, None] * feats, axis=0)   # (30,)


class TAGDEncoder(nn.Module):
    """
    Shared TAGD feature extractor used by both actor and critic.

    Input:  obs (662,)
    Output: fused (60,)  — concat of spatial + temporal attention outputs
    """

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # ── Decode observation ──────────────────────────────────────────────
        # Most-recent pose: obs[6:9] = (gdx_ego/D, gdy_ego/D, θ/π)
        goal_vec   = obs[6:8]                              # (2,)
        # Lidar frames (inv-normalised)
        lidar_prev = obs[14 + NUM_RAYS     : 14 + 2 * NUM_RAYS]  # frame 1
        lidar_curr = obs[14 + 2 * NUM_RAYS : 14 + 3 * NUM_RAYS]  # frame 2 (newest)

        # ── TAGD ───────────────────────────────────────────────────────────
        tagd, curr_sec, _ = compute_tagd(lidar_prev, lidar_curr)
        # tagd:     (30, 4)
        # curr_sec: (30, 7, 2)

        # ── Virtual waypoints (10,) ────────────────────────────────────────
        wp = _virtual_waypoints(goal_vec)                  # (10,)
        wp_tiled = jnp.tile(wp[None, :], (N_SECTORS, 1))  # (30, 10)

        # ── Spatial stream input: (30, 7*2 + 10) = (30, 24) ───────────────
        curr_flat  = curr_sec.reshape(N_SECTORS, RAYS_PER_SEC * 2)  # (30, 14)
        spatial_in = jnp.concatenate([curr_flat, wp_tiled], axis=-1)  # (30, 24)

        # ── Temporal stream input: (30, 4 + 10) = (30, 14) ────────────────
        tagd_norm   = tagd / COORD_NORM                              # (30, 4)
        temporal_in = jnp.concatenate([tagd_norm, wp_tiled], axis=-1)  # (30, 14)

        # ── Attention streams ──────────────────────────────────────────────
        sp_out = AttentionStream(name="spatial")(spatial_in)   # (30,)
        tm_out = AttentionStream(name="temporal")(temporal_in) # (30,)

        return jnp.concatenate([sp_out, tm_out], axis=-1)     # (60,)


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Actor and Critic networks
# ══════════════════════════════════════════════════════════════════════════════

class TAGDActor(nn.Module):
    """
    DDPG deterministic actor.

    Input:  obs (662,)
    Output: action (2,)  — [v ∈ [0, max_v], w ∈ [-1, 1]]
    """
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        fused = TAGDEncoder(name="encoder")(obs)   # (60,)

        h = nn.relu(nn.Dense(128)(fused))
        h = nn.relu(nn.Dense(64)(h))
        h = nn.relu(nn.Dense(64)(h))
        raw = nn.Dense(self.action_dim)(h)         # (2,) unbounded

        max_v = jnp.clip(obs[MAX_V_OBS_IDX] * 1.8 + 0.2, 0.2, 2.0)
        v = (jnp.tanh(raw[0]) + 1.0) * 0.5 * max_v  # [0, max_v]
        w = jnp.tanh(raw[1])                          # [-1, 1]
        return jnp.stack([v, w])


class TAGDCritic(nn.Module):
    """
    DDPG critic: Q(s, a).

    Input:  obs (662,), action (2,)
    Output: q_value (scalar)
    """

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        fused = TAGDEncoder(name="encoder")(obs)            # (60,)
        x     = jnp.concatenate([fused, action], axis=-1)  # (62,)

        h = nn.relu(nn.Dense(128)(x))
        h = nn.relu(nn.Dense(64)(h))
        h = nn.relu(nn.Dense(64)(h))
        return nn.Dense(1)(h)[..., 0]   # scalar


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Convenience wrappers for evaluation
# ══════════════════════════════════════════════════════════════════════════════

def make_tagd_act_fn(actor_params: dict, v_max: float = 1.0):
    """
    Return a JIT+vmapped action function from loaded actor parameters.

    Usage:
        act_vmap = make_tagd_act_fn(params, v_max=1.0)
        actions  = act_vmap(obs_batch)   # (N, 662) → (N, 2)
    """
    actor = TAGDActor()

    @jax.jit
    def act_vmap(obs_batch: jnp.ndarray) -> jnp.ndarray:
        def _single(obs):
            return actor.apply({"params": actor_params}, obs)
        return jax.vmap(_single)(obs_batch)

    return act_vmap
