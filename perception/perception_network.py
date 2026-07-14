"""
perception/perception_network.py — LegDetNet: leg-specialized people detector
===============================================================================
A network designed *specifically* for finding human legs in 2D LiDAR rings —
deliberately different from the navigation encoder (jax_network.py / CALF,
which uses a shared per-frame strided CNN + frame-stack self-attention and
compresses each scan to a single feature vector). LegDetNet instead keeps
full per-ray resolution end-to-end and bakes in three leg-specific priors:

1. Temporal motion channels — the 8-frame stack is collapsed into explicit
   per-ray statistics (current range, mean, std, min, max, last-frame diff,
   stack-wide flow) alongside the raw frames. Walking legs are the only
   scene element that oscillates at gait frequency, so the std / diff
   channels light up exactly on leg rays; static obstacles c

2. Angular-width-matched multi-scale convolutions — with 216 rays over
   360°, one ray ≈ 1.67°. A leg (Ø 0.16 m) subtends ~5.5 rays at 1 m and
   ~1.4 rays at 4 m; a leg *pair* (hip width 0.32 m) up to ~11 rays.
   Each block runs parallel k=3 circular convolutions at dilations
   1/2/4/8, covering 3–17-ray receptive fields in a single block — every
   scale a leg or leg pair can appear at — and fuses them with a 1×1 conv.
   Circular padding closes the 360° ring (a person can straddle ray 215→0).

3. Anchor-free leg→body voting — each ray classifies "person leg hit" and
   regresses TWO offsets from its (geometric) endpoint: one to the centre
   of the individual leg it hit, one to the HSFM body centre
   (people[:, 0:2] of the simulator state) of the person owning that leg.

4. Leg-pair verification — decode_detections() first clusters the leg
   votes into leg candidates (each needing MIN_RAYS_PER_LEG supporting
   rays), then emits a person ONLY where two leg candidates pair up
   within anatomical distance (PAIR_MAX_DIST). Isolated confident blobs
   on walls or furniture cannot form a pair and are rejected — a couple
   of legs is a person, anything else is not.

  input  (B, STACK_DIM=8, NUM_RAYS=216)  inverted-normalized scans
  output (B, NUM_RAYS) person logits
       + (B, NUM_RAYS, 2) body-centre offsets
       + (B, NUM_RAYS, 2) leg-centre offsets
"""

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import orthogonal, constant

from legnav.core.jax_env import ROBOT_RADIUS
from perception.config import PerceptionConfig as PC


# ── Temporal motion channels ──────────────────────────────────────────────────

def _temporal_channels(stack: jnp.ndarray) -> jnp.ndarray:
    """
    (..., S, R) frame stack → (..., R, S+7) per-ray channels.

    Raw frames are kept (the net can learn its own temporal filters) and
    augmented with handcrafted gait cues: oscillation amplitude (std,
    max−min), instantaneous motion (last-frame diff) and net displacement
    across the window (flow). On static geometry all of these are ≈ sensor
    noise; on swinging legs they are large — a strong leg/not-leg prior.
    """
    x    = jnp.swapaxes(stack, -1, -2)          # (..., R, S)
    cur  = x[..., -1:]
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    mx   = x.max(axis=-1, keepdims=True)
    mn   = x.min(axis=-1, keepdims=True)
    diff = x[..., -1:] - x[..., -2:-1]          # one-step motion
    flow = x[..., -1:] - x[..., :1]             # full-window motion
    return jnp.concatenate([x, cur, mean, std, mx, mn, diff, flow], axis=-1)


# ── Multi-scale circular block ────────────────────────────────────────────────

class MultiScaleLegBlock(nn.Module):
    """Parallel dilated circular convs spanning every leg angular width."""
    features: int = 64

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        b = self.features // 4
        branches = [
            nn.Conv(b, kernel_size=(3,), kernel_dilation=(d,),
                    padding='CIRCULAR')(z)
            for d in (1, 2, 4, 8)               # 3 / 5 / 9 / 17-ray fields
        ]
        out = nn.relu(jnp.concatenate(branches, axis=-1))
        out = nn.Conv(self.features, kernel_size=(1,))(out)
        if z.shape[-1] != self.features:
            z = nn.Conv(self.features, kernel_size=(1,))(z)
        return nn.LayerNorm()(nn.relu(z + out))


# ── LegDetNet ─────────────────────────────────────────────────────────────────

class PerceptionNet(nn.Module):
    """LegDetNet — per-ray leg detection + HSFM body-centre voting."""
    stack_dim: int = PC.STACK_DIM
    num_rays:  int = PC.NUM_RAYS
    feat:      int = 64
    depth:     int = 3

    @nn.compact
    def __call__(self, lidar_stack: jnp.ndarray):
        """
        lidar_stack : (..., stack_dim, num_rays)
        Returns (logits      (..., num_rays),
                 offsets     (..., num_rays, 2),   # → HSFM body centre
                 leg_offsets (..., num_rays, 2)).  # → hit leg centre
        """
        z = _temporal_channels(lidar_stack)              # (..., R, S+7)
        z = nn.relu(nn.Conv(self.feat, kernel_size=(5,),
                            padding='CIRCULAR')(z))      # local stem

        for _ in range(self.depth):
            z = MultiScaleLegBlock(features=self.feat)(z)

        h = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(z))

        logits  = nn.Dense(1, kernel_init=orthogonal(0.01),
                           bias_init=constant(0.0))(h)[..., 0]   # (..., num_rays)
        offsets = nn.Dense(2, kernel_init=orthogonal(0.01),
                           bias_init=constant(0.0))(h)           # (..., num_rays, 2)
        leg_offsets = nn.Dense(2, kernel_init=orthogonal(0.01),
                               bias_init=constant(0.0))(h)       # (..., num_rays, 2)
        return logits, offsets, leg_offsets


# ── Leg-pair vote decoding (NumPy, eval/deployment side) ─────────────────────

def _cluster_legs(leg_votes, body_votes, probs, ray_idx, nms_dist):
    """Greedy-cluster leg votes.

    Returns a list of dicts:
      leg, body : weighted vote means (2,)
      score     : mean ray confidence
      n_rays    : supporting-ray count
      rays      : original scan indices of the supporting rays
    """
    order = np.argsort(-probs)
    used = np.zeros(len(order), dtype=bool)
    legs = []
    for idx in order:
        if used[idx]:
            continue
        cluster = np.linalg.norm(leg_votes - leg_votes[idx], axis=1) < nms_dist
        cluster &= ~used
        used |= cluster
        w = probs[cluster]
        legs.append({
            "leg":    (leg_votes[cluster] * w[:, None]).sum(0) / w.sum(),
            "body":   (body_votes[cluster] * w[:, None]).sum(0) / w.sum(),
            "score":  w.mean(),
            "n_rays": int(cluster.sum()),
            "rays":   ray_idx[cluster],
        })
    return legs


def _is_free_standing(leg, dist, gap):
    """
    A leg is a free-standing object: on at least one angular side of its
    supporting rays the scan must be deeper (background) by `gap`. Rays on
    a wall have wall on both sides and fail this test.
    """
    idx = np.sort(leg["rays"])
    # Handle the 360° wrap: if the span covers more than half the ring,
    # shift the upper indices down so min/max bound the true angular span.
    if idx[-1] - idx[0] > PC.NUM_RAYS // 2:
        idx = np.sort(np.where(idx > PC.NUM_RAYS // 2, idx - PC.NUM_RAYS, idx))
    left  = (idx[0] - 1) % PC.NUM_RAYS
    right = (idx[-1] + 1) % PC.NUM_RAYS
    depth = dist[leg["rays"]].mean()
    return (dist[left] > depth + gap) or (dist[right] > depth + gap)


def decode_detections(logits, offsets, leg_offsets, current_inv_scan,
                      threshold: float = PC.DET_THRESHOLD,
                      pair_max_dist: float = PC.PAIR_MAX_DIST):
    """
    Turn per-ray predictions into detected people (ego frame) with the
    leg-pair constraint: a person is emitted only when TWO distinct
    free-standing leg candidates pair up at anatomical distance
    (PAIR_MIN_SEP–PAIR_MAX_DIST), each with depth background on a side
    (walls fail), with loose body-vote agreement and enough combined ray
    support (better leg ≥ MIN_RAYS_PER_LEG, pair ≥ MIN_RAYS_PER_PERSON).
    The person centre is the midpoint of the paired legs.

    logits           : (NUM_RAYS,)
    offsets          : (NUM_RAYS, 2)  body-centre votes (× OFFSET_SCALE)
    leg_offsets      : (NUM_RAYS, 2)  leg-centre votes (× LEG_OFFSET_SCALE)
    current_inv_scan : (NUM_RAYS,)    last frame of the input stack

    Returns (centers (K, 2), scores (K,)) sorted by descending confidence.
    """
    logits      = np.asarray(logits)
    offsets     = np.asarray(offsets)
    leg_offsets = np.asarray(leg_offsets)
    inv         = np.asarray(current_inv_scan)

    probs = 1.0 / (1.0 + np.exp(-logits))
    dist  = PC.MAX_DIST - inv * (PC.MAX_DIST - ROBOT_RADIUS)   # undo normalization
    ang   = -PC.FOV * 0.5 + np.arange(PC.NUM_RAYS) * (PC.FOV / (PC.NUM_RAYS - 1))
    endpoints  = dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    body_votes = endpoints + offsets * PC.OFFSET_SCALE
    leg_votes  = endpoints + leg_offsets * PC.LEG_OFFSET_SCALE

    keep = probs >= threshold
    if not keep.any():
        return np.zeros((0, 2)), np.zeros((0,))

    legs = _cluster_legs(leg_votes[keep], body_votes[keep], probs[keep],
                         np.where(keep)[0], PC.LEG_NMS_DIST)
    legs = [l for l in legs
            if _is_free_standing(l, dist, PC.FREE_STAND_GAP)]
    if len(legs) < 2:
        return np.zeros((0, 2)), np.zeros((0,))

    # Candidate pairs: two distinct free-standing legs at anatomical
    # distance with enough combined ray support. Greedy assignment by
    # closest leg separation.
    n = len(legs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            leg_d  = np.linalg.norm(legs[i]["leg"] - legs[j]["leg"])
            body_d = np.linalg.norm(legs[i]["body"] - legs[j]["body"])
            n_i, n_j = legs[i]["n_rays"], legs[j]["n_rays"]
            ok = (PC.PAIR_MIN_SEP < leg_d < pair_max_dist
                  and body_d < PC.BODY_AGREE_DIST
                  and max(n_i, n_j) >= PC.MIN_RAYS_PER_LEG
                  and n_i + n_j >= PC.MIN_RAYS_PER_PERSON)
            if ok:
                pairs.append((leg_d, i, j))
    pairs.sort()

    used = np.zeros(n, dtype=bool)
    centers, scores = [], []
    for _, i, j in pairs:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        # Body centre = midpoint of the two feet (geometric, unambiguous)
        centers.append(0.5 * (legs[i]["leg"] + legs[j]["leg"]))
        scores.append(0.5 * (legs[i]["score"] + legs[j]["score"]))

    if not centers:
        return np.zeros((0, 2)), np.zeros((0,))
    centers = np.asarray(centers)
    scores  = np.asarray(scores)
    order   = np.argsort(-scores)
    return centers[order], scores[order]
