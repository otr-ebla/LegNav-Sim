"""
perception/config.py — Configuration for the perception module
================================================================
The perception module is trained independently from the navigation policy:
the robot stays still while pedestrians (leg-level model) walk around it,
and the network learns to detect people and regress their position from
stacked 2D LiDAR scans.

NOTE: STACK_DIM here is 8 (perception), independent from
RobotConfig.LIDAR_STACK_DIM = 3 used by the navigation policy.
"""

import math


class PerceptionConfig:
    # ── Sensor (matches the simulator's LiDAR) ────────────────────────────
    NUM_RAYS  = 216            # full-circle scan
    FOV       = 2.0 * math.pi  # 360°
    MAX_DIST  = 12.0           # m — mirrors jax_env.MAX_LIDAR_DIST
    STACK_DIM = 8              # consecutive frames fed to the network

    # ── Ground-truth / decoding ───────────────────────────────────────────
    # A person is a supervision target only when at least one ray hits
    # one of their legs (occluded people cannot be labeled).
    GT_EPS          = 1e-3   # m — tolerance: leg hit == closest hit
    OFFSET_SCALE    = 2.0    # m — body-centre regression targets / this
    LEG_OFFSET_SCALE = 0.5   # m — leg-centre regression targets / this
    ASSOC_DIST      = 0.5    # m — detection ↔ GT association threshold (eval)
    DET_THRESHOLD   = 0.5    # per-ray person probability threshold

    # Leg-pair decoding: rays vote for leg centres; legs are clustered and a
    # person is emitted only when two distinct legs pair up at anatomical
    # distance and agree on the same body centre.
    LEG_NMS_DIST     = 0.16  # m — leg-vote clustering radius (leg Ø = 0.16)
    PAIR_MIN_SEP     = 0.10  # m — legs are distinct objects (no self-pairing)
    PAIR_MAX_DIST    = 0.80  # m — max leg-to-leg distance (hip 0.32 + stride)
    # Body votes are ambiguous per single leg (left/right unknown → the net
    # votes the leg itself), so agreement is only a loose sanity bound; the
    # person centre comes from the midpoint of the paired legs.
    BODY_AGREE_DIST  = 0.70  # m — ≥ max leg separation
    MIN_RAYS_PER_LEG = 2     # rays on the better-supported leg of a pair
    MIN_RAYS_PER_PERSON = 3  # combined ray support of the pair
    FREE_STAND_GAP   = 0.25  # m — depth gap behind a leg's angular neighbours
                             #     (walls have no background → rejected)

    # ── Static decoy clutter (hard negatives) ─────────────────────────────
    # Leg-like static geometry injected into every training scene: the
    # network must learn that thin free-standing objects without gait
    # history are NOT legs (false positives on map furniture/door jambs).
    DECOY_PAIRS   = 4     # static circle pairs at anatomical leg separation
    DECOY_POLES   = 2     # single leg-sized poles
    DECOY_SLATS   = 4     # thin boxes (door jambs / wall stubs / furniture edges)
    DECOY_R_MIN   = 0.05  # m — decoy circle radius range (leg r = 0.08)
    DECOY_R_MAX   = 0.11
    DECOY_SEP_MIN = 0.10  # m — pair separation range (mirrors PAIR_MIN/MAX_DIST)
    DECOY_SEP_MAX = 0.80

    # ── Training ──────────────────────────────────────────────────────────
    NUM_ENVS        = 256    # parallel stationary-robot scenes
    ROLLOUT_STEPS   = 16     # env steps collected per update chunk
    EPISODE_STEPS   = 200    # steps before a scene is resampled
    LR              = 3e-4
    NUM_UPDATES     = 3000   # total collect+update chunks
    MINIBATCHES     = 8      # minibatches per chunk
    POS_WEIGHT      = 2.0    # BCE weight on (sparse) person rays
    OFFSET_LOSS_W   = 5.0    # weight of the regression term
    LOG_EVERY       = 20
    CKPT_EVERY      = 200
