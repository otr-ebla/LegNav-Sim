"""Robot-side observation builder for the Shelfy real-robot policies (NumPy only, no JAX/Flax).

Reference implementation of the contract in README.md: a ROS node feeds it the raw sensors every
DT = 0.15 s and gets the exact flat vector the policy was trained on. `check_parity.py` replays the
training env through it and compares to legnav.core.jax_wrappers (must match to float32 precision).

  layout v668 : [kin(5) | goal(2x3) | ego_deltas(3x3) | lidar(216x3)]
  layout v672 : [kin(5) | cmd(4) | goal(2x3) | ego_deltas(3x3) | lidar(216x3)]
  kin  = [v_meas/max_v, w_meas, (max_v-0.2)/1.8, goal_dist/MAX_GOAL, goal_align/pi]
         v_meas, w_meas = MEASURED base velocity (odometry twist), not the last command
  cmd  = the two last PUBLISHED commands, newest first [v/max_v, w, v_prev/max_v, w_prev]; zeros at reset
  goal = goal in the robot frame / MAX_GOAL, per frame
  ego_deltas = pose of each frame in the newest robot frame (dx, dy, dtheta)
  lidar = (12 - r) / (12 - 0.2) clipped to [0, 1], 216 rays at -pi + k*2pi/215 (robot frame, CCW)
  frames = o_{t-4}, o_{t-2}, o_t of a 5-frame buffer (stride 2), oldest first
"""
import math

import numpy as np

DT = 0.15
NUM_RAYS, STACK, STRIDE = 216, 3, 2
BUF = (STACK - 1) * STRIDE + 1
MAX_LIDAR = 12.0
LIDAR_NORM_R = 0.2                                   # normalisation constant of the training env (not the footprint)
MAX_GOAL = math.sqrt(12.0 ** 2 + 12.0 ** 2)
RAY_ANGLES = -math.pi + np.arange(NUM_RAYS) * (2.0 * math.pi / (NUM_RAYS - 1))
SEL = np.arange(STACK) * STRIDE
LAYOUT_SIZE = {"v668": 668, "v672": 672}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def layout_of(fused_in):
    """Layout from the first fused Dense input of a CALF checkpoint (params['Dense_0']['kernel'].shape[0])."""
    return {203: "v668", 207: "v672"}[int(fused_in)]


class ShelfyObsBuilder:
    def __init__(self, max_v, layout="v672"):
        assert layout in LAYOUT_SIZE, layout
        self.max_v, self.layout = float(max_v), layout
        self.goal_s = self.lidar_s = self.pose_s = None
        self.cmd = np.zeros((2, 2))                  # [[v, w] older, [v, w] newest] published commands

    def record_command(self, v, w):
        """Call right after publishing (v, w) on cmd_vel (the value actually sent, after any clamp)."""
        self.cmd = np.array([self.cmd[1], [float(v), float(w)]])

    def _single(self, ranges, pose, goal, v, w):
        r = np.clip(np.asarray(ranges, np.float64), 0.0, MAX_LIDAR)
        inv = np.clip((MAX_LIDAR - r) / (MAX_LIDAR - LIDAR_NORM_R), 0.0, 1.0)
        x, y, th = pose
        dx, dy = goal[0] - x, goal[1] - y
        c, s = math.cos(-th), math.sin(-th)
        g = np.array([(c * dx - s * dy) / MAX_GOAL, (s * dx + c * dy) / MAX_GOAL])
        kin = np.array([v / max(self.max_v, 1e-3), w, (self.max_v - 0.2) / 1.8,
                        math.hypot(dx, dy) / MAX_GOAL, wrap(math.atan2(dy, dx) - th) / math.pi])
        return g, kin, inv

    def reset(self, ranges, pose, goal, v=0.0, w=0.0):
        """Episode start: buffers filled with the first frame, pending commands zeroed."""
        self.cmd = np.zeros((2, 2))
        g, kin, inv = self._single(ranges, pose, goal, v, w)
        self.goal_s, self.lidar_s = np.tile(g, (BUF, 1)), np.tile(inv, (BUF, 1))
        self.pose_s = np.tile(np.asarray(pose, np.float64), (BUF, 1))
        return self._assemble(kin)

    def step(self, ranges, pose, goal, v, w):
        """ranges: (216,) metres at RAY_ANGLES; pose: (x, y, theta) odometry; goal: (x, y) same frame;
        v, w: measured velocity."""
        g, kin, inv = self._single(ranges, pose, goal, v, w)
        self.goal_s = np.vstack([self.goal_s[1:], g])
        self.lidar_s = np.vstack([self.lidar_s[1:], inv])
        self.pose_s = np.vstack([self.pose_s[1:], np.asarray(pose, np.float64)])
        return self._assemble(kin)

    def _assemble(self, kin):
        ps = self.pose_s[SEL]
        cur = ps[-1]
        dxw, dyw = ps[:, 0] - cur[0], ps[:, 1] - cur[1]
        c, s = math.cos(cur[2]), math.sin(cur[2])
        deltas = np.stack([c * dxw + s * dyw, -s * dxw + c * dyw,
                           (ps[:, 2] - cur[2] + math.pi) % (2.0 * math.pi) - math.pi], axis=-1)
        parts = [kin]
        if self.layout == "v672":
            inv_v = 1.0 / max(self.max_v, 1e-3)
            new, old = self.cmd[1], self.cmd[0]
            parts.append(np.array([new[0] * inv_v, new[1], old[0] * inv_v, old[1]]))
        parts += [self.goal_s[SEL].ravel(), deltas.ravel(), self.lidar_s[SEL].ravel()]
        return np.concatenate(parts)
