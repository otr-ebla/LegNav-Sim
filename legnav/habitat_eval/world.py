"""
world.py — Habitat-Sim world for testing legnav LiDAR policies in 3D scans.

Contents
--------
  HabitatWorld : scene + navmeshes + TurtleBot4 URDF + camera-based 2D LiDAR
                 + chase camera + top-down map.
  HumanWalker  : Habitat 3.0 SMPL-X humanoid driven by
                 HumanoidRearrangeController along navmesh shortest paths.

Frames
------
  Habitat is Y-up with the camera looking along -Z. Everything the policy sees
  lives in a planar frame  (X, Y) = (x_hab, -z_hab),  yaw θ CCW from +X,
  identical to the training env's convention.

LiDAR
-----
  Habitat has no LiDAR sensor, so 4 pinhole depth cameras (90° hFOV each,
  facing 0/90/180/270°) are mounted at the scan height. Only the centre pixel
  row is used: there the elevation is exactly 0, so the planar range of
  column u is  r = depth · sqrt(1 + x_ndc(u)²)  (z-depth → Euclidean).
  The 216 beams follow compute_lidar():  φ_i = θ - π + i·2π/(N-1).
  Beam → (camera, column) indices are precomputed once, so each scan is a
  single numpy gather. Depth 0 (mesh hole, no geometry) → max range.
  The robot mesh is moved out of the way while the LiDAR cameras render, so
  the scan never hits the robot's own tower standoffs.
"""

import math
import os

import magnum as mn
import numpy as np
import habitat_sim
from habitat_sim.physics import MotionType

from habitat.articulated_agent_controllers import HumanoidRearrangeController

HUMAN_BASE_OFFSET = mn.Vector3(0.0, 0.9, 0.0)       # pelvis root above the feet
# KinematicHumanoid.offset_transform_base (model ↔ wrapper frame)
_HUM_OFF_BASE_INV = (mn.Matrix4.rotation(mn.Rad(-math.pi / 2), mn.Vector3(0, 0, 1))
                     @ mn.Matrix4.rotation(mn.Rad(-math.pi / 2), mn.Vector3(0, 1, 0))).inverted()
_URDF_Z_UP = mn.Matrix4.rotation_x(mn.Rad(-math.pi / 2))   # ROS Z-up → Habitat Y-up
_PARKED = mn.Matrix4.translation(mn.Vector3(0.0, -100.0, 0.0))

HUMANOID_NAMES = ["female_0", "male_0", "female_1", "male_1", "neutral_0", "female_2",
                  "male_2", "neutral_1", "female_3", "male_3", "neutral_2", "neutral_3"]


# ── Frame helpers ─────────────────────────────────────────────────────────────

def hab_to_planar(p) -> np.ndarray:
    return np.array([p[0], -p[2]], dtype=np.float64)


def planar_to_hab(xy, y_floor: float) -> np.ndarray:
    return np.array([xy[0], y_floor, -xy[1]], dtype=np.float32)


def yaw_from_hab_quat(q_xyzw) -> float:
    """Habitat agent rotation (about +Y, forward = -Z) → planar heading θ."""
    x, y, z, w = q_xyzw
    return wrap(2.0 * math.atan2(y, w) + math.pi / 2)


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def stage_floor(sim, x, y_ref, z, up=0.3, down=1.0):
    """Height of the scene mesh below (x, z), searching from y_ref+up to
    y_ref-down. Only STAGE hits count: humans and the robot are ignored.
    Navmesh vertices can float up to ~0.4 m above the mesh floor (Recast
    cell_height quantisation), hence the long downward reach."""
    r = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(x, y_ref + up, z), mn.Vector3(0, -1, 0)), up + down)
    for h in r.hits:
        if h.object_id == habitat_sim.stage_id:
            return float(h.point[1])
    return None


def geodesic(pf, a, b) -> float:
    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = a, b
    return sp.geodesic_distance if pf.find_path(sp) else math.inf


# ── Humanoid ──────────────────────────────────────────────────────────────────

class HumanWalker:
    """A Habitat 3.0 humanoid that walks to goals along the navmesh.

    The controller advances the walking mocap by whole frames per call. To get
    an exact average speed, every call advances exactly ONE mocap frame and
    the number of calls per step comes from a fractional frame accumulator.

    Goals form a queue: e.g. [point on the robot's route, random far point],
    so route-crossing humans walk THROUGH the robot's route instead of
    parking on it. Idling only happens at the last goal of the queue.
    Yielding: the human stops when the robot is in its front half-plane and
    close; after `max_yield` seconds it gives up and picks a new goal.
    """

    def __init__(self, sim, pathfinder, data_dir, name, start, rng, dt, speed, island,
                 ang_speed_deg=360.0, max_yield=1.5):
        self.sim, self.pf, self.rng, self.island, self.dt = sim, pathfinder, rng, island, dt
        hdir = os.path.join(data_dir, "humanoids", "humanoid_data")
        aom = sim.get_articulated_object_manager()
        self.ao = aom.add_articulated_object_from_urdf(
            os.path.join(hdir, name, f"{name}.urdf"), fixed_base=False, maintain_link_order=True)
        self.ao.motion_type = MotionType.KINEMATIC
        c = self.ctrl = HumanoidRearrangeController(os.path.join(hdir, "walking_motion_processed_smplx.pkl"))
        c.motion_fps = c.walk_motion.fps                      # one mocap frame per call
        self.frames_per_step = speed * dt / c.dist_per_step_size
        c.turning_step_amount = c.threshold_rotate_not_move = ang_speed_deg * c.dist_per_step_size / speed
        self.speed, self._acc = speed, 0.0
        c.reset(mn.Matrix4.translation(mn.Vector3(start) + HUMAN_BASE_OFFSET))
        c.prev_orientation = None
        self.goals: list = []
        self.wait, self.yielded, self.max_yield = 0, 0.0, max_yield
        ang = rng.uniform(-math.pi, math.pi)
        self.ctrl.calculate_walk_pose(mn.Vector3(math.cos(ang), 0, math.sin(ang)), distance_multiplier=0)
        self._apply()

    # ground position of the feet (Habitat frame)
    @property
    def pos(self) -> np.ndarray:
        return np.array(self.ctrl.obj_transform_base.translation - HUMAN_BASE_OFFSET)

    @property
    def heading_xy(self) -> np.ndarray:
        f = self.ctrl.prev_orientation
        return np.array([1.0, 0.0]) if f is None else hab_to_planar(np.array(f))

    def _apply(self):
        c = self.ctrl
        # Keep the feet on the scanned floor (it varies by ±10 cm, and the
        # controller itself only moves the root horizontally). Smoothed to
        # absorb mesh noise; jumps > 0.4 m (other storey) are ignored.
        T = c.obj_transform_base
        feet = T.translation - HUMAN_BASE_OFFSET
        fy = stage_floor(self.sim, feet[0], feet[1], feet[2], up=0.5, down=0.6)
        if fy is not None and abs(fy - feet[1]) < 0.4:
            T.translation = mn.Vector3(T.translation[0], feet[1] + 0.5 * (fy - feet[1]) + HUMAN_BASE_OFFSET[1],
                                       T.translation[2])
            c.obj_transform_base = T
        self.ao.joint_positions = c.joint_pose
        self.ao.transformation = (c.obj_transform_base @ _HUM_OFF_BASE_INV) @ c.obj_transform_offset

    def _random_goal(self):
        for _ in range(20):
            g = np.array(self.pf.get_random_navigable_point(island_index=self.island))
            if np.linalg.norm((g - self.pos)[[0, 2]]) > 3.0 and math.isfinite(geodesic(self.pf, self.pos, g)):
                return g
        return None

    def set_goals(self, via=None):
        """Queue [via (optional), random far goal]."""
        self.goals = ([np.array(via)] if via is not None else []) + [self._random_goal()]
        self.goals = [g for g in self.goals if g is not None]
        self.yielded = 0.0

    def _stand(self):
        self.ctrl.calculate_stop_pose()
        self._apply()

    def step(self, robot_xy, robot_r):
        if self.goals and np.linalg.norm((self.goals[0] - self.pos)[[0, 2]]) < 0.4:
            self.goals.pop(0)
            if not self.goals:
                self.wait = int(self.rng.integers(0, int(2.0 / self.dt)))   # idle ≤ 2 s at the final goal
        if not self.goals:
            if self.wait > 0:
                self.wait -= 1
                return self._stand()
            self.set_goals()
            if not self.goals:
                return self._stand()
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = self.pos, self.goals[0]
        if not self.pf.find_path(sp) or len(sp.points) < 2:
            self.goals.pop(0)
            return self._stand()
        nxt = next((np.array(p) for p in sp.points[1:] if np.linalg.norm((np.array(p) - self.pos)[[0, 2]]) > 0.15),
                   np.array(sp.points[-1]))
        d = nxt - self.pos
        d[1] = 0.0
        # Yield: stop while the robot is close in the front half-plane.
        rel = robot_xy - hab_to_planar(self.pos)
        dist = np.linalg.norm(rel)
        if dist < 0.6 + robot_r and rel @ hab_to_planar(d) > 0.0:
            self.yielded += self.dt
            if self.yielded > self.max_yield:            # give up: turn and go elsewhere
                self.set_goals()
            return self._stand()
        self.yielded = 0.0
        self._acc += self.frames_per_step
        n, self._acc = int(self._acc), self._acc - int(self._acc)
        for _ in range(n):
            self.ctrl.calculate_walk_pose(mn.Vector3(d))
        self._apply()


# ── World ─────────────────────────────────────────────────────────────────────

class HabitatWorld:
    def __init__(self, scene, data_dir, robot_urdf, scene_dataset_config=None,
                 robot_radius=0.18, human_radius=0.3, lidar_height=0.193, lidar_x=0.0,
                 num_rays=216, cam_width=360, chase_res=(480, 640), render=True):
        self.data_dir, self.robot_r, self.num_rays = data_dir, robot_radius, num_rays
        self.lidar_h, self.lidar_x, self.cam_w, self.render = lidar_height, lidar_x, cam_width, render

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene
        if scene_dataset_config:
            sim_cfg.scene_dataset_config_file = scene_dataset_config
        sim_cfg.enable_physics = True

        lidar_specs = []
        for i in range(4):
            s = habitat_sim.CameraSensorSpec()
            s.uuid, s.sensor_type = f"lidar_{i}", habitat_sim.SensorType.DEPTH
            s.resolution, s.hfov = [3, cam_width], 90.0
            # agent forward = -Z  →  lidar forward offset `lidar_x` is -lidar_x along Z
            s.position, s.orientation = [0.0, lidar_height, -lidar_x], [0.0, i * math.pi / 2, 0.0]
            lidar_specs.append(s)
        chase = habitat_sim.CameraSensorSpec()
        chase.uuid, chase.sensor_type = "chase", habitat_sim.SensorType.COLOR
        self.chase_res, self.chase_hfov = tuple(chase_res), 75.0
        chase.resolution, chase.hfov = list(chase_res), self.chase_hfov
        chase.position, chase.orientation = [0.0, 1.5, 1.9], [-0.55, 0.0, 0.0]
        cfg = habitat_sim.Configuration(sim_cfg, [
            habitat_sim.agent.AgentConfiguration(sensor_specifications=lidar_specs),
            habitat_sim.agent.AgentConfiguration(sensor_specifications=[chase] if render else []),
        ])
        self.sim = habitat_sim.Simulator(cfg)

        # Navmeshes. Recast erodes conservatively (boundary ≈ radius + ~8 cm from
        # geometry), so the robot navmesh is used ONLY for global planning.
        # Robot collisions are checked against the real geometry (ray fan, see
        # body_clear) and a thin navmesh only guarantees there is floor below.
        self.pf_robot = self.sim.pathfinder
        self._recompute(self.pf_robot, robot_radius, 0.4)
        self.pf_floor = habitat_sim.PathFinder()
        self._recompute(self.pf_floor, 0.05, 0.4)
        self.pf_human = habitat_sim.PathFinder()
        self._recompute(self.pf_human, human_radius, 1.8)
        # body ray fan: 5 heights × 16 directions, length = robot radius.
        # Lowest ray at 6 cm: scanned floors have a few cm of noise/rug edges.
        self._fan_h = np.array([0.06, 0.12, 0.19, 0.27, 0.34])
        a = np.arange(16) * (2.0 * math.pi / 16)
        self._fan_d = [mn.Vector3(math.cos(t), 0.0, -math.sin(t)) for t in a]

        aom = self.sim.get_articulated_object_manager()
        self.robot = aom.add_articulated_object_from_urdf(robot_urdf, fixed_base=True)
        self.robot.motion_type = MotionType.KINEMATIC
        self.humans: list[HumanWalker] = []

        # Beam → (camera k, column u) gather table, plus z-depth → range factor.
        phi = -math.pi + np.arange(num_rays) * (2.0 * math.pi / (num_rays - 1))
        k = np.round(phi / (math.pi / 2)).astype(int) % 4
        beta = wrap(phi - k * math.pi / 2)
        u = np.clip(np.round((-np.tan(beta) + 1.0) / 2.0 * cam_width - 0.5).astype(int), 0, cam_width - 1)
        xu = (u + 0.5) / cam_width * 2.0 - 1.0
        self._beam_k, self._beam_u, self._beam_scale = k, u, np.sqrt(1.0 + xu ** 2)
        self.beam_angles = phi

        self.pose = np.zeros(3)          # planar x, y, θ
        self.y_floor = 0.0

    def _recompute(self, pf, radius, height):
        s = habitat_sim.NavMeshSettings()
        s.set_defaults()
        s.agent_radius, s.agent_height = radius, height
        s.include_static_objects = True
        if not self.sim.recompute_navmesh(pf, s):
            raise RuntimeError("navmesh recompute failed")

    # ── robot ──
    def true_floor(self, pos_nav, fallback=None) -> float:
        """Recast navmesh vertices float 8–40 cm above the mesh floor: find the
        real floor with one downward stage-only ray. Mesh hole → fallback."""
        fy = stage_floor(self.sim, pos_nav[0], pos_nav[1], pos_nav[2])
        if fy is not None:
            return fy
        return float(pos_nav[1]) - 0.08 if fallback is None else fallback

    def indoor_island(self, samples=10, rays=24) -> int:
        """Navmesh island for random episodes: area × (fraction of horizontal
        rays at 0.5 m that hit a wall within 12 m)². Plain 'largest island'
        can be the roof / outer shell of a scanned building."""
        pf, best, best_score = self.pf_robot, 0, -1.0
        saved = self.robot.transformation
        self.robot.transformation = _PARKED
        dirs = [mn.Vector3(math.cos(t), 0, -math.sin(t)) for t in np.linspace(0, 2 * math.pi, rays, endpoint=False)]
        for i in range(pf.num_islands):
            area = pf.island_area(i)
            if area < 2.0:
                continue
            hit = 0
            for _ in range(samples):
                p = mn.Vector3(pf.get_random_navigable_point(island_index=i)) + mn.Vector3(0, 0.5, 0)
                hit += sum(self.sim.cast_ray(habitat_sim.geo.Ray(p, d), 12.0).has_hits() for d in dirs)
            score = area * (hit / (samples * rays)) ** 2
            if score > best_score:
                best, best_score = i, score
        self.robot.transformation = saved
        return best

    def set_robot(self, pos_nav, theta, floor_y=None):
        """pos_nav: point at navmesh height. Body/sensors go on the true floor."""
        self.y_floor = float(pos_nav[1])                 # navmesh height (planning)
        self.y_true = self.true_floor(pos_nav) if floor_y is None else floor_y
        self.pose = np.array([pos_nav[0], -pos_nav[2], wrap(theta)])
        p = mn.Vector3(pos_nav[0], self.y_true, pos_nav[2])
        self.robot.transformation = (mn.Matrix4.translation(p)
                                     @ mn.Matrix4.rotation_y(mn.Rad(self.pose[2])) @ _URDF_Z_UP)
        for i in (0, 1):
            n = self.sim.get_agent(i).scene_node
            n.translation = p
            n.rotation = mn.Quaternion.rotation(mn.Rad(self.pose[2] - math.pi / 2), mn.Vector3.y_axis())

    @property
    def robot_hab(self) -> np.ndarray:
        """Robot position at navmesh height (for pathfinder queries)."""
        return planar_to_hab(self.pose[:2], self.y_floor)

    def body_clear(self, pos_hab) -> bool:
        """No static/human geometry within robot_radius of the robot axis,
        at any height of the TB4 body (4–34 cm). Robot itself is parked."""
        saved = self.robot.transformation
        self.robot.transformation = _PARKED
        try:
            for h in self._fan_h:
                o = mn.Vector3(pos_hab) + mn.Vector3(0.0, float(h), 0.0)
                for d in self._fan_d:
                    if self.sim.cast_ray(habitat_sim.geo.Ray(o, d), self.robot_r).has_hits():
                        self.block_reason = f"body@{h:.2f}m"
                        return False
            return True
        finally:
            self.robot.transformation = saved

    def move_robot(self, v, w, dt):
        """Exact unicycle step (as jax_env.step_env). Translation is refused if
        the body would touch geometry or leave the floor; rotation is always
        applied. Returns True on a collision (blocked translation)."""
        x, y, th = self.pose
        if abs(w) > 1e-3:
            nx = x + v / w * (math.sin(th + w * dt) - math.sin(th))
            ny = y + v / w * (-math.cos(th + w * dt) + math.cos(th))
        else:
            nx, ny = x + v * math.cos(th) * dt, y + v * math.sin(th) * dt
        if math.hypot(nx - x, ny - y) < 1e-6:
            self.set_robot(self.robot_hab, th + w * dt, self.y_true)
            return False
        want = planar_to_hab((nx, ny), self.y_floor)
        nav = np.array(self.pf_floor.snap_point(want))
        self.block_reason = "no-floor"
        # (navmesh heights are quantised by cell_height, so only xz is compared;
        #  steps are checked on the true floor below)
        if np.all(np.isfinite(nav)) and np.linalg.norm((nav - want)[[0, 2]]) < 0.05:
            want[1] = nav[1]
            fy = self.true_floor(want, fallback=self.y_true)
            self.block_reason = "floor-step"
            if abs(fy - self.y_true) < 0.05 and self.body_clear(np.array([want[0], fy, want[2]])):
                self.set_robot(want, th + w * dt, fy)
                return False
        self.set_robot(self.robot_hab, th + w * dt, self.y_true)
        return True

    # ── humans ──
    def clear_humans(self):
        aom = self.sim.get_articulated_object_manager()
        for h in self.humans:
            aom.remove_object_by_handle(h.ao.handle)
        self.humans = []

    def spawn_humans(self, n, rng, dt, speed, avoid_xy, min_sep=1.5, cross_points=()):
        """speed: float, or (lo, hi) → uniform per human."""
        island = self.pf_human.get_island(self.pf_human.snap_point(self.robot_hab))
        for i in range(n):
            for _ in range(200):
                p = np.array(self.pf_human.get_random_navigable_point(island_index=island))
                if all(np.linalg.norm(hab_to_planar(p) - a) > min_sep for a in avoid_xy) and \
                   all(np.linalg.norm((p - h.pos)[[0, 2]]) > 1.0 for h in self.humans):
                    break
            p[1] = self.true_floor(p)                    # feet on the mesh floor
            spd = float(rng.uniform(*speed)) if isinstance(speed, (tuple, list)) else float(speed)
            name = HUMANOID_NAMES[int(rng.integers(len(HUMANOID_NAMES)))]
            h = HumanWalker(self.sim, self.pf_human, self.data_dir, name, p, rng, dt, spd, island)
            self.humans.append(h)
        self.retarget_humans(cross_points)

    def retarget_humans(self, cross_points=()):
        """Some humans walk through points of the robot's route (encounters)."""
        for i, h in enumerate(self.humans):
            h.set_goals(self.pf_human.snap_point(cross_points[i]) if i < len(cross_points) else None)

    def step_humans(self):
        for h in self.humans:
            h.step(self.pose[:2], self.robot_r)

    # ── sensors ──
    def lidar(self, max_range=12.0) -> np.ndarray:
        """216 planar ranges (m) from the lidar mount, training beam order."""
        saved = self.robot.transformation
        self.robot.transformation = _PARKED
        obs = self.sim.get_sensor_observations(0)
        self.robot.transformation = saved
        d = np.stack([obs[f"lidar_{i}"] for i in range(4)])[:, 1, :]      # centre row (4, W)
        r = d[self._beam_k, self._beam_u] * self._beam_scale
        r = np.where(r <= 1e-6, max_range, r)
        # pose the scan was taken from (planar x, y, θ, true floor height)
        self.scan_pose = (*self.pose, self.y_true)
        return np.minimum(r, max_range)

    def lidar_rays_3d(self, ranges, scan_pose=None):
        """Beam origin (3,) and end points (N, 3) in the Habitat frame."""
        x, y, th, fy = self.scan_pose if scan_pose is None else scan_pose
        ox, oy = x + self.lidar_x * math.cos(th), y + self.lidar_x * math.sin(th)
        a = th + self.beam_angles
        o = np.array([ox, fy + self.lidar_h, -oy])
        ends = np.stack([ox + ranges * np.cos(a), np.full_like(a, o[1]), -(oy + ranges * np.sin(a))], -1)
        return o, ends

    def chase_camera(self):
        """(world←camera 4×4, focal px, cx, cy) of the chase RGB camera."""
        node = self.sim.get_agent(1).scene_node.subtree_sensors["chase"].node
        H, W = self.chase_res
        f = (W / 2.0) / math.tan(math.radians(self.chase_hfov) / 2.0)
        return np.array(node.absolute_transformation()), f, W / 2.0, H / 2.0

    def chase_rgb(self) -> np.ndarray:
        return self.sim.get_sensor_observations(1)["chase"][..., :3]

    def topdown(self, mpp=0.05):
        """Occupancy of the robot navmesh at the robot's floor + pixel mapper."""
        lo, _ = self.pf_robot.get_bounds()
        grid = self.pf_robot.get_topdown_view(mpp, self.y_floor + 0.1)
        ox, oz = lo[0], lo[2]

        def to_px(xy):
            xy = np.atleast_2d(xy)
            return np.stack([(xy[:, 0] - ox) / mpp, (-xy[:, 1] - oz) / mpp], -1)
        return grid, to_px

    def close(self):
        self.sim.close()
