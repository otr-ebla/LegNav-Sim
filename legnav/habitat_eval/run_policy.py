"""
run_policy.py — Visual test of the CALF PPO (legs) policy in Habitat scenes.

The robot is the TurtleBot4 (URDF) with a simulated 216-beam 360° 2D LiDAR;
humans are Habitat 3.0 SMPL-X humanoids walking along the navmesh.
The policy sees EXACTLY the training observation (668-D, built with
legnav.core.jax_wrappers.assemble_stacked_obs): kin_vec, strided goal stack,
SE(2) ego-motion deltas and 3 strided LiDAR frames, at DT = 0.15 s.

Goal modes
----------
  waypoint (default): a navmesh shortest path to the final goal is planned
      every step; the policy is given a carrot `--lookahead` m along it
      (the policy is a local planner, trained in open rooms).
  direct: the policy is given the final goal directly (GPS+compass, like the
      Social-HM3D PointNav setting) — expect it to get stuck behind walls.

Episodes  (--num-episodes 0 = endless; stop with q or Ctrl+C)
--------
  --episodes <social-hm3d val.json.gz> [--shuffle] --scene-root <dir containing hm3d/...>
      uses the episode's scene, start pose and goal. --shuffle: random scene,
      then --episodes-per-scene random episodes of it, repeat.
  --scene-dir <dir>    random scene from the folder + random start/goal.
  --scene <file.glb>   random start/goal pairs (geodesic distance range).
  Humans: --num-humans 2-8 / --human-speed 0.6-1.4 re-sample every episode.
  --csv results.csv appends one row per episode.

Run (habitat conda env, from the repo root):
  python -m legnav.habitat_eval.run_policy --scene ~/habitat_data/data/scene_datasets/habitat-test-scenes/apartment_1.glb --show
  python -m legnav.habitat_eval.run_policy ... --save figures/habitat_apartment.mp4

Live window keys (--show; real-time paced):
  s           cycle speed 1x → 2x → 4x → 8x → max → 1x
  l           toggle LiDAR rays (top-down map + projected into the 3D view)
  → (or n)    next episode
  space       pause            q / Esc   quit
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import csv
import glob
import gzip
import itertools
import json
import math
import time

import cv2
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

from legnav.config import RobotConfig
from legnav.core.jax_network import EndToEndActorCritic, scale_action_to_env
from legnav.core.jax_wrappers import assemble_stacked_obs
from legnav.habitat_eval.world import HabitatWorld, geodesic, hab_to_planar, wrap, yaw_from_hab_quat

DT = RobotConfig.DT                       # 0.15 s policy period (training)
STACK, STRIDE = RobotConfig.LIDAR_STACK_DIM, RobotConfig.LIDAR_STACK_STRIDE
BUF = (STACK - 1) * STRIDE + 1
MAX_LIDAR = 12.0                          # jax_env.MAX_LIDAR_DIST
TRAIN_ROBOT_R = 0.2                       # RobotConfig.RADIUS (obs normalisation)
MAX_GOAL = math.sqrt(12.0 ** 2 + 12.0 ** 2)   # jax_env._MAX_GOAL_DIST
GOAL_RADIUS = 0.3                         # jax_env.GOAL_RADIUS
HUMAN_BODY_R = 0.2                        # human root ↔ robot contact radius (leg level)


# ── Policy ────────────────────────────────────────────────────────────────────

def load_policy(path, se2=True):
    raw = flax.serialization.msgpack_restore(open(path, "rb").read())
    params = raw["params"]
    net = EndToEndActorCritic(action_dim=2, use_se2_reproject=se2)

    @jax.jit
    def act(obs, max_v):
        mean = net.apply({"params": params}, obs[None])[0][0]
        return scale_action_to_env(mean, max_v)
    return act


# ── Observation (mirrors jax_env.get_obs + make_stacked_env) ──────────────────

class ObsBuilder:
    def __init__(self, max_v, noise, rng):
        self.max_v, self.noise, self.rng = max_v, noise, rng

    def single(self, ranges, pose, goal_xy, v, w):
        r = ranges
        if self.noise:                                   # same model as training
            r = r + self.rng.normal(0.0, 0.05, r.shape)
            r = np.where(self.rng.uniform(size=r.shape) < 0.06, MAX_LIDAR, r)
            r = np.clip(r, 0.0, MAX_LIDAR)
        inv = np.clip((MAX_LIDAR - r) / (MAX_LIDAR - TRAIN_ROBOT_R), 0.0, 1.0)
        x, y, th = pose
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        c, s = math.cos(-th), math.sin(-th)
        goal_vec = np.array([(c * dx - s * dy) / MAX_GOAL, (s * dx + c * dy) / MAX_GOAL])
        dist, align = math.hypot(dx, dy), wrap(math.atan2(dy, dx) - th)
        kin = np.array([v / max(self.max_v, 1e-3), w, (self.max_v - 0.2) / 1.8,
                        dist / MAX_GOAL, align / math.pi])
        return goal_vec, kin, inv

    def reset(self, ranges, pose, goal_xy):
        g, k, l = self.single(ranges, pose, goal_xy, 0.0, 0.0)
        self.goal_s, self.lidar_s = np.tile(g, (BUF, 1)), np.tile(l, (BUF, 1))
        self.pose_s = np.tile(pose, (BUF, 1))
        return self._assemble(k)

    def step(self, ranges, pose, goal_xy, v, w):
        g, k, l = self.single(ranges, pose, goal_xy, v, w)
        self.goal_s = np.vstack([self.goal_s[1:], g])
        self.lidar_s = np.vstack([self.lidar_s[1:], l])
        self.pose_s = np.vstack([self.pose_s[1:], pose])
        return self._assemble(k)

    def _assemble(self, kin):
        return assemble_stacked_obs(jnp.asarray(kin, jnp.float32), jnp.asarray(self.goal_s, jnp.float32),
                                    jnp.asarray(self.pose_s, jnp.float32), jnp.asarray(self.lidar_s, jnp.float32),
                                    STRIDE)


# ── Episodes ──────────────────────────────────────────────────────────────────

def load_social_episodes(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["episodes"]


def find_scenes(root):
    """HM3D habitat format → *.basis.glb; otherwise any *.glb (no semantic meshes)."""
    found = sorted(glob.glob(os.path.join(root, "**", "*.basis.glb"), recursive=True))
    if not found:
        found = sorted(p for p in glob.glob(os.path.join(root, "**", "*.glb"), recursive=True)
                       if not p.endswith(".semantic.glb"))
    if not found:
        raise FileNotFoundError(f"no .glb scenes under {root}")
    return found


class SceneResolver:
    """episode scene_id → file. Tries <root>/<scene_id>, the part after
    'scene_datasets/', then a basename index of <root> (built once)."""

    def __init__(self, root):
        self.root, self.index = root, None

    def __call__(self, scene_id):
        cands = [os.path.join(self.root, scene_id)]
        if "scene_datasets/" in scene_id:
            cands.append(os.path.join(self.root, scene_id.split("scene_datasets/", 1)[1]))
        for c in cands:
            if os.path.exists(c):
                return c
        if self.index is None:
            self.index = {}
            for p in glob.glob(os.path.join(self.root, "**", "*.glb"), recursive=True):
                self.index.setdefault(os.path.basename(p), os.path.realpath(p))
        hit = self.index.get(os.path.basename(scene_id))
        if hit is None:
            raise FileNotFoundError(f"scene {scene_id} not found under {self.root}")
        return hit


def episode_stream(args, rng):
    """Yields (scene_path, start|None, theta|None, goal|None, episode_id).
    num_episodes <= 0 → endless. None start/goal → sampled in the scene."""
    infinite = args.num_episodes <= 0
    budget = itertools.count() if infinite else range(args.num_episodes)
    k = itertools.count()
    if args.episodes:
        eps = load_social_episodes(args.episodes)
        resolve = SceneResolver(args.scene_root)

        def mk(i):
            e = eps[i]
            return (resolve(e["scene_id"]), np.array(e["start_position"]), yaw_from_hab_quat(e["start_rotation"]),
                    np.array(e["goals"][0]["position"]), i)
        if args.episode_ids:
            ids = itertools.cycle(args.episode_ids) if infinite else iter(args.episode_ids)
            for i in ids:
                yield mk(i)
        elif args.shuffle:
            by_scene = {}
            for i, e in enumerate(eps):
                by_scene.setdefault(e["scene_id"], []).append(i)
            scenes = list(by_scene)
            for _ in budget:
                ids = by_scene[scenes[int(rng.integers(len(scenes)))]]
                for i in rng.choice(ids, size=min(args.episodes_per_scene, len(ids)), replace=False):
                    yield mk(int(i))
                    if not infinite and next(k) >= args.num_episodes - 1:
                        return
        else:
            for n in budget:
                yield mk(n % len(eps))
    elif args.scene_dir:
        scenes = find_scenes(args.scene_dir)
        print(f"{len(scenes)} scenes under {args.scene_dir}")
        for _ in budget:
            sc = scenes[int(rng.integers(len(scenes)))]
            for _ in range(args.episodes_per_scene):
                n = next(k)
                yield (sc, None, None, None, n)
                if not infinite and n >= args.num_episodes - 1:
                    return
    else:
        for n in budget:
            yield (args.scene, None, None, None, n)


def parse_range(s, typ):
    """'3' → (3, 3); '2-8' → (2, 8)."""
    s = str(s)
    if "-" in s[1:]:
        a, b = s.split("-", 1)
        return typ(a), typ(b)
    return typ(s), typ(s)


def random_episode(world, rng, dmin, dmax):
    pf = world.pf_robot
    if not hasattr(world, "_episode_island"):
        world._episode_island = world.indoor_island()
    isl = world._episode_island
    for _ in range(500):
        s = np.array(pf.get_random_navigable_point(island_index=isl))
        g = np.array(pf.get_random_navigable_point(island_index=isl))
        d = geodesic(pf, s, g)
        if dmin <= d <= dmax and abs(s[1] - g[1]) < 0.3:
            return s, rng.uniform(-math.pi, math.pi), g
    raise RuntimeError("no start/goal pair found in the requested distance range")


def shortest_path(pf, a, b):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = a, b
    return (np.array(sp.points), sp.geodesic_distance) if pf.find_path(sp) else (None, math.inf)


def carrot(path_xy, lookahead):
    """Point at arc length `lookahead` along a planar polyline."""
    seg = np.diff(path_xy, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    if cum[-1] <= lookahead:
        return path_xy[-1]
    i = int(np.searchsorted(cum, lookahead) - 1)
    t = (lookahead - cum[i]) / max(L[i], 1e-9)
    return path_xy[i] + t * seg[i]


# ── Visualisation ─────────────────────────────────────────────────────────────

class Viewer:
    def __init__(self, world, height):
        self.world, self.h = world, height
        grid, self.to_px = world.topdown(0.05)
        img = np.where(grid[..., None], 235, 70).astype(np.uint8).repeat(3, -1)
        ys, xs = np.nonzero(grid)
        pad = 20
        self.crop = (max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad)
        self.base = img
        self.scale = 1.0

    def _rays_on_chase(self, chase, ranges):
        """Project the beams that hit something into the chase RGB image
        (X-ray overlay: no occlusion test)."""
        w = self.world
        hit = ranges < MAX_LIDAR - 1e-3
        if not hit.any():
            return chase
        T, f, cx, cy = w.chase_camera()
        Ti = np.linalg.inv(T)
        o, ends = w.lidar_rays_3d(ranges)
        oc = Ti[:3, :3] @ o + Ti[:3, 3]
        ec = ends[hit] @ Ti[:3, :3].T + Ti[:3, 3]
        near = -0.05                                     # camera looks along -Z
        if oc[2] > near:
            return chase
        behind = ec[:, 2] > near                         # clip at the near plane
        tc = (near - oc[2]) / (ec[behind, 2] - oc[2])
        ec[behind] = oc + tc[:, None] * (ec[behind] - oc)
        proj = lambda p: np.clip(np.stack([cx + f * p[..., 0] / -p[..., 2],
                                           cy - f * p[..., 1] / -p[..., 2]], -1), -1e4, 1e4)
        O = tuple(proj(oc).round().astype(int))
        E = proj(ec).round().astype(int)
        ov = np.ascontiguousarray(chase).copy()
        for e in E:
            cv2.line(ov, O, tuple(e), (255, 50, 50), 1, cv2.LINE_AA)
        for e in E[~behind]:
            cv2.circle(ov, tuple(e), 2, (255, 230, 0), -1, cv2.LINE_AA)
        return cv2.addWeighted(ov, 0.75, np.ascontiguousarray(chase), 0.25, 0)

    def frame(self, chase, path_xy, goal_xy, carrot_xy, ranges, hud, rays=False, footer=None):
        w = self.world
        img = self.base.copy()
        P = lambda xy: tuple(int(round(v)) for v in self.to_px(xy)[0])
        R = lambda m: max(1, int(round(m / 0.05)))
        if path_xy is not None and len(path_xy) > 1:
            cv2.polylines(img, [self.to_px(path_xy).round().astype(np.int32)], False, (60, 170, 60), 1, cv2.LINE_AA)
        # beams are drawn from the pose the scan was taken at (0.15 s cadence)
        x, y, th = w.pose
        sx, sy, sth, _ = w.scan_pose
        ox, oy = sx + w.lidar_x * math.cos(sth), sy + w.lidar_x * math.sin(sth)
        ang = sth + w.beam_angles
        ends = self.to_px(np.stack([ox + ranges * np.cos(ang), oy + ranges * np.sin(ang)], -1)).round().astype(int)
        hit = ranges < MAX_LIDAR - 1e-3
        if rays:
            ov = img.copy()
            O = P((ox, oy))
            for e, h in zip(ends, hit):
                cv2.line(ov, O, tuple(e), (40, 40, 220) if h else (160, 160, 160), 1, cv2.LINE_AA)
            img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
        for p in ends[hit]:
            cv2.circle(img, tuple(p), 1, (40, 40, 220), -1)
        for h in w.humans:
            hp = hab_to_planar(h.pos)
            cv2.circle(img, P(hp), R(0.25), (170, 60, 150), -1, cv2.LINE_AA)
            cv2.line(img, P(hp), P(hp + 0.4 * h.heading_xy), (170, 60, 150), 2, cv2.LINE_AA)
        cv2.circle(img, P(goal_xy), R(GOAL_RADIUS), (0, 0, 230), 2, cv2.LINE_AA)
        if carrot_xy is not None:
            cv2.circle(img, P(carrot_xy), 3, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(img, P((x, y)), R(w.robot_r), (200, 120, 20), -1, cv2.LINE_AA)
        cv2.line(img, P((x, y)), P((x + 0.35 * math.cos(th), y + 0.35 * math.sin(th))), (255, 255, 255), 2)
        y0, y1, x0, x1 = self.crop
        img = img[y0:y1, x0:x1]
        # letterbox into a fixed (h × 4h/3) panel so every frame has the same size
        W = self.h * 4 // 3
        s = min(self.h / img.shape[0], W / img.shape[1])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
        top = np.full((self.h, W, 3), 70, np.uint8)
        oy, ox = (self.h - img.shape[0]) // 2, (W - img.shape[1]) // 2
        top[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
        top = top[..., ::-1]  # BGR drawing colours → RGB
        if chase is not None:
            if rays:
                chase = self._rays_on_chase(chase, ranges)
            if chase.shape[0] != self.h:
                chase = cv2.resize(chase, (int(chase.shape[1] * self.h / chase.shape[0]), self.h))
            out = np.concatenate([chase, top], axis=1)
        else:
            out = top
        out = np.ascontiguousarray(out)

        def text(s, org):
            cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(hud):
            text(line, (10, 22 + 20 * i))
        if footer:
            text(footer, (10, out.shape[0] - 12))
        return out


# ── Live-window controls ──────────────────────────────────────────────────────

# cv2.waitKeyEx codes of the right arrow: macOS (Cocoa), Linux (GTK/Qt), Windows
RIGHT_ARROW = {63235, 65363, 2555904}
SPEEDS = [1, 2, 4, 8, 0]                 # × real time; 0 = as fast as possible


class UI:
    """Key handling + real-time pacing for the live window.

    s      cycle speed 1× → 2× → 4× → 8× → max → 1×
           (k× = every k-th substep is displayed, paced at k× real time)
    l      toggle LiDAR rays (2D map + projected in the 3D view)
    →, n   next episode       space  pause       q  quit
    """
    FOOTER = "s: speed   l: lidar rays   ->: next episode   space: pause   q: quit"

    def __init__(self, rays=False):
        self.i, self.rays, self.paused, self.anchor = 0, rays, False, None

    @property
    def mult(self):
        return SPEEDS[self.i]

    @property
    def stride(self):                    # display every `stride`-th substep
        return self.mult or 8

    @property
    def label(self):
        return f"{self.mult}x" if self.mult else "max"

    def reset_clock(self):
        self.anchor = None

    def wait(self, sim_t):
        """Pace to real time × speed while polling keys.
        Returns None (continue), 'redraw', 'skipped' or 'quit'."""
        while True:
            now = time.perf_counter()
            if self.anchor is None:
                self.anchor = (now, sim_t)
            due = now if not self.mult else self.anchor[0] + (sim_t - self.anchor[1]) / self.mult
            if now - due > 0.5:                      # far behind (slow render): re-anchor
                self.anchor, due = (now, sim_t), now
            delay = 50 if self.paused else max(1, int((due - now) * 1000))
            k = cv2.waitKeyEx(delay)
            if k in (ord("q"), 27):
                return "quit"
            if k in RIGHT_ARROW or k == ord("n"):
                return "skipped"
            if k == ord(" "):
                self.paused = not self.paused
                self.reset_clock()
            elif k == ord("s"):
                self.i = (self.i + 1) % len(SPEEDS)
                self.reset_clock()
                if self.paused:
                    return "redraw"
            elif k == ord("l"):
                self.rays = not self.rays
                if self.paused:
                    return "redraw"
            if not self.paused and time.perf_counter() >= due:
                return None


# ── Episode loop ──────────────────────────────────────────────────────────────

def run_episode(world, act, args, rng, start, theta0, goal, writer, ui, eid=0):
    world.set_robot(start, theta0)
    goal_xy = hab_to_planar(goal)
    path3, geo = shortest_path(world.pf_robot, start, goal)
    route = hab_to_planar(path3.T).T if path3 is not None else np.array([world.pose[:2], goal_xy])
    # Humans: fixed count/speed → spawned once per scene and re-targeted;
    # a range ("2-8", "0.6-1.4") → re-sampled and re-spawned every episode.
    (hlo, hhi), (slo, shi) = args.humans_rng, args.speed_rng
    n_h = int(rng.integers(hlo, hhi + 1))
    if (hlo != hhi or slo != shi) and world.humans:
        world.clear_humans()
    cross = [route[int(i)] for i in np.linspace(len(route) * 0.3, len(route) * 0.8, n_h // 2 + 1)[:-1].astype(int)] \
        if n_h else []
    cross = [np.array([p[0], world.y_floor, -p[1]]) for p in cross]

    n_sub = args.substeps
    if not world.humans and n_h:
        world.spawn_humans(n_h, rng, DT / n_sub, (slo, shi),
                           avoid_xy=[world.pose[:2], goal_xy], cross_points=cross)
        print(f"  humans: {len(world.humans)}  speed " +
              " ".join(f"{h.speed:.2f}" for h in world.humans) + " m/s")
    else:
        world.retarget_humans(cross)

    obsb = ObsBuilder(args.max_v, not args.clean_lidar, rng)
    max_t = args.max_time or (3.0 * geo / args.max_v + 30.0)
    v = w = 0.0
    t, path_len, n_static, n_close, n_human, min_h = 0.0, 0.0, 0, 0, 0, math.inf
    reasons, v_hist = {}, []
    dump = {"obs": [], "ranges": [], "pose": [], "act": []}
    status, viewer = "timeout", (Viewer(world, args.view_h) if (writer or ui) else None)
    frame_i = 0
    if ui is not None:
        ui.reset_clock()

    def policy_goal():
        if args.goal_mode == "direct":
            return goal_xy, None, route
        p3, _ = shortest_path(world.pf_robot, world.robot_hab, goal)
        if p3 is None:
            return goal_xy, None, route
        r = hab_to_planar(p3.T).T
        c = carrot(r, args.lookahead)
        return c, c, r

    ranges = world.lidar(MAX_LIDAR)
    pg, cxy, route = policy_goal()
    obs = obsb.reset(ranges, world.pose.copy(), pg)
    while t < max_t:
        a = np.asarray(act(obs, args.max_v))
        v, w = float(a[0]), float(a[1])
        v_hist.append((v, w))
        if args.dump_obs:
            dump["obs"].append(np.asarray(obs)); dump["ranges"].append(ranges)
            dump["pose"].append(world.pose.copy()); dump["act"].append((v, w))
        blocked = False
        for s in range(n_sub):
            prev = world.pose[:2].copy()
            b = world.move_robot(v, w, DT / n_sub)
            if b:
                reasons[world.block_reason] = reasons.get(world.block_reason, 0) + 1
            blocked |= b
            path_len += np.linalg.norm(world.pose[:2] - prev)
            world.step_humans()
            if viewer is not None and (s == n_sub - 1 or not args.fast):
                frame_i += 1
                sim_t = t + (s + 1) * DT / n_sub
                display = ui is not None and frame_i % ui.stride == 0
                if writer is None and not display:
                    continue                      # fast-forward: skip rendering entirely

                def render():
                    rays = ui.rays if ui is not None else args.rays
                    hud = [f"t={sim_t:5.1f}s  v={v:.2f} m/s  w={w:+.2f} rad/s",
                           f"goal {np.linalg.norm(goal_xy - world.pose[:2]):.2f} m  ({args.goal_mode})",
                           f"wall-push {n_static}  scan<{TRAIN_ROBOT_R:.1f}m {n_close}  human contact {n_human}",
                           f"min human dist {min_h:.2f} m"]
                    if ui is not None:
                        hud.append(f"speed {ui.label}  lidar rays {'ON' if rays else 'off'}"
                                   + ("  [PAUSED]" if ui.paused else ""))
                    return viewer.frame(world.chase_rgb() if world.render else None, route, goal_xy, cxy,
                                        ranges, hud, rays=rays, footer=UI.FOOTER if ui is not None else None)
                fr = render()
                if writer is not None:
                    writer.append_data(fr)
                while display:
                    cv2.imshow("legnav @ habitat", fr[..., ::-1])
                    r = ui.wait(sim_t)
                    if r in ("quit", "skipped"):
                        return r, {}
                    if r != "redraw":
                        break
                    fr = render()
        t += DT
        n_static += blocked
        ranges = world.lidar(MAX_LIDAR)
        min_r = float(ranges.min())
        n_close += min_r < TRAIN_ROBOT_R
        hit_human = False
        if world.humans:
            dh = min(np.linalg.norm(hab_to_planar(h.pos) - world.pose[:2]) for h in world.humans)
            min_h = min(min_h, dh)
            hit_human = dh < world.robot_r + HUMAN_BODY_R
            n_human += hit_human
        if np.linalg.norm(goal_xy - world.pose[:2]) < GOAL_RADIUS:
            status = "success"
            break
        if args.stop_on_collision and (blocked or hit_human or min_r < TRAIN_ROBOT_R):
            status = "collision"
            break
        pg, cxy, route = policy_goal()
        obs = obsb.step(ranges, world.pose.copy(), pg, v, w)

    spl = (geo / max(path_len, geo)) if status == "success" else 0.0
    if args.dump_obs and dump["obs"]:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_obs)) or ".", exist_ok=True)
        np.savez_compressed(f"{args.dump_obs}_ep{eid}.npz", goal=goal_xy, status=status,
                            **{k: np.array(v_) for k, v_ in dump.items()})
    vh = np.array(v_hist) if v_hist else np.zeros((1, 2))
    return status, dict(time=t, geodesic=geo, path=path_len, spl=spl, static_hits=n_static,
                        close_steps=n_close, human_contacts=n_human, min_human=min_h,
                        block_reasons=reasons, mean_v=float(vh[:, 0].mean()),
                        mean_abs_w=float(np.abs(vh[:, 1]).mean()))


def main():
    home = os.path.expanduser("~/habitat_data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="scene .glb (random start/goal)")
    src.add_argument("--scene-dir", help="folder of scenes (e.g. HM3D val): random scene + random start/goal")
    src.add_argument("--episodes", help="Social-HM3D episodes .json.gz")
    ap.add_argument("--scene-root", default=f"{home}/data/scene_datasets", help="prefix for episode scene_id")
    ap.add_argument("--scene-dataset-config", default=None)
    ap.add_argument("--episode-ids", type=int, nargs="*", default=None, help="episode indices (default: first N)")
    ap.add_argument("--num-episodes", type=int, default=3, help="0 = endless (Ctrl+C / q to stop)")
    ap.add_argument("--shuffle", action="store_true", help="--episodes: random scene, then random episodes of it")
    ap.add_argument("--episodes-per-scene", type=int, default=5,
                    help="--shuffle / --scene-dir: episodes before switching scene (scene load is the slow part)")
    ap.add_argument("--csv", default=None, help="append one row per episode to this CSV")
    ap.add_argument("--ckpt", default="checkpoints/ppo/ppo_legs_best.msgpack")
    ap.add_argument("--no-se2", action="store_true", help="checkpoint trained without SE(2) reprojection")
    ap.add_argument("--data-dir", default=f"{home}/data")
    ap.add_argument("--robot-urdf", default=f"{home}/robots/turtlebot4.urdf")
    ap.add_argument("--max-v", type=float, default=0.31, help="TurtleBot4 max linear speed (policy input)")
    ap.add_argument("--goal-mode", choices=["waypoint", "direct"], default="waypoint")
    ap.add_argument("--lookahead", type=float, default=2.0)
    ap.add_argument("--num-humans", default="4", help="N, or range 'A-B' re-sampled every episode")
    ap.add_argument("--human-speed", default="1.0", help="m/s, or range 'A-B' sampled per human")
    ap.add_argument("--min-dist", type=float, default=4.0)
    ap.add_argument("--max-dist", type=float, default=12.0)
    ap.add_argument("--max-time", type=float, default=None, help="s; default 3·geodesic/max_v + 30")
    ap.add_argument("--lidar-x", type=float, default=0.0,
                    help="lidar forward offset (m). Training casts from the robot centre (0); TB4 URDF: -0.04")
    ap.add_argument("--lidar-height", type=float, default=0.193, help="TB4 URDF rplidar_link height")
    ap.add_argument("--clean-lidar", action="store_true", help="disable training-style LiDAR noise")
    ap.add_argument("--stop-on-collision", action="store_true")
    ap.add_argument("--substeps", type=int, default=5, help="render/human substeps per 0.15 s policy step")
    ap.add_argument("--fast", action="store_true", help="render one frame per policy step")
    ap.add_argument("--show", action="store_true",
                    help="live window (s: speed, l: lidar rays, right arrow: next episode, space: pause, q: quit)")
    ap.add_argument("--rays", action="store_true", help="start with LiDAR rays drawn (also in --save videos)")
    ap.add_argument("--save", default=None, help="output .mp4")
    ap.add_argument("--view-h", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-obs", default=None, help="prefix: save per-episode obs/ranges/pose/action .npz")
    args = ap.parse_args()
    args.humans_rng = parse_range(args.num_humans, int)
    args.speed_rng = parse_range(args.human_speed, float)

    rng = np.random.default_rng(args.seed)
    act = load_policy(args.ckpt, se2=not args.no_se2)
    ui = UI(rays=args.rays) if args.show else None
    render = bool(args.show or args.save)
    writer = None
    if args.save:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fps = (1.0 / DT) * (1 if args.fast else args.substeps)
        writer = imageio.get_writer(args.save, fps=fps, codec="libx264", quality=7, macro_block_size=8)

    csv_f = csv_w = None
    if args.csv:
        new = not os.path.exists(args.csv)
        csv_f = open(args.csv, "a", newline="")
        csv_w = csv.writer(csv_f)
        if new:
            csv_w.writerow(["n", "scene", "episode", "status", "time_s", "geodesic_m", "path_m", "spl",
                            "wall_push", "scan_lt_0.2", "human_contacts", "min_human_m", "mean_v", "humans"])

    world, cur_scene, results = None, None, []

    def summary():
        if results:
            sr = np.mean([s_ == "success" for s_, _ in results])
            spl = np.mean([m_["spl"] for _, m_ in results])
            hc = np.mean([m_["human_contacts"] > 0 for _, m_ in results])
            return f"{len(results)} episodes  SR={sr:.2f}  SPL={spl:.2f}  episodes with human contact={hc:.2f}"
        return "no completed episodes"

    try:
        for scene, s, th, g, eid in episode_stream(args, rng):
            if scene != cur_scene:
                if world is not None:
                    world.close()
                t0 = time.time()
                world = HabitatWorld(scene, args.data_dir, args.robot_urdf, args.scene_dataset_config,
                                     lidar_height=args.lidar_height, lidar_x=args.lidar_x, render=render)
                cur_scene = scene
                print(f"scene {os.path.basename(scene)} loaded in {time.time() - t0:.1f}s")
            try:
                if s is None:
                    s, th, g = random_episode(world, rng, args.min_dist, args.max_dist)
                else:
                    s, g = world.pf_robot.snap_point(s), world.pf_robot.snap_point(g)
            except RuntimeError as e:                    # e.g. scene too small for the distance range
                print(f"  skip ({e})")
                continue
            t0 = time.time()
            status, m = run_episode(world, act, args, rng, s, th, g, writer, ui, eid)
            if status == "quit":
                break
            if not m:                                    # skipped with → / n
                print(f"ep {eid}: skipped")
                continue
            results.append((status, m))
            print(f"ep {eid}: {status:9s} t={m['time']:5.1f}s geo={m['geodesic']:.1f}m path={m['path']:.1f}m "
                  f"SPL={m['spl']:.2f} wall_push={m['static_hits']} scan<0.2={m['close_steps']} "
                  f"human_contact={m['human_contacts']} "
                  f"min_human={m['min_human']:.2f}m  (wall {time.time() - t0:.1f}s)\n"
                  f"        mean v={m['mean_v']:.2f} |w|={m['mean_abs_w']:.2f}  blocks={m['block_reasons']}\n"
                  f"        running: {summary()}")
            if csv_w is not None:
                csv_w.writerow([len(results), os.path.basename(scene), eid, status, f"{m['time']:.2f}",
                                f"{m['geodesic']:.2f}", f"{m['path']:.2f}", f"{m['spl']:.3f}", m["static_hits"],
                                m["close_steps"], m["human_contacts"], f"{m['min_human']:.2f}",
                                f"{m['mean_v']:.3f}", len(world.humans)])
                csv_f.flush()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if writer is not None:
            writer.close()
            print("saved", args.save)
        if csv_f is not None:
            csv_f.close()
        if world is not None:
            world.close()
        cv2.destroyAllWindows()
    print("\n" + summary())


if __name__ == "__main__":
    main()
