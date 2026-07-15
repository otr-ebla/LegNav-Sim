"""
floor_tour_eval.py — Floor 16 Complete Tour Evaluation
=======================================================
Tests the policy on the real-world floor plan (scenario 16, 42×67 m).
The goal is to complete a "tour": visit a sequence of waypoints spread
across the floor in order, without collision or timeout.

Usage:
  python3 floor_tour_eval.py --algo ppo  --ckpt checkpoints/ppo_tanh_inside_final.msgpack
  python3 floor_tour_eval.py --algo sac  --ckpt checkpoints_sac/sac_best.msgpack
  python3 floor_tour_eval.py --algo tqc  --ckpt checkpoints_tqc/tqc_final.msgpack

Tours (select with keyboard numbers 1-8):
  1 — Double CCW       : two counterclockwise laps (endurance)
  2 — Single CCW       : one counterclockwise lap
  3 — Single CW        : one clockwise lap
  4 — Double CW        : two clockwise laps (endurance)
  5 — Single CCW+People: one CCW lap with 8 corridor patrollers
  6 — Single CW+People : one CW  lap with 8 corridor patrollers
  7 — Double CCW+People: two CCW laps with 8 corridor patrollers
  8 — Double CW+People : two CW  laps with 8 corridor patrollers

Keys:
  1-8   Select tour   R   Reset tour (restart from WP0)
  N     Skip current waypoint (debug)
  L     Toggle LiDAR  H   Toggle arrows
  B     Toggle body ring  S  Cycle FPS
  Q/Esc Quit
"""

import argparse
import os
from legnav import paths
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import pygame
import numpy as np
import jax
import jax.numpy as jnp

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Floor 16 Tour Evaluation")
    p.add_argument("--algo", default="ppo",
                   choices=["ppo", "shac", "sac", "tqc", "mlp", "hsfm", "mppi", "dwa", "ppo_circles", "ppo_legs"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--legs",    dest="use_legs", action="store_true", default=True)
    p.add_argument("--no-legs", dest="use_legs", action="store_false")
    p.add_argument("--ghost-body", action="store_true")
    p.add_argument("--tour", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                   help="Starting tour: 1=DoubleCCW 2=SingleCCW 3=SingleCW 4=DoubleCW "
                        "5=SingleCCW+People 6=SingleCW+People 7=DoubleCCW+People 8=DoubleCW+People. Default: 1")
    return p.parse_args()

args = _parse_args()

import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS     = args.use_legs
_jax_env.SENSOR_NOISE = True
_jax_env.PEOPLE_RADIUS = 0.25   # map floor scenario uses smaller people radius

from legnav.core.jax_env import (ROBOT_RADIUS, PEOPLE_RADIUS,
                                  NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS)
from legnav.core.jax_env_multi import reset_env, step_env, NUM_PEOPLE as _NUM_PEOPLE
from legnav.core.jax_wrappers import make_stacked_env, assemble_stacked_obs
from legnav.core.jax_scenarios import MAP_ROOM_W, MAP_ROOM_H
from legnav.core.jax_legs import init_foot_state as _init_foot_state

# Reuse rendering + policy builders from the main eval script without triggering
# its own argparse (which would reject our flags).
import sys as _sys
_saved_argv = _sys.argv
_sys.argv = _sys.argv[:1]
from legnav.evaluation.jax_eval_multi import (
    draw_scene, draw_panel, make_fonts, build_policy, _DEFAULT_CKPT,
    C_BG, C_SUCCESS, C_COLLIDE, C_TEXT,
    C_WALL, C_PANEL, PANEL_W, draw_star,
)
_sys.argv = _saved_argv

OBS_SIZE     = 659
ACTION_DIM   = 2
SCENARIO_IDX = 16
# Goal-dist budget: diagonal of the 42×67 m floor ≈ 79 m; set a safe ceiling.
MAX_GOAL_DIST = 80.0

# ── Tour configurations ───────────────────────────────────────────────────────
# All tours circuit the big central empty rectangle visible in the floor plan.
# The rectangle is bounded by four corridors:
#   Bottom: y ≈ 26,  x ∈ [13, 37]
#   Right:  x ≈ 37,  y ∈ [26, 55]
#   Top:    y ≈ 55,  x ∈ [13, 37]
#   Left:   x ≈ 13,  y ∈ [26, 55]
# Robot spawn: (13.26, 26.12)  ← bottom corridor, left side.
#
# CCW (counterclockwise in world coords): spawn → right along bottom →
#   up right corridor → left along top → down left corridor → back near spawn.
# CW  (clockwise): spawn → up left corridor → right along top →
#   down right corridor → left along bottom → back near spawn.

_CCW = [
    (25.0, 26.0),   # bottom corridor, mid
    (35.0, 26.0),   # bottom corridor, right end
    (37.5, 29.0),   # right corridor, lower
    (37.5, 36.0),   # right corridor, lower
    (37.5, 47.0),   # right corridor, upper
    (37.5, 54.0),   # top corridor, right
    (25.0, 55.0),   # top corridor, middle
    (13.0, 55.0),   # top corridor, left
    (13.0, 49.0),   # left corridor, upper
    (13.0, 36.0),   # left corridor, lower
    (13.0, 29.0),   # back near spawn
]

_CW = [
    (13.0, 36.0),   # left corridor, lower
    (13.0, 49.0),   # left corridor, upper
    (13.0, 55.0),   # top corridor, left
    (25.0, 55.0),   # top corridor, middle
    (37.5, 54.0),   # top corridor, right
    (37.5, 47.0),   # right corridor, upper
    (37.5, 36.0),   # right corridor, lower
    (37.5, 29.0),   # right corridor, lower
    (35.0, 26.0),   # bottom corridor, right end
    (25.0, 26.0),   # bottom corridor, mid
    (13.0, 29.0),   # back near spawn
]

TOUR_CONFIGS = {
    1: {
        "name": "Double CCW",
        "desc": "Two counterclockwise laps: bottom→right→top→left (endurance)",
        "waypoints": _CCW + _CCW,
        "people": False,
    },
    2: {
        "name": "Single CCW",
        "desc": "One counterclockwise lap: bottom→right→top→left",
        "waypoints": _CCW,
        "people": False,
    },
    3: {
        "name": "Single CW",
        "desc": "One clockwise lap: bottom→left→top→right",
        "waypoints": _CW,
        "people": False,
    },
    4: {
        "name": "Double CW",
        "desc": "Two clockwise laps: bottom→left→top→right (endurance)",
        "waypoints": _CW + _CW,
        "people": False,
    },
    5: {
        "name": "Single CCW+People",
        "desc": "One CCW lap with 8 corridor patrollers",
        "waypoints": _CCW,
        "people": True,
    },
    6: {
        "name": "Single CW+People",
        "desc": "One CW lap with 8 corridor patrollers",
        "waypoints": _CW,
        "people": True,
    },
    7: {
        "name": "Double CCW+People",
        "desc": "Two CCW laps with 8 corridor patrollers",
        "waypoints": _CCW + _CCW,
        "people": True,
    },
    8: {
        "name": "Double CW+People",
        "desc": "Two CW laps with 8 corridor patrollers",
        "waypoints": _CW + _CW,
        "people": True,
    },
}

# ── Patrol-people helpers ──────────────────────────────────────────────────────
# 8 people total: 2 per corridor (bottom, right, top, left).
# Horizontal corridors (bottom/top): spawn at random x between the two endpoints.
# Vertical corridors (left/right):   spawn at random y between the two endpoints.

_PATROL_SPEED = 1.0   # m/s desired patrol speed

# Each entry: (g1x, g1y, g2x, g2y, axis)
# axis='h' → horizontal corridor, spawn x randomly; axis='v' → vertical.
_PATROL_CORRIDORS = [
    # bottom: y=26, x ∈ [13, 37]
    (13.0, 26.0, 37.0, 26.0, "h"),
    (13.0, 26.0, 37.0, 26.0, "h"),
    # right: x=37.5, y ∈ [26, 55]
    (37.5, 26.0, 37.5, 55.0, "v"),
    (37.5, 26.0, 37.5, 55.0, "v"),
    # top: y=55, x ∈ [13, 37.5]
    (13.0, 55.0, 37.5, 55.0, "h"),
    (13.0, 55.0, 37.5, 55.0, "h"),
    # left: x=13, y ∈ [26, 55]
    (13.0, 26.0, 13.0, 55.0, "v"),
    (13.0, 26.0, 13.0, 55.0, "v"),
]


def _make_patrol_people_arr():
    """Build a (NUM_PEOPLE, 11) JAX array with 8 active patrollers + dummies.

    People format (col indices):
      0:px  1:py  2:vx  3:vy  4:theta  5:distracted
      6:g1x  7:g1y  8:g2x  9:g2y  10:wp_idx  (>=0 active, <0 dummy)
    """
    rows = []
    for g1x, g1y, g2x, g2y, axis in _PATROL_CORRIDORS:
        wp_idx = float(np.random.randint(0, 2))
        if axis == "h":
            px = float(np.random.uniform(g1x, g2x))
            py = g1y
        else:
            px = g1x
            py = float(np.random.uniform(g1y, g2y))
        gx = g1x if wp_idx == 0 else g2x
        gy = g1y if wp_idx == 0 else g2y
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy) + 1e-7
        theta = math.atan2(dy, dx)
        vx = _PATROL_SPEED * dx / dist
        vy = _PATROL_SPEED * dy / dist
        rows.append([px, py, vx, vy, theta, 0.0, g1x, g1y, g2x, g2y, wp_idx])

    dummy = [-999.0, -999.0, 0.0, 0.0, 0.0, 0.0, -999.0, -999.0, -999.0, -999.0, -1.0]
    for _ in range(_NUM_PEOPLE - len(rows)):
        rows.append(dummy)

    return jnp.array(rows, dtype=jnp.float32)

# ── Env factory ───────────────────────────────────────────────────────────────

def build_fast_reset():
    bound_reset = (lambda key, max_goal_dist=3.0, scenario_idx=-1,
                   ghost_prob=0.0, min_goal_dist=0.8, **kw:
                   reset_env(key, MAX_GOAL_DIST, scenario_idx=SCENARIO_IDX,
                             ghost_prob=0.0, min_goal_dist=1.0, **kw))
    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
    jit_rs = jax.jit(lambda key: rs(key, MAX_GOAL_DIST, ghost_prob=0.0, min_goal_dist=1.0))
    jit_ss = jax.jit(ss)
    return jit_rs, jit_ss


# ── Waypoint rendering ────────────────────────────────────────────────────────

_WP_PALETTE = [
    (255, 100, 100),   # red
    (255, 165,  30),   # orange
    (240, 220,  40),   # yellow
    ( 70, 220,  80),   # green
    ( 60, 180, 255),   # blue
    (200,  80, 255),   # purple
    (255, 120, 180),   # pink
]

_YELLOW_BRIGHT  = (255, 230,  30)
_YELLOW_DIM     = ( 60,  55,   5)

def draw_waypoints(surface, waypoints, wp_idx, scale, sim_h):
    """Draw all waypoints as yellow stars; current target is brightest, visited are dark."""
    def W_(x, y): return int(x * scale), int(sim_h - y * scale)
    for i, (wx, wy) in enumerate(waypoints):
        sx, sy = W_(wx, wy)
        if i < wp_idx:
            col, border = _YELLOW_DIM, (30, 30, 10)
        elif i == wp_idx:
            col, border = _YELLOW_BRIGHT, (255, 255, 120)
        else:
            col, border = (180, 160, 20), (100, 90, 10)
        draw_star(surface, sx, sy, 6, 3, 5, col, border)


def draw_tour_overlay(surface, fonts, tour_cfg, wp_idx,
                      sim_x, win_h, wp_results):
    """Draw a compact tour-progress bar below the main panel."""
    n_wps   = len(tour_cfg["waypoints"])
    bar_x   = sim_x + 10
    bar_w   = PANEL_W - 20
    bar_y   = win_h - 60
    cell_w  = bar_w // max(n_wps, 1)

    pygame.draw.line(surface, C_WALL, (sim_x, bar_y - 8),
                     (sim_x + PANEL_W, bar_y - 8), 1)

    fnt = fonts["small"]
    lbl = fnt.render(f"Tour: {tour_cfg['name']}", True, C_TEXT)
    surface.blit(lbl, (bar_x, bar_y - 26))

    outcomes = {r[0]: r[3] for r in wp_results}
    for i in range(n_wps):
        cx = bar_x + i * cell_w + cell_w // 2
        cy = bar_y + 12
        base = _WP_PALETTE[i % len(_WP_PALETTE)]
        if i in outcomes:
            col = C_SUCCESS if outcomes[i] == "reached" else C_COLLIDE
        elif i == wp_idx:
            col = base
        else:
            col = tuple(max(0, c - 80) for c in base)
        pygame.draw.circle(surface, col, (cx, cy), 8)
        pygame.draw.circle(surface, C_WALL, (cx, cy), 8, 1)
        num = fnt.render(str(i + 1), True, (10, 10, 10))
        surface.blit(num, (cx - num.get_width() // 2,
                           cy - num.get_height() // 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    algo     = args.algo
    ckpt     = args.ckpt or _DEFAULT_CKPT.get(algo, "")
    use_legs = args.use_legs

    print(f"\nFloor Tour Eval — algo={algo.upper()}, scenario=16 ({MAP_ROOM_W:.0f}×{MAP_ROOM_H:.0f} m)")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Human model: {'LEGS' if use_legs else 'CYLINDERS'}\n")

    init_params, load_fn, infer_fn, _ = build_policy(algo)
    try:
        params = load_fn(ckpt)
        print(f"Loaded {algo.upper()} weights from {ckpt}")
    except FileNotFoundError:
        params = init_params
        print("Checkpoint not found — running with random weights.")

    _policy_reset = getattr(infer_fn, "reset_hook", lambda: None)

    fast_reset, fast_step = build_fast_reset()
    rng = jax.random.PRNGKey(42)

    # ── Window sizing — zoom to waypoint bounding box ─────────────────────────
    VIEWPORT_PAD = 2.5   # metres of margin around all waypoints
    _all_wps = _CCW + _CW
    vp_xmin = max(0.0,            min(wx for wx, _ in _all_wps) - VIEWPORT_PAD)
    vp_xmax = min(float(MAP_ROOM_W), max(wx for wx, _ in _all_wps) + VIEWPORT_PAD)
    vp_ymin = max(0.0,            min(wy for _, wy in _all_wps) - VIEWPORT_PAD)
    vp_ymax = min(float(MAP_ROOM_H), max(wy for _, wy in _all_wps) + VIEWPORT_PAD)

    SIM_TARGET = 800
    zoom     = SIM_TARGET / max(vp_xmax - vp_xmin, vp_ymax - vp_ymin)

    # full-floor offscreen at zoom scale
    off_w    = int(MAP_ROOM_W * zoom)
    off_h    = int(MAP_ROOM_H * zoom)
    # pixel crop within offscreen (row 0 = world top because y is flipped)
    crop_x   = int(vp_xmin * zoom)
    crop_y   = int((MAP_ROOM_H - vp_ymax) * zoom)
    sim_w_px = int((vp_xmax - vp_xmin) * zoom)
    sim_h_px = int((vp_ymax - vp_ymin) * zoom)

    win_w    = sim_w_px + PANEL_W
    win_h    = sim_h_px

    pygame.init()
    screen    = pygame.display.set_mode((win_w, win_h))
    offscreen = pygame.Surface((off_w, off_h))
    pygame.display.set_caption(f"Floor Tour Eval — {algo.upper()}")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    # ── State ─────────────────────────────────────────────────────────────────
    current_tour = max(1, min(8, args.tour))
    tour_cfg     = TOUR_CONFIGS[current_tour]
    waypoints    = tour_cfg["waypoints"]

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)
    _policy_reset()

    wp_idx           = 0
    wp_results: list = []    # (wp_idx, reward, steps, outcome)
    wp_seg_reward    = 0.0
    wp_seg_steps     = 0

    TRAJ_MIN_DIST = 0.05
    def _init_traj(ss):
        es = jax.device_get(ss.env_state)
        return [(float(es.x), float(es.y), float(es.v))]
    trajectory = _init_traj(stacked_state)

    _REW_KEYS  = ["rew_progress", "rew_step", "rew_smooth", "rew_yield"]
    ep         = 0
    ep_steps   = 0
    ep_reward  = 0.0
    ep_hist: list = []
    rew_acc    = {k: 0.0 for k in _REW_KEYS}
    ep_yz_steps = 0
    ep_yc_steps = 0
    session_yield_sum   = 0.0
    session_yield_count = 0

    YIELD_DIST   = 1.8
    YIELD_FOV    = math.pi / 4
    YIELD_STOP_V = 0.1

    paused     = False
    show_lidar = True
    show_arrows = True
    show_body  = args.ghost_body
    show_radar = True
    fps_speeds = [10, 20, 30, 60]
    fps_idx    = 0
    current_fps = fps_speeds[fps_idx]
    banner     = ""
    banner_t   = 0

    def get_stats():
        if not ep_hist: return {"suc": 0., "col": 0., "tmo": 0., "pcol": 0.}
        w = np.array(ep_hist[-50:])
        return {"suc": w[:, 1].mean()*100, "col": w[:, 2].mean()*100,
                "tmo": w[:, 3].mean()*100, "pcol": w[:, 4].mean()*100}

    # Inject the first waypoint as the initial goal in the env state.
    # The scenario resets with its own default goal; we override it here
    # so the panel and observation already point at WP0 from the start.
    def _inject_goal(ss, gx, gy, theta=None):
        """Replace goal_x/goal_y (and optionally theta) and recompute obs."""
        nonlocal obs, stacked_state, rng
        from legnav.core.jax_env import get_obs as _get_obs
        replace_kw = dict(
            goal_x=jnp.float32(gx),
            goal_y=jnp.float32(gy),
            time_step=jnp.int32(0),
        )
        if theta is not None:
            replace_kw["theta"] = jnp.float32(theta)
        es = ss.env_state.replace(**replace_kw)
        rng, obs_key = jax.random.split(rng)
        new_base_obs, sp_mask = _get_obs(es, obs_key)
        es = es.replace(sp_mask=sp_mask)
        ss = ss.replace(env_state=es)
        goal     = new_base_obs[:2]
        kin_vec  = new_base_obs[2:7]
        lidar    = new_base_obs[7:]
        ss = ss.replace(
            goal_stack=ss.goal_stack.at[-1].set(goal),
            lidar_stack=ss.lidar_stack.at[-1].set(lidar),
        )
        obs = assemble_stacked_obs(kin_vec, ss.goal_stack, ss.pose_stack, ss.lidar_stack)
        stacked_state = ss

    def _wp0_theta(ss, wps):
        """Angle from current robot position to the first waypoint."""
        es = ss.env_state
        dx = wps[0][0] - float(es.x)
        dy = wps[0][1] - float(es.y)
        return math.atan2(dy, dx)

    def _inject_patrol_people():
        """Replace env_state.people with fresh patrol patrollers and reinit foot_state."""
        nonlocal stacked_state, rng
        people_arr = _make_patrol_people_arr()
        rng, ft_key = jax.random.split(rng)
        new_foot = _init_foot_state(people_arr, ft_key)
        es = stacked_state.env_state.replace(people=people_arr, foot_state=new_foot)
        stacked_state = stacked_state.replace(env_state=es)

    # Point the robot at waypoint 0 right after the initial reset.
    _inject_goal(stacked_state, *waypoints[0], theta=_wp0_theta(stacked_state, waypoints))
    if tour_cfg["people"]:
        _inject_patrol_people()

    def reset_tour():
        nonlocal obs, stacked_state, rng, wp_idx, wp_seg_reward, wp_seg_steps
        nonlocal wp_results, ep_reward, ep_steps, rew_acc, trajectory
        nonlocal ep_yz_steps, ep_yc_steps
        rng, reset_rng = jax.random.split(rng)
        obs, stacked_state = fast_reset(reset_rng)
        _policy_reset()
        wp_idx       = 0
        wp_results   = []
        wp_seg_reward = 0.0
        wp_seg_steps  = 0
        ep_reward     = 0.0
        ep_steps      = 0
        rew_acc       = {k: 0.0 for k in _REW_KEYS}
        ep_yz_steps   = 0
        ep_yc_steps   = 0
        trajectory    = _init_traj(stacked_state)
        _inject_goal(stacked_state, *waypoints[0], theta=_wp0_theta(stacked_state, waypoints))
        if tour_cfg["people"]:
            _inject_patrol_people()

    def advance_waypoint():
        """Record result and advance wp_idx. Returns True if there are more WPs."""
        nonlocal wp_idx, wp_seg_reward, wp_seg_steps, ep_reward, ep_steps, rew_acc
        wp_results.append((wp_idx, wp_seg_reward, wp_seg_steps, "reached"))
        wp_idx += 1
        if wp_idx >= len(waypoints):
            return False   # tour complete
        _inject_goal(stacked_state, *waypoints[wp_idx])
        _policy_reset()
        wp_seg_reward = 0.0
        wp_seg_steps  = 0
        ep_reward     = 0.0
        ep_steps      = 0
        rew_acc       = {k: 0.0 for k in _REW_KEYS}
        return True

    print(f"Tour {current_tour}: {tour_cfg['name']} — {len(waypoints)} waypoints")
    print("Keys: 1-8 tour | R reset | N next WP | L lidar | H arrows | B body | Q quit")

    while True:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                if k == pygame.K_SPACE:
                    paused = not paused
                if k == pygame.K_l: show_lidar  = not show_lidar
                if k == pygame.K_h: show_arrows = not show_arrows
                if k == pygame.K_b: show_body   = not show_body
                if k == pygame.K_p: show_radar  = not show_radar
                if k == pygame.K_s:
                    fps_idx = (fps_idx + 1) % len(fps_speeds)
                    current_fps = fps_speeds[fps_idx]
                    banner = f"FPS: {current_fps}"; banner_t = 15
                if k == pygame.K_r:
                    reset_tour()
                    banner = "reset"; banner_t = 20
                if k == pygame.K_n:
                    if advance_waypoint():
                        banner = f"skip → WP{wp_idx + 1}"
                    else:
                        banner = "all WPs done (skipped)"
                    banner_t = 20
                # Tour selection (keys 1-8)
                for ki, ti in [(pygame.K_1, 1), (pygame.K_2, 2),
                               (pygame.K_3, 3), (pygame.K_4, 4),
                               (pygame.K_5, 5), (pygame.K_6, 6),
                               (pygame.K_7, 7), (pygame.K_8, 8)]:
                    if k == ki and ti != current_tour:
                        current_tour = ti
                        tour_cfg     = TOUR_CONFIGS[current_tour]
                        waypoints    = tour_cfg["waypoints"]
                        reset_tour()
                        banner = tour_cfg["name"]
                        banner_t = 30
                        print(f"Tour {current_tour}: {tour_cfg['name']} "
                              f"— {len(waypoints)} waypoints")

        if paused:
            clock.tick(10); continue

        # ── Inference ────────────────────────────────────────────────────────
        if algo == "hsfm":
            env_action = infer_fn(params, obs, stacked_state.env_state)
        else:
            env_action = infer_fn(params, obs, stacked_state.env_state.max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward   += float(reward); ep_steps   += 1
        wp_seg_reward += float(reward); wp_seg_steps += 1
        for rk in _REW_KEYS:
            rew_acc[rk] += float(info.get(rk, 0.0))

        # ── Per-step state ────────────────────────────────────────────────────
        cpu_state   = jax.device_get(stacked_state.env_state)
        raw_lidar   = (MAX_LIDAR_DIST
                       - jax.device_get(stacked_state.lidar_stack)[-1]
                       * (MAX_LIDAR_DIST - ROBOT_RADIUS))
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask       = np.array(cpu_state.sp_mask)

        cur_x = float(cpu_state.x); cur_y = float(cpu_state.y); cur_v = float(cpu_state.v)
        if (not trajectory
                or math.hypot(cur_x - trajectory[-1][0], cur_y - trajectory[-1][1]) >= TRAJ_MIN_DIST):
            trajectory.append((cur_x, cur_y, cur_v))
        elif cur_v <= 0.1 and trajectory[-1][2] > 0.1:
            trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], cur_v)

        # Yield accounting
        ppl_np = np.array(cpu_state.people)
        active_p = ppl_np[:, 10] >= 0.0
        dpx = ppl_np[:, 0] - cur_x
        dpy = ppl_np[:, 1] - cur_y
        dists_p = np.hypot(dpx, dpy)
        rel_ang = np.arctan2(dpy, dpx) - float(cpu_state.theta)
        rel_ang = (rel_ang + math.pi) % (2.0 * math.pi) - math.pi
        in_yz = (dists_p < YIELD_DIST) & (np.abs(rel_ang) < YIELD_FOV) & active_p
        if bool(np.any(in_yz)):
            ep_yz_steps += 1
            if cur_v <= YIELD_STOP_V:
                ep_yc_steps += 1

        dx = float(cpu_state.goal_x) - cur_x
        dy = float(cpu_state.goal_y) - cur_y
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        # ── Render ───────────────────────────────────────────────────────────
        offscreen.fill(C_BG)
        draw_scene(offscreen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs, show_body,
                   scale=zoom, sim_h=off_h,
                   trajectory=trajectory, show_map_bg=True)
        draw_waypoints(offscreen, waypoints, wp_idx, zoom, off_h)
        screen.fill(C_BG)
        screen.blit(offscreen, (0, 0), (crop_x, crop_y, sim_w_px, sim_h_px))
        draw_panel(screen, fonts, algo, ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), banner, banner_t,
                   SCENARIO_IDX, "fixed", use_legs, raw_lidar, sp_mask, rew_acc,
                   ep_yz_steps, ep_yc_steps,
                   session_yield_sum, session_yield_count,
                   show_radar=show_radar, sim_x=sim_w_px, win_h=sim_h_px)
        draw_tour_overlay(screen, fonts, tour_cfg, wp_idx,
                          sim_w_px, sim_h_px, wp_results)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip()
        clock.tick(current_fps)

        # ── Episode / waypoint logic ──────────────────────────────────────────
        if done:
            goal = bool(info["goal_reached"])
            col  = bool(info["collision"])
            pcol = bool(info["passive_col"])
            active_col = col and not pcol

            if goal:
                if advance_waypoint():
                    banner   = f"WP{wp_idx}/{len(waypoints)} reached!"
                    banner_t = 25
                    continue   # keep same episode, go to next WP
                else:
                    # All waypoints completed — pick a random different tour next.
                    print(f"\n  Ep {ep}: COMPLETE tour {current_tour} "
                          f"({len(waypoints)} WPs) "
                          f"| steps={ep_steps} reward={ep_reward:.1f}")
                    for r in wp_results:
                        print(f"    WP{r[0]+1}: reward={r[1]:.1f} steps={r[2]} → {r[3]}")
                    ep_hist.append((ep_reward, 1.0, 0.0, 0.0, 0.0))
                    if ep_yz_steps > 0:
                        session_yield_sum   += ep_yc_steps / ep_yz_steps
                        session_yield_count += 1
                    ep += 1
                    # Stay in the same people/no-people family when auto-switching.
                    same_family = [t for t in TOUR_CONFIGS
                                   if TOUR_CONFIGS[t]["people"] == tour_cfg["people"]
                                   and t != current_tour]
                    current_tour = int(np.random.choice(same_family))
                    tour_cfg  = TOUR_CONFIGS[current_tour]
                    waypoints = tour_cfg["waypoints"]
                    banner    = f"TOUR COMPLETE! → {tour_cfg['name']}"
                    banner_t  = 60
                    print(f"  Switching to tour {current_tour}: {tour_cfg['name']}")
                    reset_tour()
            else:
                outcome = ("collision" if active_col
                           else ("passive_col" if pcol else "timeout"))
                banner   = f"{outcome} at WP{wp_idx + 1}/{len(waypoints)}"
                banner_t = 40
                wp_results.append((wp_idx, wp_seg_reward, wp_seg_steps, outcome))
                ep_hist.append((ep_reward, 0.0, float(active_col), float(not goal and not col), float(pcol)))
                print(f"  Ep {ep}: {outcome} at WP{wp_idx + 1}/{len(waypoints)} "
                      f"| steps={ep_steps} reward={ep_reward:.1f}")
                if ep_yz_steps > 0:
                    session_yield_sum   += ep_yc_steps / ep_yz_steps
                    session_yield_count += 1
                ep += 1
                reset_tour()


if __name__ == "__main__":
    main()
