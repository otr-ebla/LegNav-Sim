"""
view_leg_scenarios.py — Visual gait-model check using the full jax_eval_multi
panel/rendering pipeline, but with no policy loaded.

Three dedicated scenarios are exposed for inspecting the jax_legs walk cycle:

  Scenario 13 — TREADMILL
    One person walks straight from the top of the room to the bottom. When
    they reach the bottom they teleport back to the top (the standard
    parallel-corridor teleport mechanism in jax_env_multi.py). The robot is
    parked in the bottom-right corner with max_v = 0 so it cannot move.

  Scenario 14 — CROSSING
    Two people walk against each other along the x-axis. Each person owns a
    pair of waypoints at the opposite walls, so when one reaches the wall
    the waypoint toggle (jax_env_multi.py:309–311) makes them walk back to
    the start — endlessly. Robot is again parked in the corner.

  Scenario 15 — CIRCLE
    One person walks on a constant-radius circle in the empty room. HSFM is
    bypassed for this person: pose is integrated directly from constant
    linear speed CIRCLE_V and angular speed CIRCLE_OMEGA (radius = V / Ω).
    Edit the two module-level constants below to change speed / radius.

Scenarios 13 and 14 are driven by HSFM (same as training); scenario 15 uses
pure unicycle kinematics so the gait model can be inspected on a curved path.

Keys (subset of jax_eval_multi.py — same look-and-feel):
  1     Lock scenario 13 (treadmill)
  2     Lock scenario 14 (crossing)
  3     Lock scenario 15 (circle)
  R     Reset current scenario
  Space Pause
  L     Toggle lidar rays
  H     Toggle arrows
  B     Toggle body ring
  P     Toggle radar panel
  Q/Esc Quit

Run:
  python view_leg_scenarios.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import sys
import pygame
import numpy as np
import jax
import jax.numpy as jnp

# Match jax_eval_multi.py: configure jax_env flags *before* importing it.
import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS    = True
_jax_env.SENSOR_NOISE = False     # clean readout for visual check

from legnav.core.jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS, DT)
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_legs import (LEG_RADIUS, SHOE_LENGTH, SHOE_WIDTH, STEP_FREQ,
                      STEP_SPEED, advance_feet)
from legnav.core.jax_wrappers import make_stacked_env

# ── Scenario 15 (CIRCLE) tunables ────────────────────────────────────────────
# Linear speed and angular speed of the lone walker. Radius = CIRCLE_V/CIRCLE_OMEGA
# (must fit inside the room — keep |V/Ω| <~ ROOM_W/2 - 1). Positive omega = CCW.
CIRCLE_V     = 1.0    # m/s
CIRCLE_OMEGA = 0.5    # rad/s

# ══════════════════════════════════════════════════════════════════════════════
# Rendering — copied verbatim from jax_eval_multi.py so the panel matches.
# ══════════════════════════════════════════════════════════════════════════════

SIM_SIZE   = 800
PANEL_W    = 300
WINDOW_W   = SIM_SIZE + PANEL_W
WINDOW_H   = SIM_SIZE
SCALE      = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET = 10
FPS_SPEEDS = [10, 20, 30]

C_BG        = (28,  28,  34); C_FLOOR    = (44,  44,  52); C_GRID     = (56,  56,  66)
C_WALL      = (190, 190, 205); C_ROBOT    = (60,  140, 255); C_ROBOT_H  = (170, 210, 255)
C_TRAJ      = (100, 200, 255)
C_GOAL      = (255, 210,  40); C_GOAL2    = (220, 150,  10)
C_LEG_L     = (90,  220, 100); C_LEG_R    = (50,  170,  65)
C_CIRCLE    = (140,  90,  60); C_CIRCLE_L = (180, 120,  80)
C_BOX       = ( 90, 110, 145); C_BOX_L    = (120, 145, 185)
C_RAY_FAR   = ( 50, 200,  50); C_RAY_NEAR = (220,  50,  50)
C_PANEL     = ( 20,  20,  26); C_TEXT     = (215, 215, 228)
C_DIM       = (120, 120, 138); C_SUCCESS  = ( 50, 215,  95)
C_COLLIDE   = (230,  60,  60); C_TIMEOUT  = (210, 165,  50)
C_BODY_RING = ( 50, 100,  50)

_SHOE_PALETTE = [
    (180,  60,  60), ( 60, 130, 220), (220, 160,  30), ( 60, 200, 160),
    (200,  80, 200), (100, 200,  60), (230, 120,  40), ( 80, 180, 230),
    (200, 200,  60), (160,  60, 200), ( 50, 200, 100), (220,  80, 130),
    ( 60, 160, 160), (200, 140,  80), (130,  80, 200), (200,  60,  80),
]


def W(x, y):
    return int(x * SCALE), int(SIM_SIZE - y * SCALE)


def _shoe_colour(i):
    c = _SHOE_PALETTE[i % len(_SHOE_PALETTE)]
    return c, tuple(max(0, v - 60) for v in c)


def make_fonts():
    return {
        "big"  : pygame.font.SysFont("monospace", 21, bold=True),
        "mid"  : pygame.font.SysFont("monospace", 15),
        "small": pygame.font.SysFont("monospace", 13),
        "tiny" : pygame.font.SysFont("monospace", 11),
    }


def draw_star(surface, cx, cy, r_out, r_in, n, color, border=None):
    pts = []
    for i in range(2 * n):
        angle = math.radians(-90 + i * 180 / n)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)
    if border:
        pygame.draw.polygon(surface, border, pts, 2)


def draw_lidar(surface, state, raw_lidar, show):
    if not show or raw_lidar is None:
        return
    rx, ry = W(float(state.x), float(state.y))
    theta  = float(state.theta)
    angles = theta - FOV / 2.0 + np.arange(NUM_RAYS) * FOV / (NUM_RAYS - 1)
    sp_mask = np.array(state.sp_mask)
    for i, (ang, dist) in enumerate(zip(angles, raw_lidar)):
        ex, ey = W(state.x + dist * math.cos(ang), state.y + dist * math.sin(ang))
        if sp_mask[i]:
            col = (180, 50, 220)
        else:
            t = max(0.0, 1.0 - dist / MAX_LIDAR_DIST)
            col = tuple(int(C_RAY_NEAR[j] * t + C_RAY_FAR[j] * (1 - t)) for j in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)


def draw_shoe(surface, fx, fy, theta, col, border):
    ct, st = math.cos(theta), math.sin(theta)
    lx, ly = -st, ct
    hL, hW = SHOE_LENGTH * 0.5, SHOE_WIDTH * 0.5
    cx, cy = fx + ct * hL, fy + st * hL
    corners = [
        (cx + ct * hL + lx * hW, cy + st * hL + ly * hW),
        (cx + ct * hL - lx * hW, cy + st * hL - ly * hW),
        (cx - ct * hL - lx * hW, cy - st * hL - ly * hW),
        (cx - ct * hL + lx * hW, cy - st * hL + ly * hW),
    ]
    pts = [W(wx, wy) for wx, wy in corners]
    pygame.draw.polygon(surface, col, pts)
    pygame.draw.polygon(surface, border, pts, 1)


def draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body):
    n = int(state.people.shape[0])
    left_legs  = foot_state_np[:, 0:2]
    right_legs = foot_state_np[:, 2:4]
    for i in range(n):
        if float(state.people[i, 10]) < 0:
            continue
        px, py  = float(state.people[i, 0]), float(state.people[i, 1])
        vx, vy  = float(state.people[i, 2]), float(state.people[i, 3])
        theta_h = float(state.people[i, 4])
        col, border = _shoe_colour(i)

        if show_body:
            sx, sy = W(px, py)
            body_r_px = max(3, int(_jax_env.PEOPLE_RADIUS * SCALE))
            pygame.draw.circle(surface, C_BODY_RING, (sx, sy), body_r_px, 1)
            ux, uy = math.cos(theta_h), math.sin(theta_h)
            dot_x, dot_y = W(px + ux * _jax_env.PEOPLE_RADIUS,
                             py + uy * _jax_env.PEOPLE_RADIUS)
            dot_r = max(2, int(body_r_px * 0.22))
            pygame.draw.circle(surface, C_BODY_RING, (dot_x, dot_y), dot_r)
            pygame.draw.circle(surface, (20, 60, 20), (dot_x, dot_y), dot_r, 1)

        if use_legs:
            left_theta  = float(foot_state_np[i, 10])
            right_theta = float(foot_state_np[i, 11])
            draw_shoe(surface, float(left_legs[i, 0]),  float(left_legs[i, 1]),
                      left_theta,  col, border)
            draw_shoe(surface, float(right_legs[i, 0]), float(right_legs[i, 1]),
                      right_theta, col, border)
            # Shift the leg circle forward by LEG_RADIUS along the foot heading
            # so it sits flush against the heel edge — fully inside the shoe
            # rectangle. (foot_xy is the back of the shoe; without the inset
            # the circle would protrude behind the heel.)
            leg_r = max(2, int(LEG_RADIUS * SCALE))
            lx_w = float(left_legs[i, 0])  + LEG_RADIUS * math.cos(left_theta)
            ly_w = float(left_legs[i, 1])  + LEG_RADIUS * math.sin(left_theta)
            rx_w = float(right_legs[i, 0]) + LEG_RADIUS * math.cos(right_theta)
            ry_w = float(right_legs[i, 1]) + LEG_RADIUS * math.sin(right_theta)
            lx, ly   = W(lx_w, ly_w)
            rx_, ry_ = W(rx_w, ry_w)
            pygame.draw.circle(surface, tuple(min(255, int(c * 1.1)) for c in col), (lx, ly), leg_r)
            pygame.draw.circle(surface, (20, 20, 20), (lx, ly), leg_r, 1)
            pygame.draw.circle(surface, tuple(max(0,   int(c * 0.75)) for c in col), (rx_, ry_), leg_r)
            pygame.draw.circle(surface, (20, 20, 20), (rx_, ry_), leg_r, 1)
        else:
            sx, sy = W(px, py)
            pr = max(3, int(_jax_env.PEOPLE_RADIUS * SCALE))
            pygame.draw.circle(surface, col,    (sx, sy), pr)
            pygame.draw.circle(surface, border, (sx, sy), pr, 1)
            speed = math.hypot(vx, vy)
            if show_arrows and speed > 0.05:
                ax, ay = W(px + math.cos(theta_h) * 0.5, py + math.sin(theta_h) * 0.5)
                pygame.draw.line(surface, (20, 120, 20), (sx, sy), (ax, ay), 2)


def draw_scene(surface, state, raw_lidar, foot_state_np, show_lidar, show_arrows, use_legs, show_body):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0); pygame.draw.line(surface, C_GRID, (sx, 0), (sx, SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j); pygame.draw.line(surface, C_GRID, (0, sy), (SIM_SIZE, sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

    draw_lidar(surface, state, raw_lidar, show_lidar)

    for box in np.array(state.obs_boxes):
        x0, y0, x1, y1, x2, y2, x3, y3 = box
        if max(x0, x1, x2, x3) - min(x0, x1, x2, x3) > 1e-5:
            pts = [W(x0, y0), W(x1, y1), W(x2, y2), W(x3, y3)]
            pygame.draw.polygon(surface, C_BOX, pts)
            pygame.draw.polygon(surface, C_BOX_L, pts, 2)
    for cir in np.array(state.obs_circles):
        cx, cy, r = cir
        if r > 0:
            sx, sy = W(cx, cy); pr = max(2, int(r * SCALE))
            pygame.draw.circle(surface, C_CIRCLE,   (sx, sy), pr)
            pygame.draw.circle(surface, C_CIRCLE_L, (sx, sy), pr, 2)

    gx, gy = W(float(state.goal_x), float(state.goal_y))
    draw_star(surface, gx, gy, int(0.30 * SCALE), int(0.12 * SCALE), 5, C_GOAL, C_GOAL2)
    draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body)

    rx, ry = W(float(state.x), float(state.y))
    rr = max(4, int(ROBOT_RADIUS * SCALE))
    pygame.draw.circle(surface, C_ROBOT,   (rx, ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx, ry), rr, 2)
    hx, hy = W(state.x + ROBOT_RADIUS * 3 * math.cos(float(state.theta)),
               state.y + ROBOT_RADIUS * 3 * math.sin(float(state.theta)))
    pygame.draw.line(surface, C_ROBOT_H, (rx, ry), (hx, hy), 3)


def draw_panel(surface, fonts, scen_idx, step, max_v, v, w,
               goal_dist, ch, banner, banner_t,
               use_legs, raw_lidar, sp_mask, show_radar=True,
               foot_dist=None, left_step=None, right_step=None,
               left_stance=None, body_speeds=None, cadence=None):
    pygame.draw.rect(surface, C_PANEL, (SIM_SIZE, 0, PANEL_W, WINDOW_H))
    pygame.draw.line(surface, C_WALL,  (SIM_SIZE, 0), (SIM_SIZE, WINDOW_H), 2)
    y = 10; lh = 19; x0 = SIM_SIZE + 10

    def txt(s, col=C_TEXT, font="mid"):
        nonlocal y
        surface.blit(fonts[font].render(s, True, col), (x0, y))
        y += lh

    def sep():
        nonlocal y
        pygame.draw.line(surface, C_DIM, (x0, y + 3), (x0 + PANEL_W - 20, y + 3), 1)
        y += 10

    txt("─ GAIT VIEW ─", C_TEXT, "big"); y += 2; sep()
    scen_name = {13: "Treadmill", 14: "Crossing", 15: "Circle"}.get(scen_idx, "?")
    leg_mode  = "LEGS" if use_legs else "CYLINDERS"
    txt(f"[FIXED] {leg_mode}", C_DIM, "small")
    txt(f"Scen     {scen_name}", C_SUCCESS)
    txt(f"Step     {step:>4d}"); sep()
    txt(f"max_v {max_v:>+6.3f} m/s")
    txt(f"v     {v:>+6.3f} m/s")
    txt(f"w     {w:>+6.3f} rad/s"); sep()
    txt(f"Goal  {goal_dist:>5.2f} m")
    txt(f"Human {ch:>5.2f} m"); sep()
    # ── Gait metrics for person 0 ─────────────────────────────────────────
    txt("── Gait (person 0) ─", C_DIM, "small"); y += 2
    if foot_dist is not None:
        txt(f"  Foot dist {foot_dist:>5.2f} m", C_TEXT, "small")
    else:
        txt("  Foot dist   ---  m",            C_DIM,  "small")
    # Highlight the foot currently in stance.
    l_col = C_SUCCESS if left_stance is True  else C_TEXT
    r_col = C_SUCCESS if left_stance is False else C_TEXT
    l_lbl = "L step (stance)" if left_stance is True  else "L step         "
    r_lbl = "R step (stance)" if left_stance is False else "R step         "
    if left_step is not None:
        txt(f"  {l_lbl} {left_step:>5.2f} m", l_col, "small")
    else:
        txt(f"  {l_lbl}   ---  m",            C_DIM, "small")
    if right_step is not None:
        txt(f"  {r_lbl} {right_step:>5.2f} m", r_col, "small")
    else:
        txt(f"  {r_lbl}   ---  m",            C_DIM, "small")
    if cadence is not None:
        txt(f"  Cadence    {cadence:>5.2f} 1/s", C_TEXT, "small")
    else:
        txt(f"  Cadence      ---  1/s",          C_DIM,  "small")
    sep()
    # ── HSFM body-centre velocity, one row per active person ──────────────
    txt("── Body vel (HSFM) ─", C_DIM, "small"); y += 2
    if body_speeds:
        for i, spd in enumerate(body_speeds):
            txt(f"  P{i}  |v| {spd:>5.2f} m/s", C_TEXT, "small")
    else:
        txt("  no active people",     C_DIM, "small")
    sep()
    txt("── Visual gait check ─", C_DIM, "small"); y += 2
    txt("  no policy loaded",     C_DIM, "small")
    txt("  robot parked at corner",C_DIM, "small")
    txt("  max_v = 0 → static",   C_DIM, "small"); sep()
    txt("1 Treadmill 2 Cross 3 Circ", C_DIM, "tiny")
    txt("R reset  Space pause",    C_DIM, "tiny")
    txt("L lidar  H arrows",       C_DIM, "tiny")
    txt("B body   P radar  Q quit",C_DIM, "tiny")

    if banner_t > 0:
        label, col = {
            "reset"   : ("↺  RESET",     C_DIM),
            "scenario": ("◉  SCENARIO",  C_SUCCESS),
            "timeout" : ("⏱  TIMEOUT",    C_TIMEOUT),
        }.get(banner, (banner, C_TEXT))
        surf = fonts["big"].render(label, True, col)
        bx, by = 20, 20
        bg = pygame.Surface((surf.get_width() + 24, surf.get_height() + 12))
        bg.fill((20, 20, 26)); bg.set_alpha(220)
        surface.blit(bg, (bx - 12, by - 6))
        surface.blit(surf, (bx, by))

    if raw_lidar is not None and show_radar:
        radar_r  = 130
        radar_cx = SIM_SIZE + PANEL_W // 2
        radar_cy = WINDOW_H - radar_r - 20
        pygame.draw.circle(surface, (15, 15, 20),  (radar_cx, radar_cy), radar_r)
        pygame.draw.circle(surface, (50, 50, 60),  (radar_cx, radar_cy), radar_r, 1)
        pygame.draw.circle(surface, (50, 50, 60),  (radar_cx, radar_cy), int(radar_r * .66), 1)
        pygame.draw.circle(surface, (50, 50, 60),  (radar_cx, radar_cy), int(radar_r * .33), 1)
        pygame.draw.circle(surface, C_ROBOT,       (radar_cx, radar_cy), 3)
        angles = np.linspace(FOV / 2, -FOV / 2, len(raw_lidar)) - math.pi / 2
        dists  = (raw_lidar / MAX_LIDAR_DIST) * radar_r
        xs = radar_cx + np.cos(angles) * dists
        ys = radar_cy + np.sin(angles) * dists
        for i, (rx2, ry2) in enumerate(zip(xs, ys)):
            col = (180, 50, 220) if sp_mask[i] else (220, 50, 50)
            pygame.draw.circle(surface, col, (int(rx2), int(ry2)), 2)
        for factor in [0.33, 0.66, 1.0]:
            dist_val = MAX_LIDAR_DIST * factor
            lbl = fonts["tiny"].render(f"{dist_val:.1f}m", True, (150, 150, 160))
            lx = radar_cx - lbl.get_width() // 2
            ly = radar_cy + int(radar_r * factor) - lbl.get_height()
            if factor == 1.0:
                ly -= 2
            surface.blit(lbl, (lx, ly))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n🎬 Visual gait check — no policy, robot parked in corner\n")
    print(f"   Scenario 15 (CIRCLE): V={CIRCLE_V:.2f} m/s, "
          f"Ω={CIRCLE_OMEGA:.2f} rad/s, R={CIRCLE_V/CIRCLE_OMEGA:.2f} m\n")
    print("   Keys: 1 = treadmill | 2 = crossing | 3 = circle | R reset "
          "| Space pause | L lidar | H arrows | B body | P radar | Q quit\n")

    jit_advance_feet = jax.jit(advance_feet, static_argnums=())

    MAX_EVAL_GOAL_DIST = 9.0

    def build_fast_reset(scen_idx):
        bound_reset = lambda key, max_goal_dist=3.0, scenario_idx=-1, min_goal_dist=0.8, **kw: \
            reset_env(key, max_goal_dist, scenario_idx=scen_idx, min_goal_dist=min_goal_dist, **kw)
        rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
        jit_rs = jax.jit(lambda key: rs(key, MAX_EVAL_GOAL_DIST, ghost_prob=0.0, min_goal_dist=0.8))
        jit_ss = jax.jit(ss)
        return jit_rs, jit_ss

    rng = jax.random.PRNGKey(7)
    current_scenario = 13                          # default to treadmill
    fast_reset, fast_step = build_fast_reset(current_scenario)
    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("jax_legs — visual gait scenarios (no policy)")
    clock = pygame.time.Clock(); fonts = make_fonts()

    show_lidar = True; show_arrows = True; show_body = False; show_radar = True
    paused = False
    fps_idx = 0; current_fps = FPS_SPEEDS[fps_idx]
    banner = ""; banner_t = 0
    step_count = 0

    # ── Initial-freeze frames after each reset of single-walker scenarios ─
    # Treadmill (13) and circle (15) both look nicer with a 2 s still pose
    # before motion kicks in. We hold the env stationary (skip fast_step) for
    # the corresponding number of viewer frames.
    INITIAL_FREEZE_S = 2.0
    def _initial_freeze_frames(scen_idx):
        return int(round(INITIAL_FREEZE_S * current_fps)) if scen_idx in (13, 15) else 0
    freeze_frames = _initial_freeze_frames(current_scenario)

    # ── Scenario 15 (CIRCLE) integrator state ─────────────────────────────
    # circle_theta tracks the walker's body heading; position is reconstructed
    # from it each frame around the room centre. Initial value matches the
    # spawn pose set in _gait_circle_scen (heading +y → start on +x of centre).
    circle_theta = float(math.pi / 2.0)
    def _reset_circle():
        nonlocal circle_theta
        circle_theta = float(math.pi / 2.0)

    zero_action = jnp.zeros((2,), dtype=jnp.float32)

    # ── Gait-metric state (tracked for person 0 across frames) ───────────────
    # A foot "plant" event is detected when its anchor (foot_state[i, 5:7] for
    # left, [7:9] for right) changes between frames. At a plant we capture the
    # alternating-foot step length = ||new anchor − last anchor of OTHER foot||.
    # The captured value persists until the next plant of that foot.
    gait_prev_left_anchor  = None
    gait_prev_right_anchor = None
    gait_left_step  = None
    gait_right_step = None
    gait_foot_dist  = None
    gait_left_stance = None

    def _reset_gait():
        nonlocal gait_prev_left_anchor, gait_prev_right_anchor
        nonlocal gait_left_step, gait_right_step, gait_foot_dist, gait_left_stance
        gait_prev_left_anchor  = None
        gait_prev_right_anchor = None
        gait_left_step  = None
        gait_right_step = None
        gait_foot_dist  = None
        gait_left_stance = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                if k == pygame.K_SPACE:
                    paused = not paused
                if k == pygame.K_l:
                    show_lidar = not show_lidar
                if k == pygame.K_h:
                    show_arrows = not show_arrows
                if k == pygame.K_b:
                    show_body = not show_body
                if k == pygame.K_p:
                    show_radar = not show_radar
                if k == pygame.K_s:
                    fps_idx = (fps_idx + 1) % len(FPS_SPEEDS)
                    current_fps = FPS_SPEEDS[fps_idx]
                    banner = f"FPS: {current_fps}"; banner_t = 15
                if k == pygame.K_1 and current_scenario != 13:
                    current_scenario = 13
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    step_count = 0; banner = "scenario"; banner_t = 15
                    _reset_gait(); _reset_circle()
                    freeze_frames = _initial_freeze_frames(current_scenario)
                if k == pygame.K_2 and current_scenario != 14:
                    current_scenario = 14
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    step_count = 0; banner = "scenario"; banner_t = 15
                    _reset_gait(); _reset_circle()
                    freeze_frames = _initial_freeze_frames(current_scenario)
                if k == pygame.K_3 and current_scenario != 15:
                    current_scenario = 15
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    step_count = 0; banner = "scenario"; banner_t = 15
                    _reset_gait(); _reset_circle()
                    freeze_frames = _initial_freeze_frames(current_scenario)
                if k == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    step_count = 0; banner = "reset"; banner_t = 15
                    _reset_gait(); _reset_circle()
                    freeze_frames = _initial_freeze_frames(current_scenario)

        if paused:
            clock.tick(10)
            continue

        if freeze_frames > 0:
            # Treadmill warm-up: hold the env stationary for ~2s so the walker
            # is visibly still before HSFM starts pulling them toward the goal.
            # We deliberately skip fast_step here; the env state we render is
            # the one returned by the most recent reset.
            freeze_frames -= 1
            info = {"closest_human": 100.0}
            done = False
        else:
            # Always send a zero action; the scenarios also set max_v = 0 so the
            # robot stays parked even if numerics introduce tiny drifts.
            rng, step_rng = jax.random.split(rng)
            obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, zero_action)
            step_count += 1

            # Scenario 15: override person 0 with pure unicycle kinematics so
            # the gait model walks a clean constant-radius circle. HSFM's
            # contribution to person 0 this frame is discarded; the foot state
            # is re-derived from the overridden pose via advance_feet so the
            # leg cycle is consistent with what's rendered.
            if current_scenario == 15:
                r_circ = (CIRCLE_V / CIRCLE_OMEGA) if abs(CIRCLE_OMEGA) > 1e-6 else 0.0
                cx, cy = ROOM_W * 0.5, ROOM_H * 0.5
                phi = circle_theta - math.pi / 2.0
                px = cx + r_circ * math.cos(phi)
                py = cy + r_circ * math.sin(phi)
                vx = CIRCLE_V * math.cos(circle_theta)
                vy = CIRCLE_V * math.sin(circle_theta)
                row0 = jnp.array([px, py, vx, vy, circle_theta, CIRCLE_OMEGA],
                                 dtype=jnp.float32)
                new_people = stacked_state.env_state.people.at[0, 0:6].set(row0)
                new_foot   = jit_advance_feet(stacked_state.env_state.foot_state,
                                              new_people, DT)
                new_env = stacked_state.env_state.replace(
                    people=new_people, foot_state=new_foot)
                stacked_state = stacked_state.replace(env_state=new_env)
                circle_theta += CIRCLE_OMEGA * DT

        cpu_state = jax.device_get(stacked_state.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * \
                    (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask       = np.array(cpu_state.sp_mask)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist = math.hypot(dx, dy)
        ch    = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        # ── Update gait metrics for person 0 ─────────────────────────────────
        if foot_state_np.shape[0] > 0:
            l_theta = float(foot_state_np[0, 10])
            r_theta = float(foot_state_np[0, 11])
            # Heel-aligned circle centres (match what's drawn / what the LiDAR sees)
            l_cx = float(foot_state_np[0, 0]) + LEG_RADIUS * math.cos(l_theta)
            l_cy = float(foot_state_np[0, 1]) + LEG_RADIUS * math.sin(l_theta)
            r_cx = float(foot_state_np[0, 2]) + LEG_RADIUS * math.cos(r_theta)
            r_cy = float(foot_state_np[0, 3]) + LEG_RADIUS * math.sin(r_theta)

            # Anchor positions encode the last plant location for each foot.
            l_anchor = (float(foot_state_np[0, 5]), float(foot_state_np[0, 6]))
            r_anchor = (float(foot_state_np[0, 7]), float(foot_state_np[0, 8]))

            # A plant event is a change in the anchor (>1mm) between frames.
            _PLANT_TOL = 1e-3
            if gait_prev_left_anchor is not None and gait_prev_right_anchor is not None:
                dl = math.hypot(l_anchor[0] - gait_prev_left_anchor[0],
                                l_anchor[1] - gait_prev_left_anchor[1])
                dr = math.hypot(r_anchor[0] - gait_prev_right_anchor[0],
                                r_anchor[1] - gait_prev_right_anchor[1])
                if dl > _PLANT_TOL:
                    # Left just planted → step length = dist to most-recent right plant.
                    gait_left_step = math.hypot(l_anchor[0] - r_anchor[0],
                                                l_anchor[1] - r_anchor[1])
                if dr > _PLANT_TOL:
                    gait_right_step = math.hypot(r_anchor[0] - l_anchor[0],
                                                 r_anchor[1] - l_anchor[1])
                # Foot distance is captured at each plant event (the "switch
                # moment" between stance feet) and held until the next plant.
                if dl > _PLANT_TOL or dr > _PLANT_TOL:
                    gait_foot_dist = math.hypot(l_cx - r_cx, l_cy - r_cy)

            gait_prev_left_anchor  = l_anchor
            gait_prev_right_anchor = r_anchor
            # Stance flag from gait phase: phase < 0.5 → left in stance.
            gait_left_stance = bool(float(foot_state_np[0, 4]) < 0.5)
        else:
            gait_foot_dist = None
            gait_left_stance = None

        # HSFM body-centre speeds for every active person (id columns: vx=2, vy=3,
        # active flag is column 10 ≥ 0).
        ppl = np.array(cpu_state.people)
        active_mask = ppl[:, 10] >= 0.0
        body_speeds = [float(math.hypot(ppl[i, 2], ppl[i, 3]))
                       for i in range(ppl.shape[0]) if active_mask[i]]

        # Gait cadence for person 0 — mirrors jax_legs._update_single:
        #   cadence = STEP_FREQ * clip(speed / STEP_SPEED, 0.3, 1.0)
        if body_speeds:
            p0_speed = float(math.hypot(ppl[0, 2], ppl[0, 3]))
            p0_cadence = STEP_FREQ * max(0.3, min(1.0, p0_speed / STEP_SPEED))
        else:
            p0_cadence = None

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, True, show_body)
        draw_panel(screen, fonts, current_scenario, step_count,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, ch, banner, banner_t, True, raw_lidar, sp_mask, show_radar,
                   foot_dist=gait_foot_dist,
                   left_step=gait_left_step, right_step=gait_right_step,
                   left_stance=gait_left_stance,
                   body_speeds=body_speeds,
                   cadence=p0_cadence)

        if banner_t > 0:
            banner_t -= 1
        pygame.display.flip()
        clock.tick(current_fps)

        # Auto-reset on timeout so the visualisation runs forever.
        if done:
            banner = "timeout"; banner_t = FPS_TARGET * 2
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            step_count = 0
            _reset_gait()
            freeze_frames = _initial_freeze_frames(current_scenario)


if __name__ == "__main__":
    main()
