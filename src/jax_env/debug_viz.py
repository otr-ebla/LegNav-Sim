#!/usr/bin/env python3
"""
debug_viz.py — Pygame Debug Visualizer (Unified Rendering Style)
=================================================================
Mirrors the exact visual style of jax_eval_multi.py but tracks the
dynamic curriculum (rolling_suc, cur_max_dist, cur_scenario).

Usage:
  python debug_viz.py                    # random policy
  python debug_viz.py --checkpoint path  # trained policy
  python debug_viz.py --speed 0.5        # slower playback
  python debug_viz.py --pause            # start paused
"""

import os
import sys
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", action="store_true")
parser.add_argument("--no-ghost", action="store_true", help="Disable ghost (eval mode)")
parser.add_argument("--ghost-body", action="store_true", help="Overlay JHSFM body ring on top of legs.")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

import jax
import jax.numpy as jnp
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import environment constants and wrappers
from jax_env_multi import reset_env, step_env
from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, GOAL_RADIUS, USE_LEGS, NUM_RAYS, MAX_LIDAR_DIST, FOV)
from jax_wrappers import make_stacked_env
from jax_legs import LEG_RADIUS, SHOE_LENGTH, SHOE_WIDTH

# ── Curriculum Logic ─────────────────────────────────────────────────────────
CURRICULUM_STAGES = [
    (25.0, 1.5),
    (38.0, 2.5),
    (50.0, 4.0),
    (60.0, 5.0),
    (70.0, 6.5),
    (80.0, 8.0),
    (101., 9.0),
]

def curriculum_max_goal_dist(suc_pct: float) -> float:
    for threshold, dist in CURRICULUM_STAGES:
        if suc_pct < threshold:
            return dist
    return CURRICULUM_STAGES[-1][1]

# ── Rendering Constants & Functions (from jax_eval_multi.py) ─────────────────
SIM_SIZE   = 800
PANEL_W    = 300
WINDOW_W   = SIM_SIZE + PANEL_W
WINDOW_H   = SIM_SIZE
SCALE      = SIM_SIZE / max(ROOM_W, ROOM_H)

C_BG        = (28,  28,  34); C_FLOOR    = (44,  44,  52); C_GRID     = (56,  56,  66)
C_WALL      = (190, 190, 205); C_ROBOT    = (60,  140, 255); C_ROBOT_H  = (170, 210, 255)
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

def W(x, y): return int(x * SCALE), int(SIM_SIZE - y * SCALE)

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
    if border: pygame.draw.polygon(surface, border, pts, 2)

def draw_lidar(surface, state, raw_lidar, show):
    if not show or raw_lidar is None: return
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
            col = tuple(int(C_RAY_NEAR[j]*t + C_RAY_FAR[j]*(1-t)) for j in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)

def draw_shoe(surface, fx, fy, theta, col, border):
    ct, st = math.cos(theta), math.sin(theta)
    lx, ly = -st, ct
    hL, hW = SHOE_LENGTH * 0.5, SHOE_WIDTH * 0.5
    cx, cy = fx + ct * hL, fy + st * hL
    corners = [
        (cx + ct*hL + lx*hW, cy + st*hL + ly*hW),
        (cx + ct*hL - lx*hW, cy + st*hL - ly*hW),
        (cx - ct*hL - lx*hW, cy - st*hL - ly*hW),
        (cx - ct*hL + lx*hW, cy - st*hL + ly*hW),
    ]
    pts = [W(wx, wy) for wx, wy in corners]
    pygame.draw.polygon(surface, col,    pts)
    pygame.draw.polygon(surface, border, pts, 1)

def draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body):
    n = int(state.people.shape[0])
    left_legs  = foot_state_np[:, 0:2]
    right_legs = foot_state_np[:, 2:4]
    for i in range(n):
        if float(state.people[i, 10]) < 0: continue
        px, py  = float(state.people[i, 0]), float(state.people[i, 1])
        vx, vy  = float(state.people[i, 2]), float(state.people[i, 3])
        theta_h = float(state.people[i, 4])
        col, border = _shoe_colour(i)

        if show_body:
            sx, sy = W(px, py)
            pygame.draw.circle(surface, C_BODY_RING, (sx, sy),
                               max(3, int(PEOPLE_RADIUS * SCALE)), 1)

        if use_legs:
            draw_shoe(surface, float(left_legs[i, 0]),  float(left_legs[i, 1]),  theta_h, col, border)
            draw_shoe(surface, float(right_legs[i, 0]), float(right_legs[i, 1]), theta_h, col, border)
            leg_r = max(2, int(LEG_RADIUS * SCALE))
            lx, ly = W(float(left_legs[i, 0]),  float(left_legs[i, 1]))
            rx_, ry_ = W(float(right_legs[i, 0]), float(right_legs[i, 1]))
            pygame.draw.circle(surface, tuple(min(255,int(c*1.1)) for c in col), (lx, ly), leg_r)
            pygame.draw.circle(surface, (20,20,20), (lx, ly), leg_r, 1)
            pygame.draw.circle(surface, tuple(max(0,int(c*0.75)) for c in col), (rx_, ry_), leg_r)
            pygame.draw.circle(surface, (20,20,20), (rx_, ry_), leg_r, 1)
        else:
            sx, sy = W(px, py)
            pr = max(3, int(PEOPLE_RADIUS * SCALE))
            pygame.draw.circle(surface, col,    (sx, sy), pr)
            pygame.draw.circle(surface, border, (sx, sy), pr, 1)
            speed = math.hypot(vx, vy)
            if show_arrows and speed > 0.05:
                ax, ay = W(px + math.cos(theta_h)*0.5, py + math.sin(theta_h)*0.5)
                pygame.draw.line(surface, (20,120,20), (sx,sy), (ax,ay), 2)

def draw_scene(surface, state, raw_lidar, foot_state_np, show_lidar, show_arrows, use_legs, show_body):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0); pygame.draw.line(surface, C_GRID, (sx,0), (sx,SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j); pygame.draw.line(surface, C_GRID, (0,sy), (SIM_SIZE,sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

    draw_lidar(surface, state, raw_lidar, show_lidar)

    for box in np.array(state.obs_boxes):
        cx, cy, hw, hh = box
        if hw > 0:
            sx, sy = W(cx - hw, cy + hh)
            pygame.draw.rect(surface, C_BOX,   (sx, sy, int(2*hw*SCALE), int(2*hh*SCALE)))
            pygame.draw.rect(surface, C_BOX_L, (sx, sy, int(2*hw*SCALE), int(2*hh*SCALE)), 2)
    for cir in np.array(state.obs_circles):
        cx, cy, r = cir
        if r > 0:
            sx, sy = W(cx, cy); pr = max(2, int(r * SCALE))
            pygame.draw.circle(surface, C_CIRCLE,   (sx, sy), pr)
            pygame.draw.circle(surface, C_CIRCLE_L, (sx, sy), pr, 2)

    gx, gy = W(float(state.goal_x), float(state.goal_y))
    draw_star(surface, gx, gy, int(0.30*SCALE), int(0.12*SCALE), 5, C_GOAL, C_GOAL2)
    draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body)

    rx, ry = W(float(state.x), float(state.y)); rr = max(4, int(ROBOT_RADIUS*SCALE))
    pygame.draw.circle(surface, C_ROBOT,   (rx, ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx, ry), rr, 2)
    hx, hy = W(state.x + ROBOT_RADIUS*3*math.cos(float(state.theta)),
               state.y + ROBOT_RADIUS*3*math.sin(float(state.theta)))
    pygame.draw.line(surface, C_ROBOT_H, (rx, ry), (hx, hy), 3)

def draw_panel(surface, fonts, ep, step, ep_ret, max_v, v, w,
               goal_dist, goal_align, ch, banner, banner_t,
               rolling_suc, cur_max_dist, cur_scenario, use_legs, raw_lidar, sp_mask):
    pygame.draw.rect(surface, C_PANEL, (SIM_SIZE, 0, PANEL_W, WINDOW_H))
    pygame.draw.line(surface, C_WALL,  (SIM_SIZE, 0), (SIM_SIZE, WINDOW_H), 2)
    y = 10; lh = 19; x0 = SIM_SIZE + 10

    def txt(s, col=C_TEXT, font="mid"):
        nonlocal y
        surface.blit(fonts[font].render(s, True, col), (x0, y)); y += lh

    def sep():
        nonlocal y
        pygame.draw.line(surface, C_DIM, (x0, y+3), (x0+PANEL_W-20, y+3), 1); y += 10

    txt("─ CURRICULUM DEBUG ─", C_TEXT, "big"); y += 2; sep()
    
    scen_name = "RANDOM (All Unlocked)" if cur_scenario == -1 else f"LOCKED (Scenario {cur_scenario})"
    txt(f"Scen  {scen_name}", C_SUCCESS, "small")
    txt(f"Ep    {ep:>5d}"); txt(f"Step  {step:>4d}"); sep()
    
    txt(f"max_v {max_v:>+6.3f} m/s")
    txt(f"v     {v:>+6.3f} m/s"); txt(f"w     {w:>+6.3f} rad/s"); sep()
    
    txt(f"Goal  {goal_dist:>5.2f} m")
    txt(f"Align {math.degrees(goal_align):>+5.1f}°")
    txt(f"Human {ch:>5.2f} m"); sep()
    
    txt(f"Ep ret {ep_ret:>+7.2f}", C_GOAL, "mid"); sep()
    
    txt("── Curriculum State ─", C_DIM, "small"); y += 2
    txt(f"  Roll. Suc {rolling_suc:>5.1f}%",  C_SUCCESS)
    txt(f"  Max Dist  {cur_max_dist:>5.1f}m",  C_TEXT)
    txt(f"  Threshold 25.0%", C_DIM); sep()
    
    txt("SPACE pause  R reset", C_DIM, "tiny")
    txt("L lidar  H arrows  B body  Q quit", C_DIM, "tiny")

    if banner_t > 0:
        label, col = {
            "success"    : ("✓  GOAL REACHED",  C_SUCCESS),
            "collision"  : ("X  COLLISION",     C_COLLIDE),
            "passive_col": ("🚶 PASSIVE COL",   (200,100,100)),
            "timeout"    : ("⏱  TIMEOUT",       C_TIMEOUT),
        }.get(banner, ("", C_TEXT))
        if label:
            surf = fonts["big"].render(label, True, col)
            radar_r   = 130
            radar_top = WINDOW_H - radar_r * 2 - 20
            bx = SIM_SIZE + PANEL_W // 2 - surf.get_width() // 2
            by = radar_top - surf.get_height() - 20
            bg = pygame.Surface((surf.get_width()+24, surf.get_height()+12))
            bg.fill((20,20,26)); bg.set_alpha(220)
            surface.blit(bg, (bx-12, by-6)); surface.blit(surf, (bx, by))

    # LiDAR radar chart
    if raw_lidar is not None:
        radar_r  = 130
        radar_cx = SIM_SIZE + PANEL_W // 2
        radar_cy = WINDOW_H - radar_r - 20
        pygame.draw.circle(surface, (15,15,20),  (radar_cx, radar_cy), radar_r)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), radar_r, 1)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), int(radar_r*.66), 1)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), int(radar_r*.33), 1)
        pygame.draw.circle(surface, C_ROBOT,     (radar_cx, radar_cy), 3)
        angles = np.linspace(FOV/2, -FOV/2, len(raw_lidar)) - math.pi/2
        dists  = (raw_lidar / MAX_LIDAR_DIST) * radar_r
        xs = radar_cx + np.cos(angles) * dists
        ys = radar_cy + np.sin(angles) * dists
        for i, (rx2, ry2) in enumerate(zip(xs, ys)):
            if sp_mask[i] and raw_lidar[i] >= MAX_LIDAR_DIST - 0.1:
                col = (180, 50, 220)
            else:
                col = (220, 50, 50)
            pygame.draw.circle(surface, col, (int(rx2), int(ry2)), 2)

# ── Main ─────────────────────────────────────────────────────────────────────

def load_policy(path):
    if path is None: return None
    from jax_network import EndToEndActorCritic
    import flax.serialization, optax
    net = EndToEndActorCritic(action_dim=2)
    dummy = jnp.zeros((1, 342))
    p = net.init(jax.random.PRNGKey(0), dummy)["params"]
    try:
        o = optax.adam(1e-4).init(p)
        b = flax.serialization.from_bytes({"params": p, "opt_state": o}, open(path, "rb").read())
        p = b["params"]
    except Exception:
        p = flax.serialization.from_bytes(p, open(path, "rb").read())
    print(f"Loaded: {path}")
    return net, p

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("RL Nav Debug (Curriculum Viz)")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    ghost = not args.no_ghost
    policy = load_policy(args.checkpoint)

    reset_s, step_s = make_stacked_env(reset_env, step_env, stack_dim=3, ghost_prob=1.0 if ghost else 0.0)
    rng = jax.random.PRNGKey(args.seed)

    def dynamic_reset(k, max_dist, scen_idx):
        return reset_s(k, max_goal_dist=max_dist, scenario_idx=scen_idx)
    
    jreset = jax.jit(dynamic_reset, static_argnums=(2,))

    rolling_suc = 0.0
    cur_max_dist = curriculum_max_goal_dist(rolling_suc)
    cur_scenario = 0 if rolling_suc < 25.0 else -1
    cur_updated_this_ep = False

    rng, rk = jax.random.split(rng)
    obs, ss = jreset(rk, cur_max_dist, cur_scenario)

    paused = args.pause
    show_lidar = True
    show_arrows = True
    show_body = args.ghost_body
    banner = ""
    banner_t = 0
    ep_ret = 0.0
    ep_steps = 0
    ep_count = 0
    last_done = False

    fps_target = 30

    running = True
    while running:
        step_once = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif ev.key == pygame.K_SPACE: paused = not paused
                elif ev.key == pygame.K_RIGHT: step_once = True
                elif ev.key == pygame.K_l: show_lidar = not show_lidar
                elif ev.key == pygame.K_h: show_arrows = not show_arrows
                elif ev.key == pygame.K_b: show_body = not show_body
                elif ev.key == pygame.K_r:
                    rng, rk = jax.random.split(rng)
                    obs, ss = jreset(rk, cur_max_dist, cur_scenario)
                    ep_ret, ep_steps, banner_t = 0.0, 0, 0
                    ep_count += 1
                    last_done = False
                    cur_updated_this_ep = False

        if paused and not step_once:
            clock.tick(10); continue

        if not last_done:
            rng, ak, sk = jax.random.split(rng, 3)
            if policy:
                net, par = policy
                m, ls, val = net.apply({"params": par}, obs[None])
                from jax_network import scale_action_to_env
                raw = m[0] # Deterministic evaluation
                act = scale_action_to_env(raw, ss.env_state.max_v)
            else:
                mv = float(ss.env_state.max_v)
                v = jax.random.uniform(ak, minval=0.0, maxval=mv)
                w = jax.random.uniform(jax.random.fold_in(ak, 1), minval=-1.0, maxval=1.0)
                act = jnp.array([v, w])

            obs, ss, rew, done, info = step_s(sk, ss, act)
            last_done = bool(done)
            ep_ret += float(rew)
            ep_steps += 1

            if last_done and not cur_updated_this_ep:
                gr = bool(info.get("goal_reached", 0))
                pc = bool(info.get("passive_col", 0))
                co = bool(info.get("collision", 0))
                
                rolling_suc = 0.9 * rolling_suc + 0.1 * float(gr * 100.0)
                cur_max_dist = curriculum_max_goal_dist(rolling_suc)
                cur_scenario = 0 if rolling_suc < 25.0 else -1
                cur_updated_this_ep = True

                act_col = co and not pc
                banner = "success" if gr else ("collision" if act_col else ("passive_col" if pc else "timeout"))
                banner_t = fps_target * 2
                print(f"Ep {ep_count}: {banner.upper()}  steps={ep_steps}  ret={ep_ret:+.1f} | Rolling Suc: {rolling_suc:.1f}%")

        cpu_state = jax.device_get(ss.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(ss.lidar_stack)[-1] * (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask = np.array(cpu_state.sp_mask)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info.get("closest_human", 10.0)) - ROBOT_RADIUS - PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, USE_LEGS, show_body)
        
        draw_panel(screen, fonts, ep_count, ep_steps, ep_ret,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, banner, banner_t,
                   rolling_suc, cur_max_dist, cur_scenario, USE_LEGS, raw_lidar, sp_mask)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip()
        
        actual_speed = max(1, int((1.0 / DT) * args.speed)) if not step_once else 30
        clock.tick(actual_speed)

    pygame.quit()

if __name__ == "__main__":
    main()