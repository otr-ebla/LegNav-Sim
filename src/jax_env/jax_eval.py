"""
jax_eval.py — Visual Evaluation with PyGame
============================================
PATCH — Leg simulation rendering + CLI flags:

  NEW --legs / --no-legs:
    Controls whether the environment uses leg-pair LiDAR geometry.
    Default: --legs (matches training default USE_LEGS=True).
    Pass --no-legs for cylinder-model baseline evaluation.
    The flag is forwarded to jax_env.USE_LEGS before any env is created.

  NEW --scenario N  (0-6 or -1 for random):
    Lock to a specific scenario at startup (single-env eval only).

  RENDERING:
    When USE_LEGS=True, draw_scene draws two small circles per human
    (left leg: slightly lighter, right leg: slightly darker) instead of
    one large circle. The body centre guide ring is drawn faintly so the
    visual collision boundary is still clear.

OBS_SIZE is 662 (216 rays × 3 stack + 9 pose + 5 state).
"""

import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

# ── CLI parsing happens BEFORE any env import so USE_LEGS is set in time ──────
def _parse_args():
    p = argparse.ArgumentParser(description="JAX LiDAR-Nav Evaluation")
    p.add_argument("--legs",    dest="use_legs", action="store_true",  default=True,
                   help="Use leg-pair LiDAR model (default: on)")
    p.add_argument("--no-legs", dest="use_legs", action="store_false",
                   help="Use single-cylinder human model (ablation)")
    p.add_argument("--ckpt",    default="checkpoints/ppo_model_best.msgpack",
                   help="Path to model checkpoint")
    p.add_argument("--scenario", type=int, default=-1,
                   help="Force a specific scenario (-1=random, 0-6=fixed)")
    return p.parse_args()

args = _parse_args()

# Override USE_LEGS before env is imported/traced
import jax_env
jax_env.USE_LEGS = args.use_legs

from jax_env import (reset_env, step_env,
                     ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, NUM_PEOPLE,
                     NUM_OBS_CIR, NUM_OBS_BOX, MAX_STEPS)
from jax_legs import get_leg_positions, LEG_RADIUS, HIP_WIDTH, SHOE_LENGTH, SHOE_WIDTH
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic, scale_action_to_env

OBS_SIZE = 662

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_SIZE   = 800
PANEL_W    = 300
WINDOW_W   = SIM_SIZE + PANEL_W
WINDOW_H   = SIM_SIZE
SCALE      = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET = 8

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG        = (28,  28,  34)
C_FLOOR     = (44,  44,  52)
C_GRID      = (56,  56,  66)
C_WALL      = (190, 190, 205)
C_ROBOT     = (60,  140, 255)
C_ROBOT_H   = (170, 210, 255)
C_GOAL      = (255, 210,  40)
C_GOAL2     = (220, 150,  10)
C_PERSON    = (70,  200,  80)
C_PERSON_D  = (255, 160,  40)
C_LEG_L     = (90,  220, 100)   # left leg — slightly brighter
C_LEG_R     = (50,  170,  65)   # right leg — slightly darker
C_LEG_DL    = (255, 180,  60)   # left leg distracted
C_LEG_DR    = (200, 130,  30)   # right leg distracted
C_BODY_RING = (50,  100,  50)   # faint body-centre guide ring
C_CIRCLE    = (140,  90,  60)
C_CIRCLE_L  = (180, 120,  80)
C_BOX       = ( 90, 110, 145)
C_BOX_L     = (120, 145, 185)
C_RAY_FAR   = ( 50, 200,  50)
C_RAY_NEAR  = (220,  50,  50)
C_PANEL     = ( 20,  20,  26)
C_TEXT      = (215, 215, 228)
C_DIM       = (120, 120, 138)
C_SUCCESS   = ( 50, 215,  95)
C_COLLIDE   = (230,  60,  60)
C_TIMEOUT   = (210, 165,  50)


def W(x, y):
    return int(x * SCALE), int(SIM_SIZE - y * SCALE)


def make_fonts():
    return {
        "big"  : pygame.font.SysFont("monospace", 21, bold=True),
        "mid"  : pygame.font.SysFont("monospace", 15),
        "small": pygame.font.SysFont("monospace", 13),
        "tiny" : pygame.font.SysFont("monospace", 11),
    }


def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    with open(filepath, "rb") as f:
        raw = f.read()
    return flax.serialization.msgpack_restore(raw)["params"]


def draw_star(surface, cx, cy, r_out, r_in, n, colour, border=None):
    pts = []
    for i in range(2 * n):
        angle = math.radians(-90 + i * 180 / n)
        r     = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, colour, pts)
    if border:
        pygame.draw.polygon(surface, border, pts, 2)


def draw_lidar(surface, state, raw_lidar, show):
    if not show or raw_lidar is None:
        return
    rx, ry = W(float(state.x), float(state.y))
    theta  = float(state.theta)
    fov    = float(FOV)
    angles = theta - fov / 2.0 + np.arange(NUM_RAYS) * fov / (NUM_RAYS - 1)

    # Extract the boolean mask from the CPU state
    sp_mask = np.array(state.sp_mask)

    for i, (ang, dist) in enumerate(zip(angles, raw_lidar)):
        ex, ey = W(float(state.x) + dist * math.cos(ang),
                   float(state.y) + dist * math.sin(ang))
        
        # Draw Salt & Pepper corrupted rays in purple
        if sp_mask[i]:
            col = (180, 50, 220)  # Purple
        else:
            t   = max(0.0, 1.0 - dist / MAX_LIDAR_DIST)
            col = tuple(int(C_RAY_NEAR[j]*t + C_RAY_FAR[j]*(1-t)) for j in range(3))
            
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)


# Per-person shoe colours — distinct palette, one colour per person
# Generated to be visually separable on the dark background
_SHOE_PALETTE = [
    (180,  60,  60),   # red
    ( 60, 130, 220),   # blue
    (220, 160,  30),   # amber
    ( 60, 200, 160),   # teal
    (200,  80, 200),   # magenta
    (100, 200,  60),   # lime
    (230, 120,  40),   # orange
    ( 80, 180, 230),   # sky
    (200, 200,  60),   # yellow
    (160,  60, 200),   # purple
    ( 50, 200, 100),   # green
    (220,  80, 130),   # pink
    ( 60, 160, 160),   # cyan
    (200, 140,  80),   # tan
    (130,  80, 200),   # violet
    (200,  60,  80),   # crimson
    ( 80, 220, 200),   # mint
    (220, 200,  80),   # gold
    (160, 120,  60),   # brown
    (100, 140, 220),   # periwinkle
    (180, 220,  80),   # yellow-green
    (220,  80, 160),   # rose
    ( 80, 120, 200),   # cornflower
    (200, 160,  60),   # mustard
    ( 60, 200, 130),   # seafoam
]

def _shoe_colour(person_idx):
    """Return (fill_colour, border_colour) for a given person index."""
    col = _SHOE_PALETTE[person_idx % len(_SHOE_PALETTE)]
    border = tuple(max(0, c - 60) for c in col)
    return col, border


def draw_shoe(surface, foot_x, foot_y, theta, colour, border_colour):
    """
    Draw a rotated rectangle shoe.
    Extends forward from the foot centre by SHOE_LENGTH,
    centred laterally. Toe points in direction theta.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    lat_x, lat_y = -sin_t, cos_t

    half_L = SHOE_LENGTH * 0.5
    half_W = SHOE_WIDTH  * 0.5

    cx = foot_x + cos_t * half_L
    cy = foot_y + sin_t * half_L

    corners = [
        (cx + cos_t * half_L + lat_x * half_W,
         cy + sin_t * half_L + lat_y * half_W),
        (cx + cos_t * half_L - lat_x * half_W,
         cy + sin_t * half_L - lat_y * half_W),
        (cx - cos_t * half_L - lat_x * half_W,
         cy - sin_t * half_L - lat_y * half_W),
        (cx - cos_t * half_L + lat_x * half_W,
         cy - sin_t * half_L + lat_y * half_W),
    ]
    pts = [W(wx, wy) for wx, wy in corners]
    pygame.draw.polygon(surface, colour, pts)
    pygame.draw.polygon(surface, border_colour, pts, 1)


def draw_humans(surface, state, foot_state_np, show_arrows, use_legs):
    """
    Draw all humans. When use_legs=True draws two small leg circles per person
    plus a faint body-centre guide ring. When False draws legacy single circle.
    """
    n_people = int(state.people.shape[0])

    if use_legs:
        # Compute leg positions on CPU (numpy friendly replication of jax_legs)
        left_legs_np, right_legs_np = _read_foot_positions_np(foot_state_np)

        for i in range(n_people):
            px, py   = float(state.people[i, 0]), float(state.people[i, 1])
            vx, vy   = float(state.people[i, 2]), float(state.people[i, 3])
            theta_h  = float(state.people[i, 4])
            dist_    = float(state.people[i, 5]) > 0.5

            sx, sy  = W(px, py)
            body_r  = max(3, int(PEOPLE_RADIUS * SCALE))
            leg_r   = max(2, int(LEG_RADIUS * SCALE))

            # Faint body centre ring (shows collision boundary)
            #pygame.draw.circle(surface, C_BODY_RING, (sx, sy), body_r, 1)

            # Left leg — tinted with person's shoe colour
            lx, ly = W(float(left_legs_np[i, 0]), float(left_legs_np[i, 1]))
            shoe_col, _ = _shoe_colour(i)
            # Brighten for left, slightly darken for right so they're distinguishable
            lc = tuple(min(255, int(c * 1.1)) for c in shoe_col) if not dist_ else C_LEG_DL
            pygame.draw.circle(surface, lc, (lx, ly), leg_r)
            pygame.draw.circle(surface, (20, 20, 20), (lx, ly), leg_r, 1)

            # Right leg
            rx_, ry_ = W(float(right_legs_np[i, 0]), float(right_legs_np[i, 1]))
            rc = tuple(max(0, int(c * 0.75)) for c in shoe_col) if not dist_ else C_LEG_DR
            pygame.draw.circle(surface, rc, (rx_, ry_), leg_r)
            pygame.draw.circle(surface, (20, 20, 20), (rx_, ry_), leg_r, 1)

   
    else:
        # Single-cylinder rendering — same per-person palette colour as leg mode
        for i in range(n_people):
            px, py   = float(state.people[i, 0]), float(state.people[i, 1])
            vx, vy   = float(state.people[i, 2]), float(state.people[i, 3])
            theta_h  = float(state.people[i, 4])
            dist_    = float(state.people[i, 5]) > 0.5
            sx, sy   = W(px, py)
            pr       = max(3, int(PEOPLE_RADIUS * SCALE))
            # Use per-person palette colour (same as leg mode) so visuals are
            # consistent regardless of USE_LEGS.  Distracted persons get their
            # palette colour darkened by 40 % to remain distinguishable.
            shoe_col, border_col = _shoe_colour(i)
            col = tuple(max(0, int(c * 0.6)) for c in shoe_col) if dist_ else shoe_col
            pygame.draw.circle(surface, col, (sx, sy), pr)
            pygame.draw.circle(surface, border_col, (sx, sy), pr, 1)
            speed = math.hypot(vx, vy)
            if show_arrows and speed > 0.05:
                ax, ay = W(px + math.cos(theta_h) * 0.5, py + math.sin(theta_h) * 0.5)
                pygame.draw.line(surface, (20, 120, 20), (sx, sy), (ax, ay), 2)


def _read_foot_positions_np(foot_state_np):
    """Read world-space foot positions directly from foot_state array.
    foot_state_np : (N, 10) — [left_xy, right_xy, phase, stance, swing_target_xy, swing_start_xy]
    Returns left_legs (N,2), right_legs (N,2) — already in world space, no math needed.
    """
    return foot_state_np[:, 0:2], foot_state_np[:, 2:4]


def draw_shoes_only(surface, state, foot_state_np, use_legs):
    """Draw just the shoe polygons (called before lidar rays so rays render on top)."""
    if not use_legs:
        return
    left_legs_np, right_legs_np = _read_foot_positions_np(foot_state_np)
    n_people = int(state.people.shape[0])
    for i in range(n_people):
        left_theta  = float(foot_state_np[i, 10])
        right_theta = float(foot_state_np[i, 11])
        col, border = _shoe_colour(i)
        draw_shoe(surface,
                  float(left_legs_np[i, 0]),  float(left_legs_np[i, 1]),
                  left_theta, col, border)
        draw_shoe(surface,
                  float(right_legs_np[i, 0]), float(right_legs_np[i, 1]),
                  right_theta, col, border)


def draw_scene(surface, state, raw_lidar, foot_state_np, show_lidar, show_arrows, use_legs):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0)
        pygame.draw.line(surface, C_GRID, (sx, 0), (sx, SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j)
        pygame.draw.line(surface, C_GRID, (0, sy), (SIM_SIZE, sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

    # Shoes drawn BEFORE lidar so rays render on top of them
    draw_shoes_only(surface, state, foot_state_np, use_legs)
    draw_lidar(surface, state, raw_lidar, show_lidar)

    for box in np.array(state.obs_boxes):
        cx, cy, hw, hh = box
        sx, sy = W(cx - hw, cy + hh)
        pw = int(2 * hw * SCALE)
        ph = int(2 * hh * SCALE)
        pygame.draw.rect(surface, C_BOX,   (sx, sy, pw, ph))
        pygame.draw.rect(surface, C_BOX_L, (sx, sy, pw, ph), 2)

    for cir in np.array(state.obs_circles):
        cx, cy, r = cir
        sx, sy = W(cx, cy)
        pr = max(2, int(r * SCALE))
        pygame.draw.circle(surface, C_CIRCLE,   (sx, sy), pr)
        pygame.draw.circle(surface, C_CIRCLE_L, (sx, sy), pr, 2)

    gx, gy = W(float(state.goal_x), float(state.goal_y))
    draw_star(surface, gx, gy, int(0.30 * SCALE), int(0.12 * SCALE),
              5, C_GOAL, C_GOAL2)

    draw_humans(surface, state, foot_state_np, show_arrows, use_legs)

    rx, ry = W(float(state.x), float(state.y))
    rr     = max(4, int(ROBOT_RADIUS * SCALE))
    pygame.draw.circle(surface, C_ROBOT,   (rx, ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx, ry), rr, 2)
    theta  = float(state.theta)
    hx, hy = W(float(state.x) + ROBOT_RADIUS * 1.5 * math.cos(theta),
               float(state.y) + ROBOT_RADIUS * 1.5 * math.sin(theta))
    pygame.draw.line(surface, C_ROBOT_H, (rx, ry), (hx, hy), 3)


def draw_panel(surface, fonts, ep, step, ep_ret, v, w, goal_dist, goal_align,
               closest_h, stats, banner, banner_t, use_legs):
    pygame.draw.rect(surface, C_PANEL, (SIM_SIZE, 0, PANEL_W, WINDOW_H))
    pygame.draw.line(surface, C_WALL, (SIM_SIZE, 0), (SIM_SIZE, WINDOW_H), 2)

    y  = 10
    lh = 19
    x0 = SIM_SIZE + 10

    def txt(s, col=C_TEXT, font="mid"):
        nonlocal y
        surface.blit(fonts[font].render(s, True, col), (x0, y))
        y += lh

    def sep():
        nonlocal y
        pygame.draw.line(surface, C_DIM, (x0, y+3), (x0 + PANEL_W-20, y+3), 1)
        y += 10

    txt("─ EVALUATION ─", C_TEXT, "big"); y += 2; sep()
    leg_mode = "LEG-PAIR" if use_legs else "CYLINDER"
    txt(f"Mode   {leg_mode}", C_SUCCESS if use_legs else C_DIM, "small")
    txt(f"Episode  {ep:>5d}",  font="mid")
    txt(f"Step     {step:>4d}", font="mid")
    sep()
    txt(f"v     {v:>+6.3f} m/s",      font="mid")
    txt(f"ω     {w:>+6.3f} rad/s",    font="mid")
    sep()
    txt(f"Goal  {goal_dist:>5.2f} m",       font="mid")
    txt(f"Align {math.degrees(goal_align):>+5.1f}°", font="mid")
    txt(f"Human {closest_h:>5.2f} m",       font="mid")
    sep()
    txt(f"Ep ret {ep_ret:>+7.2f}", C_GOAL, "mid")
    sep()
    txt("── Last 50 episodes ─", C_DIM, "small"); y += 2
    txt(f"  Success  {stats['suc']:>5.1f}%",   C_SUCCESS, "mid")
    txt(f"  Collision{stats['col']:>5.1f}%",   C_COLLIDE, "mid")
    txt(f"  Pass. Col{stats['pcol']:>5.1f}%",  (200, 100, 100), "mid")
    txt(f"  Timeout  {stats['tmo']:>5.1f}%",   C_TIMEOUT, "mid")
    txt(f"  Avg ret  {stats['ret']:>+6.1f}",   C_TEXT,    "mid")
    sep()
    txt("SPACE pause  R reset",  C_DIM, "tiny")
    txt("L lidar  H arrows",     C_DIM, "tiny")
    txt("+/- FPS   Q quit",      C_DIM, "tiny")

    if banner_t > 0:
        label, col = {
            "success"     : ("✓  GOAL REACHED", C_SUCCESS),
            "collision"   : ("✗  COLLISION",    C_COLLIDE),
            "passive_col" : ("🚶 PASSIVE COL",  (200, 100, 100)),
            "timeout"     : ("⏱  TIMEOUT",      C_TIMEOUT),
        }.get(banner, ("", C_TEXT))
        if label:
            surf = fonts["big"].render(label, True, col)
            bx   = SIM_SIZE // 2 - surf.get_width() // 2
            by   = SIM_SIZE // 2 - surf.get_height() // 2
            bg   = pygame.Surface((surf.get_width()+24, surf.get_height()+12))
            bg.fill((20, 20, 26)); bg.set_alpha(220)
            surface.blit(bg,  (bx-12, by-6))
            surface.blit(surf,(bx, by))


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    use_legs = args.use_legs
    print(f"🚀 JAX LiDAR-Nav Evaluation  (OBS_SIZE={OBS_SIZE})")
    print(f"   Human model: {'LEG-PAIR (2×8cm circles)' if use_legs else 'CYLINDER (20cm)'}")

    network = EndToEndActorCritic(action_dim=2)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    try:
        params = load_checkpoint(args.ckpt)
        print(f"✅ Loaded weights from {args.ckpt}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint at {args.ckpt} — running random policy.")

    reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
    fast_reset = jax.jit(reset_stacked)
    fast_step  = jax.jit(step_stacked)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    mode_str = "LEGS" if use_legs else "CYLINDERS"
    pygame.display.set_caption(f"JAX Indoor-Nav — Evaluation [{mode_str}]")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    ep        = 0
    ep_steps  = 0
    ep_reward = 0.0
    ep_hist   = []

    paused      = False
    show_lidar  = True
    show_arrows = True
    fps         = FPS_TARGET
    banner      = ""
    banner_t    = 0

    def get_stats():
        if not ep_hist: return {"suc":0.,"col":0.,"tmo":0.,"pcol":0.,"ret":0.}
        w = np.array(ep_hist[-50:])
        return {"suc": w[:,1].mean()*100, "col": w[:,2].mean()*100,
                "tmo": w[:,3].mean()*100, "pcol": w[:,4].mean()*100,
                "ret": w[:,0].mean()}

    print("🎮 Running. Close window or press Q to stop.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if event.key == pygame.K_SPACE:  paused = not paused
                if event.key == pygame.K_l:      show_lidar  = not show_lidar
                if event.key == pygame.K_h:      show_arrows = not show_arrows
                if event.key == pygame.K_EQUALS: fps = min(fps + 5, 60)
                if event.key == pygame.K_MINUS:  fps = max(fps - 5, 1)
                if event.key == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    ep_reward = 0.0; ep_steps = 0; banner_t = 0

        if paused:
            clock.tick(10); continue

        obs_batch  = obs[None]
        mean, _, _ = network.apply({"params": params}, obs_batch)
        raw_mean   = jnp.squeeze(mean, axis=0)
        max_v      = float(stacked_state.env_state.max_v)
        env_action = scale_action_to_env(raw_mean, max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward)
        ep_steps  += 1

        cpu_state      = jax.device_get(stacked_state.env_state)
        cpu_lidar_stack = jax.device_get(stacked_state.lidar_stack)
        foot_state_np  = np.array(cpu_state.foot_state)

        raw_lidar = MAX_LIDAR_DIST - cpu_lidar_stack[-1] * (MAX_LIDAR_DIST - ROBOT_RADIUS)

        dx     = float(cpu_state.goal_x) - float(cpu_state.x)
        dy     = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs)
        draw_panel(screen, fonts, ep, ep_steps, ep_reward,
                   float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch,
                   get_stats(), banner, banner_t, use_legs)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip()
        clock.tick(fps)

        if done:
            goal       = bool(info["goal_reached"])
            col        = bool(info["collision"])
            pcol       = bool(info["passive_col"])
            active_col = col and not pcol
            tmo        = not goal and not col

            banner   = "success" if goal else ("collision" if active_col else \
                        ("passive_col" if pcol else "timeout"))
            banner_t = fps * 2

            ep_hist.append((ep_reward, float(goal), float(active_col), float(tmo), float(pcol)))
            print(f"  Ep {ep:03d} — steps:{ep_steps} reward:{ep_reward:+.1f}  "
                  f"{'GOAL ✅' if goal else 'PASSIVE COL 🚶' if pcol else 'COLLISION 💥' if active_col else 'TIMEOUT ⏱️'}")

            ep += 1; ep_reward = 0.0; ep_steps = 0
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)


if __name__ == "__main__":
    main()
