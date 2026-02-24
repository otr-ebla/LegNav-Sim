"""
jax_eval.py — Visual Evaluation with PyGame
============================================
CHANGES vs previous version:

  IMPROVEMENT — OBS_SIZE updated to 342 (was 339):
    Matches jax_env.py new layout: 9+9+324 = 342.

  UNCHANGED — LiDAR reconstruction fix, dual-pane renderer, all colours,
  eval loop logic, banner display were already correct.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import time
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

from jax_env import (reset_env, step_env,
                     ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, NUM_PEOPLE,
                     NUM_OBS_CIR, NUM_OBS_BOX, MAX_STEPS)
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic, scale_action_to_env

# OBS_SIZE defined here directly to avoid importing jax_train,
# which calls _verify_gpu() at module level and crashes on CPU-only eval.
# Must match jax_train.OBS_SIZE: 9 (pose×3) + 9 (state_vec) + 324 (lidar×3) = 342
OBS_SIZE = 342

# ── Configuration ─────────────────────────────────────────────────────────────
SIM_SIZE   = 800
PANEL_W    = 300
WINDOW_W   = SIM_SIZE + PANEL_W
WINDOW_H   = SIM_SIZE
SCALE      = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET = 30

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG       = (28,  28,  34)
C_FLOOR    = (44,  44,  52)
C_GRID     = (56,  56,  66)
C_WALL     = (190, 190, 205)
C_ROBOT    = (60,  140, 255)
C_ROBOT_H  = (170, 210, 255)
C_GOAL     = (255, 210,  40)
C_GOAL2    = (220, 150,  10)
C_PERSON   = (70,  200,  80)
C_PERSON_D = (255, 160,  40)
C_CIRCLE   = (140,  90,  60)
C_CIRCLE_L = (180, 120,  80)
C_BOX      = ( 90, 110, 145)
C_BOX_L    = (120, 145, 185)
C_RAY_FAR  = ( 50, 200,  50)
C_RAY_NEAR = (220,  50,  50)
C_PANEL    = ( 20,  20,  26)
C_TEXT     = (215, 215, 228)
C_DIM      = (120, 120, 138)
C_SUCCESS  = ( 50, 215,  95)
C_COLLIDE  = (230,  60,  60)
C_TIMEOUT  = (210, 165,  50)


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
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle["params"]


# ── Rendering Helpers ─────────────────────────────────────────────────────────

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

    for ang, dist in zip(angles, raw_lidar):
        ex, ey = W(float(state.x) + dist * math.cos(ang),
                   float(state.y) + dist * math.sin(ang))
        t   = max(0.0, 1.0 - dist / MAX_LIDAR_DIST)
        col = tuple(int(C_RAY_NEAR[i]*t + C_RAY_FAR[i]*(1-t)) for i in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)

    for side in [-1, 1]:
        ang = theta + side * fov / 2.0
        ex, ey = W(float(state.x) + MAX_LIDAR_DIST * math.cos(ang),
                   float(state.y) + MAX_LIDAR_DIST * math.sin(ang))
        pygame.draw.line(surface, C_DIM, (rx, ry), (ex, ey), 1)


def draw_scene(surface, state, raw_lidar, show_lidar, show_arrows):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0); pygame.draw.line(surface, C_GRID, (sx, 0), (sx, SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j); pygame.draw.line(surface, C_GRID, (0, sy), (SIM_SIZE, sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

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

    for i in range(int(state.people.shape[0])):
        px, py   = float(state.people[i, 0]), float(state.people[i, 1])
        vx, vy   = float(state.people[i, 2]), float(state.people[i, 3])
        dist_    = float(state.people[i, 5]) > 0.5
        sx, sy   = W(px, py)
        pr       = max(3, int(PEOPLE_RADIUS * SCALE))
        col      = C_PERSON_D if dist_ else C_PERSON
        pygame.draw.circle(surface, col, (sx, sy), pr)
        pygame.draw.circle(surface, (20, 20, 20), (sx, sy), pr, 1)
        if show_arrows:
            spd = math.hypot(vx, vy)
            if spd > 0.05:
                ax, ay = W(px + vx / spd * 0.5, py + vy / spd * 0.5)
                pygame.draw.line(surface, (20, 120, 20), (sx, sy), (ax, ay), 2)

    rx, ry = W(float(state.x), float(state.y))
    rr     = max(4, int(ROBOT_RADIUS * SCALE))
    pygame.draw.circle(surface, C_ROBOT,   (rx, ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx, ry), rr, 2)
    theta  = float(state.theta)
    hx, hy = W(float(state.x) + ROBOT_RADIUS * 3 * math.cos(theta),
               float(state.y) + ROBOT_RADIUS * 3 * math.sin(theta))
    pygame.draw.line(surface, C_ROBOT_H, (rx, ry), (hx, hy), 3)


def draw_panel(surface, fonts, ep, step, ep_ret, v, w, goal_dist, goal_align,
               closest_h, stats, banner, banner_t):
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
            "success"  : ("✓  GOAL REACHED", C_SUCCESS),
            "collision": ("✗  COLLISION",    C_COLLIDE),
            "timeout"  : ("⏱  TIMEOUT",      C_TIMEOUT),
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
    print(f"🚀 JAX LiDAR-Nav Evaluation  (OBS_SIZE={OBS_SIZE})")

    network = EndToEndActorCritic(action_dim=2)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    ckpt = "checkpoints/ppo_model_best.msgpack"
    try:
        params = load_checkpoint(ckpt)
        print(f"✅ Loaded weights from {ckpt}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint at {ckpt} — running random policy.")

    reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
    fast_reset = jax.jit(reset_stacked)
    fast_step  = jax.jit(step_stacked)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("JAX Indoor-Nav — Evaluation")
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
                "tmo": w[:,3].mean()*100, "pcol": w[:,4].mean()*100, "ret": w[:,0].mean()}

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
                    ep_reward = 0.0; ep_steps = 0
                    banner_t = 0

        if paused:
            clock.tick(10); continue

        # Forward pass (deterministic)
        obs_batch  = obs[None]
        mean, _, _ = network.apply({"params": params}, obs_batch)
        raw_mean   = jnp.squeeze(mean, axis=0)

        max_v      = float(stacked_state.env_state.max_v)
        env_action = scale_action_to_env(raw_mean, max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward)
        ep_steps  += 1

        cpu_state = jax.device_get(stacked_state.env_state)
        cpu_lidar_stack = jax.device_get(stacked_state.lidar_stack)

        # Reconstruct raw lidar distances from normalised stack (inverse of get_obs formula)
        raw_lidar = MAX_LIDAR_DIST - cpu_lidar_stack[-1] * (MAX_LIDAR_DIST - ROBOT_RADIUS)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, show_lidar, show_arrows)
        draw_panel(screen, fonts, ep, ep_steps, ep_reward,
                   float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch,
                   get_stats(), banner, banner_t)

        if banner_t > 0: banner_t -= 1

        pygame.display.flip()
        clock.tick(fps)

        if done:
            goal = bool(info["goal_reached"])
            col  = bool(info["collision"])
            pcol = bool(info["passive_col"])
            tmo  = not goal and not col

            banner   = "success" if goal else ("collision" if col else "timeout")
            banner_t = fps * 2

            ep_hist.append((ep_reward, float(goal), float(col), float(tmo), float(pcol)))

            print(f"  Ep {ep:03d} finished — steps:{ep_steps} reward:{ep_reward:+.1f}  "
                  f"{'GOAL ✅' if goal else 'COLLISION 💥' if col else 'TIMEOUT ⏱️'}")

            ep += 1
            ep_reward = 0.0
            ep_steps  = 0
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)

if __name__ == "__main__":
    main()