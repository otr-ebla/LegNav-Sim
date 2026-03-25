#!/usr/bin/env python3
"""
debug_viz.py — Visual single-env PPO training debugger
========================================================
Runs exactly the same logic as jax_ppo.py (curriculum, ghost_prob, rewards)
but with 1 environment rendered using jax_eval_multi.py graphics.

Usage:
  python debug_viz.py
  python debug_viz.py --speed 0.5
  python debug_viz.py --pause

Controls:
  SPACE — pause / resume      RIGHT — single step (when paused)
  R     — reset episode       L     — toggle LiDAR
  H     — toggle arrows       Q/ESC — quit
"""

import os, sys, argparse, math, functools, random as pyrandom
os.environ["JAX_PLATFORMS"] = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import jax, jax.numpy as jnp, numpy as np, pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax_env as _jax_env
_jax_env.USE_LEGS = True

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS, DT)
from jax_env_multi import reset_env, step_env
from jax_legs import LEG_RADIUS, SHOE_LENGTH, SHOE_WIDTH
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic, sample_action, scale_action_to_env

# ── Curriculum (copied from jax_ppo.py) ──────────────────────────────────────
CURRICULUM_STAGES = [
    (25.0, 1.5), (38.0, 2.5), (50.0, 4.0), (60.0, 5.0),
    (70.0, 6.5), (80.0, 8.0), (101., 9.0),
]
GHOST_PROB_STAGES = [
    (50.0, 1.0), (65.0, 0.8), (78.0, 0.6), (101., 0.4),
]

def curriculum_min_goal_dist(suc_pct):
    for t, d in CURRICULUM_STAGES:
        if suc_pct < t: return d
    return CURRICULUM_STAGES[-1][1]

def curriculum_ghost_prob(suc_pct):
    for t, p in GHOST_PROB_STAGES:
        if suc_pct < t: return p
    return GHOST_PROB_STAGES[-1][1]

def _curriculum_stage(suc_pct):
    for i, (t, _) in enumerate(CURRICULUM_STAGES):
        if suc_pct < t: return i
    return len(CURRICULUM_STAGES) - 1

# ── Rendering (from jax_eval_multi.py) ───────────────────────────────────────
SIM_SIZE = 800; PANEL_W = 300
WINDOW_W = SIM_SIZE + PANEL_W; WINDOW_H = SIM_SIZE
SCALE = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET = 30

C_BG=(28,28,34); C_FLOOR=(44,44,52); C_GRID=(56,56,66)
C_WALL=(190,190,205); C_ROBOT=(60,140,255); C_ROBOT_H=(170,210,255)
C_GOAL=(255,210,40); C_GOAL2=(220,150,10)
C_LEG_L=(90,220,100); C_LEG_R=(50,170,65)
C_CIRCLE=(140,90,60); C_CIRCLE_L=(180,120,80)
C_BOX=(90,110,145); C_BOX_L=(120,145,185)
C_RAY_FAR=(50,200,50); C_RAY_NEAR=(220,50,50)
C_PANEL=(20,20,26); C_TEXT=(215,215,228); C_DIM=(120,120,138)
C_SUCCESS=(50,215,95); C_COLLIDE=(230,60,60); C_TIMEOUT=(210,165,50)
C_BODY_RING=(50,100,50)

_SHOE_PALETTE = [
    (180,60,60),(60,130,220),(220,160,30),(60,200,160),
    (200,80,200),(100,200,60),(230,120,40),(80,180,230),
    (200,200,60),(160,60,200),(50,200,100),(220,80,130),
    (60,160,160),(200,140,80),(130,80,200),(200,60,80),
]

def W(x, y): return int(x * SCALE), int(SIM_SIZE - y * SCALE)
def _shoe_colour(i):
    c = _SHOE_PALETTE[i % len(_SHOE_PALETTE)]
    return c, tuple(max(0, v-60) for v in c)

def make_fonts():
    return {
        "big": pygame.font.SysFont("monospace", 21, bold=True),
        "mid": pygame.font.SysFont("monospace", 15),
        "small": pygame.font.SysFont("monospace", 13),
        "tiny": pygame.font.SysFont("monospace", 11),
    }

def draw_star(surface, cx, cy, r_out, r_in, n, color, border=None):
    pts = []
    for i in range(2*n):
        angle = math.radians(-90 + i*180/n)
        r = r_out if i%2==0 else r_in
        pts.append((cx + r*math.cos(angle), cy + r*math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)
    if border: pygame.draw.polygon(surface, border, pts, 2)

def draw_lidar(surface, state, raw_lidar, show):
    if not show or raw_lidar is None: return
    rx, ry = W(float(state.x), float(state.y))
    theta = float(state.theta)
    angles = theta - FOV/2.0 + np.arange(NUM_RAYS)*FOV/(NUM_RAYS-1)
    sp_mask = np.array(state.sp_mask)
    for i, (ang, dist) in enumerate(zip(angles, raw_lidar)):
        ex, ey = W(state.x + dist*math.cos(ang), state.y + dist*math.sin(ang))
        if sp_mask[i]:
            col = (180,50,220)
        else:
            t = max(0.0, 1.0 - dist/MAX_LIDAR_DIST)
            col = tuple(int(C_RAY_NEAR[j]*t + C_RAY_FAR[j]*(1-t)) for j in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)

def draw_shoe(surface, fx, fy, theta, col, border):
    ct, st_ = math.cos(theta), math.sin(theta)
    lx, ly = -st_, ct
    hL, hW = SHOE_LENGTH*0.5, SHOE_WIDTH*0.5
    cx, cy = fx + ct*hL, fy + st_*hL
    corners = [
        (cx+ct*hL+lx*hW, cy+st_*hL+ly*hW), (cx+ct*hL-lx*hW, cy+st_*hL-ly*hW),
        (cx-ct*hL-lx*hW, cy-st_*hL-ly*hW), (cx-ct*hL+lx*hW, cy-st_*hL+ly*hW),
    ]
    pts = [W(wx, wy) for wx, wy in corners]
    pygame.draw.polygon(surface, col, pts)
    pygame.draw.polygon(surface, border, pts, 1)

def draw_humans(surface, state, foot_state_np, show_arrows, use_legs=True):
    n = int(state.people.shape[0])
    left_legs = foot_state_np[:, 0:2]; right_legs = foot_state_np[:, 2:4]
    for i in range(n):
        if float(state.people[i, 10]) < 0: continue
        px, py = float(state.people[i,0]), float(state.people[i,1])
        vx, vy = float(state.people[i,2]), float(state.people[i,3])
        theta_h = float(state.people[i,4])
        col, border = _shoe_colour(i)
        if use_legs:
            draw_shoe(surface, float(left_legs[i,0]), float(left_legs[i,1]), theta_h, col, border)
            draw_shoe(surface, float(right_legs[i,0]), float(right_legs[i,1]), theta_h, col, border)
            leg_r = max(2, int(LEG_RADIUS*SCALE))
            lx, ly = W(float(left_legs[i,0]), float(left_legs[i,1]))
            rx_, ry_ = W(float(right_legs[i,0]), float(right_legs[i,1]))
            pygame.draw.circle(surface, tuple(min(255,int(c*1.1)) for c in col), (lx, ly), leg_r)
            pygame.draw.circle(surface, (20,20,20), (lx, ly), leg_r, 1)
            pygame.draw.circle(surface, tuple(max(0,int(c*0.75)) for c in col), (rx_, ry_), leg_r)
            pygame.draw.circle(surface, (20,20,20), (rx_, ry_), leg_r, 1)
        else:
            sx, sy = W(px, py)
            pr = max(3, int(_jax_env.PEOPLE_RADIUS*SCALE))
            pygame.draw.circle(surface, col, (sx,sy), pr)
            pygame.draw.circle(surface, border, (sx,sy), pr, 1)
            speed = math.hypot(vx, vy)
            if show_arrows and speed > 0.05:
                ax, ay = W(px+math.cos(theta_h)*0.5, py+math.sin(theta_h)*0.5)
                pygame.draw.line(surface, (20,120,20), (sx,sy), (ax,ay), 2)

def draw_scene(surface, state, raw_lidar, foot_state_np, show_lidar, show_arrows):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W)+1):
        sx, _ = W(i,0); pygame.draw.line(surface, C_GRID, (sx,0), (sx,SIM_SIZE))
    for j in range(int(ROOM_H)+1):
        _, sy = W(0,j); pygame.draw.line(surface, C_GRID, (0,sy), (SIM_SIZE,sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)
    draw_lidar(surface, state, raw_lidar, show_lidar)
    for box in np.array(state.obs_boxes):
        cx_, cy_, hw, hh = box
        if hw > 0:
            sx, sy = W(cx_-hw, cy_+hh)
            pygame.draw.rect(surface, C_BOX, (sx,sy,int(2*hw*SCALE),int(2*hh*SCALE)))
            pygame.draw.rect(surface, C_BOX_L, (sx,sy,int(2*hw*SCALE),int(2*hh*SCALE)), 2)
    for cir in np.array(state.obs_circles):
        cx_, cy_, r = cir
        if r > 0:
            sx, sy = W(cx_, cy_); pr = max(2, int(r*SCALE))
            pygame.draw.circle(surface, C_CIRCLE, (sx,sy), pr)
            pygame.draw.circle(surface, C_CIRCLE_L, (sx,sy), pr, 2)
    gx, gy = W(float(state.goal_x), float(state.goal_y))
    draw_star(surface, gx, gy, int(0.30*SCALE), int(0.12*SCALE), 5, C_GOAL, C_GOAL2)
    draw_humans(surface, state, foot_state_np, show_arrows)
    rx, ry = W(float(state.x), float(state.y)); rr = max(4, int(ROBOT_RADIUS*SCALE))
    pygame.draw.circle(surface, C_ROBOT, (rx,ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx,ry), rr, 2)
    hx, hy = W(state.x + ROBOT_RADIUS*3*math.cos(float(state.theta)),
               state.y + ROBOT_RADIUS*3*math.sin(float(state.theta)))
    pygame.draw.line(surface, C_ROBOT_H, (rx,ry), (hx,hy), 3)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("PPO Training Debug (1 env)")
    clock = pygame.time.Clock(); fonts = make_fonts()
    rng = jax.random.PRNGKey(args.seed)

    # ── Network (random init — this is untrained) ────────────────────────────
    OBS_SIZE = 342
    net = EndToEndActorCritic(action_dim=2)
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jnp.zeros((1, OBS_SIZE)))["params"]

    # ── Curriculum state ─────────────────────────────────────────────────────
    rolling_suc = 0.0
    cur_min_dist = curriculum_min_goal_dist(0.0)
    cur_ghost = curriculum_ghost_prob(0.0)
    cur_stage = _curriculum_stage(0.0)

    def build_env(ghost_prob):
        ghost_robot = (ghost_prob >= 0.5)
        rs, ss = make_stacked_env(reset_env, step_env, stack_dim=3,
                                   ghost_robot=ghost_robot)
        return rs, ss, ghost_robot

    reset_s, step_s, ghost_robot = build_env(cur_ghost)

    def do_reset(k):
        return reset_s(k, min_goal_dist=cur_min_dist)

    rng, rk = jax.random.split(rng)
    obs, stacked_state = do_reset(rk)

    # ── Episode tracking ─────────────────────────────────────────────────────
    paused = args.pause
    show_lidar = True; show_arrows = True
    ep = 0; ep_steps = 0; ep_ret = 0.0
    ep_hist = []  # list of (ret, goal, acol, tmo, pcol)
    banner = ""; banner_t = 0
    last_r = 0.0

    def get_stats():
        if not ep_hist: return {"suc":0.,"col":0.,"tmo":0.,"pcol":0.}
        w = np.array(ep_hist[-50:])
        return {"suc":w[:,1].mean()*100, "col":w[:,2].mean()*100,
                "tmo":w[:,3].mean()*100, "pcol":w[:,4].mean()*100}

    print(f"Training debug | stage={cur_stage} dist={cur_min_dist:.1f}m ghost={cur_ghost:.1f}")
    print(f"Keys: SPACE=pause  R=reset  L=lidar  H=arrows  Q=quit")

    while True:
        step_once = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if k == pygame.K_SPACE: paused = not paused
                if k == pygame.K_RIGHT: step_once = True
                if k == pygame.K_l: show_lidar = not show_lidar
                if k == pygame.K_h: show_arrows = not show_arrows
                if k == pygame.K_r:
                    rng, rk = jax.random.split(rng)
                    obs, stacked_state = do_reset(rk)
                    ep_ret, ep_steps = 0.0, 0

        # ── Step (same as jax_ppo: sample from policy, scale, step) ──────────
        if (not paused or step_once):
            rng, ak, sk = jax.random.split(rng, 3)

            mean, logstd, value = net.apply({"params": params}, obs[None])
            raw_action, log_prob = sample_action(ak, mean[0], logstd[0])
            env_action = scale_action_to_env(raw_action,
                                              stacked_state.env_state.max_v)

            obs, stacked_state, rew, done, info = step_s(sk, stacked_state,
                                                          env_action)
            last_r = float(rew)
            ep_ret += last_r
            ep_steps += 1

            if bool(done):
                info_f = {k: (float(v) if hasattr(v,'ndim') and v.ndim==0 else v)
                          for k, v in info.items()}
                gr = info_f.get("goal_reached", 0)
                co = info_f.get("collision", 0)
                pc = info_f.get("passive_col", 0)
                tmo = (not gr and not co)
                acol = (co and not pc)

                banner = ("success" if gr else
                          "collision" if acol else
                          "passive_col" if pc else "timeout")
                banner_t = FPS_TARGET * 2
                ep_hist.append((ep_ret, float(bool(gr)), float(bool(acol)),
                                float(bool(tmo)), float(bool(pc))))

                # ── Curriculum update (same as jax_ppo.py) ───────────────────
                suc_pct = get_stats()["suc"]
                rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct

                new_dist = curriculum_min_goal_dist(rolling_suc)
                new_ghost = curriculum_ghost_prob(rolling_suc)
                new_stage = _curriculum_stage(rolling_suc)

                if new_dist > cur_min_dist or new_ghost < cur_ghost:
                    cur_min_dist = new_dist
                    cur_ghost = new_ghost
                    cur_stage = new_stage
                    reset_s, step_s, ghost_robot = build_env(cur_ghost)
                    print(f"  → Curriculum: stage={cur_stage} dist={cur_min_dist:.1f}m "
                          f"ghost={cur_ghost:.1f}")

                tag = banner.upper()
                print(f"Ep {ep:>4d}: {tag:<12s} steps={ep_steps:>4d} "
                      f"ret={ep_ret:>+7.1f} | "
                      f"suc={suc_pct:.0f}% stage={cur_stage} "
                      f"dist={cur_min_dist:.1f}m")

                ep += 1; ep_ret = 0.0; ep_steps = 0
                rng, rk = jax.random.split(rng)
                obs, stacked_state = do_reset(rk)

        # ── Render ───────────────────────────────────────────────────────────
        cpu_state = jax.device_get(stacked_state.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * \
                    (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows)

        # ── Panel ────────────────────────────────────────────────────────────
        pygame.draw.rect(screen, C_PANEL, (SIM_SIZE, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, C_WALL, (SIM_SIZE,0), (SIM_SIZE,WINDOW_H), 2)
        y = 10; lh = 19; x0 = SIM_SIZE + 10
        stats = get_stats()

        def txt(s, col=C_TEXT, font="mid"):
            nonlocal y
            screen.blit(fonts[font].render(s, True, col), (x0, y)); y += lh
        def sep():
            nonlocal y
            pygame.draw.line(screen, C_DIM, (x0,y+3), (x0+PANEL_W-20,y+3), 1); y += 10

        txt("─ PPO TRAIN DEBUG ─", C_TEXT, "big"); y += 2; sep()
        txt(f"Ghost: {'YES' if ghost_robot else 'NO'}   LEGS: YES", C_DIM, "small")
        txt(f"Curriculum stage {cur_stage}", C_SUCCESS)
        txt(f"min_goal_dist {cur_min_dist:.1f} m")
        txt(f"ghost_prob    {cur_ghost:.1f}"); sep()
        txt(f"Episode  {ep:>5d}")
        txt(f"Step     {ep_steps:>4d}")

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist = math.hypot(dx, dy)
        txt(f"Goal     {gdist:>5.2f} m")
        txt(f"max_v    {float(cpu_state.max_v):>5.2f} m/s")
        txt(f"Reward   {last_r:>+7.2f}")
        txt(f"Ep ret   {ep_ret:>+7.2f}", C_GOAL); sep()

        txt(f"Rolling suc {rolling_suc:.1f}%", C_DIM, "small"); y += 2
        txt("── Last 50 episodes ─", C_DIM, "small"); y += 2
        txt(f"  Success  {stats['suc']:>5.1f}%", C_SUCCESS)
        txt(f"  Collision{stats['col']:>5.1f}%", C_COLLIDE)
        txt(f"  Pass.Col {stats['pcol']:>5.1f}%", (200,100,100))
        txt(f"  Timeout  {stats['tmo']:>5.1f}%", C_TIMEOUT); sep()
        txt("SPACE=pause  R=reset  L=lidar", C_DIM, "tiny")
        txt("H=arrows  →=step  Q=quit", C_DIM, "tiny")

        # Banner
        if banner_t > 0:
            label, col = {
                "success":     ("✓  GOAL REACHED",  C_SUCCESS),
                "collision":   ("X  COLLISION",      C_COLLIDE),
                "passive_col": ("🚶 PASSIVE COL",    (200,100,100)),
                "timeout":     ("⏱  TIMEOUT",        C_TIMEOUT),
            }.get(banner, ("", C_TEXT))
            if label:
                surf = fonts["big"].render(label, True, col)
                bx = SIM_SIZE + PANEL_W//2 - surf.get_width()//2
                by = WINDOW_H - 80
                bg = pygame.Surface((surf.get_width()+24, surf.get_height()+12))
                bg.fill((20,20,26)); bg.set_alpha(220)
                screen.blit(bg, (bx-12, by-6)); screen.blit(surf, (bx, by))
            banner_t -= 1

        if paused:
            ps = fonts["big"].render("PAUSED", True, (255,255,100))
            screen.blit(ps, (SIM_SIZE//2 - ps.get_width()//2, 10))

        pygame.display.flip()
        fps = max(1, int((1.0/DT)*args.speed)) if (not paused or step_once) else 30
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main()