#!/usr/bin/env python3
"""
debug_viz.py — Pygame Debug Visualizer
=======================================
Simple clean rendering of a single environment.

Usage:
  python debug_viz.py                    # random policy
  python debug_viz.py --checkpoint path  # trained policy
  python debug_viz.py --scenario 0       # force scenario
  python debug_viz.py --speed 0.5        # slower playback
  python debug_viz.py --pause            # start paused

Controls:
  SPACE  — pause / resume
  RIGHT  — single step (when paused)
  R      — reset episode
  Q/ESC  — quit
"""

import os
import sys
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--scenario", type=int, default=-1)
parser.add_argument("--min-goal-dist", type=float, default=1.5)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", action="store_true")
parser.add_argument("--no-ghost", action="store_true", help="Disable ghost (eval mode)")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_env_multi import reset_env, step_env, NUM_PEOPLE
from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, GOAL_RADIUS, USE_LEGS)
from jax_wrappers import make_stacked_env
from jax_legs import get_shoe_boxes, LEG_RADIUS

import pygame

# ── Display ──────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 800, 850
HUD_H = 50


def w2s(x, y, sc, ox, oy):
    return int(ox + x * sc), int(oy + (ROOM_H - y) * sc)


def r2px(r, sc):
    return max(1, int(r * sc))


def load_policy(path):
    if path is None:
        return None
    from jax_network import EndToEndActorCritic
    import flax.serialization, optax
    net = EndToEndActorCritic(action_dim=2)
    dummy = jnp.zeros((1, 342))
    p = net.init(jax.random.PRNGKey(0), dummy)["params"]
    try:
        o = optax.adam(1e-4).init(p)
        b = flax.serialization.from_bytes({"params": p, "opt_state": o},
                                           open(path, "rb").read())
        p = b["params"]
    except Exception:
        p = flax.serialization.from_bytes(p, open(path, "rb").read())
    print(f"Loaded: {path}")
    return net, p


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RL Nav Debug")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)

    ghost = not args.no_ghost
    policy = load_policy(args.checkpoint)

    reset_s, step_s = make_stacked_env(reset_env, step_env, stack_dim=3,
                                        ghost_robot=ghost)
    rng = jax.random.PRNGKey(args.seed)

    def do_reset(k):
        return reset_s(k, min_goal_dist=args.min_goal_dist)

    rng, rk = jax.random.split(rng)
    obs, ss = do_reset(rk)

    paused = args.pause
    ep_ret = 0.0
    ep_steps = 0
    ep_count = 0
    last_r = 0.0
    last_done = False
    last_info = {}
    last_v = 0.0
    last_w = 0.0

    running = True
    while running:
        step_once = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    step_once = True
                elif ev.key == pygame.K_r:
                    rng, rk = jax.random.split(rng)
                    obs, ss = do_reset(rk)
                    ep_ret, ep_steps = 0.0, 0
                    ep_count += 1
                    last_done = False
                    last_info = {}

        # ── Step ─────────────────────────────────────────────────────────────
        if (not paused or step_once) and not last_done:
            rng, ak, sk = jax.random.split(rng, 3)
            if policy:
                net, par = policy
                m, ls, val = net.apply({"params": par}, obs[None])
                from jax_network import sample_action, scale_action_to_env
                raw, _ = sample_action(ak, m[0], ls[0])
                act = scale_action_to_env(raw, ss.env_state.max_v)
            else:
                mv = float(ss.env_state.max_v)
                v = jax.random.uniform(ak, minval=0.0, maxval=mv)
                w = jax.random.uniform(jax.random.fold_in(ak, 1),
                                        minval=-1.0, maxval=1.0)
                act = jnp.array([v, w])

            last_v, last_w = float(act[0]), float(act[1])
            obs, ss, rew, done, info = step_s(sk, ss, act)
            last_r = float(rew)
            last_done = bool(done)
            last_info = {k: (float(v) if hasattr(v, 'ndim') and v.ndim == 0 else v)
                         for k, v in info.items()}
            ep_ret += last_r
            ep_steps += 1

            if last_done:
                gr = last_info.get("goal_reached", 0)
                pc = last_info.get("passive_col", 0)
                co = last_info.get("collision", 0)
                tag = ("SUCCESS" if gr else
                       "PASSIVE" if pc else
                       "COLLISION" if co else "TIMEOUT")
                print(f"Ep {ep_count}: {tag}  steps={ep_steps}  ret={ep_ret:+.1f}")

        # ── Extract state ────────────────────────────────────────────────────
        st = ss.env_state
        rx, ry, rt = float(st.x), float(st.y), float(st.theta)
        gx, gy = float(st.goal_x), float(st.goal_y)
        ppl = np.array(st.people)
        cirs = np.array(st.obs_circles)
        bxs = np.array(st.obs_boxes)

        room_px = WIN_H - HUD_H
        sc = room_px / max(ROOM_W, ROOM_H)
        ox = (WIN_W - ROOM_W * sc) / 2
        oy = (room_px - ROOM_H * sc) / 2

        def _w(x, y):
            return w2s(x, y, sc, ox, oy)

        def _r(r):
            return r2px(r, sc)

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill((25, 25, 30))

        # Grid
        for i in range(int(ROOM_W) + 1):
            pygame.draw.line(screen, (40, 40, 45), _w(i, 0), _w(i, ROOM_H), 1)
        for j in range(int(ROOM_H) + 1):
            pygame.draw.line(screen, (40, 40, 45), _w(0, j), _w(ROOM_W, j), 1)

        # Walls
        pygame.draw.polygon(screen, (160, 160, 160),
            [_w(0, 0), _w(ROOM_W, 0), _w(ROOM_W, ROOM_H), _w(0, ROOM_H)], 2)

        # Obstacle circles (filled)
        for i in range(cirs.shape[0]):
            cx, cy, r = cirs[i]
            if r > 0:
                pygame.draw.circle(screen, (90, 90, 100), _w(cx, cy), _r(r))

        # Obstacle boxes (filled)
        for i in range(bxs.shape[0]):
            cx, cy, hw, hh = bxs[i]
            if hw > 0:
                tl = _w(cx - hw, cy + hh)
                br = _w(cx + hw, cy - hh)
                pygame.draw.rect(screen, (90, 90, 100),
                    pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]))

        # Goal
        gsx, gsy = _w(gx, gy)
        pygame.draw.circle(screen, (50, 220, 80), (gsx, gsy), _r(GOAL_RADIUS), 2)
        pygame.draw.circle(screen, (50, 220, 80), (gsx, gsy), 3)

        # Humans (body circles + heading)
        for i in range(ppl.shape[0]):
            px, py = ppl[i, 0], ppl[i, 1]
            if px < -100:
                continue
            gi = ppl[i, 10] if ppl.shape[1] > 10 else 0
            if gi < 0:
                continue

            hsx, hsy = _w(px, py)
            pygame.draw.circle(screen, (220, 80, 60), (hsx, hsy),
                               _r(PEOPLE_RADIUS), 2)

            th = ppl[i, 4]
            spd = math.sqrt(ppl[i, 2]**2 + ppl[i, 3]**2)
            if spd > 0.05:
                al = _r(0.4)
                pygame.draw.line(screen, (220, 80, 60), (hsx, hsy),
                    (int(hsx + al * math.cos(th)),
                     int(hsy - al * math.sin(th))), 2)

        # Shoe boxes (USE_LEGS only)
        if USE_LEGS:
            sbs = np.array(get_shoe_boxes(st.people, st.foot_state))
            for i in range(sbs.shape[0]):
                cx, cy, hw, hh = sbs[i]
                if cx < -100:
                    continue
                tl = _w(cx - hw, cy + hh)
                br = _w(cx + hw, cy - hh)
                pygame.draw.rect(screen, (255, 160, 40),
                    pygame.Rect(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]), 1)

        # Robot
        rsx, rsy = _w(rx, ry)
        pygame.draw.circle(screen, (60, 140, 255), (rsx, rsy), _r(ROBOT_RADIUS))
        al = _r(ROBOT_RADIUS) + _r(0.15)
        pygame.draw.line(screen, (255, 255, 255), (rsx, rsy),
            (int(rsx + al * math.cos(rt)),
             int(rsy - al * math.sin(rt))), 2)

        # ── DONE overlay ─────────────────────────────────────────────────────
        if last_done:
            ov = pygame.Surface((WIN_W, WIN_H - HUD_H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 100))
            screen.blit(ov, (0, 0))

            gr = last_info.get("goal_reached", 0)
            pc = last_info.get("passive_col", 0)
            co = last_info.get("collision", 0)
            if gr:
                txt, col = "GOAL REACHED", (50, 220, 80)
            elif pc:
                txt, col = "PASSIVE COLLISION", (255, 160, 40)
            elif co:
                txt, col = "ACTIVE COLLISION", (255, 50, 50)
            else:
                txt, col = "TIMEOUT", (180, 180, 60)

            big = pygame.font.SysFont("monospace", 28, bold=True)
            ts = big.render(txt, True, col)
            screen.blit(ts, ((WIN_W - ts.get_width()) // 2,
                             (WIN_H - HUD_H) // 2 - 20))
            sub = font.render(f"ret={ep_ret:+.1f}  steps={ep_steps}  [R]=reset",
                              True, (160, 160, 160))
            screen.blit(sub, ((WIN_W - sub.get_width()) // 2,
                              (WIN_H - HUD_H) // 2 + 15))

        # ── HUD ──────────────────────────────────────────────────────────────
        hud_y = WIN_H - HUD_H
        pygame.draw.rect(screen, (20, 20, 25), (0, hud_y, WIN_W, HUD_H))

        gd = math.sqrt((rx - gx)**2 + (ry - gy)**2)
        status = "PAUSED" if paused else ("DONE" if last_done else "RUNNING")
        mode = "TRAINED" if policy else "RANDOM"
        gh = "ghost" if ghost else "visible"

        l1 = (f"{mode} {gh} | s={ep_steps} r={last_r:+.2f} ret={ep_ret:+.1f}"
              f" | goal={gd:.1f}m v={last_v:.2f} w={last_w:.2f} | {status}")
        screen.blit(font.render(l1, True, (200, 200, 200)), (8, hud_y + 5))

        l2 = (f"ep={ep_count} max_v={float(st.max_v):.1f}"
              f" | [SPACE]=pause [R]=reset [→]=step [Q]=quit")
        screen.blit(font.render(l2, True, (120, 120, 120)), (8, hud_y + 22))

        pygame.display.flip()

        fps = (max(1, int((1.0 / DT) * args.speed))
               if (not paused or step_once) else 30)
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()