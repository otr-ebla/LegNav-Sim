"""
perception/live_eval.py — Live PPO navigation + online people perception
=========================================================================
Interactive viewer in the style of legnav/evaluation/jax_eval_multi.py:
the trained PPO policy drives the robot in the background while the
perception network (LegDetNet) runs online on its own 8-frame LiDAR stack
and draws a dashed red circle around every detected human.

The simulation, policy inference and rendering are reused from
jax_eval_multi; this script adds the rolling 8-frame perception buffer,
per-frame LegDetNet inference + vote decoding, and the detection overlay.

Usage (from the repo root):
    python -m perception.live_eval
    python -m perception.live_eval --ppo-ckpt checkpoints/ppo/ppo_tanh_inside_final.msgpack \
                                   --perception-ckpt checkpoints/perception/perception_best.msgpack

Keys:
  0-6   Lock scenario    7     Random mode
  R     Reset episode    Space Pause
  L     Toggle LiDAR     H     Toggle arrows
  B     Toggle body ring D     Toggle detections
  S     Cycle FPS        Q/Esc Quit
"""

import argparse
import math
import os
import random
import sys

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="Live PPO + perception eval")
    p.add_argument("--ppo-ckpt", default="",
                   help="PPO checkpoint (default: same as jax_eval_multi --algo ppo).")
    p.add_argument("--perception-ckpt", default="",
                   help="Perception checkpoint (default: checkpoints/perception/perception_best.msgpack).")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Exit after N rendered frames (0 = run forever; used for smoke tests).")
    return p.parse_args()


args = _parse_args()

# jax_eval_multi parses its own CLI at import time — hand it an empty argv so
# it falls back to defaults (legs on, sensor noise on, JAX on CPU).
_argv, sys.argv = sys.argv, sys.argv[:1]
from legnav.evaluation import jax_eval_multi as JEM
sys.argv = _argv

import jax
import jax.numpy as jnp
import pygame

from perception.config import PerceptionConfig as PC
from perception.perception_network import PerceptionNet, decode_detections
from perception.train_perception import load_checkpoint, CKPT_BEST

MAX_EVAL_GOAL_DIST = 9.0
C_DETECT = (255, 60, 60)   # dashed detection circle


# ── Detection overlay ─────────────────────────────────────────────────────────

def draw_dashed_circle(surface, color, cx, cy, radius_px,
                       dash_len=7, gap_len=5, width=2):
    """Dashed circle via the dashed-polyline helper from jax_eval_multi."""
    n = max(24, int(radius_px))
    pts = [(cx + radius_px * math.cos(2 * math.pi * i / n),
            cy + radius_px * math.sin(2 * math.pi * i / n))
           for i in range(n + 1)]
    JEM.draw_dashed_polyline(surface, color, pts, dash_len, gap_len, width)


def draw_detections(surface, dets_world, scale, sim_h, fonts, scores):
    r_px = max(6, int(JEM.PEOPLE_RADIUS * scale))
    for (wx, wy), sc in zip(dets_world, scores):
        px, py = int(wx * scale), int(sim_h - wy * scale)
        draw_dashed_circle(surface, C_DETECT, px, py, r_px)
        pygame.draw.line(surface, C_DETECT, (px - 4, py), (px + 4, py), 2)
        pygame.draw.line(surface, C_DETECT, (px, py - 4), (px, py + 4), 2)
        lbl = fonts["tiny"].render(f"{sc:.2f}", True, C_DETECT)
        surface.blit(lbl, (px + r_px - 2, py - r_px - 2))


def ego_to_world(dets_ego, rx, ry, rtheta):
    if len(dets_ego) == 0:
        return np.zeros((0, 2))
    c, s = math.cos(rtheta), math.sin(rtheta)
    wx = rx + c * dets_ego[:, 0] - s * dets_ego[:, 1]
    wy = ry + s * dets_ego[:, 0] + c * dets_ego[:, 1]
    return np.stack([wx, wy], axis=-1)


# ── Compact side panel ────────────────────────────────────────────────────────

def draw_panel(surface, fonts, ep, step, v, w, n_det, n_people, show_det,
               sim_x, win_h):
    pygame.draw.rect(surface, JEM.C_PANEL, (sim_x, 0, JEM.PANEL_W, win_h))
    pygame.draw.line(surface, JEM.C_WALL, (sim_x, 0), (sim_x, win_h), 2)
    y = 10
    x0 = sim_x + 10

    def txt(s, col=JEM.C_TEXT, font="mid", dy=19):
        nonlocal y
        surface.blit(fonts[font].render(s, True, col), (x0, y))
        y += dy

    txt("─ PPO + PERCEPTION ─", JEM.C_TEXT, "big", 26)
    txt(f"Episode  {ep:>4d}")
    txt(f"Step     {step:>4d}")
    txt(f"v  {v:>+6.3f} m/s")
    txt(f"w  {w:>+6.3f} rad/s", dy=26)
    txt("── Perception ─", JEM.C_DIM, "small")
    txt(f"detections {n_det:>3d}", C_DETECT if n_det else JEM.C_DIM)
    txt(f"people     {n_people:>3d}", JEM.C_SUCCESS)
    txt(f"overlay    {'ON' if show_det else 'OFF'}",
        JEM.C_SUCCESS if show_det else JEM.C_DIM, dy=26)
    txt("0-6 lock  7 random  R reset", JEM.C_DIM, "tiny", 15)
    txt("L lidar  B body  H arrows", JEM.C_DIM, "tiny", 15)
    txt("D detections  S fps  Q quit", JEM.C_DIM, "tiny", 15)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ppo_ckpt  = args.ppo_ckpt or JEM._DEFAULT_CKPT["ppo"]
    perc_ckpt = args.perception_ckpt or CKPT_BEST

    print("\n🚀 Live eval — PPO drives, perception detects")
    print(f"   PPO checkpoint        : {ppo_ckpt}")
    print(f"   Perception checkpoint : {perc_ckpt}\n")

    # PPO policy (same builder as jax_eval_multi --algo ppo)
    init_params, load_fn, infer_fn, _ = JEM._build_ppo_shac()
    try:
        params = load_fn(ppo_ckpt)
        print(f"✅ Loaded PPO weights from {ppo_ckpt}")
    except FileNotFoundError:
        params = init_params
        print("⚠️  PPO checkpoint not found — random policy.")

    # Perception network
    net = PerceptionNet()
    template = net.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, PC.STACK_DIM, PC.NUM_RAYS)))["params"]
    perc_params = load_checkpoint(template, perc_ckpt)
    perc_apply = jax.jit(lambda s: net.apply({"params": perc_params}, s))
    print(f"✅ Loaded perception weights from {perc_ckpt}")

    def build_fast_reset(scen_idx, max_goal_dist=MAX_EVAL_GOAL_DIST,
                         min_goal_dist=8.0):
        bound_reset = lambda key, max_goal_dist=3.0, scenario_idx=-1, \
                             min_goal_dist=0.8, **kw: \
            JEM.reset_env(key, max_goal_dist, scenario_idx=scen_idx,
                          min_goal_dist=min_goal_dist, **kw)
        rs, ss = JEM.make_stacked_env(bound_reset, JEM.step_env, stack_dim=3)
        jit_rs = jax.jit(lambda key: rs(key, max_goal_dist, ghost_prob=0.0,
                                        min_goal_dist=min_goal_dist))
        return jit_rs, jax.jit(ss)

    rng = jax.random.PRNGKey(42)
    current_scenario = random.randint(0, 6)
    eval_mode = "random"
    fast_reset, fast_step = build_fast_reset(current_scenario)

    pygame.init()
    screen = pygame.display.set_mode((JEM.WINDOW_W, JEM.WINDOW_H))
    pygame.display.set_caption("Live eval — PPO + perception")
    clock = pygame.time.Clock()
    fonts = JEM.make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    def _fresh_buffer(ss):
        """Perception keeps its own 8-frame stack (policy only stacks 3)."""
        frame = np.asarray(ss.lidar_stack[-1])
        return np.tile(frame[None, :], (PC.STACK_DIM, 1))

    perc_buffer = _fresh_buffer(stacked_state)

    ep = 0
    ep_steps = 0
    frames = 0
    paused = False
    show_lidar = True
    show_arrows = True
    show_body = False
    show_det = True
    fps_speeds = [10, 20, 30]
    fps_idx = 0

    def _reset_episode(scen=None):
        nonlocal fast_reset, fast_step, obs, stacked_state, perc_buffer, \
                 ep_steps, rng, current_scenario
        if scen is not None:
            current_scenario = scen
            fast_reset, fast_step = build_fast_reset(scen)
        rng, reset_rng = jax.random.split(rng)
        obs, stacked_state = fast_reset(reset_rng)
        perc_buffer = _fresh_buffer(stacked_state)
        ep_steps = 0

    print("🎮 Keys: 0-6 lock | 7 random | R reset | L lidar | B body | "
          "H arrows | D detections | S fps | Q quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                if k == pygame.K_SPACE:  paused      = not paused
                if k == pygame.K_l:      show_lidar  = not show_lidar
                if k == pygame.K_h:      show_arrows = not show_arrows
                if k == pygame.K_b:      show_body   = not show_body
                if k == pygame.K_d:      show_det    = not show_det
                if k == pygame.K_s:      fps_idx     = (fps_idx + 1) % len(fps_speeds)
                if pygame.K_0 <= k <= pygame.K_6:
                    eval_mode = "fixed"
                    _reset_episode(k - pygame.K_0)
                if k == pygame.K_7:
                    eval_mode = "random"
                    _reset_episode(random.randint(0, 6))
                if k == pygame.K_r:
                    _reset_episode()

        if paused:
            clock.tick(10); continue

        # ── PPO drives the robot ─────────────────────────────────────────────
        env_action = infer_fn(params, obs, stacked_state.env_state.max_v)
        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(
            step_rng, stacked_state, env_action)
        ep_steps += 1

        cpu_state = jax.device_get(stacked_state.env_state)
        inv_frame = np.asarray(stacked_state.lidar_stack[-1])
        raw_lidar = JEM.MAX_LIDAR_DIST - inv_frame * \
                    (JEM.MAX_LIDAR_DIST - JEM.ROBOT_RADIUS)

        # ── Perception runs on top (own 8-frame stack) ───────────────────────
        perc_buffer = np.concatenate([perc_buffer[1:], inv_frame[None]], axis=0)
        logits, offsets, leg_offsets = perc_apply(jnp.asarray(perc_buffer))
        dets_ego, scores = decode_detections(logits, offsets, leg_offsets,
                                             perc_buffer[-1])
        dets_world = ego_to_world(dets_ego, float(cpu_state.x),
                                  float(cpu_state.y), float(cpu_state.theta))

        # ── Render ───────────────────────────────────────────────────────────
        dyn_scale = JEM.SIM_SIZE / max(float(cpu_state.room_w),
                                       float(cpu_state.room_h))
        sim_w_px = int(float(cpu_state.room_w) * dyn_scale)
        sim_h_px = int(float(cpu_state.room_h) * dyn_scale)
        if (sim_w_px + JEM.PANEL_W, sim_h_px) != screen.get_size():
            screen = pygame.display.set_mode((sim_w_px + JEM.PANEL_W, sim_h_px))
        screen.fill(JEM.C_BG)

        JEM.draw_scene(screen, cpu_state, raw_lidar,
                       np.array(cpu_state.foot_state),
                       show_lidar, show_arrows, True, show_body,
                       scale=dyn_scale, sim_h=sim_h_px)
        if show_det:
            draw_detections(screen, dets_world, dyn_scale, sim_h_px,
                            fonts, scores)

        n_people = int((np.array(cpu_state.people)[:, 10] >= 0).sum())
        draw_panel(screen, fonts, ep, ep_steps,
                   float(cpu_state.v), float(cpu_state.w),
                   len(dets_world), n_people, show_det,
                   sim_x=sim_w_px, win_h=sim_h_px)

        pygame.display.flip()
        clock.tick(fps_speeds[fps_idx])

        frames += 1
        if args.max_steps and frames >= args.max_steps:
            print(f"max-steps reached ({frames} frames) — exiting.")
            pygame.quit(); return

        if bool(done):
            ep += 1
            if eval_mode == "random":
                _reset_episode(random.randint(0, 6))
            else:
                _reset_episode()


if __name__ == "__main__":
    main()
