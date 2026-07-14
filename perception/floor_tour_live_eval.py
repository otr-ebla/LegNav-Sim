"""
perception/floor_tour_live_eval.py — Floor 16 tour + online people perception
==============================================================================
Live viewer on the real-world floor plan (scenario 16, 42×67 m), in the style
of legnav/evaluation/floor_tour_eval.py: the trained PPO policy drives the
robot through waypoint tours while the perception network (LegDetNet) runs
online on its own 8-frame LiDAR stack and draws a dashed red circle around
every detected human, as in perception/live_eval.py.

Usage (from the repo root):
    python -m perception.floor_tour_live_eval
    python -m perception.floor_tour_live_eval --tour 5 \
        --ppo-ckpt checkpoints/ppo/ppo_tanh_inside_final.msgpack \
        --perception-ckpt checkpoints/perception/perception_best.msgpack

Keys:
  1-8   Select tour     R     Reset tour
  N     Skip waypoint   Space Pause
  L     Toggle LiDAR    H     Toggle arrows
  B     Toggle body     D     Toggle detections
  S     Cycle FPS       Q/Esc Quit
"""

import argparse
import math
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="Floor tour + perception live eval")
    p.add_argument("--ppo-ckpt", default="",
                   help="PPO checkpoint (default: same as jax_eval_multi --algo ppo).")
    p.add_argument("--perception-ckpt", default="",
                   help="Perception checkpoint (default: checkpoints/perception/perception_best.msgpack).")
    p.add_argument("--tour", type=int, default=5, choices=range(1, 9),
                   help="Starting tour (see floor_tour_eval.py; 5-8 include people). Default: 5")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Exit after N rendered frames (0 = run forever; used for smoke tests).")
    return p.parse_args()


args = _parse_args()

# floor_tour_eval / jax_eval_multi / live_eval all parse their own CLI at
# import time — hand them an empty argv so they fall back to defaults.
# floor_tour_eval must come first: it patches PEOPLE_RADIUS=0.25 for the map
# floor before jax_eval_multi binds its constants.
_argv, sys.argv = sys.argv, sys.argv[:1]
from legnav.evaluation import floor_tour_eval as FTE
from legnav.evaluation import jax_eval_multi as JEM
from perception import live_eval as LE
sys.argv = _argv

import jax
import jax.numpy as jnp
import pygame

from legnav.core.jax_env import get_obs, MAX_LIDAR_DIST, ROBOT_RADIUS
from legnav.core.jax_scenarios import MAP_ROOM_W, MAP_ROOM_H
from perception.config import PerceptionConfig as PC
from perception.perception_network import PerceptionNet, decode_detections
from perception.train_perception import load_checkpoint, CKPT_BEST


def main():
    ppo_ckpt  = args.ppo_ckpt or JEM._DEFAULT_CKPT["ppo"]
    perc_ckpt = args.perception_ckpt or CKPT_BEST

    print("\n🚀 Floor tour live eval — PPO drives, perception detects")
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

    fast_reset, fast_step = FTE.build_fast_reset()
    rng = jax.random.PRNGKey(42)

    # ── Window sizing — zoom to waypoint bounding box (as floor_tour_eval) ──
    VIEWPORT_PAD = 2.5
    _all_wps = FTE._CCW + FTE._CW
    vp_xmin = max(0.0,               min(wx for wx, _ in _all_wps) - VIEWPORT_PAD)
    vp_xmax = min(float(MAP_ROOM_W), max(wx for wx, _ in _all_wps) + VIEWPORT_PAD)
    vp_ymin = max(0.0,               min(wy for _, wy in _all_wps) - VIEWPORT_PAD)
    vp_ymax = min(float(MAP_ROOM_H), max(wy for _, wy in _all_wps) + VIEWPORT_PAD)

    SIM_TARGET = 800
    zoom     = SIM_TARGET / max(vp_xmax - vp_xmin, vp_ymax - vp_ymin)
    off_w    = int(MAP_ROOM_W * zoom)
    off_h    = int(MAP_ROOM_H * zoom)
    crop_x   = int(vp_xmin * zoom)
    crop_y   = int((MAP_ROOM_H - vp_ymax) * zoom)
    sim_w_px = int((vp_xmax - vp_xmin) * zoom)
    sim_h_px = int((vp_ymax - vp_ymin) * zoom)

    pygame.init()
    screen    = pygame.display.set_mode((sim_w_px + JEM.PANEL_W, sim_h_px))
    offscreen = pygame.Surface((off_w, off_h))
    pygame.display.set_caption("Floor tour live eval — PPO + perception")
    clock = pygame.time.Clock()
    fonts = JEM.make_fonts()

    # ── Tour state ──────────────────────────────────────────────────────────
    current_tour = args.tour
    tour_cfg     = FTE.TOUR_CONFIGS[current_tour]
    waypoints    = tour_cfg["waypoints"]

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    def _inject_goal(gx, gy, theta=None):
        """Replace goal_x/goal_y (and optionally theta) and recompute obs."""
        nonlocal obs, stacked_state, rng
        replace_kw = dict(
            goal_x=jnp.float32(gx),
            goal_y=jnp.float32(gy),
            time_step=jnp.int32(0),
        )
        if theta is not None:
            replace_kw["theta"] = jnp.float32(theta)
        es = stacked_state.env_state.replace(**replace_kw)
        rng, obs_key = jax.random.split(rng)
        new_base_obs, sp_mask = get_obs(es, obs_key)
        es = es.replace(sp_mask=sp_mask)
        ss = stacked_state.replace(env_state=es)
        goal      = new_base_obs[:2]
        kin_vec   = new_base_obs[2:7]
        lidar     = new_base_obs[7:]
        ss = ss.replace(
            goal_stack=ss.goal_stack.at[-1].set(goal),
            lidar_stack=ss.lidar_stack.at[-1].set(lidar),
        )
        obs = jnp.concatenate([
            ss.goal_stack.flatten(), kin_vec, ss.lidar_stack.flatten()
        ])
        stacked_state = ss

    def _wp0_theta():
        es = stacked_state.env_state
        dx = waypoints[0][0] - float(es.x)
        dy = waypoints[0][1] - float(es.y)
        return math.atan2(dy, dx)

    def _inject_patrol_people():
        nonlocal stacked_state, rng
        people_arr = FTE._make_patrol_people_arr()
        rng, ft_key = jax.random.split(rng)
        new_foot = FTE._init_foot_state(people_arr, ft_key)
        es = stacked_state.env_state.replace(people=people_arr,
                                             foot_state=new_foot)
        stacked_state = stacked_state.replace(env_state=es)

    def _fresh_buffer():
        """Perception keeps its own 8-frame stack (policy only stacks 3)."""
        frame = np.asarray(stacked_state.lidar_stack[-1])
        return np.tile(frame[None, :], (PC.STACK_DIM, 1))

    wp_idx           = 0
    wp_results: list = []
    ep        = 0
    ep_steps  = 0
    frames    = 0
    perc_buffer = None

    def reset_tour():
        nonlocal obs, stacked_state, rng, wp_idx, wp_results, ep_steps, \
                 perc_buffer, trajectory
        rng, reset_rng = jax.random.split(rng)
        obs, stacked_state = fast_reset(reset_rng)
        wp_idx     = 0
        wp_results = []
        ep_steps   = 0
        trajectory = []
        _inject_goal(*waypoints[0], theta=_wp0_theta())
        if tour_cfg["people"]:
            _inject_patrol_people()
        perc_buffer = _fresh_buffer()

    def advance_waypoint():
        """Record result and advance wp_idx. Returns True if there are more WPs."""
        nonlocal wp_idx
        wp_results.append((wp_idx, 0.0, ep_steps, "reached"))
        wp_idx += 1
        if wp_idx >= len(waypoints):
            return False
        _inject_goal(*waypoints[wp_idx])
        return True

    _inject_goal(*waypoints[0], theta=_wp0_theta())
    if tour_cfg["people"]:
        _inject_patrol_people()
    perc_buffer = _fresh_buffer()

    paused      = False
    show_lidar  = True
    show_arrows = True
    show_body   = False
    show_det    = True
    fps_speeds  = [10, 20, 30, 60]
    fps_idx     = 0
    trajectory  = []

    print(f"Tour {current_tour}: {tour_cfg['name']} — {len(waypoints)} waypoints")
    print("🎮 Keys: 1-8 tour | R reset | N next WP | L lidar | B body | "
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
                if k == pygame.K_r:
                    reset_tour()
                if k == pygame.K_n:
                    if not advance_waypoint():
                        ep += 1
                        reset_tour()
                if pygame.K_1 <= k <= pygame.K_8:
                    ti = k - pygame.K_0
                    if ti != current_tour:
                        current_tour = ti
                        tour_cfg     = FTE.TOUR_CONFIGS[current_tour]
                        waypoints    = tour_cfg["waypoints"]
                        reset_tour()
                        print(f"Tour {current_tour}: {tour_cfg['name']} "
                              f"— {len(waypoints)} waypoints")

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
        raw_lidar = MAX_LIDAR_DIST - inv_frame * (MAX_LIDAR_DIST - ROBOT_RADIUS)

        cur_x, cur_y = float(cpu_state.x), float(cpu_state.y)
        if (not trajectory
                or math.hypot(cur_x - trajectory[-1][0],
                              cur_y - trajectory[-1][1]) >= 0.05):
            trajectory.append((cur_x, cur_y, float(cpu_state.v)))

        # ── Perception runs on top (own 8-frame stack) ───────────────────────
        perc_buffer = np.concatenate([perc_buffer[1:], inv_frame[None]], axis=0)
        logits, offsets, leg_offsets = perc_apply(jnp.asarray(perc_buffer))
        dets_ego, scores = decode_detections(logits, offsets, leg_offsets,
                                             perc_buffer[-1])
        dets_world = LE.ego_to_world(dets_ego, cur_x, cur_y,
                                     float(cpu_state.theta))

        # ── Render ───────────────────────────────────────────────────────────
        offscreen.fill(JEM.C_BG)
        JEM.draw_scene(offscreen, cpu_state, raw_lidar,
                       np.array(cpu_state.foot_state),
                       show_lidar, show_arrows, True, show_body,
                       scale=zoom, sim_h=off_h,
                       trajectory=trajectory, show_map_bg=True)
        FTE.draw_waypoints(offscreen, waypoints, wp_idx, zoom, off_h)
        if show_det:
            LE.draw_detections(offscreen, dets_world, zoom, off_h,
                               fonts, scores)
        screen.fill(JEM.C_BG)
        screen.blit(offscreen, (0, 0), (crop_x, crop_y, sim_w_px, sim_h_px))

        n_people = int((np.array(cpu_state.people)[:, 10] >= 0).sum())
        LE.draw_panel(screen, fonts, ep, ep_steps,
                      float(cpu_state.v), float(cpu_state.w),
                      len(dets_world), n_people, show_det,
                      sim_x=sim_w_px, win_h=sim_h_px)
        FTE.draw_tour_overlay(screen, fonts, tour_cfg, wp_idx,
                              sim_w_px, sim_h_px, wp_results)

        pygame.display.flip()
        clock.tick(fps_speeds[fps_idx])

        frames += 1
        if args.max_steps and frames >= args.max_steps:
            print(f"max-steps reached ({frames} frames) — exiting.")
            pygame.quit(); return

        # ── Episode / waypoint logic ──────────────────────────────────────────
        if bool(done):
            if bool(info["goal_reached"]):
                if advance_waypoint():
                    continue
                print(f"  Ep {ep}: COMPLETE tour {current_tour} "
                      f"({len(waypoints)} WPs) | steps={ep_steps}")
            else:
                outcome = ("collision" if bool(info["collision"])
                           else "timeout")
                print(f"  Ep {ep}: {outcome} at WP{wp_idx + 1}/"
                      f"{len(waypoints)} | steps={ep_steps}")
            ep += 1
            trajectory = []
            reset_tour()


if __name__ == "__main__":
    main()
