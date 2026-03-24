#!/usr/bin/env python3
"""
debug_viz.py — Pygame Debug Visualizer for Single-Environment Rollouts
========================================================================
Renders one environment instance in real-time with full state visibility:
  - Robot (blue circle + heading arrow)
  - Goal (green star)
  - Humans (red circles + velocity arrows + ID labels)
  - Dummy humans (grey, at -999) are hidden
  - Obstacle circles (dark grey)
  - Obstacle boxes (dark grey)
  - Walls (room boundary)
  - LiDAR rays (yellow fans)
  - Shoe boxes (orange outlines) when USE_LEGS=True
  - Comfort zone (translucent ring around each human)
  - Collision flash (screen border turns red on collision)
  - HUD: step, reward, done, goal_dist, closest_human, episode return,
         action (v, w), scenario info, collision type

Usage:
  python debug_viz.py                    # random policy
  python debug_viz.py --checkpoint path  # trained policy
  python debug_viz.py --scenario 0       # force scenario 0
  python debug_viz.py --min-goal-dist 5  # override goal distance
  python debug_viz.py --speed 1.0        # playback speed multiplier
  python debug_viz.py --pause            # start paused (space to step)

Controls:
  SPACE  — pause / unpause
  RIGHT  — step one frame (when paused)
  R      — reset episode
  Q/ESC  — quit
  +/-    — zoom in/out
  L      — toggle LiDAR ray rendering
  S      — toggle shoe box rendering
"""

import os
import sys
import argparse
import math
import time

# ── Parse args BEFORE importing JAX (GPU config) ────────────────────────────
parser = argparse.ArgumentParser(description="Debug Visualizer")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .msgpack checkpoint")
parser.add_argument("--scenario", type=int, default=-1, help="Force scenario index (0-6, -1=random)")
parser.add_argument("--min-goal-dist", type=float, default=1.5, help="Minimum goal distance")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--pause", action="store_true", help="Start paused")
parser.add_argument("--ghost", action="store_true", default=True, help="Ghost robot (training mode)")
parser.add_argument("--no-ghost", action="store_false", dest="ghost", help="Visible robot (eval mode)")
parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

import jax
import jax.numpy as jnp
import numpy as np

# Ensure we can import the env modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add project root for JHSFM imports
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from jax_env_multi import reset_env, step_env, NUM_PEOPLE
from jax_env import (EnvState, ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     DT, MAX_STEPS, GOAL_RADIUS, NUM_RAYS, FOV,
                     MAX_LIDAR_DIST, USE_LEGS)

# Comfort distance for visualization — matches jax_env_multi._COMFORT_DIST
COMFORT_DIST = 1.2
from jax_wrappers import make_stacked_env, StackedEnvState, POSE_SIZE, STATE_VEC_SIZE
from jax_legs import get_leg_positions, get_shoe_boxes, LEG_RADIUS
from jax_physics import compute_lidar
from jax_legs import get_leg_circles

import pygame

# ── Display constants ────────────────────────────────────────────────────────
WIN_W, WIN_H = 900, 900
MARGIN = 40              # pixels around the room
BG_COLOR      = (30, 30, 35)
WALL_COLOR    = (180, 180, 180)
GRID_COLOR    = (50, 50, 55)
ROBOT_COLOR   = (60, 140, 255)
GOAL_COLOR    = (50, 220, 80)
HUMAN_COLOR   = (220, 80, 60)
HUMAN_DUMMY   = (80, 80, 80)
OBS_CIR_COLOR = (100, 100, 110)
OBS_BOX_COLOR = (100, 100, 110)
LIDAR_COLOR   = (255, 220, 50, 60)
SHOE_COLOR    = (255, 160, 40)
COMFORT_COLOR = (255, 200, 100, 40)
COLLISION_FLASH = (255, 30, 30)
TEXT_COLOR     = (220, 220, 220)
TEXT_DIM       = (140, 140, 140)

# ── Coordinate transform ────────────────────────────────────────────────────

def world_to_screen(x, y, scale, offset_x, offset_y):
    """World coordinates → screen pixels (y-flipped)."""
    sx = int(offset_x + x * scale)
    sy = int(offset_y + (ROOM_H - y) * scale)
    return sx, sy

def world_radius_to_px(r, scale):
    return max(1, int(r * scale))


# ── LiDAR computation (single env, for rendering) ───────────────────────────

def compute_lidar_for_render(state):
    """Compute raw LiDAR distances for visualization (not the obs version)."""
    human_circles = get_leg_circles(state.people, state.foot_state, use_legs=USE_LEGS)
    all_circles = jnp.concatenate([human_circles, state.obs_circles], axis=0)
    raw_lidar = compute_lidar(
        state.x, state.y, state.theta,
        all_circles, state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    )
    return raw_lidar


# ── Network loading ─────────────────────────────────────────────────────────

def load_policy(checkpoint_path):
    """Load trained policy from checkpoint. Returns (apply_fn, params) or None."""
    if checkpoint_path is None:
        return None

    from jax_network import EndToEndActorCritic
    import flax.serialization

    OBS_SIZE = 342
    network = EndToEndActorCritic(action_dim=2)
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    rng = jax.random.PRNGKey(0)
    dummy_params = network.init(rng, dummy_obs)["params"]

    with open(checkpoint_path, "rb") as f:
        raw = f.read()

    # Try loading with opt_state (training checkpoint format)
    try:
        import optax
        dummy_opt = optax.adam(1e-4).init(dummy_params)
        bundle = flax.serialization.from_bytes(
            {"params": dummy_params, "opt_state": dummy_opt}, raw
        )
        params = bundle["params"]
    except Exception:
        # Try params-only format
        params = flax.serialization.from_bytes(dummy_params, raw)

    print(f"Loaded checkpoint: {checkpoint_path}")
    return network, params


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RL Nav Debug Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    font_big = pygame.font.SysFont("monospace", 16, bold=True)

    # ── Load policy (or use random) ──────────────────────────────────────────
    policy = load_policy(args.checkpoint)
    policy_name = "TRAINED" if policy else "RANDOM"

    # ── Build stacked env (same as training) ─────────────────────────────────
    reset_stacked, step_stacked = make_stacked_env(
        reset_env, step_env, stack_dim=3,
        ghost_robot=args.ghost
    )

    # ── Init env ─────────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)

    def do_reset(rng_key):
        return reset_stacked(rng_key, min_goal_dist=args.min_goal_dist)

    rng, reset_key = jax.random.split(rng)
    obs, stacked_state = do_reset(reset_key)

    # ── State ────────────────────────────────────────────────────────────────
    paused = args.pause
    show_lidar = True
    show_shoes = USE_LEGS
    zoom = 1.0
    episode_return = 0.0
    episode_steps = 0
    episode_count = 0
    last_reward = 0.0
    last_done = False
    last_info = {}
    last_action_v = 0.0
    last_action_w = 0.0

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────────
        step_once = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    step_once = True
                elif event.key == pygame.K_r:
                    rng, reset_key = jax.random.split(rng)
                    obs, stacked_state = do_reset(reset_key)
                    episode_return = 0.0
                    episode_steps = 0
                    episode_count += 1
                    last_done = False
                    last_info = {}
                elif event.key == pygame.K_l:
                    show_lidar = not show_lidar
                elif event.key == pygame.K_s:
                    show_shoes = not show_shoes
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom = min(3.0, zoom + 0.1)
                elif event.key == pygame.K_MINUS:
                    zoom = max(0.3, zoom - 0.1)

        # ── Step env ─────────────────────────────────────────────────────────
        if (not paused or step_once) and not last_done:
            rng, act_key, step_key = jax.random.split(rng, 3)

            if policy is not None:
                network, params = policy
                mean, logstd, value = network.apply({"params": params}, obs[None])
                mean, logstd = mean[0], logstd[0]
                # Sample action (training-like)
                from jax_network import sample_action, scale_action_to_env
                raw_action, _ = sample_action(act_key, mean, logstd)
                env_action = scale_action_to_env(raw_action, stacked_state.env_state.max_v)
            else:
                # Random action
                max_v = float(stacked_state.env_state.max_v)
                v = jax.random.uniform(act_key, minval=0.0, maxval=max_v)
                w = jax.random.uniform(jax.random.fold_in(act_key, 1), minval=-1.0, maxval=1.0)
                env_action = jnp.array([v, w])

            last_action_v = float(env_action[0])
            last_action_w = float(env_action[1])

            obs, stacked_state, reward, done, info = step_stacked(
                step_key, stacked_state, env_action
            )
            last_reward = float(reward)
            last_done = bool(done)
            last_info = jax.tree_util.tree_map(lambda x: float(x) if x.ndim == 0 else x, info)
            episode_return += last_reward
            episode_steps += 1

            # ── Terminal diagnostics ─────────────────────────────────────────
            # Compute closest human distance for console debug
            _people = np.array(stacked_state.env_state.people)
            _rx = float(stacked_state.env_state.x)
            _ry = float(stacked_state.env_state.y)
            _dists = []
            for hi in range(_people.shape[0]):
                px, py = _people[hi, 0], _people[hi, 1]
                if px < -100:
                    continue
                _dists.append(math.sqrt((px - _rx)**2 + (py - _ry)**2))
            _min_d = min(_dists) if _dists else 99.0
            _body_thresh = ROBOT_RADIUS + PEOPLE_RADIUS  # 0.6m with USE_LEGS=False

            if episode_steps % 20 == 0 or last_done:
                _col = last_info.get("collision", 0)
                _pcol = last_info.get("passive_col", 0)
                _acol = last_info.get("active_col", 0)
                _gr = last_info.get("goal_reached", 0)
                _tmo = last_info.get("timeout", 0)
                print(f"  step={episode_steps:>4d}  r={last_reward:>+7.2f}  ret={episode_return:>+8.1f}  "
                      f"min_d={_min_d:.3f}  thresh={_body_thresh:.2f}  "
                      f"done={last_done}  col={_col:.0f}  acol={_acol:.0f}  pcol={_pcol:.0f}  "
                      f"goal={_gr:.0f}  tmo={_tmo:.0f}  "
                      f"v={last_action_v:.2f}  w={last_action_w:.2f}")

            if last_done:
                print(f"  ═══ EPISODE {episode_count} ENDED at step {episode_steps}: "
                      f"return={episode_return:+.1f}  min_human_dist={_min_d:.3f}m  "
                      f"thresh={_body_thresh:.2f}m ═══")
                print(f"      Press R to reset, SPACE to unpause")

        # ── Extract state for rendering ──────────────────────────────────────
        state = stacked_state.env_state
        rx, ry = float(state.x), float(state.y)
        rtheta = float(state.theta)
        gx, gy = float(state.goal_x), float(state.goal_y)
        people = np.array(state.people)
        obs_circles = np.array(state.obs_circles)
        obs_boxes = np.array(state.obs_boxes)
        foot_st = state.foot_state

        # ── Compute scale and offset ─────────────────────────────────────────
        effective_w = WIN_W - 2 * MARGIN
        effective_h = WIN_H - 2 * MARGIN - 120  # reserve bottom for HUD
        scale = min(effective_w / ROOM_W, effective_h / ROOM_H) * zoom
        offset_x = MARGIN + (effective_w - ROOM_W * scale) / 2
        offset_y = MARGIN + (effective_h - ROOM_H * scale) / 2

        def w2s(x, y):
            return world_to_screen(x, y, scale, offset_x, offset_y)

        def wr2px(r):
            return world_radius_to_px(r, scale)

        # ── Clear ────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # ── Collision flash ──────────────────────────────────────────────────
        if last_done and last_info.get("collision", 0) > 0.5:
            pygame.draw.rect(screen, COLLISION_FLASH, (0, 0, WIN_W, WIN_H), 6)

        # ── Grid ─────────────────────────────────────────────────────────────
        for i in range(int(ROOM_W) + 1):
            p1 = w2s(i, 0)
            p2 = w2s(i, ROOM_H)
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)
        for j in range(int(ROOM_H) + 1):
            p1 = w2s(0, j)
            p2 = w2s(ROOM_W, j)
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)

        # ── Walls ────────────────────────────────────────────────────────────
        corners = [w2s(0, 0), w2s(ROOM_W, 0), w2s(ROOM_W, ROOM_H), w2s(0, ROOM_H)]
        pygame.draw.polygon(screen, WALL_COLOR, corners, 2)

        # ── Obstacle circles ─────────────────────────────────────────────────
        for i in range(obs_circles.shape[0]):
            cx, cy, r = obs_circles[i]
            if r > 0:
                pygame.draw.circle(screen, OBS_CIR_COLOR, w2s(cx, cy), wr2px(r))

        # ── Obstacle boxes ───────────────────────────────────────────────────
        for i in range(obs_boxes.shape[0]):
            cx, cy, hw, hh = obs_boxes[i]
            if hw > 0 and hh > 0:
                tl = w2s(cx - hw, cy + hh)
                br = w2s(cx + hw, cy - hh)
                rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
                pygame.draw.rect(screen, OBS_BOX_COLOR, rect)

        # ── Goal ─────────────────────────────────────────────────────────────
        gsx, gsy = w2s(gx, gy)
        gr = wr2px(GOAL_RADIUS)
        pygame.draw.circle(screen, GOAL_COLOR, (gsx, gsy), gr + 4, 2)
        # Star shape
        for a in range(5):
            angle = math.radians(a * 72 - 90)
            x1 = gsx + int(gr * math.cos(angle))
            y1 = gsy + int(gr * math.sin(angle))
            angle2 = math.radians(a * 72 - 90 + 144)
            x2 = gsx + int(gr * math.cos(angle2))
            y2 = gsy + int(gr * math.sin(angle2))
            pygame.draw.line(screen, GOAL_COLOR, (x1, y1), (x2, y2), 2)

        # ── Comfort zones (translucent) ──────────────────────────────────────
        comfort_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for i in range(people.shape[0]):
            px, py = people[i, 0], people[i, 1]
            if px < -100:
                continue  # dummy
            csx, csy = w2s(px, py)
            cr = wr2px(COMFORT_DIST)
            pygame.draw.circle(comfort_surf, COMFORT_COLOR, (csx, csy), cr, 1)
        screen.blit(comfort_surf, (0, 0))

        # ── Humans ───────────────────────────────────────────────────────────
        # Collision threshold depends on USE_LEGS flag
        if USE_LEGS:
            body_thresh_viz = ROBOT_RADIUS + LEG_RADIUS  # shoe AABB is primary, body is tighter
        else:
            body_thresh_viz = ROBOT_RADIUS + PEOPLE_RADIUS
        
        for i in range(people.shape[0]):
            px, py = people[i, 0], people[i, 1]
            if px < -100:
                continue  # dummy — skip rendering

            vx, vy = people[i, 2], people[i, 3]
            theta_h = people[i, 4]

            # Determine if active
            goal_idx = people[i, 10] if people.shape[1] > 10 else 0
            is_dummy = goal_idx < 0
            color = HUMAN_DUMMY if is_dummy else HUMAN_COLOR

            hsx, hsy = w2s(px, py)
            hr = wr2px(PEOPLE_RADIUS)

            # Distance from robot to this human (center-to-center)
            d_to_robot = math.sqrt((px - rx)**2 + (py - ry)**2)

            # Highlight if within collision threshold
            in_collision = d_to_robot < body_thresh_viz
            if in_collision:
                # Red filled circle = collision!
                pygame.draw.circle(screen, (255, 0, 0), (hsx, hsy), hr + 4)
                color = (255, 255, 255)

            # Body circle
            pygame.draw.circle(screen, color, (hsx, hsy), hr, 2)

            # Velocity arrow (shows actual movement direction)
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 0.05:
                arrow_len = wr2px(speed * 0.5)
                ex = int(hsx + arrow_len * math.cos(theta_h))
                ey = int(hsy - arrow_len * math.sin(theta_h))  # y-flip
                pygame.draw.line(screen, color, (hsx, hsy), (ex, ey), 2)

            # ── Waypoint visualization (shows where this human is going) ─────
            if people.shape[1] > 10 and not is_dummy:
                g1x_h, g1y_h = people[i, 6], people[i, 7]
                g2x_h, g2y_h = people[i, 8], people[i, 9]
                cur_idx = people[i, 10]
                # Active waypoint
                wpx = g1x_h if cur_idx == 0 else g2x_h
                wpy = g1y_h if cur_idx == 0 else g2y_h
                if wpx > -100:  # not a dummy waypoint
                    wpsx, wpsy = w2s(wpx, wpy)
                    # Dashed line from human to waypoint (cyan)
                    pygame.draw.line(screen, (0, 180, 180, 100), (hsx, hsy), (wpsx, wpsy), 1)
                    pygame.draw.circle(screen, (0, 180, 180), (wpsx, wpsy), 4, 1)

            # Distance label (shows center-to-center dist, RED if below threshold)
            d_color = (255, 60, 60) if d_to_robot < body_thresh_viz + 0.2 else TEXT_DIM
            d_label = font.render(f"{d_to_robot:.2f}", True, d_color)
            screen.blit(d_label, (hsx + hr + 2, hsy - 6))

            # Line from robot to human if close
            if d_to_robot < 2.0:
                line_color = (255, 50, 50) if in_collision else (255, 200, 100, 80)
                pygame.draw.line(screen, line_color, (rsx, rsy), (hsx, hsy), 1)

        # ── Shoe boxes ───────────────────────────────────────────────────────
        if show_shoes and USE_LEGS:
            shoe_boxes_np = np.array(get_shoe_boxes(state.people, foot_st))
            for i in range(shoe_boxes_np.shape[0]):
                cx, cy, hw, hh = shoe_boxes_np[i]
                if cx < -100:
                    continue
                tl = w2s(cx - hw, cy + hh)
                br = w2s(cx + hw, cy - hh)
                rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
                pygame.draw.rect(screen, SHOE_COLOR, rect, 1)

        # ── LiDAR rays ──────────────────────────────────────────────────────
        if show_lidar:
            raw_lidar = np.array(compute_lidar_for_render(state))
            lidar_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            fov = float(FOV)
            for j in range(NUM_RAYS):
                angle = rtheta - fov * 0.5 + j * (fov / (NUM_RAYS - 1))
                dist = raw_lidar[j]
                hx = rx + dist * math.cos(angle)
                hy = ry + dist * math.sin(angle)
                p1 = w2s(rx, ry)
                p2 = w2s(hx, hy)
                # Color by distance: close=red, far=green
                t = min(dist / MAX_LIDAR_DIST, 1.0)
                r_c = int(255 * (1 - t))
                g_c = int(200 * t)
                pygame.draw.line(lidar_surf, (r_c, g_c, 50, 50), p1, p2, 1)
                # Hit point
                if dist < MAX_LIDAR_DIST - 0.1:
                    pygame.draw.circle(lidar_surf, (255, 100, 50, 120), p2, 2)
            screen.blit(lidar_surf, (0, 0))

        # ── Robot ────────────────────────────────────────────────────────────
        rsx, rsy = w2s(rx, ry)
        rr = wr2px(ROBOT_RADIUS)

        # Body
        pygame.draw.circle(screen, ROBOT_COLOR, (rsx, rsy), rr)

        # Heading arrow
        arrow_len = rr + wr2px(0.3)
        ax = int(rsx + arrow_len * math.cos(rtheta))
        ay = int(rsy - arrow_len * math.sin(rtheta))
        pygame.draw.line(screen, (255, 255, 255), (rsx, rsy), (ax, ay), 3)

        # Collision threshold ring (matches what the env actually checks)
        if USE_LEGS:
            col_r = wr2px(ROBOT_RADIUS + LEG_RADIUS)
        else:
            col_r = wr2px(ROBOT_RADIUS + PEOPLE_RADIUS)
        pygame.draw.circle(screen, (255, 80, 80, 80), (rsx, rsy), col_r, 1)

        # ── DONE overlay ─────────────────────────────────────────────────────
        if last_done:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            
            # Big centered text
            if last_info.get("goal_reached", 0) > 0.5:
                big_text = "GOAL REACHED!"
                big_color = GOAL_COLOR
            elif last_info.get("passive_col", 0) > 0.5:
                big_text = "PASSIVE COLLISION"
                big_color = (255, 160, 40)
            elif last_info.get("collision", 0) > 0.5:
                big_text = "ACTIVE COLLISION"
                big_color = COLLISION_FLASH
            elif last_info.get("timeout", 0) > 0.5:
                big_text = "TIMEOUT"
                big_color = (180, 180, 60)
            else:
                big_text = "DONE"
                big_color = TEXT_COLOR
            
            font_huge = pygame.font.SysFont("monospace", 36, bold=True)
            text_surf = font_huge.render(big_text, True, big_color)
            tx = (WIN_W - text_surf.get_width()) // 2
            ty = (WIN_H - 120) // 2 - 30
            screen.blit(text_surf, (tx, ty))
            
            sub_text = f"Return: {episode_return:+.1f}  |  Steps: {episode_steps}  |  Press R to reset"
            sub_surf = font_big.render(sub_text, True, TEXT_DIM)
            screen.blit(sub_surf, ((WIN_W - sub_surf.get_width()) // 2, ty + 45))

        # ── HUD ──────────────────────────────────────────────────────────────
        hud_y = WIN_H - 115
        pygame.draw.rect(screen, (20, 20, 25), (0, hud_y - 5, WIN_W, 120))
        pygame.draw.line(screen, WALL_COLOR, (0, hud_y - 5), (WIN_W, hud_y - 5), 1)

        goal_dist = math.sqrt((rx - gx)**2 + (ry - gy)**2)
        closest_h = float(last_info.get("closest_human", 99.0))

        # Determine episode outcome label
        if last_done:
            if last_info.get("goal_reached", 0) > 0.5:
                outcome = "SUCCESS"
                outcome_color = GOAL_COLOR
            elif last_info.get("passive_col", 0) > 0.5:
                outcome = "PASSIVE COLLISION"
                outcome_color = (255, 160, 40)
            elif last_info.get("collision", 0) > 0.5:
                outcome = "ACTIVE COLLISION"
                outcome_color = COLLISION_FLASH
            elif last_info.get("timeout", 0) > 0.5:
                outcome = "TIMEOUT"
                outcome_color = (180, 180, 60)
            else:
                outcome = "DONE"
                outcome_color = TEXT_DIM
        else:
            outcome = "RUNNING"
            outcome_color = ROBOT_COLOR

        lines = [
            f"Policy: {policy_name}  |  Ghost: {args.ghost}  |  Episode: {episode_count}  |  Scenario: {args.scenario}",
            f"Step: {episode_steps:>4d}/{MAX_STEPS}  |  Reward: {last_reward:>+7.2f}  |  EpReturn: {episode_return:>+8.1f}  |  Goal: {goal_dist:.2f}m  |  ClosestH: {closest_h:.2f}m",
            f"Action: v={last_action_v:.3f}  w={last_action_w:.3f}  |  max_v={float(state.max_v):.2f}  |  robot=({rx:.1f},{ry:.1f})  θ={math.degrees(rtheta):.0f}°",
            f"{'PAUSED (SPACE=resume, RIGHT=step)' if paused else 'L=lidar  S=shoes  R=reset  +/-=zoom  SPACE=pause'}"
        ]

        for i, line in enumerate(lines):
            color = TEXT_COLOR if i < 3 else TEXT_DIM
            surf = font.render(line, True, color)
            screen.blit(surf, (10, hud_y + i * 18))

        # Outcome badge
        badge = font_big.render(f" {outcome} ", True, (0, 0, 0), outcome_color)
        screen.blit(badge, (WIN_W - badge.get_width() - 10, hud_y + 5))

        # ── Flip ─────────────────────────────────────────────────────────────
        pygame.display.flip()

        # ── Frame rate ───────────────────────────────────────────────────────
        target_fps = max(1, int((1.0 / DT) * args.speed))
        if paused and not step_once:
            clock.tick(30)  # idle
        else:
            clock.tick(target_fps)

    pygame.quit()


if __name__ == "__main__":
    main()