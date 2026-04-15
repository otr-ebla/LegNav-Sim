"""
test_scenarios_eval.py — Evaluate trained policies on test scenarios (7-12)
==========================================================================
Runs the 6 test scenarios with multi-waypoint support for the robot.
When the robot reaches a waypoint, the goal is updated to the next one
and metrics (reward, steps) reset — as if a new episode started — but
the environment state (people, obstacles, robot position) continues.

Usage:
  python3 test_scenarios_eval.py --algo sac  --ckpt checkpoints_sac/sac_best.msgpack
  python3 test_scenarios_eval.py --algo tqc  --ckpt checkpoints_tqc/tqc_best.msgpack
  python3 test_scenarios_eval.py --algo ppo  --ckpt checkpoints/ppo_attn_final.msgpack
  python3 test_scenarios_eval.py --algo mlp  --ckpt checkpoints_vanilla_ppo/ppo_mlp_best.msgpack

Keys:
  7-9, 0(=10), -(=11), =(=12)   Select test scenario
  R     Reset episode    →   Skip to random test scenario
  N     Next waypoint (skip to next, for debugging)
  L     Toggle LiDAR     H   Toggle arrows
  B     Toggle body ring S   Cycle FPS
  Q/Esc Quit
"""

import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import random
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization

# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Test Scenario Evaluation")
    p.add_argument("--algo", default="sac",
                   choices=["ppo", "shac", "sac", "tqc", "mlp"])
    p.add_argument("--ckpt", default="")
    p.add_argument("--legs",    dest="use_legs", action="store_true",  default=True)
    p.add_argument("--no-legs", dest="use_legs", action="store_false")
    p.add_argument("--ghost-body", action="store_true")
    p.add_argument("--scenario", type=int, default=7,
                   help="Starting test scenario (7-12). Default: 7")
    p.add_argument("--episodes", type=int, default=0,
                   help="Episodes per scenario for batch/headless mode. 0 = interactive (default).")
    return p.parse_args()

args = _parse_args()

import jax_env as _jax_env
_jax_env.USE_LEGS     = args.use_legs
_jax_env.SENSOR_NOISE = True

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS, GOAL_RADIUS)
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env
from jax_scenarios import TEST_ROBOT_WAYPOINTS, TEST_SCENARIO_NAMES

# ── Import eval rendering from jax_eval_multi ────────────────────────────────
# We reuse the drawing functions from the main eval script.
try:
    # Temporarily patch sys.argv so jax_eval_multi's argparse doesn't see our flags
    import sys as _sys
    _saved_argv = _sys.argv
    _sys.argv = _sys.argv[:1]
    from jax_eval_multi import (draw_scene, draw_panel, make_fonts,
                                C_BG, WINDOW_W, WINDOW_H, FPS_TARGET)
    _sys.argv = _saved_argv
    HAS_VIZ = True
except BaseException:
    try:
        _sys.argv = _saved_argv
    except Exception:
        pass
    HAS_VIZ = False
    print("Warning: Could not import draw functions from jax_eval_multi. "
          "Run with --episodes N for headless batch mode.")

# ── Reuse network builders from jax_eval_multi ───────────────────────────────
from jax_eval_multi import build_policy, _DEFAULT_CKPT

OBS_SIZE   = 662
ACTION_DIM = 2
use_legs   = args.use_legs

# ══════════════════════════════════════════════════════════════════════════════
# Core logic
# ══════════════════════════════════════════════════════════════════════════════

MAX_EVAL_GOAL_DIST = 20.0   # large enough for all test scenarios


def build_fast_reset(scen_idx):
    # ghost_prob is absorbed from the wrapper's kwargs and overridden to 0.0
    bound_reset = lambda key, max_goal_dist=3.0, scenario_idx=-1, ghost_prob=0.0, **kw: \
        reset_env(key, MAX_EVAL_GOAL_DIST, scenario_idx=scen_idx, ghost_prob=0.0, **kw)
    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
    jit_rs = jax.jit(lambda key: rs(key, MAX_EVAL_GOAL_DIST, ghost_prob=0.0))
    jit_ss = jax.jit(ss)
    return jit_rs, jit_ss


def run_interactive():
    """Interactive pygame evaluation with live rendering."""
    if not HAS_VIZ:
        print("ERROR: Could not import draw functions from jax_eval_multi.py.")
        print("Make sure jax_eval_multi.py is in the same directory.")
        print("Use --episodes N (e.g. --episodes 50) to run in headless batch mode instead.")
        return

    algo = args.algo
    result = build_policy(algo)
    if len(result) == 4:
        init_params, load_fn, infer_fn, _ = result
    else:
        init_params, load_fn, infer_fn = result

    ckpt = args.ckpt or _DEFAULT_CKPT.get(algo, "")
    try:
        params = load_fn(ckpt)
        print(f"Loaded {algo.upper()} checkpoint: {ckpt}")
    except FileNotFoundError:
        params = init_params
        print(f"Checkpoint not found — running with random weights.")

    current_scenario = max(7, min(12, args.scenario))
    fast_reset, fast_step = build_fast_reset(current_scenario)

    rng = jax.random.PRNGKey(42)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Test Scenarios — {algo.upper()}")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    # Multi-waypoint tracking
    waypoints = TEST_ROBOT_WAYPOINTS[current_scenario]
    wp_idx = 0   # current waypoint index (0 = first goal, already set by scenario)
    wp_segment_reward = 0.0
    wp_segment_steps = 0
    wp_results = []   # (wp_idx, reward, steps, outcome) per segment

    _REW_KEYS = ["rew_progress", "rew_step", "rew_smooth", "rew_yield"]
    ep = 0; ep_steps = 0; ep_reward = 0.0; ep_hist = []
    rew_acc = {k: 0.0 for k in _REW_KEYS}
    paused = False; show_lidar = True; show_arrows = True
    show_body = args.ghost_body; show_radar = True
    fps_idx = 0; fps_speeds = [10, 20, 30, 60, 120]; current_fps = fps_speeds[fps_idx]
    banner = ""; banner_t = 0

    def get_stats():
        if not ep_hist:
            return {"suc": 0., "col": 0., "tmo": 0., "pcol": 0.}
        w = np.array(ep_hist[-50:])
        return {"suc": w[:, 1].mean()*100, "col": w[:, 2].mean()*100,
                "tmo": w[:, 3].mean()*100, "pcol": w[:, 4].mean()*100}

    def reset_episode():
        nonlocal obs, stacked_state, wp_idx, wp_segment_reward, wp_segment_steps
        nonlocal ep_reward, ep_steps, rew_acc, wp_results
        nonlocal rng
        rng, reset_rng = jax.random.split(rng)
        obs, stacked_state = fast_reset(reset_rng)
        wp_idx = 0
        wp_segment_reward = 0.0
        wp_segment_steps = 0
        wp_results = []
        ep_reward = 0.0
        ep_steps = 0
        rew_acc = {k: 0.0 for k in _REW_KEYS}

    def advance_waypoint():
        """Advance to next robot waypoint. Returns True if there are more waypoints."""
        nonlocal obs, stacked_state, wp_idx, wp_segment_reward, wp_segment_steps
        nonlocal ep_reward, ep_steps, rew_acc, rng
        wp_results.append((wp_idx, wp_segment_reward, wp_segment_steps, "reached"))
        wp_idx += 1
        if wp_idx >= len(waypoints):
            return False  # all waypoints done

        # Update goal in the env state to next waypoint.
        # Reset time_step so the 400-step timeout budget is fresh for each segment.
        next_gx, next_gy = waypoints[wp_idx]
        env_state = stacked_state.env_state
        env_state = env_state.replace(
            goal_x=jnp.float32(next_gx),
            goal_y=jnp.float32(next_gy),
            time_step=jnp.int32(0),
        )
        stacked_state = stacked_state.replace(env_state=env_state)

        # Recompute observation with new goal
        from jax_env import get_obs as _get_obs
        rng, obs_key = jax.random.split(rng)
        new_base_obs, sp_mask = _get_obs(env_state, obs_key)
        env_state = env_state.replace(sp_mask=sp_mask)
        stacked_state = stacked_state.replace(env_state=env_state)

        # Update the stacked observation's pose and state_vec (goal-relative)
        pose = new_base_obs[:3]
        state_vec = new_base_obs[3:8]
        lidar = new_base_obs[8:]
        new_pose_stack = stacked_state.pose_stack.at[-1].set(pose)
        new_lidar_stack = stacked_state.lidar_stack.at[-1].set(lidar)
        stacked_state = stacked_state.replace(
            pose_stack=new_pose_stack,
            lidar_stack=new_lidar_stack,
        )
        # Rebuild flat obs
        obs = jnp.concatenate([
            new_pose_stack.flatten(), state_vec, new_lidar_stack.flatten()
        ])

        # Reset per-segment counters — new episode starts from here
        wp_segment_reward = 0.0
        wp_segment_steps  = 0
        ep_reward = 0.0
        ep_steps  = 0
        rew_acc   = {k: 0.0 for k in _REW_KEYS}
        return True

    scen_name = TEST_SCENARIO_NAMES[current_scenario]
    print(f"Test Scenarios Eval — {algo.upper()}")
    print(f"Keys: 7-9 select scenario | 0=10, -=11, ==12 | R reset | N next wp | Q quit")
    print(f"Starting scenario {current_scenario}: {scen_name} ({len(waypoints)} waypoints)")

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
                    fps_idx = (fps_idx+1) % len(fps_speeds)
                    current_fps = fps_speeds[fps_idx]
                    banner = f"FPS: {current_fps}"; banner_t = 15
                if k == pygame.K_r:
                    reset_episode()
                    banner = "reset"; banner_t = 15
                if k == pygame.K_RIGHT:
                    # Skip → random test scenario (7-12), different from current
                    _choices = [s for s in range(7, 13) if s != current_scenario]
                    current_scenario = int(np.random.choice(_choices))
                    waypoints = TEST_ROBOT_WAYPOINTS[current_scenario]
                    scen_name = TEST_SCENARIO_NAMES[current_scenario]
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    reset_episode()
                    banner = f"skip → {scen_name}"; banner_t = 25
                if k == pygame.K_n:
                    # Debug: skip to next waypoint
                    if advance_waypoint():
                        banner = f"wp {wp_idx}/{len(waypoints)}"
                    else:
                        banner = "all wps done"
                    banner_t = 20
                # Scenario selection
                new_scen = None
                if k == pygame.K_7: new_scen = 7
                if k == pygame.K_8: new_scen = 8
                if k == pygame.K_9: new_scen = 9
                if k == pygame.K_0: new_scen = 10
                if k == pygame.K_MINUS: new_scen = 11
                if k == pygame.K_EQUALS: new_scen = 12
                if new_scen is not None:
                    current_scenario = new_scen
                    waypoints = TEST_ROBOT_WAYPOINTS[current_scenario]
                    scen_name = TEST_SCENARIO_NAMES[current_scenario]
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    reset_episode()
                    banner = scen_name; banner_t = 30
                    print(f"Switched to scenario {current_scenario}: {scen_name} ({len(waypoints)} wps)")

        if paused:
            clock.tick(10); continue

        # ── Inference ────────────────────────────────────────────────────────
        env_action = infer_fn(params, obs, stacked_state.env_state.max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward); ep_steps += 1
        wp_segment_reward += float(reward); wp_segment_steps += 1
        for rk in _REW_KEYS:
            rew_acc[rk] += float(info.get(rk, 0.0))

        # ── Render ───────────────────────────────────────────────────────────
        cpu_state = jax.device_get(stacked_state.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * \
                    (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask = np.array(cpu_state.sp_mask)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs, show_body)

        # Overlay waypoint info on the panel
        wp_banner = f"WP {wp_idx+1}/{len(waypoints)}"
        if banner_t > 0:
            wp_banner = f"{banner} | {wp_banner}"

        draw_panel(screen, fonts, f"{algo.upper()} TEST", ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), wp_banner, max(banner_t, 1),
                   current_scenario, "fixed", use_legs, raw_lidar, sp_mask, rew_acc, show_radar)

        if banner_t > 0:
            banner_t -= 1
        pygame.display.flip()
        clock.tick(current_fps)

        # ── Episode logic ────────────────────────────────────────────────────
        if done:
            goal = bool(info["goal_reached"])
            col = bool(info["collision"])
            pcol = bool(info["passive_col"])
            tmo = not goal and not col

            if goal:
                # Robot reached current waypoint — advance to next
                if advance_waypoint():
                    banner = f"wp {wp_idx}/{len(waypoints)} reached!"
                    banner_t = 20
                    continue  # don't reset — keep going to next waypoint
                else:
                    # All waypoints completed!
                    banner = "ALL WAYPOINTS REACHED!"
                    banner_t = 40
                    print(f"  Episode {ep}: ALL {len(waypoints)} waypoints completed! "
                          f"Total reward: {ep_reward:.1f}, steps: {ep_steps}")
                    for wr in wp_results:
                        print(f"    WP{wr[0]}: reward={wr[1]:.1f} steps={wr[2]} outcome={wr[3]}")
                    ep_hist.append((ep_reward, 1.0, 0.0, 0.0, 0.0))
            else:
                outcome = "collision" if (col and not pcol) else ("passive_col" if pcol else "timeout")
                banner = f"{outcome} at wp {wp_idx+1}/{len(waypoints)}"
                banner_t = 30
                wp_results.append((wp_idx, wp_segment_reward, wp_segment_steps, outcome))
                ep_hist.append((ep_reward, 0.0, float(col and not pcol), float(tmo), float(pcol)))

            ep += 1
            # After every episode (success or failure) jump to a random test scenario
            _choices = [s for s in range(7, 13) if s != current_scenario]
            current_scenario = int(np.random.choice(_choices))
            waypoints = TEST_ROBOT_WAYPOINTS[current_scenario]
            scen_name = TEST_SCENARIO_NAMES[current_scenario]
            fast_reset, fast_step = build_fast_reset(current_scenario)
            reset_episode()


def run_headless():
    """Headless batch evaluation: run N episodes per scenario, print stats."""
    algo = args.algo
    result = build_policy(algo)
    if len(result) == 4:
        init_params, load_fn, infer_fn, _ = result
    else:
        init_params, load_fn, infer_fn = result

    ckpt = args.ckpt or _DEFAULT_CKPT.get(algo, "")
    try:
        params = load_fn(ckpt)
        print(f"Loaded {algo.upper()} checkpoint: {ckpt}")
    except FileNotFoundError:
        params = init_params
        print(f"Checkpoint not found — running with random weights.")

    n_episodes = args.episodes
    rng = jax.random.PRNGKey(42)

    print(f"\n{'Scenario':25s} | {'Success':>8s} | {'Collision':>10s} | {'Timeout':>8s} | "
          f"{'Avg Reward':>11s} | {'Avg Steps':>10s} | {'WP Completed':>12s}")
    print("-" * 105)

    for scen_idx in range(7, 13):
        scen_name = TEST_SCENARIO_NAMES[scen_idx]
        waypoints = TEST_ROBOT_WAYPOINTS[scen_idx]
        fast_reset, fast_step = build_fast_reset(scen_idx)

        successes = 0; collisions = 0; timeouts = 0
        total_reward = 0.0; total_steps = 0; total_wp_completed = 0

        for ep in range(n_episodes):
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            wp_idx = 0
            ep_reward = 0.0; ep_steps = 0   # reset at start of every segment
            episode_done = False

            while not episode_done:
                # Each iteration of this while-loop is one waypoint segment.
                # ep_steps / ep_reward reset to zero here, exactly as if a
                # new episode had started.
                ep_reward = 0.0
                ep_steps  = 0

                for step in range(MAX_STEPS):
                    env_action = infer_fn(params, obs, stacked_state.env_state.max_v)
                    rng, step_rng = jax.random.split(rng)
                    obs, stacked_state, reward, done, info = fast_step(
                        step_rng, stacked_state, env_action)
                    ep_reward += float(reward)
                    ep_steps += 1

                    if done:
                        goal = bool(info["goal_reached"])
                        col = bool(info["collision"])

                        if goal and wp_idx + 1 < len(waypoints):
                            # Advance to next waypoint.
                            # Reset time_step so the 400-step budget is fresh per segment.
                            wp_idx += 1
                            total_wp_completed += 1
                            next_gx, next_gy = waypoints[wp_idx]
                            env_state = stacked_state.env_state
                            env_state = env_state.replace(
                                goal_x=jnp.float32(next_gx),
                                goal_y=jnp.float32(next_gy),
                                time_step=jnp.int32(0),
                            )
                            stacked_state = stacked_state.replace(env_state=env_state)
                            # Recompute obs with new goal
                            from jax_env import get_obs as _get_obs
                            rng, obs_key = jax.random.split(rng)
                            new_base_obs, sp_mask = _get_obs(env_state, obs_key)
                            env_state = env_state.replace(sp_mask=sp_mask)
                            stacked_state = stacked_state.replace(env_state=env_state)
                            pose = new_base_obs[:3]
                            state_vec = new_base_obs[3:8]
                            lidar = new_base_obs[8:]
                            stacked_state = stacked_state.replace(
                                pose_stack=stacked_state.pose_stack.at[-1].set(pose),
                                lidar_stack=stacked_state.lidar_stack.at[-1].set(lidar),
                            )
                            obs = jnp.concatenate([
                                stacked_state.pose_stack.flatten(),
                                state_vec,
                                stacked_state.lidar_stack.flatten()
                            ])
                            break  # break inner for-loop; while resets counters
                        elif goal:
                            # All waypoints reached
                            total_wp_completed += 1
                            successes += 1
                            episode_done = True
                            break
                        else:
                            # Collision or timeout
                            if col:
                                collisions += 1
                            else:
                                timeouts += 1
                            episode_done = True
                            break
                else:
                    # MAX_STEPS exhausted without done — timeout for this segment
                    timeouts += 1
                    episode_done = True

            total_reward += ep_reward
            total_steps  += ep_steps

        avg_reward = total_reward / n_episodes
        avg_steps = total_steps / n_episodes
        avg_wp = total_wp_completed / n_episodes
        suc_pct = successes / n_episodes * 100
        col_pct = collisions / n_episodes * 100
        tmo_pct = timeouts / n_episodes * 100

        print(f"{scen_idx:2d}. {scen_name:22s} | {suc_pct:7.1f}% | {col_pct:9.1f}% | "
              f"{tmo_pct:7.1f}% | {avg_reward:10.1f} | {avg_steps:9.1f} | "
              f"{avg_wp:5.1f}/{len(waypoints)}")

    print()


if __name__ == "__main__":
    if args.episodes > 0:
        run_headless()
    else:
        run_interactive()
