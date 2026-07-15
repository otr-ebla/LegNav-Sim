"""
test_scenarios_eval.py — Evaluate trained policies on test scenarios (7-12)
==========================================================================
Runs the 6 test scenarios (7-12) with multi-waypoint support for the robot.
When the robot reaches a waypoint, the goal is updated to the next one and
the environment state (people, obstacles, robot position) continues. Headless
mode reports the standard metrics (SR/ACR/PCR/TR/SPL/TTG/MHD/AJ/SC/YS),
accumulated over the whole episode across all waypoint segments.

Usage:
  python3 test_scenarios_eval.py --algo sac  --ckpt checkpoints_sac/sac_final.msgpack
  python3 test_scenarios_eval.py --algo tqc  --ckpt checkpoints_tqc/tqc_final.msgpack
  python3 test_scenarios_eval.py --algo ppo  --ckpt checkpoints/ppo_tanh_inside_final.msgpack
  python3 test_scenarios_eval.py --algo mlp  --ckpt checkpoints_vanilla_ppo/ppo_mlp_best.msgpack
  python3 test_scenarios_eval.py --algo dwa   # Dynamic Window Approach (no checkpoint)
  python3 test_scenarios_eval.py --algo mppi  # Model Predictive Path Integral (no checkpoint)
  python3 test_scenarios_eval.py --algo hsfm  # HSFM: robot driven as a human circle (no checkpoint)

Keys:
  7-9, 0(=10), -/'(=11), =(=12)   Select test scenario
  R     Reset episode    →   Skip to random test scenario
  N     Next waypoint (skip to next, for debugging)
  L     Toggle LiDAR     H   Toggle arrows
  B     Toggle body ring S   Cycle FPS
  Q/Esc Quit
"""

import argparse
import os
from legnav import paths
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
                   choices=["ppo", "shac", "sac", "tqc", "mlp", "navrep", "tagd", "mppi", "dwa", "hsfm", "ppo_circles"])
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

import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS     = args.use_legs
_jax_env.SENSOR_NOISE = True

from legnav.core.jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS, GOAL_RADIUS)
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, _ego_deltas, assemble_stacked_obs
from legnav.core.jax_scenarios import TEST_ROBOT_WAYPOINTS, TEST_SCENARIO_NAMES

# --- INIZIO MODIFICA: Sovrascriviamo i waypoint dello scenario 9 ---
# -1.0 = door1_y, -2.0 = door2_y (calcolati dinamicamente da obs_boxes).
# WP1: stanza 1 davanti porta 1 | WP2: stanza 2 davanti porta 2
TEST_ROBOT_WAYPOINTS[9] = [(7.0, -1.0), (10.0, -2.0)]
# --- FINE MODIFICA ---

# ── Import eval rendering from jax_eval_multi ────────────────────────────────
# We reuse the drawing functions from the main eval script.
try:
    # Temporarily patch sys.argv so jax_eval_multi's argparse doesn't see our flags
    import sys as _sys
    _saved_argv = _sys.argv
    _sys.argv = _sys.argv[:1]
    from legnav.evaluation.jax_eval_multi import (draw_scene, draw_panel, make_fonts,
                                C_BG, WINDOW_W, WINDOW_H, SIM_SIZE, PANEL_W, FPS_TARGET)
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
from legnav.evaluation.jax_eval_multi import build_policy, _DEFAULT_CKPT

def _build_navrep():
    from legnav.baselines.navrep_network import NavRepActorCritic
    from legnav.core.jax_network import scale_action_to_env

    net = NavRepActorCritic(action_dim=ACTION_DIM)
    rng = jax.random.PRNGKey(0)
    init_params = net.init(rng, jnp.zeros((1, OBS_SIZE)))["params"]

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        bundle = flax.serialization.msgpack_restore(raw)
        return bundle.get("params", bundle)

    def infer(params, obs, max_v):
        mean, _, _ = net.apply({"params": params}, obs[None])
        return scale_action_to_env(jnp.squeeze(mean, 0), float(max_v))

    def reset_cb(): pass
    return init_params, load, infer, reset_cb

_DEFAULT_CKPT["navrep"] = paths.checkpoint("navrep", "navrep_best.msgpack")


def _build_tagd():
    from legnav.baselines.tagd_network import TAGDActor

    actor = TAGDActor()
    rng   = jax.random.PRNGKey(0)
    init_params = actor.init(rng, jnp.zeros(OBS_SIZE))["params"]
    jit_act = jax.jit(lambda params, obs: actor.apply({"params": params}, obs))

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        bundle = flax.serialization.msgpack_restore(raw)
        return bundle.get("actor_params", bundle)

    def infer(params, obs, _max_v):
        # TAGDActor reads max_v from obs[11] internally
        return jit_act(params, obs)

    def reset_cb(): pass

    return init_params, load, infer, reset_cb

_DEFAULT_CKPT["tagd"] = paths.checkpoint("tagd", "tagd_best.msgpack")

OBS_SIZE   = 668   # kin(5) + goal_stack(2*3) + ego_deltas(3*3) + lidar_stack(216*3)
ACTION_DIM = 2
use_legs   = args.use_legs

# ══════════════════════════════════════════════════════════════════════════════
# Core logic
# ══════════════════════════════════════════════════════════════════════════════

MAX_EVAL_GOAL_DIST = 20.0   # large enough for all test scenarios


def build_fast_reset(scen_idx, min_goal_dist: float = 8.0):
    # ghost_prob is absorbed from the wrapper's kwargs and overridden to 0.0
    bound_reset = lambda key, max_goal_dist=3.0, scenario_idx=-1, ghost_prob=0.0, min_goal_dist=0.8, **kw: \
        reset_env(key, MAX_EVAL_GOAL_DIST, scenario_idx=scen_idx, ghost_prob=0.0, min_goal_dist=min_goal_dist, **kw)
    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
    jit_rs = jax.jit(lambda key: rs(key, MAX_EVAL_GOAL_DIST, ghost_prob=0.0, min_goal_dist=min_goal_dist))
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
    if algo == "navrep":
        init_params, load_fn, infer_fn, _ = _build_navrep()
    elif algo == "tagd":
        init_params, load_fn, infer_fn, _ = _build_tagd()
    else:
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

    # Optional per-policy reset (MPPI reseeds its warm-start u_mean).
    _policy_reset = getattr(infer_fn, "reset_hook", lambda: None)

    current_scenario = max(7, min(12, args.scenario))
    fast_reset, fast_step = build_fast_reset(current_scenario)

    rng = jax.random.PRNGKey(42)

    pygame.init()

    def _viz_params(room_h_val):
        """Return (scale, sim_w, sim_h, win_w, win_h) for the given room height."""
        sc  = SIM_SIZE / max(ROOM_W, room_h_val)
        sw  = int(ROOM_W * sc)
        sh  = int(room_h_val * sc)
        return sc, sw, sh, sw + PANEL_W, sh

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)
    _init_rh = float(stacked_state.env_state.room_h)
    _sc, _sw, _sh, _ww, _wh = _viz_params(_init_rh)

    screen = pygame.display.set_mode((_ww, _wh))
    pygame.display.set_caption(f"Test Scenarios — {algo.upper()}")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    # Multi-waypoint tracking
    waypoints = TEST_ROBOT_WAYPOINTS[current_scenario]
    wp_idx = 0   # current waypoint index (0 = first goal, already set by scenario)
    wp_segment_reward = 0.0
    wp_segment_steps = 0
    wp_results = []   # (wp_idx, reward, steps, outcome) per segment

    # Trajectory recording (mirrors jax_eval_multi.py): list of (x, y, v) points;
    # persists across waypoint segments so the full multi-wp path is visible.
    TRAJ_MIN_DIST = 0.05
    def _init_trajectory(ss):
        es = jax.device_get(ss.env_state)
        return [(float(es.x), float(es.y), float(es.v))]
    trajectory = _init_trajectory(stacked_state)

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
        nonlocal ep_reward, ep_steps, rew_acc, wp_results, trajectory
        nonlocal rng
        rng, reset_rng = jax.random.split(rng)
        obs, stacked_state = fast_reset(reset_rng)
        _policy_reset()
        wp_idx = 0
        wp_segment_reward = 0.0
        wp_segment_steps = 0
        wp_results = []
        ep_reward = 0.0
        ep_steps = 0
        rew_acc = {k: 0.0 for k in _REW_KEYS}
        trajectory = _init_trajectory(stacked_state)

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
        
        # --- INIZIO MODIFICA: Calcolo dinamico Y per lo scenario 9 ---
        if current_scenario == 9:
            obs_boxes = stacked_state.env_state.obs_boxes
            # 8-vertex layout (CCW from bottom-left): index 5 is the top-right
            # vertex's y, i.e. the wall's top edge (cy + hh).
            if next_gy == -1.0:
                # Porta 1: Y basata sul muro inferiore 1
                next_gy = obs_boxes[0, 5] + 1.0
            elif next_gy == -2.0:
                # Porta 2: Y basata sul muro inferiore 2
                next_gy = obs_boxes[2, 5] + 1.0
        # --- FINE MODIFICA ---

        env_state = stacked_state.env_state
        env_state = env_state.replace(
            goal_x=jnp.float32(next_gx),
            goal_y=jnp.float32(next_gy),
            time_step=jnp.int32(0),
        )
        stacked_state = stacked_state.replace(env_state=env_state)

        # Recompute observation with new goal
        from legnav.core.jax_env import get_obs as _get_obs
        rng, obs_key = jax.random.split(rng)
        new_base_obs, sp_mask = _get_obs(env_state, obs_key)
        env_state = env_state.replace(sp_mask=sp_mask)
        stacked_state = stacked_state.replace(env_state=env_state)

        # Update the stacked observation's goal and kin_vec (goal-relative)
        goal = new_base_obs[:2]
        kin_vec = new_base_obs[2:7]
        lidar = new_base_obs[7:]
        new_goal_stack = stacked_state.goal_stack.at[-1].set(goal)
        new_lidar_stack = stacked_state.lidar_stack.at[-1].set(lidar)
        stacked_state = stacked_state.replace(
            goal_stack=new_goal_stack,
            lidar_stack=new_lidar_stack,
        )
        # Rebuild flat obs (668 layout, strided frame selection) via shared helper
        obs = assemble_stacked_obs(
            kin_vec, new_goal_stack, stacked_state.pose_stack, new_lidar_stack
        )

        # Waypoint change = goal change, so any stateful policy (MPPI) must
        # drop its warm-start plan: the previous u_mean was aiming at the
        # old waypoint and is now stale.
        _policy_reset()

        # Reset per-segment counters — new episode starts from here
        wp_segment_reward = 0.0
        wp_segment_steps  = 0
        ep_reward = 0.0
        ep_steps  = 0
        rew_acc   = {k: 0.0 for k in _REW_KEYS}
        return True

    scen_name = TEST_SCENARIO_NAMES[current_scenario]
    print(f"Test Scenarios Eval — {algo.upper()}")
    print(f"Keys: 7-9 select scenario | 0=10, -/'=11, ==12 | R reset | N next wp | Q quit")
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
                if k in (pygame.K_MINUS, pygame.K_QUOTE): new_scen = 11
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
        # HSFM drives the robot from the true env state (it overloads the 3rd arg
        # to be the EnvState, not max_v) — same convention as jax_eval_multi.py.
        if algo == "hsfm":
            env_action = infer_fn(params, obs, stacked_state.env_state)
        else:
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

        # Append to trajectory (downsampled by TRAJ_MIN_DIST; refresh v if stopped)
        cur_x, cur_y, cur_v = float(cpu_state.x), float(cpu_state.y), float(cpu_state.v)
        if not trajectory or math.hypot(cur_x - trajectory[-1][0], cur_y - trajectory[-1][1]) >= TRAJ_MIN_DIST:
            trajectory.append((cur_x, cur_y, cur_v))
        elif cur_v <= 0.1 and trajectory[-1][2] > 0.1:
            trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], cur_v)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        # ── Rescale viewport to fit the current room (24 m corridor → narrower sim) ──
        rh_val = float(cpu_state.room_h)
        sc, sw, sh, ww, wh = _viz_params(rh_val)
        if screen.get_size() != (ww, wh):
            screen = pygame.display.set_mode((ww, wh))

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs, show_body,
                   scale=sc, sim_h=sh, trajectory=trajectory)

        # Overlay waypoint info on the panel
        wp_banner = f"WP {wp_idx+1}/{len(waypoints)}"
        if banner_t > 0:
            wp_banner = f"{banner} | {wp_banner}"

        draw_panel(screen, fonts, f"{algo.upper()} TEST", ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), wp_banner, max(banner_t, 1),
                   current_scenario, "fixed", use_legs, raw_lidar, sp_mask, rew_acc, show_radar,
                   sim_x=sw, win_h=wh)

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
    if algo == "navrep":
        init_params, load_fn, infer_fn, _ = _build_navrep()
    elif algo == "tagd":
        init_params, load_fn, infer_fn, _ = _build_tagd()
    else:
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

    # Optional per-policy reset (MPPI reseeds its warm-start u_mean).
    _policy_reset = getattr(infer_fn, "reset_hook", lambda: None)

    n_episodes = args.episodes
    rng = jax.random.PRNGKey(42)

    # Metric constants (aligned with benchmark_eval.py / paper_comparison_eval.py).
    DT = _jax_env.DT
    YIELD_DIST = 1.5
    YIELD_FOV  = 0.785          # ±45° (90° cone), matches yielding reward
    SPACE_COMP_DIST = 0.5       # surface distance (m) for a "space-compliant" step

    hdr = (f"{'Scenario':21s} | {'SR%':>5s} | {'ACR%':>5s} | {'PCR%':>5s} | {'TR%':>5s} | "
           f"{'SPL':>4s} | {'TTG':>5s} | {'MHD':>5s} | {'AJ':>6s} | {'SC':>4s} | {'YS':>4s}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for scen_idx in range(7, 13):
        scen_name = TEST_SCENARIO_NAMES[scen_idx]
        waypoints = TEST_ROBOT_WAYPOINTS[scen_idx]
        fast_reset, fast_step = build_fast_reset(scen_idx)

        # Per-episode metric samples.
        m_sr, m_acr, m_pcr, m_tr = [], [], [], []
        m_spl, m_ttg, m_mhd, m_aj, m_sc, m_ys = [], [], [], [], [], []

        for ep in range(n_episodes):
            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            _policy_reset()
            wp_idx = 0
            episode_done = False

            # Whole-episode accumulators (persist across waypoint segments).
            path_len = 0.0          # distance travelled (SPL denominator)
            d_star   = 0.0          # sum of per-segment straight-line distances
            mhd      = np.inf       # min human surface distance
            jw_sum   = 0.0; n_jerk = 0     # angular jerk
            sc_compliant = 0; sc_total = 0 # space compliance
            yz_steps = 0; yc_steps = 0     # yield situations / compliant
            prev_w   = 0.0; prev_aw = 0.0  # angular jerk history (reset per episode)
            n_steps_ep = 0
            outcome  = "tr"         # default outcome if budget exhausted

            while not episode_done:
                # Straight-line distance for this segment (shortest-path proxy).
                es0 = stacked_state.env_state
                d_star += float(jnp.sqrt(
                    (es0.goal_x - es0.x) ** 2 + (es0.goal_y - es0.y) ** 2))

                for step in range(MAX_STEPS):
                    # HSFM overloads the 3rd arg to be the EnvState (see jax_eval_multi.py).
                    if algo == "hsfm":
                        env_action = infer_fn(params, obs, stacked_state.env_state)
                    else:
                        env_action = infer_fn(params, obs, stacked_state.env_state.max_v)
                    rng, step_rng = jax.random.split(rng)
                    obs, stacked_state, reward, done, info = fast_step(
                        step_rng, stacked_state, env_action)

                    es = stacked_state.env_state
                    v  = float(es.v); w = float(es.w)

                    # ── Per-step metrics (state continues even on done) ──────────
                    if _jax_env.USE_LEGS:
                        ch = float(info["closest_shoe_surface"])
                    else:
                        ch = float(info["closest_human"]) - ROBOT_RADIUS - PEOPLE_RADIUS
                    mhd = min(mhd, ch)
                    sc_total += 1
                    if ch > SPACE_COMP_DIST:
                        sc_compliant += 1

                    path_len += v * DT
                    aw = (w - prev_w) / DT
                    jw_sum += abs((aw - prev_aw) / DT)   # angular jerk (rad/s^3)
                    n_jerk += 1
                    prev_w, prev_aw = w, aw
                    n_steps_ep += 1

                    # Yield: a human in the frontal cone within YIELD_DIST means the
                    # robot should stop (v <= 0.1 m/s).
                    ppl = np.asarray(es.people)
                    dpx = ppl[:, 0] - float(es.x)
                    dpy = ppl[:, 1] - float(es.y)
                    dists = np.sqrt(dpx**2 + dpy**2 + 1e-8)
                    rel = np.arctan2(dpy, dpx) - float(es.theta)
                    rel = (rel + np.pi) % (2*np.pi) - np.pi
                    active_p = ppl[:, 10] >= 0.0
                    in_yz = (dists < YIELD_DIST) & (np.abs(rel) < YIELD_FOV) & active_p
                    if in_yz.any():
                        yz_steps += 1
                        if v <= 0.1:
                            yc_steps += 1

                    if done:
                        goal = bool(info["goal_reached"])
                        col = bool(info["collision"])

                        if goal and wp_idx + 1 < len(waypoints):
                            # Advance to next waypoint.
                            # Reset time_step so the 400-step budget is fresh per segment.
                            wp_idx += 1
                            next_gx, next_gy = waypoints[wp_idx]

                            # --- INIZIO MODIFICA: Calcolo dinamico Y per lo scenario 9 ---
                            if scen_idx == 9:
                                obs_boxes = stacked_state.env_state.obs_boxes
                                # 8-vertex layout: index 5 = top-right y = cy + hh
                                if next_gy == -1.0:
                                    next_gy = obs_boxes[0, 5] + 1.0
                                elif next_gy == -2.0:
                                    next_gy = obs_boxes[2, 5] + 1.0
                            # --- FINE MODIFICA ---

                            env_state = stacked_state.env_state
                            env_state = env_state.replace(
                                goal_x=jnp.float32(next_gx),
                                goal_y=jnp.float32(next_gy),
                                time_step=jnp.int32(0),
                            )
                            stacked_state = stacked_state.replace(env_state=env_state)
                            # Recompute obs with new goal
                            from legnav.core.jax_env import get_obs as _get_obs
                            rng, obs_key = jax.random.split(rng)
                            new_base_obs, sp_mask = _get_obs(env_state, obs_key)
                            env_state = env_state.replace(sp_mask=sp_mask)
                            stacked_state = stacked_state.replace(env_state=env_state)
                            goal_vec = new_base_obs[:2]
                            kin_vec = new_base_obs[2:7]
                            lidar = new_base_obs[7:]
                            stacked_state = stacked_state.replace(
                                goal_stack=stacked_state.goal_stack.at[-1].set(goal_vec),
                                lidar_stack=stacked_state.lidar_stack.at[-1].set(lidar),
                            )
                            obs = assemble_stacked_obs(
                                kin_vec,
                                stacked_state.goal_stack,
                                stacked_state.pose_stack,
                                stacked_state.lidar_stack,
                            )
                            # Waypoint changed → drop stale warm-start plan.
                            _policy_reset()
                            break  # break inner for-loop; while continues
                        elif goal:
                            outcome = "success"      # all waypoints reached
                            episode_done = True
                            break
                        else:
                            if bool(info["passive_col"]) and not bool(info["active_col"]):
                                outcome = "pcr"
                            elif col:
                                outcome = "acr"      # active human or static obstacle
                            else:
                                outcome = "tr"
                            episode_done = True
                            break
                else:
                    # MAX_STEPS exhausted without done — timeout for the episode.
                    outcome = "tr"
                    episode_done = True

            # ── Episode-level aggregation ───────────────────────────────────────
            success = (outcome == "success")
            m_sr.append(1.0 if success else 0.0)
            m_acr.append(1.0 if outcome == "acr" else 0.0)
            m_pcr.append(1.0 if outcome == "pcr" else 0.0)
            m_tr.append(1.0 if outcome == "tr" else 0.0)
            m_spl.append((d_star / max(path_len, d_star, 1e-8)) if success else 0.0)
            if success:
                m_ttg.append(n_steps_ep * DT)
            if np.isfinite(mhd):
                m_mhd.append(mhd)
            m_aj.append(jw_sum / max(n_jerk, 1))
            m_sc.append(sc_compliant / max(sc_total, 1))
            if yz_steps > 0:
                m_ys.append(yc_steps / yz_steps)

        # ── Scenario row ────────────────────────────────────────────────────────
        sr  = np.mean(m_sr) * 100
        acr = np.mean(m_acr) * 100
        pcr = np.mean(m_pcr) * 100
        tr  = np.mean(m_tr) * 100
        spl = np.mean(m_spl)
        ttg = np.mean(m_ttg) if m_ttg else float("nan")
        mhd_v = np.mean(m_mhd) if m_mhd else float("nan")
        aj  = np.mean(m_aj)
        sc  = np.mean(m_sc)
        ys  = np.mean(m_ys) if m_ys else float("nan")

        label = f"{scen_idx:2d}.{scen_name}"[:21]
        print(f"{label:21s} | {sr:4.0f}% | {acr:4.0f}% | {pcr:4.0f}% | {tr:4.0f}% | "
              f"{spl:4.2f} | {ttg:5.1f} | {mhd_v:5.2f} | {aj:6.1f} | {sc:4.2f} | {ys:4.2f}")

    print()


if __name__ == "__main__":
    if args.episodes > 0:
        run_headless()
    else:
        run_interactive()