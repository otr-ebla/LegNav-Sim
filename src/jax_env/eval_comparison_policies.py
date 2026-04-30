"""
eval_comparison_policies.py — Visual evaluation for comparison baselines.
=========================================================================
Mirrors jax_eval_multi.py in style; adds DWA, MPPI and NavRep policies.

Usage:
  python3 eval_comparison_policies.py --algo dwa
  python3 eval_comparison_policies.py --algo mppi
  python3 eval_comparison_policies.py --algo navrep
  python3 eval_comparison_policies.py --algo navrep --ckpt checkpoints_navrep/navrep_best.msgpack

Keys:
  0-6   Lock scenario    7   Random mode
  R     Reset episode    →   Skip episode (no stats)
  L     Toggle LiDAR     H   Toggle arrows
  B     Toggle body ring Q/Esc Quit
"""

import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import random
import sys
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_THIS_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_THIS_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Visual evaluation — comparison policies")
    p.add_argument("--algo", default="dwa",
                   choices=["dwa", "mppi", "navrep", "mlp", "tagd"],
                   help="Policy to evaluate: dwa | mppi | navrep | mlp | tagd")
    p.add_argument("--ckpt", default="",
                   help="Checkpoint path (navrep only). "
                        "Default: checkpoints_navrep/navrep_best.msgpack")
    p.add_argument("--legs",     dest="use_legs", action="store_true",  default=True)
    p.add_argument("--no-legs",  dest="use_legs", action="store_false")
    p.add_argument("--ghost-body", action="store_true",
                   help="Overlay JHSFM body ring on top of legs.")
    p.add_argument("--sensor-noise", action="store_true", default=True)
    return p.parse_args()

args = _parse_args()

# Flags must be set BEFORE importing jax_env
import jax_env as _jax_env
_jax_env.USE_LEGS     = args.use_legs
_jax_env.SENSOR_NOISE = args.sensor_noise

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS)
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env

OBS_SIZE   = 662
ACTION_DIM = 2

# ── Borrow all rendering from jax_eval_multi (patch argv to silence its parser)
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
from jax_eval_multi import (
    draw_scene, draw_panel, make_fonts,
    C_BG, WINDOW_W, WINDOW_H, FPS_TARGET,
    SIM_SIZE, PANEL_W, SCALE, W,
)
sys.argv = _saved_argv

# ══════════════════════════════════════════════════════════════════════════════
# Policy builders — each returns (init_params, load_fn, infer_fn)
# ══════════════════════════════════════════════════════════════════════════════

def _build_dwa():
    from comparison_policies.dwa_planner import DWA
    planner = DWA()
    jit_act = jax.jit(planner.act)

    def load(_path):   return None
    def infer(_p, obs, _mv): return jit_act(obs)
    def reset_cb():    pass   # stateless — nothing to reset

    return None, load, infer, reset_cb


def _build_mppi():
    from comparison_policies.mppi_planner import MPPI
    planner = MPPI()
    # MPPI is stateful: it carries a warm-start control sequence across steps.
    # State is kept in a 1-element list so the closure can mutate it.
    _rng  = [jax.random.PRNGKey(0)]
    _umean = [planner.init_u_mean()]
    jit_act = jax.jit(planner.act)

    def load(_path): return None

    def infer(_p, obs, _mv):
        _rng[0], k = jax.random.split(_rng[0])
        action, new_umean = jit_act(obs, _umean[0], k)
        _umean[0] = new_umean
        return action

    def reset_cb():
        _umean[0] = planner.init_u_mean()

    return None, load, infer, reset_cb


def _build_navrep():
    from comparison_policies.navrep_network import NavRepActorCritic
    from jax_network import scale_action_to_env

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


def _build_mlp():
    from comparison_policies.vanilla_mlp_network import VanillaMLPActorCritic
    from jax_network import scale_action_to_env

    net = VanillaMLPActorCritic(action_dim=ACTION_DIM, hidden_dim=128)
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


def _build_tagd():
    from comparison_policies.tagd_network import TAGDActor

    actor = TAGDActor()
    rng   = jax.random.PRNGKey(0)
    init_params = actor.init(rng, jnp.zeros(OBS_SIZE))["params"]
    jit_act = jax.jit(lambda params, obs: actor.apply({"params": params}, obs))

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        bundle = flax.serialization.msgpack_restore(raw)
        # train_tagd_ddpg.py saves under "actor_params"
        return bundle.get("actor_params", bundle)

    def infer(params, obs, _max_v):
        # TAGDActor reads max_v from obs[11] internally — no external scaling needed
        return jit_act(params, obs)

    def reset_cb(): pass   # stateless

    return init_params, load, infer, reset_cb


_DEFAULT_CKPT = {
    "navrep": "checkpoints_navrep/navrep_best.msgpack",
    "mlp":    "checkpoints_vanilla_ppo/ppo_mlp_best.msgpack",
    "tagd":   "checkpoints_tagd/tagd_best.msgpack",
}

_BUILDERS = {
    "dwa":    _build_dwa,
    "mppi":   _build_mppi,
    "navrep": _build_navrep,
    "mlp":    _build_mlp,
    "tagd":   _build_tagd,
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    algo     = args.algo
    use_legs = args.use_legs
    ckpt     = args.ckpt or _DEFAULT_CKPT.get(algo, "")

    print(f"\nComparison Policy Eval — algo={algo.upper()}")
    print(f"   Checkpoint : {ckpt or 'N/A (zero-shot)'}")
    print(f"   Human model: {'LEGS' if use_legs else 'CYLINDERS'}")
    print(f"   Sensor noise: {'ON' if args.sensor_noise else 'OFF'}\n")

    init_params, load_fn, infer_fn, reset_cb = _BUILDERS[algo]()

    if ckpt:
        try:
            params = load_fn(ckpt)
            print(f"Loaded {algo.upper()} weights from {ckpt}")
        except FileNotFoundError:
            params = init_params
            print(f"Checkpoint not found — running with random weights.")
    else:
        params = init_params  # None for DWA/MPPI

    MAX_EVAL_GOAL_DIST = 9.0

    def build_fast_reset(scen_idx, max_goal_dist=MAX_EVAL_GOAL_DIST):
        bound_reset = lambda key, max_goal_dist=3.0, scenario_idx=-1, **kw: \
            reset_env(key, max_goal_dist, scenario_idx=scen_idx, **kw)
        rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
        jit_rs = jax.jit(lambda key: rs(key, max_goal_dist, ghost_prob=0.0))
        jit_ss = jax.jit(ss)
        return jit_rs, jit_ss

    rng = jax.random.PRNGKey(42)
    evaluation_mode  = "random"
    current_scenario = random.randint(0, 6)
    fast_reset, fast_step = build_fast_reset(current_scenario)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Comparison Eval — {algo.upper()}")
    clock  = pygame.time.Clock()
    fonts  = make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    _REW_KEYS = ["rew_progress", "rew_step", "rew_smooth", "rew_yield"]
    ep=0; ep_steps=0; ep_reward=0.0; ep_hist=[]
    rew_acc = {k: 0.0 for k in _REW_KEYS}
    paused=False; show_lidar=True; show_arrows=True
    show_body=args.ghost_body; show_radar=True
    fps_idx=0; fps_speeds=[10, 20, 30]; current_fps=fps_speeds[fps_idx]
    banner=""; banner_t=0

    def get_stats():
        if not ep_hist: return {"suc": 0., "col": 0., "tmo": 0., "pcol": 0.}
        w = np.array(ep_hist[-50:])
        return {"suc": w[:,1].mean()*100, "col": w[:,2].mean()*100,
                "tmo": w[:,3].mean()*100, "pcol": w[:,4].mean()*100}

    print("Keys: 0-6 lock scenario | 7 random | R reset | → skip | L lidar | H arrows | B body | Q quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if k == pygame.K_SPACE: paused = not paused
                if k == pygame.K_l:    show_lidar  = not show_lidar
                if k == pygame.K_h:    show_arrows = not show_arrows
                if k == pygame.K_b:    show_body   = not show_body
                if k == pygame.K_p:    show_radar  = not show_radar
                if k == pygame.K_s:
                    fps_idx = (fps_idx + 1) % len(fps_speeds)
                    current_fps = fps_speeds[fps_idx]
                    banner = f"FPS: {current_fps}"; banner_t = 15
                if pygame.K_0 <= k <= pygame.K_6:
                    evaluation_mode  = "fixed"
                    current_scenario = k - pygame.K_0
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    reset_cb(); ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_7:
                    evaluation_mode  = "random"
                    current_scenario = random.randint(0, 6)
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    reset_cb(); ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    reset_cb(); ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_RIGHT:
                    if evaluation_mode == "random":
                        current_scenario = random.randint(0, 6)
                        fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    reset_cb(); ep_reward=0.0; ep_steps=0
                    banner="skipped"; banner_t=FPS_TARGET

        if paused:
            clock.tick(10); continue

        # ── Inference ─────────────────────────────────────────────────────────
        env_action = infer_fn(params, obs, stacked_state.env_state.max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward); ep_steps += 1
        for k in _REW_KEYS:
            rew_acc[k] += float(info.get(k, 0.0))

        cpu_state     = jax.device_get(stacked_state.env_state)
        raw_lidar     = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * \
                        (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask       = np.array(cpu_state.sp_mask)

        dx     = float(cpu_state.goal_x) - float(cpu_state.x)
        dy     = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs, show_body)
        draw_panel(screen, fonts, algo, ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), banner, banner_t,
                   current_scenario, evaluation_mode, use_legs,
                   raw_lidar, sp_mask, rew_acc, show_radar)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip(); clock.tick(current_fps)

        if done:
            goal       = bool(info["goal_reached"])
            col        = bool(info["collision"])
            pcol       = bool(info["passive_col"])
            active_col = col and not pcol
            tmo        = not goal and not col
            banner     = ("success"     if goal       else
                          "collision"   if active_col else
                          "passive_col" if pcol        else "timeout")
            banner_t   = FPS_TARGET * 2
            ep_hist.append((ep_reward, float(goal), float(active_col),
                            float(tmo), float(pcol)))

            if evaluation_mode == "random":
                current_scenario = random.randint(0, 6)
                fast_reset, fast_step = build_fast_reset(current_scenario)

            ep += 1; ep_reward = 0.0; ep_steps = 0
            rew_acc = {k: 0.0 for k in _REW_KEYS}
            reset_cb()

            if ep >= 300:
                s = get_stats()
                print(f"\n300 episodes done. "
                      f"Success: {s['suc']:.1f}% | "
                      f"Collision: {s['col']:.1f}% | "
                      f"Pass.Col: {s['pcol']:.1f}% | "
                      f"Timeout: {s['tmo']:.1f}%")
                pygame.quit(); return

            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            reset_cb()


if __name__ == "__main__":
    main()
