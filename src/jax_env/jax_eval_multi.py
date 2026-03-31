"""
jax_eval_multi.py — Universal Interactive Evaluation
=====================================================
Loads any PPO / SHAC / SAC / TQC checkpoint via argparse.

Usage:
  # PPO or SHAC (same EndToEndActorCritic architecture, same ckpt format)
  python3 jax_eval_multi.py --algo ppo  --ckpt checkpoints/ppo_model_best.msgpack
  python3 jax_eval_multi.py --algo shac --ckpt checkpoints/shac_best.msgpack

  # SAC (split encoder + head)
  python3 jax_eval_multi.py --algo sac  --ckpt checkpoints_sac/sac_best.msgpack

  # TQC (monolithic actor)
  python3 jax_eval_multi.py --algo tqc  --ckpt checkpoints_tqc/tqc_best.msgpack

Keys:
  0-6   Lock scenario    7   Random mode
  R     Reset episode    L   Toggle LiDAR
  H     Toggle arrows    B   Toggle body ring
  Q/Esc Quit
"""

import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import functools
import random
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization

# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Universal JAX Evaluation")
    p.add_argument("--algo",     default="ppo",
                   choices=["ppo", "shac", "sac", "tqc"],
                   help="Algorithm whose checkpoint to load.")
    p.add_argument("--ckpt",     default="",
                   help="Path to the checkpoint file (msgpack). "
                        "If empty, a sensible default for the chosen algo is used.")
    p.add_argument("--legs",     dest="use_legs", action="store_true",  default=True)
    p.add_argument("--no-legs",  dest="use_legs", action="store_false")
    p.add_argument("--ghost-body", action="store_true",
                   help="Overlay JHSFM body ring on top of legs.")
    p.add_argument("--sensor-noise", action="store_true", default=True,
                   help="Enable Salt&Pepper sensor noise (off by default for clean eval).")
    p.add_argument("--watch", action="store_true", default=False,
                   help="Watch the checkpoint file and hot-reload weights when modified.")
    return p.parse_args()

args = _parse_args()

# Sensor noise must be set BEFORE importing jax_env
import jax_env as _jax_env
_jax_env.USE_LEGS    = args.use_legs
_jax_env.SENSOR_NOISE = args.sensor_noise

from jax_env import (ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS)
from jax_env_multi import reset_env, step_env
from jax_legs import LEG_RADIUS, SHOE_LENGTH, SHOE_WIDTH
from jax_wrappers import make_stacked_env

OBS_SIZE   = 342
ACTION_DIM = 2

# ══════════════════════════════════════════════════════════════════════════════
# Network definitions (each algo has its own architecture)
# ══════════════════════════════════════════════════════════════════════════════

# ── PPO / SHAC ─────────────────────────────────────────────────────────────────
# Uses EndToEndActorCritic from jax_network.py (imported only when needed to
# avoid triggering GPU pinning code in training scripts).
# Also includes a fallback `_OldPPOActor` for older checkpoints where the critic
# branched directly from the `fused` representation instead of `shared` trunk.

class _OldPPOActor(nn.Module):
    action_dim: int
    stack_dim:  int = 3
    num_rays:   int = 108

    @nn.compact
    def __call__(self, x):
        pose_size  = 3 * self.stack_dim
        state_size = 9

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        shared = nn.relu(nn.Dense(128)(shared))

        actor_mean = nn.Dense(self.action_dim)(shared)
        actor_logstd = nn.Dense(self.action_dim, bias_init=nn.initializers.constant(-1.0))(shared)
        actor_logstd = jnp.clip(actor_logstd, -4.0, 0.5)

        critic = nn.relu(nn.Dense(128)(fused))
        critic = nn.relu(nn.Dense(64)(critic))
        value  = nn.Dense(1)(critic)

        return actor_mean, actor_logstd, jnp.squeeze(value, axis=-1)


def _build_ppo_shac():
    from jax_network import EndToEndActorCritic, scale_action_to_env

    net_new = EndToEndActorCritic(action_dim=ACTION_DIM)
    net_old = _OldPPOActor(action_dim=ACTION_DIM)
    rng = jax.random.PRNGKey(0)
    init_params = net_new.init(rng, jnp.zeros((1, OBS_SIZE)))["params"]

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        bundle = flax.serialization.msgpack_restore(raw)
        # SHAC saves the full training bundle; PPO saves just params.
        return bundle.get("params", bundle)

    def infer(params, obs, max_v):
        is_old_arch = ("Dense_6" in params and params["Dense_6"]["kernel"].shape[0] == 960)
        net = net_old if is_old_arch else net_new
        mean, _, _ = net.apply({"params": params}, obs[None])
        return scale_action_to_env(jnp.squeeze(mean, 0), float(max_v))

    return init_params, load, infer


# ── SAC ────────────────────────────────────────────────────────────────────────

class _SACEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x):
        pose_size = 3 * self.stack_dim
        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + 9]
        lidar_flat = x[..., pose_size + 9:]
        bs = lidar_flat.shape[:-1]
        cnn = lidar_flat.reshape((*bs, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(32, (7,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.relu(nn.Conv(64, (5,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.relu(nn.Conv(64, (3,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.LayerNorm()(cnn.reshape((*bs, -1)))
        g   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        g   = nn.relu(nn.Dense(128)(g))
        g   = nn.relu(nn.Dense(64)(g))
        f   = jnp.concatenate([cnn, g], axis=-1)
        f   = nn.relu(nn.Dense(256)(f))
        return nn.relu(nn.Dense(128)(f))

class _SACActorHead(nn.Module):
    @nn.compact
    def __call__(self, feat):
        mean    = nn.Dense(ACTION_DIM, name="mean")(feat)
        log_std = nn.Dense(ACTION_DIM, name="log_std")(feat)
        return mean, jnp.clip(log_std, -5.0, 2.0)

def _build_sac():
    enc  = _SACEncoder()
    head = _SACActorHead()
    rng  = jax.random.PRNGKey(0)
    dummy_obs  = jnp.zeros((1, OBS_SIZE))
    dummy_feat = jnp.zeros((1, 128))
    enc_params  = enc.init(rng, dummy_obs)["params"]
    head_params = head.init(rng, dummy_feat)["params"]
    init_params = (enc_params, head_params)

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        b = flax.serialization.msgpack_restore(raw)
        return b["actor_enc_params"], b["actor_head_params"]

    def infer(params, obs, max_v):
        enc_p, head_p = params
        feat = enc.apply({"params": enc_p}, obs[None])
        mean, _ = head.apply({"params": head_p}, feat)
        mean = jnp.squeeze(mean, 0)
        tanh_mean = jnp.tanh(mean)
        v = (tanh_mean[0] + 1.0) * 0.5 * float(max_v)
        w = tanh_mean[1]
        return jnp.stack([v, w])

    return init_params, load, infer


# ── TQC ────────────────────────────────────────────────────────────────────────

class _TQCActor(nn.Module):
    @nn.compact
    def __call__(self, x):
        pose_size = 9
        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + 9]
        lidar_flat = x[..., pose_size + 9:]
        bs = lidar_flat.shape[:-1]
        cnn = lidar_flat.reshape((*bs, 108, 3))
        cnn = nn.relu(nn.Conv(32, (7,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.relu(nn.Conv(64, (5,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.relu(nn.Conv(64, (3,), strides=(2,), padding="SAME")(cnn))
        cnn = nn.LayerNorm()(cnn.reshape((*bs, -1)))
        g   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        g   = nn.relu(nn.Dense(128)(g))
        g   = nn.relu(nn.Dense(64)(g))
        f   = nn.relu(nn.Dense(256)(jnp.concatenate([cnn, g], axis=-1)))
        f   = nn.relu(nn.Dense(128)(f))
        mean    = nn.Dense(ACTION_DIM)(f)
        log_std = nn.Dense(ACTION_DIM)(f)
        return mean, jnp.clip(log_std, -5.0, 2.0)

def _build_tqc():
    net = _TQCActor()
    rng = jax.random.PRNGKey(0)
    init_params = net.init(rng, jnp.zeros((1, OBS_SIZE)))["params"]

    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        b = flax.serialization.msgpack_restore(raw)
        return b["actor_params"]

    def infer(params, obs, max_v):
        mean, _ = net.apply({"params": params}, obs[None])
        mean = jnp.squeeze(mean, 0)
        tanh_mean = jnp.tanh(mean)
        v = (tanh_mean[0] + 1.0) * 0.5 * float(max_v)
        w = tanh_mean[1]
        return jnp.stack([v, w])

    return init_params, load, infer


# ── Default checkpoint paths ───────────────────────────────────────────────────

_DEFAULT_CKPT = {
    "ppo":  "checkpoints/ppo_classic_best.msgpack",#"checkpoints/ppo_model_best.msgpack",
    "shac": "checkpoints/shac_best.msgpack",
    "sac":  "checkpoints_sac/sac_best.msgpack",
    "tqc":  "checkpoints_tqc/tqc_best.msgpack",
}

# ── Policy factory ─────────────────────────────────────────────────────────────

def build_policy(algo):
    if algo in ("ppo", "shac"):
        return _build_ppo_shac()
    elif algo == "sac":
        return _build_sac()
    elif algo == "tqc":
        return _build_tqc()
    else:
        raise ValueError(f"Unknown algo: {algo}")


# ══════════════════════════════════════════════════════════════════════════════
# Rendering (unchanged from original jax_eval_multi.py)
# ══════════════════════════════════════════════════════════════════════════════

SIM_SIZE   = 800
PANEL_W    = 300
WINDOW_W   = SIM_SIZE + PANEL_W
WINDOW_H   = SIM_SIZE
SCALE      = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET = 10

C_BG        = (28,  28,  34); C_FLOOR    = (44,  44,  52); C_GRID     = (56,  56,  66)
C_WALL      = (190, 190, 205); C_ROBOT    = (60,  140, 255); C_ROBOT_H  = (170, 210, 255)
C_GOAL      = (255, 210,  40); C_GOAL2    = (220, 150,  10)
C_LEG_L     = (90,  220, 100); C_LEG_R    = (50,  170,  65)
C_CIRCLE    = (140,  90,  60); C_CIRCLE_L = (180, 120,  80)
C_BOX       = ( 90, 110, 145); C_BOX_L    = (120, 145, 185)
C_RAY_FAR   = ( 50, 200,  50); C_RAY_NEAR = (220,  50,  50)
C_PANEL     = ( 20,  20,  26); C_TEXT     = (215, 215, 228)
C_DIM       = (120, 120, 138); C_SUCCESS  = ( 50, 215,  95)
C_COLLIDE   = (230,  60,  60); C_TIMEOUT  = (210, 165,  50)
C_BODY_RING = ( 50, 100,  50)

_SHOE_PALETTE = [
    (180,  60,  60), ( 60, 130, 220), (220, 160,  30), ( 60, 200, 160),
    (200,  80, 200), (100, 200,  60), (230, 120,  40), ( 80, 180, 230),
    (200, 200,  60), (160,  60, 200), ( 50, 200, 100), (220,  80, 130),
    ( 60, 160, 160), (200, 140,  80), (130,  80, 200), (200,  60,  80),
]

def W(x, y): return int(x * SCALE), int(SIM_SIZE - y * SCALE)

def _shoe_colour(i):
    c = _SHOE_PALETTE[i % len(_SHOE_PALETTE)]
    return c, tuple(max(0, v - 60) for v in c)

def make_fonts():
    return {
        "big"  : pygame.font.SysFont("monospace", 21, bold=True),
        "mid"  : pygame.font.SysFont("monospace", 15),
        "small": pygame.font.SysFont("monospace", 13),
        "tiny" : pygame.font.SysFont("monospace", 11),
    }


def draw_star(surface, cx, cy, r_out, r_in, n, color, border=None):
    pts = []
    for i in range(2 * n):
        angle = math.radians(-90 + i * 180 / n)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)
    if border: pygame.draw.polygon(surface, border, pts, 2)


def draw_lidar(surface, state, raw_lidar, show):
    if not show or raw_lidar is None: return
    rx, ry = W(float(state.x), float(state.y))
    theta  = float(state.theta)
    angles = theta - FOV / 2.0 + np.arange(NUM_RAYS) * FOV / (NUM_RAYS - 1)
    sp_mask = np.array(state.sp_mask)
    for i, (ang, dist) in enumerate(zip(angles, raw_lidar)):
        ex, ey = W(state.x + dist * math.cos(ang), state.y + dist * math.sin(ang))
        if sp_mask[i]:
            col = (180, 50, 220)
        else:
            t = max(0.0, 1.0 - dist / MAX_LIDAR_DIST)
            col = tuple(int(C_RAY_NEAR[j]*t + C_RAY_FAR[j]*(1-t)) for j in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)


def draw_shoe(surface, fx, fy, theta, col, border):
    ct, st = math.cos(theta), math.sin(theta)
    lx, ly = -st, ct
    hL, hW = SHOE_LENGTH * 0.5, SHOE_WIDTH * 0.5
    cx, cy = fx + ct * hL, fy + st * hL
    corners = [
        (cx + ct*hL + lx*hW, cy + st*hL + ly*hW),
        (cx + ct*hL - lx*hW, cy + st*hL - ly*hW),
        (cx - ct*hL - lx*hW, cy - st*hL - ly*hW),
        (cx - ct*hL + lx*hW, cy - st*hL + ly*hW),
    ]
    pts = [W(wx, wy) for wx, wy in corners]
    pygame.draw.polygon(surface, col,    pts)
    pygame.draw.polygon(surface, border, pts, 1)


def draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body):
    n = int(state.people.shape[0])
    left_legs  = foot_state_np[:, 0:2]
    right_legs = foot_state_np[:, 2:4]
    for i in range(n):
        if float(state.people[i, 10]) < 0: continue
        px, py  = float(state.people[i, 0]), float(state.people[i, 1])
        vx, vy  = float(state.people[i, 2]), float(state.people[i, 3])
        theta_h = float(state.people[i, 4])
        col, border = _shoe_colour(i)

        if show_body:
            sx, sy = W(px, py)
            pygame.draw.circle(surface, C_BODY_RING, (sx, sy),
                                max(3, int(_jax_env.PEOPLE_RADIUS * SCALE)), 1)

        if use_legs:
            left_theta  = float(foot_state_np[i, 10])
            right_theta = float(foot_state_np[i, 11])
            draw_shoe(surface, float(left_legs[i, 0]),  float(left_legs[i, 1]),  left_theta, col, border)
            draw_shoe(surface, float(right_legs[i, 0]), float(right_legs[i, 1]), right_theta, col, border)
            leg_r = max(2, int(LEG_RADIUS * SCALE))
            lx, ly = W(float(left_legs[i, 0]),  float(left_legs[i, 1]))
            rx_, ry_ = W(float(right_legs[i, 0]), float(right_legs[i, 1]))
            pygame.draw.circle(surface, tuple(min(255,int(c*1.1)) for c in col), (lx, ly), leg_r)
            pygame.draw.circle(surface, (20,20,20), (lx, ly), leg_r, 1)
            pygame.draw.circle(surface, tuple(max(0,int(c*0.75)) for c in col), (rx_, ry_), leg_r)
            pygame.draw.circle(surface, (20,20,20), (rx_, ry_), leg_r, 1)
        else:
            sx, sy = W(px, py)
            pr = max(3, int(_jax_env.PEOPLE_RADIUS * SCALE))
            pygame.draw.circle(surface, col,    (sx, sy), pr)
            pygame.draw.circle(surface, border, (sx, sy), pr, 1)
            speed = math.hypot(vx, vy)
            if show_arrows and speed > 0.05:
                ax, ay = W(px + math.cos(theta_h)*0.5, py + math.sin(theta_h)*0.5)
                pygame.draw.line(surface, (20,120,20), (sx,sy), (ax,ay), 2)


def draw_scene(surface, state, raw_lidar, foot_state_np, show_lidar, show_arrows, use_legs, show_body):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0); pygame.draw.line(surface, C_GRID, (sx,0), (sx,SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j); pygame.draw.line(surface, C_GRID, (0,sy), (SIM_SIZE,sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

    draw_lidar(surface, state, raw_lidar, show_lidar)

    for box in np.array(state.obs_boxes):
        cx, cy, hw, hh = box
        if hw > 0:
            sx, sy = W(cx - hw, cy + hh)
            pygame.draw.rect(surface, C_BOX,   (sx, sy, int(2*hw*SCALE), int(2*hh*SCALE)))
            pygame.draw.rect(surface, C_BOX_L, (sx, sy, int(2*hw*SCALE), int(2*hh*SCALE)), 2)
    for cir in np.array(state.obs_circles):
        cx, cy, r = cir
        if r > 0:
            sx, sy = W(cx, cy); pr = max(2, int(r * SCALE))
            pygame.draw.circle(surface, C_CIRCLE,   (sx, sy), pr)
            pygame.draw.circle(surface, C_CIRCLE_L, (sx, sy), pr, 2)

    gx, gy = W(float(state.goal_x), float(state.goal_y))
    draw_star(surface, gx, gy, int(0.30*SCALE), int(0.12*SCALE), 5, C_GOAL, C_GOAL2)
    draw_humans(surface, state, foot_state_np, show_arrows, use_legs, show_body)

    rx, ry = W(float(state.x), float(state.y)); rr = max(4, int(ROBOT_RADIUS*SCALE))
    pygame.draw.circle(surface, C_ROBOT,   (rx, ry), rr)
    pygame.draw.circle(surface, C_ROBOT_H, (rx, ry), rr, 2)
    hx, hy = W(state.x + ROBOT_RADIUS*3*math.cos(float(state.theta)),
               state.y + ROBOT_RADIUS*3*math.sin(float(state.theta)))
    pygame.draw.line(surface, C_ROBOT_H, (rx, ry), (hx, hy), 3)


def draw_panel(surface, fonts, algo, ep, step, ep_ret, max_v, v, w,
               goal_dist, goal_align, ch, stats, banner, banner_t,
               scen_idx, eval_mode, use_legs, raw_lidar, sp_mask):
    pygame.draw.rect(surface, C_PANEL, (SIM_SIZE, 0, PANEL_W, WINDOW_H))
    pygame.draw.line(surface, C_WALL,  (SIM_SIZE, 0), (SIM_SIZE, WINDOW_H), 2)
    y = 10; lh = 19; x0 = SIM_SIZE + 10

    def txt(s, col=C_TEXT, font="mid"):
        nonlocal y
        surface.blit(fonts[font].render(s, True, col), (x0, y)); y += lh

    def sep():
        nonlocal y
        pygame.draw.line(surface, C_DIM, (x0, y+3), (x0+PANEL_W-20, y+3), 1); y += 10

    txt(f"─ {algo.upper()} EVAL ─", C_TEXT, "big"); y += 2; sep()
    scen_name = {0:"Random",1:"Parallel",2:"Perpend",3:"Circular",
                 4:"Bottleneck",5:"Intersect",6:"Groups"}.get(scen_idx, "?")
    mode_text = "[RANDOM]" if eval_mode == "random" else "[FIXED]"
    leg_mode  = "LEGS" if use_legs else "CYLINDERS"
    txt(f"{mode_text} {leg_mode}", C_DIM, "small")
    txt(f"Scen     {scen_name}", C_SUCCESS)
    txt(f"Episode  {ep:>5d}"); txt(f"Step     {step:>4d}"); sep()
    txt(f"max_v {max_v:>+6.3f} m/s")
    txt(f"v     {v:>+6.3f} m/s"); txt(f"w     {w:>+6.3f} rad/s"); sep()
    txt(f"Goal  {goal_dist:>5.2f} m")
    txt(f"Align {math.degrees(goal_align):>+5.1f}°")
    txt(f"Human {ch:>5.2f} m"); sep()
    txt(f"Ep ret {ep_ret:>+7.2f}", C_GOAL, "mid"); sep()
    txt("── Last 50 episodes ─", C_DIM, "small"); y += 2
    txt(f"  Success  {stats['suc']:>5.1f}%",  C_SUCCESS)
    txt(f"  Collision{stats['col']:>5.1f}%",  C_COLLIDE)
    txt(f"  Pass.Col {stats['pcol']:>5.1f}%", (200,100,100))
    txt(f"  Timeout  {stats['tmo']:>5.1f}%",  C_TIMEOUT); sep()
    txt("0-6 lock  7 random  R reset", C_DIM, "tiny")
    txt("L lidar  H arrows  B body  Q quit", C_DIM, "tiny")

    if banner_t > 0:
        label, col = {
            "success"    : ("✓  GOAL REACHED",  C_SUCCESS),
            "collision"  : ("X  COLLISION",     C_COLLIDE),
            "passive_col": ("🚶 PASSIVE COL",   (200,100,100)),
            "timeout"    : ("⏱  TIMEOUT",       C_TIMEOUT),
        }.get(banner, ("", C_TEXT))
        if label:
            surf = fonts["big"].render(label, True, col)
            radar_r   = 130
            radar_top = WINDOW_H - radar_r * 2 - 20
            bx = SIM_SIZE + PANEL_W // 2 - surf.get_width() // 2
            by = radar_top - surf.get_height() - 20
            bg = pygame.Surface((surf.get_width()+24, surf.get_height()+12))
            bg.fill((20,20,26)); bg.set_alpha(220)
            surface.blit(bg, (bx-12, by-6)); surface.blit(surf, (bx, by))

    # LiDAR radar chart
    if raw_lidar is not None:
        radar_r  = 130
        radar_cx = SIM_SIZE + PANEL_W // 2
        radar_cy = WINDOW_H - radar_r - 20
        pygame.draw.circle(surface, (15,15,20),  (radar_cx, radar_cy), radar_r)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), radar_r, 1)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), int(radar_r*.66), 1)
        pygame.draw.circle(surface, (50,50,60),  (radar_cx, radar_cy), int(radar_r*.33), 1)
        pygame.draw.circle(surface, C_ROBOT,     (radar_cx, radar_cy), 3)
        angles = np.linspace(FOV/2, -FOV/2, len(raw_lidar)) - math.pi/2
        dists  = (raw_lidar / MAX_LIDAR_DIST) * radar_r
        xs = radar_cx + np.cos(angles) * dists
        ys = radar_cy + np.sin(angles) * dists
        for i, (rx2, ry2) in enumerate(zip(xs, ys)):
            if sp_mask[i] and raw_lidar[i] >= MAX_LIDAR_DIST - 0.1:
                col = (180, 50, 220)
            else:
                col = (220, 50, 50)
            pygame.draw.circle(surface, col, (int(rx2), int(ry2)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    algo     = args.algo
    ckpt     = args.ckpt or _DEFAULT_CKPT[algo]
    use_legs = args.use_legs

    print(f"\n🚀 Universal Eval — algo={algo.upper()}")
    print(f"   Checkpoint : {ckpt}")
    print(f"   Human model: {'LEGS' if use_legs else 'CYLINDERS'}")
    print(f"   Sensor noise: {'ON' if args.sensor_noise else 'OFF'}\n")

    # Build policy (init params, loader, inference fn)
    init_params, load_fn, infer_fn = build_policy(algo)

    try:
        params = load_fn(ckpt)
        print(f"✅ Loaded {algo.upper()} weights from {ckpt}")
    except FileNotFoundError:
        params = init_params
        print(f"⚠️  Checkpoint not found — running with random weights.")

    last_mtime = os.path.getmtime(ckpt) if os.path.exists(ckpt) else 0

    def build_fast_reset(scen_idx):
        bound_reset = functools.partial(reset_env, scenario_idx=scen_idx)
        rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3, ghost_robot=False)
        return jax.jit(rs), jax.jit(ss)

    rng = jax.random.PRNGKey(42)
    evaluation_mode  = "random"
    current_scenario = random.randint(0, 6)
    fast_reset, fast_step = build_fast_reset(current_scenario)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"JAX Eval — {algo.upper()}")
    clock = pygame.time.Clock(); fonts = make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    ep=0; ep_steps=0; ep_reward=0.0; ep_hist=[]
    paused=False; show_lidar=True; show_arrows=True; show_body=args.ghost_body
    banner=""; banner_t=0

    def get_stats():
        if not ep_hist: return {"suc":0.,"col":0.,"tmo":0.,"pcol":0.}
        w = np.array(ep_hist[-50:])
        return {"suc":w[:,1].mean()*100,"col":w[:,2].mean()*100,
                "tmo":w[:,3].mean()*100,"pcol":w[:,4].mean()*100}

    print("🎮 Keys: 0-6 lock scenario | 7 random | R reset | L lidar | H arrows | B body | Q quit")
    if args.watch: print(f"👀 WATCH MODE ENABLED: Polling {ckpt} for updates.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if k == pygame.K_SPACE: paused = not paused
                if k == pygame.K_l:    show_lidar  = not show_lidar
                if k == pygame.K_h:    show_arrows = not show_arrows
                if k == pygame.K_b:    show_body   = not show_body
                if pygame.K_0 <= k <= pygame.K_6:
                    evaluation_mode  = "fixed"
                    current_scenario = k - pygame.K_0
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_7:
                    evaluation_mode  = "random"
                    current_scenario = random.randint(0, 6)
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    ep_reward=0.0; ep_steps=0; banner_t=0

        if paused: clock.tick(10); continue

        if args.watch and ep_steps % 30 == 0:
            try:
                mtime = os.path.getmtime(ckpt)
                if mtime > last_mtime:
                    params = load_fn(ckpt)
                    last_mtime = mtime
                    print(f"🔄 Reloaded weights from {ckpt}!")
            except Exception:
                pass

        # ── Inference ────────────────────────────────────────────────────────
        env_action = infer_fn(params, obs, stacked_state.env_state.max_v)

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward); ep_steps += 1

        cpu_state = jax.device_get(stacked_state.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * \
                    (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask       = np.array(cpu_state.sp_mask)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np,
                   show_lidar, show_arrows, use_legs, show_body)
        draw_panel(screen, fonts, algo, ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), banner, banner_t,
                   current_scenario, evaluation_mode, use_legs, raw_lidar, sp_mask)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip(); clock.tick(FPS_TARGET)

        if done:
            goal = bool(info["goal_reached"]); col = bool(info["collision"])
            pcol = bool(info["passive_col"]); tmo = not goal and not col
            active_col = col and not pcol
            banner   = "success" if goal else ("collision" if active_col else
                        ("passive_col" if pcol else "timeout"))
            banner_t = FPS_TARGET * 2
            ep_hist.append((ep_reward, float(goal), float(active_col), float(tmo), float(pcol)))

            if evaluation_mode == "random":
                current_scenario = random.randint(0, 6)
                fast_reset, fast_step = build_fast_reset(current_scenario)

            ep+=1; ep_reward=0.0; ep_steps=0

            if ep >= 300:
                s = get_stats()
                print(f"\n✅ 300 episodes done. "
                      f"Success: {s['suc']:.1f}% | "
                      f"Collision: {s['col']:.1f}% | "
                      f"Pass.Col: {s['pcol']:.1f}% | "
                      f"Timeout: {s['tmo']:.1f}%")
                pygame.quit(); return

            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)


if __name__ == "__main__":
    main()