"""
jax_debug_train.py — Single-Environment Visual Debug Training
=============================================================
FIXES & IMPROVEMENTS vs previous version:

  FIX (carried) — BUG 6: logstd[0] strip of batch dim in infer().

  FIX (carried) — BUG 8: deque(maxlen=BUF_SIZE) replacing O(N) list.pop(0).

  FIX A — ENTROPY_COEF mismatch (NEW):
    Was 0.05 here vs 0.002 in jax_ppo.py — 25× mismatch meant debug training
    produced a completely different policy than the full trainer.
    Fixed: now uses shared ENTROPY_COEF = 0.002 imported concept.
    The constant is defined here explicitly to match jax_ppo.py.

  FIX B — GAE bootstrap ignores non-terminal last step (NEW):
    gae() initialised nv = 0.0 regardless of whether the buffer's last
    transition was terminal. For a non-terminal last step the value should
    be bootstrapped from the critic's estimate of the NEXT state.
    Fixed: nv = values[-1] if not done, 0.0 if done (standard correct GAE).

  IMPROVEMENT — OBS_SIZE updated to 342 (was 339):
    Matches jax_env.py new layout: 9+9+324 = 342.

  UNCHANGED — Render functions, ppo_update, infer, sample_action_raw
  are all correct.
"""

import os
import math
import warnings
import collections
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_physics import compute_lidar as _compute_lidar

from jax_env import (reset_env, step_env,
                     ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS,
                     NUM_RAYS, MAX_LIDAR_DIST, FOV, NUM_PEOPLE,
                     NUM_OBS_CIR, NUM_OBS_BOX, MAX_STEPS)
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_train import OBS_SIZE

# FIX A: must match jax_ppo.py exactly
ENTROPY_COEF = 0.002
CLIP_EPS     = 0.2

# ── Layout ────────────────────────────────────────────────────────────────────
SIM_SIZE  = 800
PANEL_W   = 300
WINDOW_W  = SIM_SIZE + PANEL_W
WINDOW_H  = SIM_SIZE
SCALE     = SIM_SIZE / max(ROOM_W, ROOM_H)
FPS_DEF   = 20

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


# ── Draw helpers ──────────────────────────────────────────────────────────────

def draw_star(surface, cx, cy, r_out, r_in, n, colour, border=None):
    pts = []
    for i in range(2 * n):
        angle = math.radians(-90 + i * 180 / n)
        r     = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, colour, pts)
    if border:
        pygame.draw.polygon(surface, border, pts, 2)


def draw_lidar(surface, state, show):
    if not show:
        return
    rx, ry = W(float(state.x), float(state.y))
    theta  = float(state.theta)
    fov    = float(FOV)
    angles = theta - fov / 2 + np.arange(NUM_RAYS) * fov / (NUM_RAYS - 1)

    people_cir = np.column_stack([
        np.array(state.people[:, 0]),
        np.array(state.people[:, 1]),
        np.full(NUM_PEOPLE, PEOPLE_RADIUS)
    ])
    all_cir = np.vstack([people_cir, np.array(state.obs_circles)])
    raw = np.array(_compute_lidar(
        state.x, state.y, state.theta,
        jnp.array(all_cir), state.obs_boxes,
        NUM_RAYS, float(FOV), MAX_LIDAR_DIST, ROOM_W, ROOM_H
    ))

    for ang, dist in zip(angles, raw):
        ex, ey = W(float(state.x) + dist * np.cos(ang),
                   float(state.y) + dist * np.sin(ang))
        t   = max(0.0, 1.0 - dist / MAX_LIDAR_DIST)
        col = tuple(int(C_RAY_NEAR[i]*t + C_RAY_FAR[i]*(1-t)) for i in range(3))
        pygame.draw.line(surface, col, (rx, ry), (ex, ey), 1)

    for side in [-1, 1]:
        ang = theta + side * fov / 2
        ex, ey = W(float(state.x) + MAX_LIDAR_DIST * np.cos(ang),
                   float(state.y) + MAX_LIDAR_DIST * np.sin(ang))
        pygame.draw.line(surface, C_DIM, (rx, ry), (ex, ey), 1)


def draw_scene(surface, state, show_lidar, show_arrows):
    pygame.draw.rect(surface, C_FLOOR, (0, 0, SIM_SIZE, SIM_SIZE))
    for i in range(int(ROOM_W) + 1):
        sx, _ = W(i, 0); pygame.draw.line(surface, C_GRID, (sx, 0), (sx, SIM_SIZE))
    for j in range(int(ROOM_H) + 1):
        _, sy = W(0, j); pygame.draw.line(surface, C_GRID, (0, sy), (SIM_SIZE, sy))
    pygame.draw.rect(surface, C_WALL, (0, 0, SIM_SIZE, SIM_SIZE), 3)

    draw_lidar(surface, state, show_lidar)

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


def draw_panel(surface, fonts, px0, ep, step, ep_ret, v, w,
               goal_dist, goal_align, closest_h,
               last_rew, n_upd, stats, banner, banner_t):
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

    txt("─ DEBUG TRAIN ─", C_TEXT, "big"); y += 2; sep()
    txt(f"Episode  {ep:>5d}",  font="mid")
    txt(f"Step     {step:>4d}/{MAX_STEPS}", font="mid")
    txt(f"Updates  {n_upd:>5d}", font="mid")
    sep()
    txt(f"v     {v:>+6.3f} m/s",      font="mid")
    txt(f"ω     {w:>+6.3f} rad/s",    font="mid")
    sep()
    txt(f"Goal  {goal_dist:>5.2f} m",       font="mid")
    txt(f"Align {math.degrees(goal_align):>+5.1f}°", font="mid")
    txt(f"Human {closest_h:>5.2f} m",       font="mid")
    sep()
    txt(f"Step R {last_rew:>+7.3f}", font="mid")
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


# ── PPO helpers ───────────────────────────────────────────────────────────────

network   = EndToEndActorCritic(action_dim=2)
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(3e-4, eps=1e-5)
)


@jax.jit
def infer(params, obs):
    mean, logstd, value = network.apply({"params": params}, obs[None])
    # Strip batch dim from all three outputs
    return mean[0], logstd[0], value[0]


@jax.jit
def sample_action_raw(key, mean, logstd):
    """Returns (raw_action, log_prob) — both in raw Gaussian space."""
    noise  = jax.random.normal(key, mean.shape)
    action = mean + jnp.exp(logstd) * noise
    lp     = jnp.sum(-0.5*(noise**2 + jnp.log(2*jnp.pi)) - logstd)
    return action, lp


@jax.jit
def ppo_update(params, opt_state, obs, raw_actions, returns, advantages, old_lp):
    """
    raw_actions are Gaussian samples (pre-squash), consistent with old_lp.
    Uses ENTROPY_COEF = 0.002 (matches jax_ppo.py — was 0.05 before, now fixed).
    """
    def loss_fn(p):
        mean, logstd, vals = network.apply({"params": p}, obs)
        std  = jnp.exp(logstd)
        z    = (raw_actions - mean) / (std + 1e-8)
        lp   = jnp.sum(-0.5*(z**2 + jnp.log(2*jnp.pi)) - logstd, axis=-1)
        adv  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        r    = jnp.exp(jnp.clip(lp - old_lp, -5.0, 5.0))
        pi_l = -jnp.mean(jnp.minimum(
            r * adv,
            jnp.clip(r, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        ))
        v_l  = 0.5 * jnp.mean((returns - vals)**2)
        # FIX A: ENTROPY_COEF = 0.002, not 0.05
        ent  = -ENTROPY_COEF * jnp.mean(jnp.sum(0.5*jnp.log(2*jnp.pi*jnp.e) + logstd, axis=-1))
        return pi_l + v_l + ent, (pi_l, v_l)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, new_os = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), new_os, float(loss)


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Standard GAE.
    FIX B: bootstrap from values[-1] if last transition is non-terminal.
    Old code always initialised nv=0.0 — wrong for non-terminal buffer ends.
    """
    T   = len(rewards)
    adv = np.zeros(T, np.float32)
    # FIX B: correct bootstrap value
    nv  = 0.0 if bool(dones[-1]) else float(values[-1])
    g   = 0.0
    for t in reversed(range(T)):
        nd     = 1.0 - float(dones[t])
        d      = rewards[t] + gamma * nv * nd - values[t]
        g      = d + gamma * lam * nd * g
        adv[t] = g
        nv     = float(values[t])
    ret = adv + np.array(values, np.float32)
    return ret, adv


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    CKPT      = "checkpoints/ppo_model_best.msgpack"
    STACK     = 3
    TRAIN_EPS = 5
    BUF_SIZE  = 512

    reset_stk, step_stk = make_stacked_env(reset_env, step_env, stack_dim=STACK)
    jreset = jax.jit(reset_stk)
    jstep  = jax.jit(step_stk)

    rng = jax.random.PRNGKey(0)
    rng, k = jax.random.split(rng)
    params    = network.init(k, jnp.zeros((1, OBS_SIZE)))["params"]
    opt_state = optimizer.init(params)

    if os.path.exists(CKPT):
        try:
            with open(CKPT,"rb") as f:
                b = flax.serialization.from_bytes(
                    {"params":params,"opt_state":opt_state}, f.read())
            params, opt_state = b["params"], b["opt_state"]
            print(f"📥 Checkpoint loaded: {CKPT}")
        except Exception as e:
            print(f"⚠  Could not load checkpoint: {e}")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("JAX Nav — Debug")
    clock  = pygame.time.Clock()
    fonts  = make_fonts()

    rng, k = jax.random.split(rng)
    obs, state = jreset(k)

    buf_obs     = collections.deque(maxlen=BUF_SIZE)
    buf_raw_act = collections.deque(maxlen=BUF_SIZE)
    buf_rew     = collections.deque(maxlen=BUF_SIZE)
    buf_done    = collections.deque(maxlen=BUF_SIZE)
    buf_lp      = collections.deque(maxlen=BUF_SIZE)
    buf_val     = collections.deque(maxlen=BUF_SIZE)

    ep, step, ep_ret = 0, 0, 0.0
    n_upd         = 0
    last_rew      = 0.0
    paused        = False
    show_lidar    = True
    show_arrows   = True
    fps           = FPS_DEF
    banner        = ""
    banner_t      = 0
    ep_hist       = []

    def get_stats():
        if not ep_hist: return {"suc":0.,"col":0.,"tmo":0.,"pcol":0.,"ret":0.}
        w = np.array(ep_hist[-50:])
        return {"suc": w[:,1].mean()*100, "col": w[:,2].mean()*100,
                "tmo": w[:,3].mean()*100, "pcol": w[:,4].mean()*100, "ret": w[:,0].mean()}

    print(f"🎮  Debug train started (OBS_SIZE={OBS_SIZE}). SPACE=pause  R=reset  Q=quit")

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if e.key == pygame.K_SPACE:  paused = not paused
                if e.key == pygame.K_l:      show_lidar  = not show_lidar
                if e.key == pygame.K_h:      show_arrows = not show_arrows
                if e.key == pygame.K_EQUALS: fps = min(fps + 5, 60)
                if e.key == pygame.K_MINUS:  fps = max(fps - 5, 1)
                if e.key == pygame.K_r:
                    rng, k = jax.random.split(rng)
                    obs, state = jreset(k)
                    ep_ret = 0.; step = 0
                    buf_obs.clear(); buf_raw_act.clear(); buf_rew.clear()
                    buf_done.clear(); buf_lp.clear(); buf_val.clear()

        if paused:
            clock.tick(10); continue

        rng, ak, sk = jax.random.split(rng, 3)
        mean, logstd, value = infer(params, obs)

        raw_action, lp = sample_action_raw(ak, mean, logstd)

        mv = float(state.env_state.max_v)
        env_action = scale_action_to_env(raw_action, mv)

        obs, state, reward, done, info = jstep(sk, state, env_action)
        ep_ret += float(reward); step += 1; last_rew = float(reward)

        buf_obs.append(np.array(obs))
        buf_raw_act.append(np.array(raw_action))
        buf_rew.append(float(reward))
        buf_done.append(bool(done))
        buf_lp.append(float(lp))
        buf_val.append(float(value))

        if done:
            goal = bool(info["goal_reached"])
            col  = bool(info["collision"])
            pcol = bool(info["passive_col"])
            tmo  = not goal and not col
            banner  = "success" if goal else ("collision" if col else "timeout")
            banner_t= fps * 2

            ep_hist.append((ep_ret, float(goal), float(col), float(tmo), float(pcol)))
            ep += 1

            sym = "✓" if goal else ("✗" if col else "⏱")
            print(f"  Ep {ep:04d} | {sym} | steps:{step:4d} | ret:{ep_ret:+7.2f}")

            if ep % TRAIN_EPS == 0 and len(buf_obs) >= 32:
                b_obs     = jnp.array(np.array(buf_obs))
                b_raw_act = jnp.array(np.array(buf_raw_act))
                # FIX B: pass values list for correct non-terminal bootstrap
                b_ret, b_adv = gae(list(buf_rew), list(buf_val), list(buf_done))
                b_ret  = jnp.array(b_ret)
                b_adv  = jnp.array(b_adv)
                b_lp   = jnp.array(list(buf_lp))
                params, opt_state, loss = ppo_update(
                    params, opt_state,
                    b_obs, b_raw_act,
                    b_ret, b_adv, b_lp
                )
                n_upd += 1
                print(f"  → update #{n_upd}  loss={loss:.4f}")
                os.makedirs("checkpoints", exist_ok=True)
                bundle = {"params": jax.device_get(params),
                          "opt_state": jax.device_get(opt_state)}
                with open(CKPT,"wb") as f:
                    f.write(flax.serialization.to_bytes(bundle))

            rng, k = jax.random.split(rng)
            obs, state = jreset(k)
            ep_ret = 0.; step = 0

        screen.fill(C_BG)
        cpu_state = jax.device_get(state.env_state)
        draw_scene(screen, cpu_state, show_lidar, show_arrows)

        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch     = float(info["closest_human"]) - ROBOT_RADIUS - PEOPLE_RADIUS

        draw_panel(screen, fonts, SIM_SIZE + 10,
                   ep, step, ep_ret,
                   float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch,
                   last_rew, n_upd, get_stats(),
                   banner, banner_t)

        if banner_t > 0: banner_t -= 1

        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()