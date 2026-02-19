"""
jax_eval.py — Visual Evaluation with PyGame
============================================
Fixes vs original:
  - dummy_obs size corrected to OBS_SIZE (338)
  - Checkpoint loading updated for new bundle format (params + opt_state)
  - draw_env draws room walls, circular pillars, LiDAR rays, and heading arrows
  - Deterministic action (no sampling noise) via get_deterministic_action
  - Graceful handling when no checkpoint exists (runs random policy)
  - Clock-controlled loop with configurable FPS
"""

import os
import time
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

from jax_env import reset_env, step_env, ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, NUM_RAYS
from jax_wrappers import make_stacked_env
from jax_network import EndToEndActorCritic, get_deterministic_action
from jax_train import OBS_SIZE

# ── Display config ────────────────────────────────────────────────────────────
WINDOW_SIZE = 800
SCALE       = WINDOW_SIZE / max(ROOM_W, ROOM_H)
FPS_TARGET  = 20   # real-time playback speed

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG       = (245, 245, 245)
C_WALL     = (40,  40,  40)
C_ROBOT    = (30,  80, 220)
C_GOAL     = (255, 165,   0)
C_PERSON   = (50, 180,  50)
C_OBSTACLE = (120, 120, 120)
C_LIDAR    = (200,  60,  60, 60)   # semi-transparent


def to_screen(x, y):
    """Sim coords → PyGame pixels (Y-flipped)."""
    return int(x * SCALE), int(WINDOW_SIZE - y * SCALE)


def load_checkpoint(dummy_params, dummy_opt_state, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    with open(filepath, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes(
        {"params": dummy_params, "opt_state": dummy_opt_state}, raw
    )
    return bundle["params"]


def draw_env(screen, state, lidar_raw=None):
    """Render a single frame onto `screen`."""
    screen.fill(C_BG)

    # Room border
    pygame.draw.rect(screen, C_WALL, pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE), 3)

    # Circular pillar obstacles
    for i in range(state.obstacles.shape[0]):
        cx, cy, r = float(state.obstacles[i, 0]), float(state.obstacles[i, 1]), float(state.obstacles[i, 2])
        sx, sy = to_screen(cx, cy)
        pygame.draw.circle(screen, C_OBSTACLE, (sx, sy), max(1, int(r * SCALE)))

    # Goal
    gx, gy = to_screen(float(state.goal_x), float(state.goal_y))
    pygame.draw.circle(screen, C_GOAL, (gx, gy), int(0.3 * SCALE))
    pygame.draw.circle(screen, (200, 120, 0), (gx, gy), int(0.3 * SCALE), 2)

    # LiDAR rays (optional, semi-transparent overlay)
    if lidar_raw is not None:
        fov   = 2.0 * np.pi
        theta = float(state.theta)
        angles = theta - fov / 2.0 + np.arange(NUM_RAYS) * (fov / (NUM_RAYS - 1))
        for i, (ang, dist) in enumerate(zip(angles, lidar_raw)):
            ex = float(state.x) + dist * np.cos(ang)
            ey = float(state.y) + dist * np.sin(ang)
            sx_, sy_ = to_screen(float(state.x), float(state.y))
            ex_, ey_ = to_screen(ex, ey)
            pygame.draw.line(screen, (220, 80, 80), (sx_, sy_), (ex_, ey_), 1)

    # People
    for i in range(state.people.shape[0]):
        px, py  = float(state.people[i, 0]), float(state.people[i, 1])
        ang     = float(state.people[i, 4])
        distracted = float(state.people[i, 5]) > 0.5
        cx_, cy_ = to_screen(px, py)
        colour = (255, 140, 0) if distracted else C_PERSON
        pygame.draw.circle(screen, colour, (cx_, cy_), max(1, int(PEOPLE_RADIUS * SCALE)))
        hx_, hy_ = to_screen(px + 0.35 * np.cos(ang), py + 0.35 * np.sin(ang))
        pygame.draw.line(screen, (0, 80, 0), (cx_, cy_), (hx_, hy_), 2)

    # Robot body
    rx_, ry_ = to_screen(float(state.x), float(state.y))
    pygame.draw.circle(screen, C_ROBOT, (rx_, ry_), max(1, int(ROBOT_RADIUS * SCALE)))
    # Heading arrow
    theta = float(state.theta)
    hx_, hy_ = to_screen(
        float(state.x) + ROBOT_RADIUS * 2.0 * np.cos(theta),
        float(state.y) + ROBOT_RADIUS * 2.0 * np.sin(theta)
    )
    pygame.draw.line(screen, (0, 0, 100), (rx_, ry_), (hx_, hy_), 3)

    pygame.display.flip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🚀 JAX LiDAR-Nav Evaluation")

    # Network
    network = EndToEndActorCritic(action_dim=2)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    dummy_obs  = jnp.zeros((1, OBS_SIZE))
    params     = network.init(init_rng, dummy_obs)["params"]
    from optax import adam
    dummy_opt  = adam(3e-4).init(params)

    ckpt = "checkpoints/ppo_model_best.msgpack"
    try:
        params = load_checkpoint(params, dummy_opt, ckpt)
        print(f"✅ Loaded weights from {ckpt}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint at {ckpt} — running random policy.")

    # Env (stacked, no auto-reset for evaluation so we see episode boundaries)
    reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
    fast_reset = jax.jit(reset_stacked)
    fast_step  = jax.jit(step_stacked)

    # PyGame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("JAX Indoor-Nav — Evaluation")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 16)

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)
    ep_reward  = 0.0
    ep_steps   = 0
    ep_count   = 0

    print("🎮 Running. Close window to stop.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Deterministic inference
        obs_batch  = obs[None]   # add batch dim
        mean, _, _ = network.apply({"params": params}, obs_batch)
        action     = get_deterministic_action(jnp.squeeze(mean, axis=0))

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, action)
        ep_reward += float(reward)
        ep_steps  += 1

        # Pull state to CPU for rendering
        cpu_state = jax.device_get(stacked_state.env_state)

        # Extract raw lidar from current obs (last num_rays elements of the latest frame)
        # obs layout: pose_stack(9) + state_vec(5) + lidar_stack(324)
        # Latest lidar frame = last NUM_RAYS values of lidar_stack portion
        cpu_lidar_stack = jax.device_get(stacked_state.lidar_stack)  # (3, 108)
        cpu_lidar = cpu_lidar_stack[-1]   # most recent frame (108,)

        # Convert normalised lidar back to raw distances for rendering
        from jax_env import MAX_LIDAR_DIST
        raw_lidar = MAX_LIDAR_DIST * (1.0 - cpu_lidar)

        draw_env(screen, cpu_state, raw_lidar)

        # HUD overlay
        hud = font.render(
            f"Ep {ep_count:03d}  Step {ep_steps:04d}  R {ep_reward:+.1f}  "
            f"v={float(stacked_state.env_state.v):.2f}  w={float(stacked_state.env_state.w):.2f}",
            True, (20, 20, 20)
        )
        screen.blit(hud, (8, 8))
        pygame.display.flip()
        clock.tick(FPS_TARGET)

        if done:
            goal = bool(info["goal_reached"])
            col  = bool(info["collision"])
            print(f"  Ep {ep_count:03d} finished — steps:{ep_steps} reward:{ep_reward:.1f}  "
                  f"{'GOAL ✅' if goal else 'COLLISION 💥' if col else 'TIMEOUT ⏱️'}")
            time.sleep(0.5)

            ep_count  += 1
            ep_reward  = 0.0
            ep_steps   = 0

            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)


if __name__ == "__main__":
    main()