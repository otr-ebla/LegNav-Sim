import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import math
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pygame

import jax_env as _jax_env
import jax_network as network
from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, MAX_LIDAR_DIST, FOV
from jax_env_multi import reset_env, step_env
from jax_eval_multi import (
    draw_scene, draw_panel, make_fonts, WINDOW_W, WINDOW_H, SIM_SIZE, PANEL_W, FPS_TARGET
)
from jax_wrappers import make_stacked_env

# ── Hyperparameters for Single-Env Debug ─────────────────────────────────────
OBS_SIZE       = 342
ACTION_DIM     = 2
ROLLOUT_STEPS  = 150    # Steps before network update
MINI_BATCH_SIZE= 150
PPO_EPOCHS     = 4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25
ENTROPY_COEF   = 0.003
LR             = 3e-4

def calculate_gae(rewards, values, dones):
    """Calculate Generalized Advantage Estimation for a single rollout line."""
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    next_value = 0.0 # simple bootstrap 0

    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_v = values[t + 1] if t + 1 < len(rewards) else next_value
        delta = rewards[t] + GAMMA * next_v * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        
    returns = advantages + values
    return advantages, returns


def ppo_loss_fn(params, obs_batch, act_batch, adv_batch, ret_batch, old_lp_batch):
    mean, logstd, values = network.EndToEndActorCritic(action_dim=ACTION_DIM).apply({"params": params}, obs_batch)
    
    # log_prob for normal distribution
    variance = jnp.exp(2 * logstd)
    log_probs = -0.5 * ((act_batch - mean) ** 2 / variance + 2 * logstd + jnp.log(2 * jnp.pi))
    log_prob  = jnp.sum(log_probs, axis=-1)

    ratio = jnp.exp(jnp.clip(log_prob - old_lp_batch, -5.0, 5.0))
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * adv_batch,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_batch
    ))

    value_loss = VF_COEF * jnp.mean((ret_batch - values) ** 2)
    entropy = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -ENTROPY_COEF * entropy

    return policy_loss + value_loss + entropy_loss

@jax.jit
def ppo_update(opt_state, params, obs_batch, act_batch, adv_batch, ret_batch, old_lp_batch):
    adv_batch = (adv_batch - jnp.mean(adv_batch)) / (jnp.std(adv_batch) + 1e-8)
    grads = jax.grad(ppo_loss_fn)(params, obs_batch, act_batch, adv_batch, ret_batch, old_lp_batch)
    updates, new_opt_state = optax.adam(LR).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def main():
    print("🚀 Single-Environment Host Debug Training")

    rng = jax.random.PRNGKey(42)

    # Networks & Optimizer
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    net = network.EndToEndActorCritic(action_dim=ACTION_DIM)
    
    rng, net_rng = jax.random.split(rng)
    params = net.init(net_rng, dummy_obs)["params"]
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    # Environment
    rs, ss = make_stacked_env(reset_env, step_env, stack_dim=3, ghost_robot=False)
    fast_reset = jax.jit(rs)
    fast_step  = jax.jit(ss)

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    # Inference Fn
    @jax.jit
    def get_action_and_value(p, rng_act, o):
        mean, logstd, value = net.apply({"params": p}, o[None])
        std = jnp.exp(logstd)
        # Sample normal
        noise = jax.random.normal(rng_act, mean.shape)
        action = mean + noise * std
        
        # Log prob
        variance = jnp.exp(2 * logstd)
        log_prob = -0.5 * ((action - mean) ** 2 / variance + 2 * logstd + jnp.log(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=-1)
        
        # Scale to environment bounds (v: 0 to max_v, w: -1 to 1)
        # Tanh squashing
        tanh_act = jnp.tanh(action)
        max_v = o[0] # Roughly max_v feature might be missing here but we can extract from env_state below
        v = (tanh_act[0,0] + 1.0) * 0.5
        w = tanh_act[0,1]
        env_action = jnp.stack([v, w])

        return action[0], log_prob[0], value[0], env_action

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Host Debug RL")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    ep = 0
    ep_steps = 0
    ep_reward = 0.0
    ep_hist = []

    update = 0

    while True:
        update += 1
        
        # ── Rollout Collection (Host Loop) ───────────────────────────────────────
        b_obs, b_act, b_rew, b_val, b_don, b_logp = [], [], [], [], [], []

        for step in range(ROLLOUT_STEPS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            rng, act_rng = jax.random.split(rng)
            
            # 1. Infer Action
            raw_action, log_prob, val, env_act = get_action_and_value(params, act_rng, obs)
            
            # The true max_v is kept in the JAX env_state, so we must correct the V scale:
            v_val = float(env_act[0]) * float(stacked_state.env_state.max_v)
            w_val = float(env_act[1])
            final_env_action = jnp.stack([v_val, w_val])

            # 2. Step Env
            rng, step_rng = jax.random.split(rng)
            next_obs, next_stacked_state, reward, done, info = fast_step(step_rng, stacked_state, final_env_action)
            ep_reward += float(reward)
            ep_steps += 1

            # Save transitions
            b_obs.append(obs)
            b_act.append(raw_action)
            b_rew.append(float(reward))
            b_val.append(float(val))
            b_don.append(int(done))
            b_logp.append(float(log_prob))

            # 3. Render directly using the visualizer from jax_eval_multi
            cpu_state = jax.device_get(stacked_state.env_state)
            raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * (MAX_LIDAR_DIST - ROBOT_RADIUS)
            foot_state_np = np.array(cpu_state.foot_state)
            sp_mask       = np.array(cpu_state.sp_mask)

            dx = float(cpu_state.goal_x) - float(cpu_state.x)
            dy = float(cpu_state.goal_y) - float(cpu_state.y)
            gdist  = math.hypot(dx, dy)
            galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
            ch     = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

            screen.fill((10, 10, 15))
            draw_scene(screen, cpu_state, raw_lidar, foot_state_np, True, True, True, False)
            
            # Draw overlay data
            banner = ""
            if done:
                goal = bool(info["goal_reached"]); col = bool(info["collision"])
                pcol = bool(info["passive_col"]); tmo = not goal and not col
                active_col = col and not pcol
                banner = "success" if goal else ("collision" if active_col else ("passive_col" if pcol else "timeout"))
                ep_hist.append((ep_reward, float(goal), float(active_col), float(tmo), float(pcol)))
            
            stats = {"suc":0.,"col":0.,"tmo":0.,"pcol":0.} if not ep_hist else \
                    {"suc":np.mean(np.array(ep_hist)[-50:,1])*100, 
                     "col":np.mean(np.array(ep_hist)[-50:,2])*100,
                     "tmo":np.mean(np.array(ep_hist)[-50:,3])*100,
                     "pcol":np.mean(np.array(ep_hist)[-50:,4])*100}

            draw_panel(screen, fonts, "debug", ep, ep_steps, ep_reward,
                       float(cpu_state.max_v), v_val, w_val,
                       gdist, galign, ch, stats, banner, 0,
                       -1, "random", True, raw_lidar, sp_mask)
            
            pygame.display.flip()
            clock.tick(60) # Visual throttle so user can see it!

            if done:
                rng, reset_rng = jax.random.split(rng)
                obs, stacked_state = fast_reset(reset_rng)
                ep += 1
                ep_reward = 0.0
                ep_steps = 0
            else:
                obs = next_obs
                stacked_state = next_stacked_state


        # ── Optimize / Learn ──────────────────────────────────────────────────────
        b_obs = jnp.stack(b_obs)
        b_act = jnp.stack(b_act)
        b_val = np.array(b_val)
        b_rew = np.array(b_rew)
        b_don = np.array(b_don)
        b_logp = jnp.stack(b_logp)

        adv, ret = calculate_gae(b_rew, b_val, b_don)
        adv = jnp.array(adv)
        ret = jnp.array(ret)

        for _ in range(PPO_EPOCHS):
            params, opt_state = ppo_update(opt_state, params, b_obs, b_act, adv, ret, b_logp)
        
        print(f"Update: {update} | Last 50 Suc: {stats['suc']:.1f}% Col: {stats['col']:.1f}%")

if __name__ == "__main__":
    main()
