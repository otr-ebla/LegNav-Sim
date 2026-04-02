"""
dreamer_eval.py — Interactive Evaluation for DreamerV3

Fixes vs submitted version:
  - BUG E FIXED: build_fast_reset passed ghost_robot=False to make_stacked_env,
    but the parameter is named ghost_prob (a float). At eval time ghost_prob=0.0
    means humans are fully aware of the robot — the correct evaluation setting.
"""

import argparse
import os
# Disabilitiamo la GPU per PyGame in modo da non occupare VRAM inutile
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import functools
import random
import pygame
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

# Importazioni di ambiente e rendering
import jax_env as _jax_env
_jax_env.USE_LEGS = True
_jax_env.SENSOR_NOISE = False

from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, NUM_RAYS, MAX_LIDAR_DIST, FOV, MAX_STEPS
from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env
from jax_network import scale_action_to_env
from jax_eval_multi import (
    C_BG, WINDOW_W, WINDOW_H, FPS_TARGET, 
    draw_scene, draw_panel, make_fonts
)

# Importazioni architettura Dreamer
from dreamer_rssm import DreamerEncoder, RSSM, DETERMINISTIC_SIZE, LATENT_SIZE
from dreamer_behavior import DreamerActor

# --- Costruzione Policy Dreamer ---
def build_dreamer_policy():
    encoder = DreamerEncoder()
    rssm = RSSM(action_dim=2)
    actor = DreamerActor(action_dim=2)
    
    def load_fn(path):
        with open(path, "rb") as f:
            raw = f.read()
        # Inizializza dummy params per capire la struttura
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 666))
        dummy_h = jnp.zeros((1, DETERMINISTIC_SIZE))
        dummy_z = jnp.zeros((1, LATENT_SIZE))
        dummy_act = jnp.zeros((1, 2))
        
        dummy_params = {
            'encoder': encoder.init(rng, dummy_obs)['params'],
            'rssm': rssm.init(rng, dummy_h, dummy_z, dummy_act, dummy_h)['params'],
            'actor': actor.init(rng, dummy_h, dummy_z)['params']
        }
        return flax.serialization.from_bytes(dummy_params, raw)

    @jax.jit
    def infer_step(params, obs_b, prev_h, prev_z, prev_action, max_v):
        """
        Step di inferenza puro. Batched per dimensione 1.
        Usa la variante greedy del posterior per l'evaluazione.
        """
        obs_embed = encoder.apply({'params': params['encoder']}, obs_b)
        
        # 1. Avanza lo stato deterministico (memoria)
        h_next = rssm.apply({'params': params['rssm']}, prev_h, prev_z, prev_action, method=rssm.step_gru)
        
        # 2. Calcola lo stato stocastico usando l'osservazione reale (senza rumore Gumbel)
        z_next, _ = rssm.apply({'params': params['rssm']}, h_next, obs_embed, method=rssm.posterior_greedy)
        
        # 3. L'attore decide in base all'immaginazione latente
        mean, _ = actor.apply({'params': params['actor']}, h_next, z_next)
        
        raw_action = jnp.tanh(mean)
        env_action = scale_action_to_env(raw_action[0], max_v)
        
        return env_action, raw_action, h_next, z_next

    return load_fn, infer_step

# --- Main Loop ---
def main():
    print(f"\n🚀 DreamerV3 Eval")
    ckpt = "checkpoints/dreamer_best.msgpack"
    
    load_fn, infer_step = build_dreamer_policy()
    
    try:
        params = load_fn(ckpt)
        print(f"✅ Loaded Dreamer weights from {ckpt}")
    except FileNotFoundError:
        print(f"❌ Checkpoint {ckpt} non trovato! Esegui prima dreamer_train.py con il salvataggio.")
        return

    def build_fast_reset(scen_idx):
        bound_reset = functools.partial(reset_env, scenario_idx=scen_idx)
        # BUG E FIX: make_stacked_env accepts ghost_prob (float), not ghost_robot (bool).
        rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3, ghost_prob=0.0)
        return jax.jit(rs), jax.jit(ss)

    rng = jax.random.PRNGKey(42)
    evaluation_mode  = "random"
    current_scenario = random.randint(0, 6)
    fast_reset, fast_step = build_fast_reset(current_scenario)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"JAX Eval — DreamerV3")
    clock = pygame.time.Clock(); fonts = make_fonts()

    rng, reset_rng = jax.random.split(rng)
    obs, stacked_state = fast_reset(reset_rng)

    # Memoria Latente (inizializzata a zero ad ogni nuovo episodio)
    current_h = jnp.zeros((1, DETERMINISTIC_SIZE))
    current_z = jnp.zeros((1, LATENT_SIZE))
    prev_act  = jnp.zeros((1, 2))

    ep=0; ep_steps=0; ep_reward=0.0; ep_hist=[]
    paused=False; show_lidar=True; show_arrows=True; show_body=False
    banner=""; banner_t=0

    def get_stats():
        if not ep_hist: return {"suc":0.,"col":0.,"tmo":0.,"pcol":0.}
        w = np.array(ep_hist[-50:])
        return {"suc":w[:,1].mean()*100,"col":w[:,2].mean()*100,
                "tmo":w[:,3].mean()*100,"pcol":w[:,4].mean()*100}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); return
                if k == pygame.K_SPACE: paused = not paused
                if k == pygame.K_l: show_lidar = not show_lidar
                if k == pygame.K_h: show_arrows = not show_arrows
                if k == pygame.K_b: show_body = not show_body
                if pygame.K_0 <= k <= pygame.K_6:
                    evaluation_mode = "fixed"
                    current_scenario = k - pygame.K_0
                    fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    # RESET MEMORIA LATENTE
                    current_h = jnp.zeros((1, DETERMINISTIC_SIZE))
                    current_z = jnp.zeros((1, LATENT_SIZE))
                    prev_act  = jnp.zeros((1, 2))
                    ep_reward=0.0; ep_steps=0; banner_t=0
                if k == pygame.K_7 or k == pygame.K_r:
                    if k == pygame.K_7:
                        evaluation_mode = "random"
                        current_scenario = random.randint(0, 6)
                        fast_reset, fast_step = build_fast_reset(current_scenario)
                    rng, reset_rng = jax.random.split(rng)
                    obs, stacked_state = fast_reset(reset_rng)
                    # RESET MEMORIA LATENTE
                    current_h = jnp.zeros((1, DETERMINISTIC_SIZE))
                    current_z = jnp.zeros((1, LATENT_SIZE))
                    prev_act  = jnp.zeros((1, 2))
                    ep_reward=0.0; ep_steps=0; banner_t=0

        if paused: clock.tick(10); continue

        # ── Inferenza Ricorsiva ──
        # Aggiungiamo la batch dimension per l'RSSM
        obs_b = obs[None, :] 
        env_action, raw_action, current_h, current_z = infer_step(
            params, obs_b, current_h, current_z, prev_act, stacked_state.env_state.max_v
        )
        prev_act = raw_action

        rng, step_rng = jax.random.split(rng)
        obs, stacked_state, reward, done, info = fast_step(step_rng, stacked_state, env_action)
        ep_reward += float(reward); ep_steps += 1

        # Raccogli dati per PyGame
        cpu_state = jax.device_get(stacked_state.env_state)
        raw_lidar = MAX_LIDAR_DIST - jax.device_get(stacked_state.lidar_stack)[-1] * (MAX_LIDAR_DIST - ROBOT_RADIUS)
        foot_state_np = np.array(cpu_state.foot_state)
        sp_mask = np.array(cpu_state.sp_mask)
        dx = float(cpu_state.goal_x) - float(cpu_state.x)
        dy = float(cpu_state.goal_y) - float(cpu_state.y)
        gdist  = math.hypot(dx, dy)
        galign = (math.atan2(dy, dx) - float(cpu_state.theta) + math.pi) % (2*math.pi) - math.pi
        ch = float(info["closest_human"]) - ROBOT_RADIUS - _jax_env.PEOPLE_RADIUS

        screen.fill(C_BG)
        draw_scene(screen, cpu_state, raw_lidar, foot_state_np, show_lidar, show_arrows, True, show_body)
        draw_panel(screen, fonts, "dreamer", ep, ep_steps, ep_reward,
                   float(cpu_state.max_v), float(cpu_state.v), float(cpu_state.w),
                   gdist, galign, ch, get_stats(), banner, banner_t,
                   current_scenario, evaluation_mode, True, raw_lidar, sp_mask)

        if banner_t > 0: banner_t -= 1
        pygame.display.flip(); clock.tick(FPS_TARGET)

        if done:
            goal = bool(info["goal_reached"]); col = bool(info["collision"])
            pcol = bool(info["passive_col"]); tmo = not goal and not col
            active_col = col and not pcol
            banner = "success" if goal else ("collision" if active_col else ("passive_col" if pcol else "timeout"))
            banner_t = FPS_TARGET * 2
            ep_hist.append((ep_reward, float(goal), float(active_col), float(tmo), float(pcol)))

            if evaluation_mode == "random":
                current_scenario = random.randint(0, 6)
                fast_reset, fast_step = build_fast_reset(current_scenario)

            rng, reset_rng = jax.random.split(rng)
            obs, stacked_state = fast_reset(reset_rng)
            
            # RESET MEMORIA LATENTE DOPO IL DONE
            current_h = jnp.zeros((1, DETERMINISTIC_SIZE))
            current_z = jnp.zeros((1, LATENT_SIZE))
            prev_act  = jnp.zeros((1, 2))
            
            ep+=1; ep_reward=0.0; ep_steps=0

if __name__ == "__main__":
    main()