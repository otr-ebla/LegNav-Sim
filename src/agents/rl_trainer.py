import os

# --- RIMOSSO: Non forzare path manuali, usiamo l'ambiente Conda configurato correttamente ---
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=..." 

# Imposta il backend preferito (opzionale, JAX lo fa in automatico se vede le GPU, ma aiuta)
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import time
import optax
from functools import partial
from flax.serialization import to_bytes, from_bytes
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple

# Import envs
from src.jax_env.jax_env import (
    EnvState, StaticConfig, reset, Obstacles, RectObstacles, 
    auto_reset_step, 
    step, render 
)

ACTION_DIM = 2

# --- 1. RETE NEURALE ---
class ActorCritic(nn.Module):
    cfg: StaticConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        mu = nn.Dense(features=ACTION_DIM)(x)
        # Clipping delle azioni
        mu_lin = jnp.clip(mu[..., 0], 0.0, self.cfg.max_lin_vel) 
        mu_ang = jnp.clip(mu[..., 1], -self.cfg.max_ang_vel, self.cfg.max_ang_vel)
        mu = jnp.stack([mu_lin, mu_ang], axis=-1)
        
        log_std = self.param('log_std', nn.initializers.zeros, (ACTION_DIM,))
        std = jnp.exp(log_std)
        
        V = nn.Dense(features=1)(x)
        return mu, std, V.squeeze(-1)

# --- 2. TRAINING STATE ---
class TrainState(NamedTuple):
    params: any
    opt_state: any
    env_state: EnvState
    obs: jnp.ndarray
    obstacles: Obstacles
    rect_obstacles: RectObstacles
    rng: jnp.ndarray

# --- 3. FUNZIONI PPO CORE ---

def calculate_gae(rewards, dones, values, next_value, gamma, lambda_):
    def scan_gae(carry, x):
        gae, next_val = carry
        r, d, v = x
        delta = r + gamma * next_val * (1.0 - d) - v
        gae = delta + gamma * lambda_ * (1.0 - d) * gae
        return (gae, v), (gae, gae + v)

    _, (advantages, targets) = jax.lax.scan(
        scan_gae,
        (jnp.zeros_like(next_value), next_value),
        (rewards, dones, values),
        reverse=True
    )
    return advantages, targets

def ppo_loss_fn(params, batch, cfg):
    model = ActorCritic(cfg=cfg)
    mu, std, values = model.apply({'params': params}, batch['obs'])
    
    def log_prob_fn(action, mu, std):
        return -0.5 * jnp.sum(((action - mu) / std)**2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    
    # in_axes=(0, 0, None) perché std è un parametro fisso non batchato qui
    log_probs = jax.vmap(log_prob_fn, in_axes=(0, 0, None))(batch['actions'], mu, std)
    
    entropy = jnp.sum(0.5 * (jnp.log(2 * jnp.pi * std**2) + 1.0))
    
    ratio = jnp.exp(log_probs - batch['old_log_probs'])
    adv = batch['advantages']
    # Normalizzazione vantaggi
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    
    clip_eps = 0.2
    loss_actor = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * adv).mean()
    loss_critic = jnp.mean((values - batch['targets'])**2) * 0.5
    loss_entropy = -0.01 * entropy.mean()
    
    total_loss = loss_actor + 0.5 * loss_critic + loss_entropy
    return total_loss, (loss_actor, loss_critic, loss_entropy)


# --- 4. TRAINING LOOP ---

# JIT compila l'intero loop. NOTA: static_argnums deve corrispondere agli argomenti non-array
@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def run_training_loop(
    cfg: StaticConfig,
    num_envs: int,
    num_updates: int,
    steps_per_rollout: int
):
    # Inizializzazione RNG
    rng = jax.random.PRNGKey(int(time.time()))
    rng, init_rng, reset_rng = jax.random.split(rng, 3)

    # Reset Ambiente
    reset_keys = jax.random.split(reset_rng, num_envs)
    _, env_state, obs, obstacles, rect_obst = jax.vmap(reset, in_axes=(0, None))(reset_keys, cfg)

    # Init Rete
    network = ActorCritic(cfg=cfg)
    OBS_DIM = cfg.num_rays + 2
    init_obs_dummy = obs[0][:OBS_DIM]
    
    params = network.init(init_rng, init_obs_dummy)['params']
    
    # Init Optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    init_train_state = TrainState(
        params=params,
        opt_state=opt_state,
        env_state=env_state,
        obs=obs, 
        obstacles=obstacles,
        rect_obstacles=rect_obst,
        rng=rng
    )

    # --- RIMOSSO BLOCCO DI VERIFICA GPU QUI ---
    # (Non si può fare .device() dentro una funzione JIT)

    def update_step(train_state: TrainState, _):
        
        # --- A. ROLLOUT (Raccolta Dati) ---
        def rollout_body(carry, _):
            ts, rng = carry
            rng, act_rng = jax.random.split(rng)
            
            current_obs = ts.obs[:, :OBS_DIM]

            # Inferenza rete
            mu, std_vec, value = network.apply({'params': ts.params}, current_obs)
            std = jnp.exp(std_vec)
            
            # Campionamento azione
            act_keys = jax.random.split(act_rng, num_envs)
            noise = jax.random.normal(act_rng, shape=mu.shape)
            action = mu + std * noise
            
            def log_prob_fn(a, m, s):
                return -0.5 * jnp.sum(((a - m) / s)**2 + 2 * jnp.log(s) + jnp.log(2 * jnp.pi))
            
            old_log_prob = jax.vmap(log_prob_fn, in_axes=(0, 0, None))(action, mu, std)

            step_keys = jax.random.split(rng, num_envs)
            
            # Step Environment
            next_state, next_obs, reward, done, next_obst, next_rect = jax.vmap(
                auto_reset_step, 
                in_axes=(0, 0, 0, None, 0, 0, 0)
            )(ts.env_state, ts.obs, action, cfg, ts.obstacles, ts.rect_obstacles, step_keys)

            transition = {
                'obs': current_obs,
                'actions': action,
                'rewards': reward,
                'dones': done,
                'values': value,
                'old_log_probs': old_log_prob
            }

            new_ts = ts._replace(
                env_state=next_state,
                obs=next_obs,
                obstacles=next_obst,
                rect_obstacles=next_rect
            )
            
            return (new_ts, rng), transition

        # Esegui Rollout
        (final_ts_rollout, rng_after_rollout), batch = jax.lax.scan(
            rollout_body, 
            (train_state, train_state.rng), 
            None, 
            length=steps_per_rollout
        )

        # --- B. GAE (Generalized Advantage Estimation) ---
        last_obs = final_ts_rollout.obs[:, :OBS_DIM]
        _, _, last_val = network.apply({'params': train_state.params}, last_obs)
        
        advantages, targets = jax.vmap(calculate_gae, in_axes=(1, 1, 1, 0, None, None))(
            batch['rewards'], 
            batch['dones'], 
            batch['values'], 
            last_val, 
            0.99, 0.95
        )
        
        # Flattening dei batch per l'update PPO
        flat_batch = jax.tree_util.tree_map(
            lambda x: x.reshape((steps_per_rollout * num_envs,) + x.shape[2:]),
            batch
        )
        flat_adv = advantages.reshape(-1)
        flat_tar = targets.reshape(-1)
        
        flat_batch['advantages'] = flat_adv
        flat_batch['targets'] = flat_tar

        # --- C. UPDATE PPO (Epoche multiple) ---
        def train_epoch(carry, _):
            params, opt_st = carry
            grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(params, flat_batch, cfg)
            updates, new_opt_st = optimizer.update(grads, opt_st, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_st), (loss, metrics)

        (new_params, new_opt_state), (loss, metrics) = jax.lax.scan(
            train_epoch, 
            (train_state.params, train_state.opt_state),
            None,
            length=10 # 4 Epoche PPO
        )

        new_train_state = final_ts_rollout._replace(
            params=new_params,
            opt_state=new_opt_state,
            rng=rng_after_rollout
        )
        
        avg_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        return new_train_state, avg_metrics

    # Loop principale di training (jax.lax.scan è molto più veloce di un for loop python)
    final_state, metrics_history = jax.lax.scan(
        update_step, 
        init_train_state, 
        None, 
        length=num_updates
    )
    
    return final_state.params, metrics_history


def eval_and_render(cfg: StaticConfig, params):
    print("\n>>> INIZIO VALUTAZIONE E RENDERING <<<")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    rng = jax.random.PRNGKey(int(time.time()))
    rng, reset_key = jax.random.split(rng)
    
    _, state, obs, obstacles, rect_obst = reset(reset_key, cfg)
    
    network = ActorCritic(cfg=cfg)
    OBS_DIM = cfg.num_rays + 2

    @jax.jit
    def get_deterministic_action(params, obs):
        obs_in = obs[:OBS_DIM] 
        mu, _, _ = network.apply({'params': params}, obs_in)
        return mu

    step_jit = jax.jit(step, static_argnums=(2,)) 
    
    max_steps = 500
    total_reward = 0.0
    
    try:
        for t in range(max_steps):
            action = get_deterministic_action(params, obs)
            state, obs, reward, done = step_jit(state, action, cfg, obstacles, rect_obst)
            total_reward += float(reward)
            render(state, cfg, ax=ax, obstacles=obstacles, rect_obst=rect_obst)
            ax.set_title(f"Step: {t} | Reward: {total_reward:.2f} | Done: {done}")
            plt.pause(0.05) 
            
            if done:
                print(f"Episodio terminato al passo {t}. Reward totale: {total_reward:.2f}")
                time.sleep(1.0)
                break
                
    except KeyboardInterrupt:
        print("Visualizzazione interrotta dall'utente.")
    
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # --- VERIFICA GPU (SPOSTATA QUI) ---
    print("\n--- VERIFICA SISTEMA JAX ---")
    devices = jax.devices()
    print(f"Dispositivi visibili: {devices}")
    try:
        # Verifica allocazione reale
        dummy = jax.numpy.zeros(1)
        print(f"Test allocazione memoria riuscito su: {dummy.device()}")
        if "cpu" in str(dummy.device()).lower():
            print("⚠️  ATTENZIONE: Stai usando la CPU! Il training sarà lento.")
        else:
            print("✅  OTTIMO: GPU rilevata e funzionante.")
    except Exception as e:
        print(f"Errore verifica: {e}")
    print("----------------------------\n")

    # Configurazione 
    static_cfg = StaticConfig(
        dt=0.1, room_width=15.0, room_height=15.0, max_lin_vel=1.0, max_ang_vel=jnp.pi,
        robot_radius=0.2, num_rays=216, max_lidar_distance=20.0, num_people=5, people_radius=0.2,
        min_circ_obstacles=2, max_circ_obstacles=5, obst_min_radius=0.3, obst_max_radius=1.5, obst_clearance=0.2,
        min_rect_obstacles=1, max_rect_obstacles=3, rect_min_width=1.0, rect_max_width=3.0, rect_min_height=1.0, rect_max_height=3.0,
        goal_radius=0.3, goal_min_robot_dist=3.0, goal_reward=10.0,
    )
    
    print(">>> AVVIO TRAINING PURE JAX <<<")
    
    # Hyperparameters
    # NOTA: 100M step sono tanti. Per un test riduciamo a 1M o 5M.
    NUM_ENVS = 512
    STEPS_PER_ROLLOUT = 256
    # TOTAL_TIMESTEPS = 100_000_000 
    TOTAL_TIMESTEPS = 50_000_000 # Riduco per test, rimetti 100M quando sei sicuro
    
    NUM_UPDATES = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_ROLLOUT)
    print(f"Totale Updates da compilare: {NUM_UPDATES}")

    t0 = time.time()
    
    # 1. Training
    print("Inizio compilazione JIT e training (potrebbe volerci 1-2 minuti per partire)...")
    trained_params, metrics = run_training_loop(
        static_cfg, 
        num_envs=NUM_ENVS, 
        num_updates=NUM_UPDATES, 
        steps_per_rollout=STEPS_PER_ROLLOUT
    )
    
    # 2. Sincronizzazione GPU
    print("Training finito. Attendo sincronizzazione GPU...")
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), trained_params)
    
    t1 = time.time()
    total_time = t1 - t0
    print(f"Training REALE completato in {total_time:.2f}s")
    print(f"FPS Stimati: {TOTAL_TIMESTEPS / total_time:.2f}")
    
    # 3. Salvataggio
    with open("ppo_pure_jax_params.msgpack", "wb") as f:
        f.write(to_bytes(trained_params))
    print("Parametri salvati.")
    
    # 4. Visualizzazione
    input("Premi INVIO per avviare la visualizzazione...")
    eval_and_render(static_cfg, params=trained_params)