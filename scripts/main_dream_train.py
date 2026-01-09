import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# --- IMPORTIAMO I TUOI FILE ---
from envs.jax_env import StaticConfig, reset, auto_reset_step
from dreamerv3_lidar import Dreamer, train_world_model, train_actor_critic

# --- 1. CONFIGURAZIONE AMBIENTE ---
env_cfg = StaticConfig(
    dt=0.1,
    room_width=15.0,
    room_height=15.0,
    max_lin_vel=1.0,
    max_ang_vel=2.0,
    robot_radius=0.2,
    num_rays=108,          # DEVE ESSERE 108
    max_lidar_distance=15.0,
    num_people=15,
    people_radius=0.3,
    min_circ_obstacles=2,
    max_circ_obstacles=5,
    obst_min_radius=0.3,
    obst_max_radius=0.8,
    obst_clearance=0.5,
    min_rect_obstacles=1,
    max_rect_obstacles=3,
    rect_min_width=0.5,
    rect_max_width=2.0,
    rect_min_height=0.5,
    rect_max_height=2.0,
    goal_radius=0.5,
    goal_min_robot_dist=2.0,
    goal_reward=10.0
)

BATCH_SIZE = 512
SEQ_LEN = 50

# Vettorizziamo l'ambiente (4 ot insieme)
reset_vmap = jax.vmap(reset, in_axes=(0, None))
step_vmap  = jax.vmap(auto_reset_step, in_axes=(0, 0, 0, None, 0, 0, 0))

# --- 2. FUNZIONE PER RACCOGLIERE DATI (ROLLOUT) ---
@jax.jit
def collect_rollout(agent_state, env_state, obs, prev_rssm_state, prev_action, obst, rects, rng):
    
    def step_fn(carry, _):
        env_st, curr_obs, rssm_st, p_act, key = carry
        
        # Determiniamo il Batch Size dinamicamente dalle osservazioni
        batch_n = curr_obs.shape[0]
        
        # Prepara input
        goal = curr_obs[:, :2]
        lidar = curr_obs[:, 2:]
        
        # AGENTE: Sceglie azione
        action, next_rssm_st = agent_state.apply_fn(
            agent_state.params,
            rssm_st, lidar, goal, p_act,
            method=Dreamer.policy_step
        )
        
        # --- CORREZIONE CHIAVI RANDOM ---
        # Dobbiamo generare (Batch + 1) chiavi:
        # 1 chiave serve per il 'carry' del prossimo passo (next_key)
        # 'batch_n' chiavi servono per i robot paralleli (step_keys)
        splits = jax.random.split(key, batch_n + 1)
        next_key = splits[0]
        step_keys = splits[1:] # Shape (Batch, 2) -> Perfetto per vmap!
        
        # AMBIENTE: Esegue azione
        next_env_st, next_obs, reward, done, next_obst, next_rect = step_vmap(
            env_st, curr_obs, action, env_cfg, obst, rects, step_keys
        )
        
        # Salva dati
        transition = {
            'lidar': lidar,
            'goal': goal,
            'actions': action,
            'rewards': reward,
            'done': done
        }
        
        return (next_env_st, next_obs, next_rssm_st, action, next_key), transition

    # Scan loop
    init_carry = (env_state, obs, prev_rssm_state, prev_action, rng)
    final_carry, batch_history = jax.lax.scan(step_fn, init_carry, None, length=SEQ_LEN)
    
    # Riordina assi: (Time, Batch, ...) -> (Batch, Time, ...)
    # CORREZIONE: Usa jax.tree_util.tree_map
    batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch_history)
    
    # Estrai stato finale
    fin_env, fin_obs, fin_rssm, fin_act, fin_rng = final_carry
    return batch, fin_env, fin_obs, fin_rssm, fin_act, fin_rng

# --- 3. MAIN LOOP ---
if __name__ == "__main__":
    print("--- START TRAINING ---")
    key = jax.random.PRNGKey(0)
    
    # A. Init Ambiente
    key, *env_keys = jax.random.split(key, BATCH_SIZE + 1)
    env_keys = jnp.array(env_keys)
    rng_keys, env_states, env_obs, obst, rects = reset_vmap(env_keys, env_cfg)
    
    # B. Init Agente
    dummy_goal = jnp.zeros((BATCH_SIZE, 1, 2))
    dummy_lidar = jnp.zeros((BATCH_SIZE, 1, 108))
    dummy_action = jnp.zeros((BATCH_SIZE, 1, 2))
    
    agent = Dreamer()
    variables = agent.init(key, dummy_lidar, dummy_goal, dummy_action)
    
    tx = optax.chain(
        optax.clip_by_global_norm(100.0),  # Taglia i gradienti esplosivi
        optax.adam(learning_rate=1e-4)
    )
    agent_state = train_state.TrainState.create(
        apply_fn=agent.apply, params=variables, tx=tx
    )
    
    # C. Init Stati Memoria
    # Ricaviamo le dimensioni da una istanza dummy
    # Se hai impostato le variabili in setup(), possiamo accedere alle dimensioni, 
    # altrimenti usiamo i valori di default che conosciamo:
    init_h = jnp.zeros((BATCH_SIZE, 512))
    init_z = jnp.zeros((BATCH_SIZE, 32, 32)) 
    rssm_state = (init_h, init_z)
    prev_action = jnp.zeros((BATCH_SIZE, 2))
    
    # LOOP INFINITO
    for step in range(1, 10001):
        # 1. Raccogli Dati (Guida nell'ambiente)
        batch_data, env_states, env_obs, rssm_state, prev_action, key = collect_rollout(
            agent_state, env_states, env_obs, rssm_state, prev_action, obst, rects, key
        )
        
        # 2. Addestra World Model (Capisci la fisica)
        agent_state, logs_wm, features = train_world_model(agent_state, batch_data)
        
        # 3. Addestra Actor (Impara a guidare nel sogno)
        agent_state, logs_ac = train_actor_critic(agent_state, features)
        
        if step % 10 == 0:
            print(f"Step {step} | LossWM: {logs_wm['loss_total']:.3f} | "
                  f"LossAct: {logs_ac['loss_actor']:.3f} | "
                  f"Rew: {jnp.mean(batch_data['rewards']):.3f}")