import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import matplotlib.pyplot as plt # Per il rendering

# --- IMPORTIAMO I TUOI FILE ---
from envs.jax_env import StaticConfig, reset, auto_reset_step, render
from dreamerv3_lidar import Dreamer, train_world_model, train_actor_critic

# --- 1. CONFIGURAZIONE ---
env_cfg = StaticConfig(
    dt=0.1,
    room_width=10.0,
    room_height=10.0,
    max_lin_vel=1.0,
    max_ang_vel=2.0,
    robot_radius=0.3,
    num_rays=216,
    max_lidar_distance=10.0,
    num_people=3,
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

BATCH_SIZE = 4 # Ne visualizzeremo solo 1, ma ne alleniamo 4
SEQ_LEN = 50

reset_vmap = jax.vmap(reset, in_axes=(0, None))
step_vmap  = jax.vmap(auto_reset_step, in_axes=(0, 0, 0, None, 0, 0, 0))

# --- 2. ROLLOUT (Invariato) ---
@jax.jit
def collect_rollout(agent_state, env_state, obs, prev_rssm_state, prev_action, obst, rects, rng):
    def step_fn(carry, _):
        env_st, curr_obs, rssm_st, p_act, key = carry
        batch_n = curr_obs.shape[0]
        
        goal = curr_obs[:, :2]
        lidar = curr_obs[:, 2:]
        
        action, next_rssm_st = agent_state.apply_fn(
            agent_state.params,
            rssm_st, lidar, goal, p_act,
            method=Dreamer.policy_step
        )
        
        splits = jax.random.split(key, batch_n + 1)
        next_key = splits[0]
        step_keys = splits[1:]
        
        next_env_st, next_obs, reward, done, next_obst, next_rect = step_vmap(
            env_st, curr_obs, action, env_cfg, obst, rects, step_keys
        )
        
        transition = {
            'lidar': lidar,
            'goal': goal,
            'actions': action,
            'rewards': reward,
            'done': done
        }
        return (next_env_st, next_obs, next_rssm_st, action, next_key), transition

    init_carry = (env_state, obs, prev_rssm_state, prev_action, rng)
    final_carry, batch_history = jax.lax.scan(step_fn, init_carry, None, length=SEQ_LEN)
    batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch_history)
    fin_env, fin_obs, fin_rssm, fin_act, fin_rng = final_carry
    return batch, fin_env, fin_obs, fin_rssm, fin_act, fin_rng

# --- 3. VISUALIZER LOOP ---
if __name__ == "__main__":
    print("--- START VISUAL TRAINING ---")
    
    # Setup Grafico interattivo
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))

    key = jax.random.PRNGKey(42)
    key, *env_keys = jax.random.split(key, BATCH_SIZE + 1)
    env_keys = jnp.array(env_keys)
    rng_keys, env_states, env_obs, obst, rects = reset_vmap(env_keys, env_cfg)
    
    dummy_goal = jnp.zeros((BATCH_SIZE, 1, 2))
    dummy_lidar = jnp.zeros((BATCH_SIZE, 1, 216))
    dummy_action = jnp.zeros((BATCH_SIZE, 1, 2))
    
    agent = Dreamer()
    variables = agent.init(key, dummy_lidar, dummy_goal, dummy_action)
    
    # Ottimizzatore Safe (con clipping)
    tx = optax.chain(
        optax.clip_by_global_norm(100.0),
        optax.adam(learning_rate=1e-4)
    )
    agent_state = train_state.TrainState.create(
        apply_fn=agent.apply, params=variables, tx=tx
    )
    
    init_h = jnp.zeros((BATCH_SIZE, 512))
    init_z = jnp.zeros((BATCH_SIZE, 32, 32)) 
    rssm_state = (init_h, init_z)
    prev_action = jnp.zeros((BATCH_SIZE, 2))
    
    for step in range(1, 100001):
        # 1. Rollout
        batch_data, env_states, env_obs, rssm_state, prev_action, key = collect_rollout(
            agent_state, env_states, env_obs, rssm_state, prev_action, obst, rects, key
        )
        
        # 2. Train World Model
        agent_state, logs_wm, features = train_world_model(agent_state, batch_data)
        
        # 3. Train Actor
        agent_state, logs_ac = train_actor_critic(agent_state, features)
        
        # 4. LOG & RENDER (Ogni 10 step)
        if step % 10 == 0:
            mean_rew = jnp.mean(batch_data['rewards'])
            print(f"Step {step} | LossWM: {logs_wm['loss_total']:.3f} | "
                  f"Act: {logs_ac['loss_actor']:.3f} | Rew: {mean_rew:.3f}")

            # --- RENDERING ---
            # Prendiamo il primo environment del batch (indice 0)
            # Dobbiamo usare tree_map per estrarre l'indice 0 da tutte le strutture (EnvState, Obstacles...)
            single_env_state = jax.tree_util.tree_map(lambda x: x[0], env_states)
            single_obst = jax.tree_util.tree_map(lambda x: x[0], obst)
            single_rect = jax.tree_util.tree_map(lambda x: x[0], rects)
            
            # Chiamiamo la tua funzione render
            # Nota: render usa matplotlib che è lento, quindi rallenterà il training.
            # Ma è fondamentale per vedere se impara.
            render(single_env_state, env_cfg, ax=ax, obstacles=single_obst, rect_obst=single_rect)
            plt.pause(0.01)