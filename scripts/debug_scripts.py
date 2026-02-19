import numpy as np
import jax
import jax.numpy as jnp
import math
from src.envs.jhsfm_nav_env import SimpleNavEnv
from src.jhsfm_utils.JHSFM.jhsfm.hsfm import vectorized_single_update, get_linear_velocity

def debug_physics_step():
    print("\n🔬 --- DIAGNOSTICA FISICA HSFM ---")
    
    # Inizializza ambiente
    env = SimpleNavEnv(scenario_type="intersection", num_people=1, force_static=False)
    env.reset()
    
    # Estrai stato JAX
    state = env.humans_state_jax[0] # Umano 0
    goal = env.humans_goal_jax[0]
    params = env.hsfm_params[0]
    obs = env.static_obstacles_jax
    
    # Parametri vitali
    print(f"\n1. STATO AGENTE")
    print(f"   Pos: ({state[0]:.3f}, {state[1]:.3f})")
    print(f"   Vel (Body): ({state[2]:.3f}, {state[3]:.3f})")
    print(f"   Theta: {state[4]:.3f} rad")
    print(f"   Goal: ({goal[0]:.3f}, {goal[1]:.3f})")
    print(f"   V_Max (Param[2]): {params[2]}")
    print(f"   Massa (Param[1]): {params[1]}")

    # Calcolo manuale vettori
    diff = goal - state[:2]
    dist = np.linalg.norm(diff)
    desired_dir = diff / (dist + 1e-6)
    
    print(f"\n2. VETTORI GEOMETRICI")
    print(f"   Distanza Goal: {dist:.3f}")
    print(f"   Direzione Goal: ({desired_dir[0]:.3f}, {desired_dir[1]:.3f})")
    
    # Esegui Step HSFM e intercetta valori (simulato)
    # Nota: non possiamo 'entrare' in JIT facilmente, quindi ricalcoliamo le forze qui
    # usando la logica numpy per vedere cosa succede
    
    # --- DESIRED FORCE ---
    # Formula: mass * (v_max * n_0 - v) / tau
    lin_vel = _get_lin_vel_numpy(state[4], state[2:4])
    v_des_vec = desired_dir * params[2]
    f_des = (params[1] * (v_des_vec - lin_vel)) / params[3]
    
    print(f"\n3. ANALISI FORZE (Ricostruzione)")
    print(f"   Linear Vel (World): ({lin_vel[0]:.3f}, {lin_vel[1]:.3f})")
    print(f"   Desired Force: ({f_des[0]:.3f}, {f_des[1]:.3f}) <--- DEVE ESSERE NON-ZERO")
    
    # --- OBSTACLE FORCE ---
    # Cerchiamo l'ostacolo più vicino
    min_dist_obs = 999.0
    closest_pt = None
    
    if len(obs) > 0:
        for o in obs:
            # Circle obstacle: cx, cy, r
            center = o[:2]
            r = o[2]
            d_center = np.linalg.norm(state[:2] - center)
            d_surf = d_center - r - params[0] # Distanza dalla 'pelle' dell'agente
            if d_surf < min_dist_obs:
                min_dist_obs = d_surf
                closest_pt = center
        
        print(f"   Dist. Ostacolo Più Vicino: {min_dist_obs:.3f} m")
        if min_dist_obs < 0:
            print("   ⚠️  ATTENZIONE: AGENTE DENTRO OSTACOLO!")
    
    # --- STEP REALE ---
    print("\n4. ESECUZIONE STEP REALE JAX")
    new_state = env.hsfm_step_fn(
        env.humans_state_jax, 
        env.humans_goal_jax, 
        env.hsfm_params, 
        # Tile obstacles
        jnp.tile(env.static_obstacles_jax[None, :, :], (len(env.humans_state_jax), 1, 1)), 
        0.1
    )
    ns = new_state[0]
    
    dx = ns[0] - state[0]
    dy = ns[1] - state[1]
    moved_dist = math.hypot(dx, dy)
    
    print(f"   Nuova Pos: ({ns[0]:.3f}, {ns[1]:.3f})")
    print(f"   Spostamento: {moved_dist:.4f} m")
    
    if moved_dist < 0.001:
        print("\n❌ ESITO: L'AGENTE È BLOCCATO.")
        if params[2] == 0: print("   -> Causa Probabile: V_Max è 0.")
        elif min_dist_obs < 0.1: print("   -> Causa Probabile: Collisione Muro/Ostacolo.")
        else: print("   -> Causa Probabile: Errore nel calcolo forze (es. NaN).")
    else:
        print("\n✅ ESITO: L'AGENTE SI MUOVE!")

def _get_lin_vel_numpy(theta, b_vel):
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return rot @ b_vel

if __name__ == "__main__":
    debug_physics_step()