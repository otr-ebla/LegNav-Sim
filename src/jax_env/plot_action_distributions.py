"""
plot_random_test_actions_gpu.py
Script ad altissime prestazioni su GPU per analizzare la distribuzione delle azioni.
Campiona scenari di test casuali (7-12) alla massima velocità usando la GPU.
"""

import os
import time
import argparse

# ==============================================================================
# 🚀 PRE-INIZIALIZZAZIONE GPU (Stessa configurazione di jax_ppo.py)
# ==============================================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
import matplotlib.pyplot as plt

# Forza JAX a usare esplicitamente la prima GPU CUDA
try:
    jax.config.update("jax_default_device", jax.devices("cuda")[0])
    print(f"✅ GPU CUDA rilevata e forzata: {jax.devices('cuda')[0]}")
except RuntimeError:
    print("❌ ERRORE CRITICO: JAX non vede la GPU CUDA e sta ripiegando su CPU!")

from jax_network import EndToEndActorCritic
from jax_train import init_env_state, collect_rollouts, NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE
from jax_scenarios import TEST_SCENARIO_NAMES

def load_checkpoint(filepath, model, dummy_obs):
    """Carica i parametri della rete."""
    rng = jax.random.PRNGKey(0)
    init_vars = model.init(rng, dummy_obs)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint non trovato: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()
    
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle.get("params", bundle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/ppo_tanh_fix_final.msgpack", help="Percorso del checkpoint")
    parser.add_argument("--episodes", type=int, default=5000, help="Numero target di episodi di test da raccogliere")
    parser.add_argument("--max_dist", type=float, default=20.0, help="Distanza massima del goal")
    parser.add_argument("--v_max_plot", type=float, default=1.5, help="v_max fisso per simulare lo squashing nel plot")
    args = parser.parse_args()
    
    # 1. Setup Rete e Checkpoint
    network = EndToEndActorCritic(action_dim=2)
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    
    print(f"📂 Caricamento pesi da: {args.ckpt}")
    params = load_checkpoint(args.ckpt, network, dummy_obs)
    
    rng = jax.random.PRNGKey(42)
    
    all_raw_v = []
    all_raw_w = []
    
    collected_dones = 0
    test_scenarios = [7, 8, 9, 10, 11, 12]
    
    print(f"\n🌍 Inizio campionamento su GPU fino a raggiungere {args.episodes} episodi...")
    t_start = time.time()

    # 2. Ciclo di raccolta veloce
    # 2. Ciclo di raccolta equo per tutti gli scenari
    episodes_per_scenario = args.episodes // len(test_scenarios)
    
    for scen_idx in test_scenarios:
        scen_name = TEST_SCENARIO_NAMES.get(scen_idx, f"Sconosciuto ({scen_idx})")
        scen_collected = 0
        
        rng, env_rng = jax.random.split(rng)
        
        # Inizializza 1024 ambienti paralleli per QUESTO specifico scenario
        env_obs, env_state, vmap_step = init_env_state(
            env_rng, max_goal_dist=args.max_dist, ghost_prob=0.0, scenario_idx=scen_idx
        )
        
        # Continua a raccogliere finché non raggiungiamo la quota per questo scenario
        while scen_collected < episodes_per_scenario:
            rng, run_rng = jax.random.split(rng)
            t_chunk_start = time.time()
            
            # Esecuzione JIT su GPU
            rollout_history, env_state, env_obs, _ = collect_rollouts(
                run_rng,
                params,
                network.apply,
                vmap_step,
                env_state,
                env_obs,
                max_goal_dist=args.max_dist,
                scenario_idx=jnp.int32(scen_idx),
                ghost_prob=0.0,
                max_scenario=jnp.int32(12) 
            )
            
            raw_actions = np.array(rollout_history["actions"]) 
            dones = np.array(rollout_history["dones"])
            
            all_raw_v.append(raw_actions[..., 0].flatten())
            all_raw_w.append(raw_actions[..., 1].flatten())
            
            chunk_dones = int(dones.sum())
            scen_collected += chunk_dones
            collected_dones += chunk_dones
            
            fps = (NUM_ENVS * ROLLOUT_STEPS) / (time.time() - t_chunk_start)
            print(f"  ⚡ {scen_name} ({scen_idx}) | Raccolti {chunk_dones} | Tot Scen: {scen_collected}/{episodes_per_scenario} | Tot Gen: {collected_dones}/{args.episodes}")

    tot_time = time.time() - t_start
    print(f"\n✅ Target raggiunto in {tot_time:.1f} secondi!")

    # 3. Aggregazione e Squashing
    raw_v_total = np.concatenate(all_raw_v)
    raw_w_total = np.concatenate(all_raw_w)
    
    #squashed_v_total = (np.tanh(raw_v_total) * 0.5 + 0.5) * args.v_max_plot
    #squashed_w_total = np.tanh(raw_w_total)
    squashed_v_total = np.clip(raw_v_total, 0.0, 1.0) * args.v_max_plot
    squashed_w_total = np.clip(raw_w_total, -1.0, 1.0)

    print(f"📊 Generazione dei grafici con {len(raw_v_total):,} sample totali...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Test Scenarios Action Distributions - {args.ckpt.split('/')[-1]}\n({collected_dones} Episodi Casuali)", fontsize=16)

    bins = 150
    alpha = 0.75

    # --- Plot Velocità Lineare (v) ---
    axes[0, 0].hist(raw_v_total, bins=bins, density=True, color='gray', alpha=alpha)
    axes[0, 0].set_title("Raw Action V (Uscita Gaussiana)")
    axes[0, 0].set_xlabel("Valore Raw")
    axes[0, 0].grid(alpha=0.3)
    v_p1, v_p99 = np.percentile(raw_v_total, [0.1, 99.9])
    axes[0, 0].set_xlim(v_p1 - 0.5, v_p99 + 0.5)

    axes[0, 1].hist(squashed_v_total, bins=bins, density=True, color='dodgerblue', alpha=alpha)
    axes[0, 1].set_title(f"Squashed Action V (tanh mappata in [0, {args.v_max_plot}])")
    axes[0, 1].set_xlabel("Metri al secondo (m/s)")
    axes[0, 1].set_xlim(0, args.v_max_plot)
    axes[0, 1].grid(alpha=0.3)

    # --- Plot Velocità Angolare (w) ---
    axes[1, 0].hist(raw_w_total, bins=bins, density=True, color='gray', alpha=alpha)
    axes[1, 0].set_title("Raw Action W (Uscita Gaussiana)")
    axes[1, 0].set_xlabel("Valore Raw")
    axes[1, 0].grid(alpha=0.3)
    w_p1, w_p99 = np.percentile(raw_w_total, [0.1, 99.9])
    axes[1, 0].set_xlim(w_p1 - 0.5, w_p99 + 0.5)

    axes[1, 1].hist(squashed_w_total, bins=bins, density=True, color='seagreen', alpha=alpha)
    axes[1, 1].set_title("Squashed Action W (tanh mappata in [-1, 1])")
    axes[1, 1].set_xlabel("Radianti al secondo (rad/s)")
    axes[1, 1].set_xlim(-1.1, 1.1)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = "tanh_test_action_distributions.png"
    plt.savefig(out_file, dpi=200)
    print(f"🎉 Finito! Grafico salvato come: {out_file}")

if __name__ == "__main__":
    main()