"""
pretrain_navrep.py — Collect expert transitions and pretrain NavRep's V and M.

Pipeline
--------
1. Collect millions of (lidar, next-lidar) transitions by rolling out the
   HumanPilot expert (JHSFM Social Force Model toward goal) in the vectorised
   JAX environment (NUM_ENVS in parallel).  The robot moves like a pedestrian:
   SFM goal force + obstacle repulsion from LiDAR → unicycle [v, w] command.
2. Train Module V (VAE: 1D-CNN encoder + ConvTranspose decoder) on shuffled
   single-frame LiDAR scans with reconstruction + β·KL loss.
3. Encode full per-env trajectories into z-sequences and train Module M
   (causal Transformer) with a next-z MSE predictive loss.
4. Save the combined {encoder, M} weights to checkpoints_navrep/navrep_vm.msgpack
   so train_navrep.py can load them and PPO-train the controller only.

Usage
-----
    cd src/jax_env
    python comparison_policies/pretrain_navrep.py \
        [--frames 2_000_000] [--v-epochs 5] [--m-epochs 5]
"""

import os
import csv

os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import time
import argparse
import sys

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_default_device", jax.devices("cuda")[0])
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization

from jax_train import NUM_ENVS
from jax_wrappers import make_stacked_env
from jax_env import reset_env, step_env, NUM_RAYS as ENV_NUM_RAYS
from comparison_policies.jhsfm_planner import HumanPilot
from comparison_policies.navrep_network import (
    VAE, MWrapper, LidarEncoder, Z_DIM, NUM_RAYS, STACK_DIM,
)

assert ENV_NUM_RAYS == NUM_RAYS, "NUM_RAYS mismatch between env and navrep_network"

_STATE_END = 14   # obs[_STATE_END:] = flattened lidar_stack (3×216)


# ── Defaults ──────────────────────────────────────────────────────────────────
CHUNK_LEN          = 64           # env steps per jit-compiled collection chunk
DEFAULT_FRAMES     = 2_000_000     # total single-frame samples to collect
DEFAULT_V_EPOCHS   = 5
DEFAULT_M_EPOCHS   = 5
V_BATCH            = 256
V_LR               = 1e-3
KL_BETA            = 1e-3

M_SEQ_LEN          = 16
M_BATCH            = 128
M_LR               = 1e-3

CKPT_DIR  = os.path.join(_JAX_ENV_DIR, "checkpoints_navrep")
VM_CKPT   = os.path.join(CKPT_DIR, "navrep_vm.msgpack")


# ══════════════════════════════════════════════════════════════════════════════
# 1) Expert rollout collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_expert_lidar(total_frames: int, start_rng: jax.Array) -> jnp.ndarray:
    """
    Esegue in un'unica chiamata JIT su GPU la generazione completa del dataset.
    Niente ritorni a Python, nessun loop For su CPU. Pura velocità.
    """
    N_STEPS = total_frames // NUM_ENVS
    
    # ── 1. Preparazione dell'Esperto e dell'Ambiente ──────────────────────────
    pilot = HumanPilot()
    vmap_act = jax.vmap(pilot.act)

    bound_reset = lambda key, max_goal_dist, scenario_idx, ghost_prob: \
        reset_env(key, max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob)
    
    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
    vmap_reset = jax.vmap(rs, in_axes=(0, None, 0, None))
    vmap_step  = jax.vmap(ss)

    # ── 2. Il Grafo Monolitico (100% GPU) ─────────────────────────────────────
    @jax.jit
    def collect_all_data_gpu(rng):
        rng_reset, rng_scan = jax.random.split(rng)
        env_rngs = jax.random.split(rng_reset, NUM_ENVS)
        
        # FORZATURA: Solo scenari aperti (0-6) per evitare che JHSFM si blocchi nei muri!
        scenarios = jax.random.randint(rng_reset, (NUM_ENVS,), 0, 7)
        obs, stacked_state = vmap_reset(env_rngs, 9.0, scenarios, 0.0)

        def step_fn(carry, step_rng):
            current_obs, current_state = carry
            
            # Inferenza dell'esperto (tutto su GPU)
            actions = vmap_act(current_state.env_state)
            
            # Step dell'ambiente
            step_rngs = jax.random.split(step_rng, NUM_ENVS)
            next_obs, next_state, reward, done, info = vmap_step(step_rngs, current_state, actions)
            
            # Estrazione precisa del LiDAR (salviamo solo i 216 raggi per non esplodere la RAM)
            # Nel layout a 662D, l'ultimo frame LiDAR occupa esattamente gli ultimi 216 indici
            lidar_frame = next_obs[:, -216:]
            
            return (next_obs, next_state), lidar_frame

        step_rngs = jax.random.split(rng_scan, N_STEPS)
        
        # Un singolo loop JAX che gira N_STEPS volte senza MAI parlare con Python
        _, lidar_history = jax.lax.scan(step_fn, (obs, stacked_state), step_rngs)
        
        return lidar_history  # Shape: (N_STEPS, NUM_ENVS, 216)

    # ── 3. Esecuzione ─────────────────────────────────────────────────────────
    print(f"\n[collect] target {total_frames:,} frames ({N_STEPS} env-steps)")
    print(f"          esperto: HumanPilot (JHSFM) - Avvio generazione 100% su GPU...")
    t0 = time.time()
    
    # Esegue il calcolo e lo mantiene direttamente sulla memoria VRAM
    lidar_frames_gpu = collect_all_data_gpu(start_rng)
    
    # Sincronizzazione per misurare il tempo effettivo
    jax.block_until_ready(lidar_frames_gpu)
    
    print(f"[collect] Completato in {time.time() - t0:.1f} secondi. Boom.")
    
    # 1. Per il VAE (Modulo V) serve un array 2D piatto: (N_totali, 216)
    lidar_frames = lidar_frames_gpu.reshape(-1, 216)
    
    # 2. Per il Transformer (Modulo M) serve l'array 3D sequenziale: (Tempo, Ambienti, 216)
    lidar_seq = lidar_frames_gpu
    
    return lidar_frames, lidar_seq


# ══════════════════════════════════════════════════════════════════════════════
# 2) VAE training (Module V)
# ══════════════════════════════════════════════════════════════════════════════

def train_vae(lidar_frames: np.ndarray, epochs: int, rng: jax.Array):
    """
    lidar_frames : (N_total, 216)  shuffled single frames
    Returns: encoder params (dict) from the trained VAE.
    """
    vae = VAE(z_dim=Z_DIM)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, NUM_RAYS))
    params = vae.init(init_rng, dummy, sample_rng)["params"]

    optimizer = optax.adam(V_LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_epoch(params, opt_state, epoch_data, epoch_rng):
        def batch_step(carry, step_data):
            p, os_ = carry
            batch, sample_rng = step_data
            def loss_fn(p_):
                recon, z_mean, z_logvar = vae.apply({"params": p_}, batch, sample_rng)
                recon_loss = jnp.mean(jnp.sum((recon - batch) ** 2, axis=-1))
                kl = -0.5 * jnp.mean(jnp.sum(
                    1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1
                ))
                return recon_loss + KL_BETA * kl, (recon_loss, kl)
            (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, new_os = optimizer.update(grads, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), (loss, recon, kl)
        
        batch_rngs = jax.random.split(epoch_rng, epoch_data.shape[0])
        (new_params, new_opt_state), (losses, recons, kls) = jax.lax.scan(
            batch_step, (params, opt_state), (epoch_data, batch_rngs)
        )
        return new_params, new_opt_state, jnp.mean(losses), jnp.mean(recons), jnp.mean(kls)

    N = lidar_frames.shape[0]
    n_batches = N // V_BATCH
    valid_N = n_batches * V_BATCH
    print(f"\n[V] training VAE: {N:,} frames, batch={V_BATCH}, "
          f"{n_batches} batches/epoch, {epochs} epochs")

    # Trasferimento massivo del dataset pulito in VRAM (zero overhead interno)
    device_data = jnp.asarray(lidar_frames[:valid_N])

    for epoch in range(epochs):
        rng, perm_rng, epoch_rng = jax.random.split(rng, 3)
        perm = jax.random.permutation(perm_rng, valid_N)
        epoch_data = device_data[perm].reshape(n_batches, V_BATCH, -1)
        
        t0 = time.time()
        params, opt_state, loss_val, recon_val, kl_val = train_epoch(
            params, opt_state, epoch_data, epoch_rng
        )
        # Sincronizzazione unica a fine epoca!
        print(f"[V] epoch {epoch+1}/{epochs}  "
              f"loss={float(loss_val):.4f}  "
              f"recon={float(recon_val):.4f}  "
              f"kl={float(kl_val):.4f}  "
              f"({time.time()-t0:.1f}s)")

    return params["encoder"]


# ══════════════════════════════════════════════════════════════════════════════
# 3) Transformer M training on z-sequences
# ══════════════════════════════════════════════════════════════════════════════

def _encode_all(encoder_params, lidar_seq: np.ndarray) -> np.ndarray:
    """
    lidar_seq : (T, N, 216)  per-env time-ordered
    → z_seq   : (T, N, Z)    deterministic (z_mean)
    """
    encoder = LidarEncoder(z_dim=Z_DIM)
    T, N, R = lidar_seq.shape

    @jax.jit
    def enc(batch):
        z_mean, _ = encoder.apply({"params": encoder_params}, batch)
        return z_mean

    BATCH = 4096
    flat = lidar_seq.reshape(T * N, R)
    out = np.empty((T * N, Z_DIM), dtype=np.float32)
    for i in range(0, flat.shape[0], BATCH):
        out[i:i+BATCH] = np.asarray(enc(jnp.asarray(flat[i:i+BATCH])))
    return out.reshape(T, N, Z_DIM)


def train_transformer(encoder_params, lidar_seq: np.ndarray,
                      epochs: int, rng: jax.Array):
    """
    Train TransformerM to predict z_{t+1} from z_{t}, causally.
    Returns M params (dict) under the key 'M'.
    """
    print("\n[M] encoding all frames to z-sequences with frozen V ...")
    z_all = _encode_all(encoder_params, lidar_seq)           # (T, N, Z)
    T, N, Z = z_all.shape
    assert T > M_SEQ_LEN, f"need T > {M_SEQ_LEN} for M training"

    m_model = MWrapper(z_dim=Z)
    rng, init_rng = jax.random.split(rng)
    dummy = jnp.zeros((1, M_SEQ_LEN, Z))
    params = m_model.init(init_rng, dummy)["params"]

    optimizer = optax.adam(M_LR)
    opt_state = optimizer.init(params)

    # Manteniamo l'intera sequenza latente nella VRAM per indicizzazione ad alta velocità
    device_z = jnp.asarray(z_all)

    @jax.jit
    def train_epoch(params, opt_state, epoch_env_ids, epoch_start_ts):
        def batch_step(carry, step_data):
            p, os_ = carry
            e_ids, s_ts = step_data
            
            # Costruzione della batch generata DIRETTAMENTE su GPU
            def get_seq(e, s):
                return jax.lax.dynamic_slice(device_z, (s, e, 0), (M_SEQ_LEN, 1, Z))[:, 0, :]
            
            z_batch = jax.vmap(get_seq)(e_ids, s_ts)
            
            def loss_fn(p_):
                next_z, _ = m_model.apply({"params": p_}, z_batch)
                pred   = next_z[:, :-1, :]
                target = z_batch[:, 1:, :]
                return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))
                
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_os = optimizer.update(grads, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), loss

        (new_params, new_opt_state), losses = jax.lax.scan(
            batch_step, (params, opt_state), (epoch_env_ids, epoch_start_ts)
        )
        return new_params, new_opt_state, jnp.mean(losses)

    # Sample subsequences: pick (env_idx, start_t) pairs.
    starts_per_env = T - M_SEQ_LEN
    total_subseqs  = starts_per_env * N
    n_batches = max(1, total_subseqs // M_BATCH)

    print(f"[M] training Transformer: T={T}, N={N}, Z={Z}  "
          f"subseq_len={M_SEQ_LEN}  batch={M_BATCH}  "
          f"{n_batches} batches/epoch, {epochs} epochs")

    for epoch in range(epochs):
        rng, env_rng, start_rng = jax.random.split(rng, 3)
        # JAX genera in modo nativo su GPU senza blocchi numpy
        env_ids  = jax.random.randint(env_rng, (n_batches, M_BATCH), 0, N)
        start_ts = jax.random.randint(start_rng, (n_batches, M_BATCH), 0, starts_per_env)

        t0 = time.time()
        params, opt_state, loss_val = train_epoch(
            params, opt_state, env_ids, start_ts
        )
        # Sincronizzazione unica a fine epoca!
        print(f"[M] epoch {epoch+1}/{epochs}  "
              f"mse={float(loss_val):.5f}  "
              f"({time.time()-t0:.1f}s)")

    return params["M"]


# ══════════════════════════════════════════════════════════════════════════════
# 4) Save combined V + M checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def save_vm(encoder_params, m_params, path=VM_CKPT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "encoder": jax.device_get(encoder_params),
        "M":       jax.device_get(m_params),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"\n[save] V+M → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",   type=int, default=DEFAULT_FRAMES,
                    help="Total single-frame LiDAR samples to collect.")
    ap.add_argument("--v-epochs", type=int, default=DEFAULT_V_EPOCHS)
    ap.add_argument("--m-epochs", type=int, default=DEFAULT_M_EPOCHS)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    print(f"NavRep Pretraining  (V + M)")
    print(f"  expert   : HumanPilot (JHSFM Social Force Model toward goal)")
    print(f"  frames   : {args.frames:,}")
    print(f"  V epochs : {args.v_epochs}")
    print(f"  M epochs : {args.m_epochs}")
    print(f"  seed     : {args.seed}\n")

    rng = jax.random.PRNGKey(args.seed)
    rng, coll_rng, v_rng, m_rng = jax.random.split(rng, 4)

    flat_frames, lidar_seq = collect_expert_lidar(args.frames, coll_rng)

    enc_params = train_vae(flat_frames, args.v_epochs, v_rng)
    m_params   = train_transformer(enc_params, lidar_seq, args.m_epochs, m_rng)

    save_vm(enc_params, m_params)


if __name__ == "__main__":
    main()
