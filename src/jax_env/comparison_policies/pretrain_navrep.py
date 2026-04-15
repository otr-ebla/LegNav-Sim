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
import sys
import time
import argparse

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization

from jax_train import init_env_state, NUM_ENVS
from jax_env import NUM_RAYS as ENV_NUM_RAYS
from comparison_policies.jhsfm_planner import HumanPilot
from comparison_policies.navrep_network import (
    VAE, MWrapper, LidarEncoder, Z_DIM, NUM_RAYS, STACK_DIM,
)

assert ENV_NUM_RAYS == NUM_RAYS, "NUM_RAYS mismatch between env and navrep_network"

_STATE_END = 14   # obs[_STATE_END:] = flattened lidar_stack (3×216)


# ── Defaults ──────────────────────────────────────────────────────────────────
CHUNK_LEN          = 256           # env steps per jit-compiled collection chunk
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

def _extract_latest_lidar(obs_chunk: jnp.ndarray) -> jnp.ndarray:
    """obs_chunk: (T, N, 662) → most-recent lidar frame (T, N, 216)."""
    return obs_chunk[..., _STATE_END:].reshape(
        *obs_chunk.shape[:-1], STACK_DIM, NUM_RAYS
    )[..., -1, :]


def _build_chunk_runner(vmap_step, pilot_act, max_goal_dist, ghost_prob, max_scen):
    """JIT-compile one chunk of CHUNK_LEN env steps with the HumanPilot expert."""
    vmap_act = jax.vmap(pilot_act)

    @jax.jit
    def run_chunk(state, obs, rng):
        def _step(carry, _):
            s, o, r = carry
            r, step_rng = jax.random.split(r)
            actions = vmap_act(o)
            step_keys = jax.random.split(step_rng, NUM_ENVS)
            next_obs, next_state, _, _, _ = vmap_step(
                step_keys, s, actions,
                max_goal_dist, jnp.int32(-1), ghost_prob, jnp.int32(max_scen),
            )
            return (next_state, next_obs, r), o
        (fs, fo, _), obs_seq = jax.lax.scan(
            _step, (state, obs, rng), None, length=CHUNK_LEN
        )
        return obs_seq, fs, fo
    return run_chunk


def collect_expert_lidar(total_frames: int, rng: jax.Array) -> np.ndarray:
    """
    Roll out HumanPilot (JHSFM) and collect most-recent-frame LiDAR
    as an array of shape (T_total, NUM_ENVS, 216).

    The robot moves toward its goal using the Social Force Model —
    the same dynamics used for pedestrians in jax_humans.py —
    producing naturalistic LiDAR trajectories for V/M pretraining.
    """
    print(f"[collect] target {total_frames:,} frames "
          f"({total_frames // NUM_ENVS} env-steps across {NUM_ENVS} envs)\n"
          f"         expert: HumanPilot (JHSFM — SFM toward goal)")

    n_chunks = max(1, (total_frames + NUM_ENVS * CHUNK_LEN - 1)
                   // (NUM_ENVS * CHUNK_LEN))

    rng, env_rng = jax.random.split(rng)
    env_obs, env_state, vmap_step = init_env_state(
        env_rng, max_goal_dist=9.0, ghost_prob=0.0
    )

    pilot = HumanPilot(lidar_n_frames=1)
    run_chunk = _build_chunk_runner(vmap_step, pilot.act,
                                    max_goal_dist=9.0,
                                    ghost_prob=0.0,
                                    max_scen=12)

    buffer = []
    t0 = time.time()
    for ci in range(n_chunks):
        rng, chunk_rng = jax.random.split(rng)
        obs_seq, env_state, env_obs = run_chunk(env_state, env_obs, chunk_rng)
        lidar = np.asarray(jax.device_get(_extract_latest_lidar(obs_seq)))
        buffer.append(lidar)
        elapsed = time.time() - t0
        done = (ci + 1) * CHUNK_LEN * NUM_ENVS
        print(f"[collect] chunk {ci+1}/{n_chunks}  frames={done:,}  "
              f"fps={int(done/elapsed):,}")

    data = np.concatenate(buffer, axis=0)    # (T_total, N, 216)
    print(f"[collect] done → shape {data.shape}, "
          f"dtype {data.dtype}, size {data.nbytes/1e6:.1f} MB")
    return data


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
    def train_step(params, opt_state, batch, rng):
        def loss_fn(p):
            recon, z_mean, z_logvar = vae.apply({"params": p}, batch, rng)
            recon_loss = jnp.mean(jnp.sum((recon - batch) ** 2, axis=-1))
            kl = -0.5 * jnp.mean(jnp.sum(
                1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1
            ))
            return recon_loss + KL_BETA * kl, (recon_loss, kl)
        (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, recon, kl

    N = lidar_frames.shape[0]
    n_batches = N // V_BATCH
    print(f"\n[V] training VAE: {N:,} frames, batch={V_BATCH}, "
          f"{n_batches} batches/epoch, {epochs} epochs")

    for epoch in range(epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = np.asarray(jax.random.permutation(perm_rng, N))
        shuffled = lidar_frames[perm]
        t0 = time.time()
        sum_loss = sum_recon = sum_kl = 0.0
        for bi in range(n_batches):
            batch = jnp.asarray(shuffled[bi*V_BATCH:(bi+1)*V_BATCH])
            rng, sample_rng = jax.random.split(rng)
            params, opt_state, loss, recon, kl = train_step(
                params, opt_state, batch, sample_rng
            )
            sum_loss  += float(loss)
            sum_recon += float(recon)
            sum_kl    += float(kl)
        print(f"[V] epoch {epoch+1}/{epochs}  "
              f"loss={sum_loss/n_batches:.4f}  "
              f"recon={sum_recon/n_batches:.4f}  "
              f"kl={sum_kl/n_batches:.4f}  "
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

    @jax.jit
    def train_step(params, opt_state, z_batch):
        def loss_fn(p):
            next_z, _ = m_model.apply({"params": p}, z_batch)
            pred   = next_z[:, :-1, :]
            target = z_batch[:, 1:, :]
            return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    # Sample subsequences: pick (env_idx, start_t) pairs.
    starts_per_env = T - M_SEQ_LEN
    total_subseqs  = starts_per_env * N
    n_batches = max(1, total_subseqs // M_BATCH)

    print(f"[M] training Transformer: T={T}, N={N}, Z={Z}  "
          f"subseq_len={M_SEQ_LEN}  batch={M_BATCH}  "
          f"{n_batches} batches/epoch, {epochs} epochs")

    z_host = z_all  # keep on host, pull slices as needed

    for epoch in range(epochs):
        rng, env_rng, start_rng = jax.random.split(rng, 3)
        env_ids  = np.asarray(jax.random.randint(env_rng, (n_batches * M_BATCH,), 0, N))
        start_ts = np.asarray(jax.random.randint(start_rng, (n_batches * M_BATCH,), 0, starts_per_env))

        t0 = time.time()
        sum_loss = 0.0
        for bi in range(n_batches):
            idx_e = env_ids[bi*M_BATCH:(bi+1)*M_BATCH]
            idx_s = start_ts[bi*M_BATCH:(bi+1)*M_BATCH]
            # Gather (M_BATCH, M_SEQ_LEN, Z) subsequences
            batch = np.stack(
                [z_host[s:s+M_SEQ_LEN, e] for s, e in zip(idx_s, idx_e)],
                axis=0,
            )
            params, opt_state, loss = train_step(params, opt_state, jnp.asarray(batch))
            sum_loss += float(loss)
        print(f"[M] epoch {epoch+1}/{epochs}  "
              f"mse={sum_loss/n_batches:.5f}  "
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

    lidar_seq = collect_expert_lidar(args.frames, coll_rng)   # (T, N, 216)
    T, N, _   = lidar_seq.shape
    flat_frames = lidar_seq.reshape(T * N, NUM_RAYS).astype(np.float32)

    enc_params = train_vae(flat_frames, args.v_epochs, v_rng)
    m_params   = train_transformer(enc_params, lidar_seq, args.m_epochs, m_rng)

    save_vm(enc_params, m_params)


if __name__ == "__main__":
    main()
