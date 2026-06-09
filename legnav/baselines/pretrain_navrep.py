"""
pretrain_navrep.py — Collect expert transitions and JOINT-pretrain NavRep's V+M.

Training regime: JOINT V+M (the paper's recommended NavRep configuration). V (a
VAE) and M (a causal Transformer predicting the next state) are trained together
to minimise the combined loss (Dugas et al., ICRA 2021):

    L = L_rec            reconstruction of the current LiDAR scan       (Eq. 1)
      + β · KL(z^t)       VAE prior on the latent                       (Eq. 1)
      + L_{V+M}          next-state prediction  MSE(V(ẑ_{t+1}), s_{t+1})
      + L_sr             next goal-velocity prediction  MSE(ŝ_r, s_r)   (Eq. 2)

For the env's 1-D LiDAR (216 rays normalised to [0,1]) all terms use MSE, as the
paper does for its 1-D variant (BCE is only used for the 2-D "rings" variant).

Pipeline
--------
1. Roll out the HSFM expert (JHSFM Social Force Model toward goal) in the
   vectorised JAX environment (NUM_ENVS in parallel) and record, per step, the
   newest LiDAR frame (216), the goal-velocity state s_r (8 = newest pose(3) +
   state_vec(5)) and the done flag (autoreset boundary).
2. JOINT-train NavRepWorldModel (encoder + decoder + Transformer M) on sampled
   sub-sequences of those trajectories with the combined loss above.
3. Save {encoder, decoder, M} to checkpoints_navrep/navrep_vm.msgpack so
   train_navrep.py can load encoder + M (frozen) and PPO-train the controller.

Usage
-----
    cd src/jax_env
    python comparison_policies/pretrain_navrep.py \
        [--frames 2_000_000] [--epochs 5] [--batches-per-epoch 3000]
"""

import os
from legnav import paths

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

from legnav.algorithms.jax_train import NUM_ENVS
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env
from legnav.core.jax_env import NUM_RAYS as ENV_NUM_RAYS
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.baselines.jhsfm_planner import HumanPilot
from legnav.baselines.navrep_network import (
    NavRepWorldModel, Z_DIM, SR_DIM, NUM_RAYS,
)

assert ENV_NUM_RAYS == NUM_RAYS, "NUM_RAYS mismatch between env and navrep_network"

# Stacked-obs slices (662D = pose_stack(9) | state_vec(5) | lidar_stack(3*216)).
# Newest pose = obs[6:9]; state_vec = obs[9:14]; s_r = obs[6:14]; newest LiDAR
# frame = obs[-216:].
_SR_LO, _SR_HI = 6, 14


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_FRAMES     = 2_000_000     # total env-steps collected (across NUM_ENVS)
DEFAULT_EPOCHS     = 5
DEFAULT_BATCHES    = 3000          # sampled sub-sequence batches per epoch

SEQ_LEN            = 16            # sub-sequence length fed to M
BATCH              = 128           # sub-sequences per batch
BATCHES_PER_BLOCK  = 100           # batches per jitted block. An epoch is run as
                                   # a host loop over blocks so XLA only ever
                                   # plans ~BATCHES_PER_BLOCK scan iterations at a
                                   # time; compiling the full 3000-batch epoch as
                                   # one scan made XLA estimate a ~65 GiB peak and
                                   # spam rematerialization probes on a 16 GB GPU.
LR                 = 1e-3
KL_BETA            = 1e-1          # VAE prior weight (β); 10x up from 1e-2 to
                                   # pull the latent back toward the unit-Gaussian
                                   # prior (KL was drifting to ~55). Not 1.0: with
                                   # MSE recon ~1.7, β=1 would crush reconstruction.
PRED_COEF          = 1.0          # next-state prediction loss weight
SR_COEF            = 1.0          # goal-velocity prediction loss weight
COLLECT_CHUNK      = 128           # env-steps per jitted collection chunk. The
                                   # rollout runs as a host loop over chunks so
                                   # XLA never fuses the full 1953-step physics
                                   # scan into one graph (that estimated a ~65 GiB
                                   # peak and spammed rematerialization probes).

CKPT_DIR  = os.path.join(_JAX_ENV_DIR, paths.checkpoint("navrep"))
VM_CKPT   = os.path.join(CKPT_DIR, "navrep_vm.msgpack")


# ══════════════════════════════════════════════════════════════════════════════
# 1) Expert rollout collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_expert_data(total_frames: int, start_rng: jax.Array):
    """Generate the full dataset in a single JIT call on GPU.

    Returns
      lidar_seq : (T, N, 216)  per-env time-ordered newest LiDAR frame
      sr_seq    : (T, N, 8)    per-env time-ordered goal-velocity state s_r
      done_seq  : (T, N)       autoreset boundary flags
    """
    N_STEPS = total_frames // NUM_ENVS

    # Pretraining wants clean LiDAR: with SENSOR_NOISE on, 6% of rays jump to
    # max-range and every ray gets 0.05 m Gaussian — i.i.d. *per frame*. That
    # noise flows into z and dominates the predictive target, which becomes
    # unpredictable. Encode clean scans so the dynamics are learnable.
    import legnav.core.jax_env as _jax_env
    _jax_env.SENSOR_NOISE = False

    pilot = HumanPilot()
    vmap_act = jax.vmap(pilot.act)

    bound_reset = lambda key, max_goal_dist, scenario_idx, ghost_prob, min_goal_dist=0.8, **kwargs: \
        reset_env(key, max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob, min_goal_dist=min_goal_dist, **kwargs)

    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=3)
    # Autoreset chains fresh episodes so M sees real dynamics throughout the
    # rollout (episodes terminate well before the N_STEPS-long scan ends).
    step_ar = make_autoreset_env(rs, ss)
    # ghost_prob=0.5: in ~half the episodes the robot is invisible to humans, so
    # they cross its path — giving M the varied human motion the controller
    # faces during training.
    _reset_one = lambda key, scenario_idx: rs(
        key, max_goal_dist=9.0, scenario_idx=scenario_idx, ghost_prob=0.5
    )
    _step_one = lambda key, state, action: step_ar(
        key, state, action, max_goal_dist=9.0, scenario_idx=-1,
        ghost_prob=0.5, max_scenario=6, min_goal_dist=0.8,
    )
    vmap_reset = jax.vmap(_reset_one, in_axes=(0, 0))
    vmap_step  = jax.vmap(_step_one)

    @jax.jit
    def reset_all(rng):
        env_rngs = jax.random.split(rng, NUM_ENVS)
        # Scenarios 0-6 — the same training curriculum range (jax_env_multi).
        scenarios = jax.random.randint(rng, (NUM_ENVS,), 0, 7)
        return vmap_reset(env_rngs, scenarios)        # obs, stacked_state

    @jax.jit
    def collect_chunk(rng_scan, obs, stacked_state):
        def step_fn(carry, step_rng):
            current_obs, current_state = carry
            actions = vmap_act(current_state.env_state)
            step_rngs = jax.random.split(step_rng, NUM_ENVS)
            next_obs, next_state, reward, done, info = vmap_step(step_rngs, current_state, actions)

            lidar_frame = next_obs[:, -NUM_RAYS:]          # (N, 216)
            sr_frame    = next_obs[:, _SR_LO:_SR_HI]       # (N, 8)
            # done[t]=True ⇒ next_obs is the FIRST frame of a fresh episode
            # (scene teleports); the prediction into it must be masked.
            return (next_obs, next_state), (lidar_frame, sr_frame, done)

        step_rngs = jax.random.split(rng_scan, COLLECT_CHUNK)
        (final_obs, final_state), chunk = jax.lax.scan(
            step_fn, (obs, stacked_state), step_rngs
        )
        return final_obs, final_state, chunk   # chunk: (CHUNK,N,216),(CHUNK,N,8),(CHUNK,N)

    n_chunks = max(1, N_STEPS // COLLECT_CHUNK)
    eff_steps = n_chunks * COLLECT_CHUNK
    print(f"\n[collect] target {eff_steps * NUM_ENVS:,} frames "
          f"({eff_steps} env-steps = {n_chunks}×{COLLECT_CHUNK})")
    print(f"          expert: HumanPilot (JHSFM) — generating 100% on GPU...")
    t0 = time.time()

    rng_reset, rng_scan = jax.random.split(start_rng)
    obs, stacked_state = reset_all(rng_reset)
    lidar_chunks, sr_chunks, done_chunks = [], [], []
    for i in range(n_chunks):
        rng_scan, chunk_rng = jax.random.split(rng_scan)
        obs, stacked_state, (lid, sr, dn) = collect_chunk(chunk_rng, obs, stacked_state)
        lidar_chunks.append(lid); sr_chunks.append(sr); done_chunks.append(dn)

    lidar_seq = jnp.concatenate(lidar_chunks, axis=0)     # (T, N, 216)
    sr_seq    = jnp.concatenate(sr_chunks,    axis=0)     # (T, N, 8)
    done_seq  = jnp.concatenate(done_chunks,  axis=0)     # (T, N)
    jax.block_until_ready(lidar_seq)
    print(f"[collect] Completed in {time.time() - t0:.1f}s. "
          f"lidar={tuple(lidar_seq.shape)}  sr={tuple(sr_seq.shape)}")
    return lidar_seq, sr_seq, done_seq


# ══════════════════════════════════════════════════════════════════════════════
# 2) JOINT V + M training
# ══════════════════════════════════════════════════════════════════════════════

def train_joint(lidar_seq, sr_seq, done_seq, epochs: int,
                batches_per_epoch: int, rng: jax.Array):
    """Jointly train V + M (NavRepWorldModel) on sampled sub-sequences."""
    T, N, R = lidar_seq.shape
    assert T > SEQ_LEN, f"need T > {SEQ_LEN} for joint training"
    assert R == NUM_RAYS and sr_seq.shape[-1] == SR_DIM

    model = NavRepWorldModel(z_dim=Z_DIM, sr_dim=SR_DIM)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, SEQ_LEN, NUM_RAYS))
    params = model.init(init_rng, dummy, sample_rng)["params"]

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    # Keep the full dataset resident in VRAM for fast on-device slicing.
    device_lidar = jnp.asarray(lidar_seq)                 # (T, N, 216)
    device_sr    = jnp.asarray(sr_seq)                    # (T, N, 8)
    device_done  = jnp.asarray(done_seq)                  # (T, N) bool

    starts_per_env = T - SEQ_LEN

    @jax.jit
    def train_block(params, opt_state, env_ids, start_ts, batch_rngs,
                    dataset_lidar, dataset_sr, dataset_done):
        # Scans over a single block of BATCHES_PER_BLOCK batches. The (T, N, ·)
        # datasets are passed as ARGUMENTS, not closed over: a closed-over array
        # is baked into the executable as a compile-time constant, which XLA
        # then tries to replicate (the 1.7 GB LiDAR tensor inflated allocations).
        def batch_step(carry, step_data):
            p, os_ = carry
            e_ids, s_ts, b_rng = step_data

            def get_seq(e, s):
                lid = jax.lax.dynamic_slice(dataset_lidar, (s, e, 0), (SEQ_LEN, 1, R))[:, 0, :]
                sr  = jax.lax.dynamic_slice(dataset_sr,    (s, e, 0), (SEQ_LEN, 1, SR_DIM))[:, 0, :]
                d   = jax.lax.dynamic_slice(dataset_done,  (s, e),    (SEQ_LEN, 1))[:, 0]
                return lid, sr, d

            lid_b, sr_b, d_b = jax.vmap(get_seq)(e_ids, s_ts)   # (B,L,216),(B,L,8),(B,L)

            def loss_fn(p_):
                recon, next_recon, sr_pred, z_mean, z_logvar = model.apply(
                    {"params": p_}, lid_b, b_rng
                )
                # Reconstruction of the current frames (every frame valid).
                rec_loss = jnp.mean(jnp.sum((recon - lid_b) ** 2, axis=-1))
                kl = -0.5 * jnp.mean(jnp.sum(
                    1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1
                ))
                # Next-state prediction: step t predicts frame t+1. Mask windows
                # whose next frame is a fresh-episode start (scene teleport).
                valid   = (~d_b[:, 1:]).astype(recon.dtype)         # (B, L-1)
                vsum    = jnp.clip(jnp.sum(valid), 1.0)
                pred_se = jnp.sum((next_recon[:, :-1] - lid_b[:, 1:]) ** 2, axis=-1)
                pred_loss = jnp.sum(pred_se * valid) / vsum
                sr_se   = jnp.sum((sr_pred[:, :-1] - sr_b[:, 1:]) ** 2, axis=-1)
                sr_loss = jnp.sum(sr_se * valid) / vsum

                total = (rec_loss + KL_BETA * kl
                         + PRED_COEF * pred_loss + SR_COEF * sr_loss)
                return total, (rec_loss, kl, pred_loss, sr_loss)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, new_os = optimizer.update(grads, os_, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), (loss, *aux)

        (new_params, new_opt_state), metrics = jax.lax.scan(
            batch_step, (params, opt_state), (env_ids, start_ts, batch_rngs)
        )
        # Block-mean of each metric → (5,): loss, rec, kl, pred, sr.
        return new_params, new_opt_state, jnp.stack([jnp.mean(m) for m in metrics])

    n_blocks = max(1, batches_per_epoch // BATCHES_PER_BLOCK)
    eff_batches = n_blocks * BATCHES_PER_BLOCK
    print(f"\n[joint] T={T} N={N} Z={Z_DIM} SR={SR_DIM}  seq_len={SEQ_LEN}  "
          f"batch={BATCH}  {eff_batches} batches/epoch "
          f"({n_blocks}×{BATCHES_PER_BLOCK})  {epochs} epochs")

    for epoch in range(epochs):
        t0 = time.time()
        acc = np.zeros(5, dtype=np.float64)
        for _ in range(n_blocks):
            rng, env_rng, start_rng, batch_rng = jax.random.split(rng, 4)
            env_ids   = jax.random.randint(env_rng,   (BATCHES_PER_BLOCK, BATCH), 0, N)
            start_ts  = jax.random.randint(start_rng, (BATCHES_PER_BLOCK, BATCH), 0, starts_per_env)
            batch_rngs = jax.random.split(batch_rng, BATCHES_PER_BLOCK)
            params, opt_state, m = train_block(
                params, opt_state, env_ids, start_ts, batch_rngs,
                device_lidar, device_sr, device_done
            )
            acc += np.asarray(m)
        loss, rec, kl, pred, sr = acc / n_blocks
        print(f"[joint] epoch {epoch+1}/{epochs}  "
              f"loss={loss:.4f}  rec={rec:.4f}  "
              f"kl={kl:.4f}  pred={pred:.4f}  "
              f"sr={sr:.4f}  ({time.time()-t0:.1f}s)")

    return params   # {"encoder", "decoder", "M"}


# ══════════════════════════════════════════════════════════════════════════════
# 3) Save combined V + M checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def save_vm(params, path=VM_CKPT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "encoder": jax.device_get(params["encoder"]),
        "decoder": jax.device_get(params["decoder"]),
        "M":       jax.device_get(params["M"]),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"\n[save] V+M (joint) → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help="Total env-steps to collect (across NUM_ENVS).")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES,
                    help="Sampled sub-sequence batches per epoch.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("NavRep Pretraining  —  JOINT V + M (paper-recommended)")
    print(f"  expert            : HumanPilot (JHSFM Social Force Model)")
    print(f"  frames            : {args.frames:,}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batches/epoch     : {args.batches_per_epoch}")
    print(f"  seed              : {args.seed}\n")

    rng = jax.random.PRNGKey(args.seed)
    rng, coll_rng, train_rng = jax.random.split(rng, 3)

    lidar_seq, sr_seq, done_seq = collect_expert_data(args.frames, coll_rng)
    params = train_joint(lidar_seq, sr_seq, done_seq,
                         args.epochs, args.batches_per_epoch, train_rng)
    save_vm(params)


if __name__ == "__main__":
    main()
