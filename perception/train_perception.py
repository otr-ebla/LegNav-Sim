"""
perception/train_perception.py — Supervised training of the perception module
===============================================================================
Trains PerceptionNet to detect people and regress their body-centre position
from stacked 2D LiDAR scans (216 rays, 360°, stack of 8). Data is generated
on-the-fly on the GPU by NUM_ENVS parallel stationary-robot scenes
(perception_env): the robot never moves, pedestrians walk/stop around it.

Loss:
  - per-ray sigmoid BCE (person / not-person), positives up-weighted
  - smooth-L1 on the (dx, dy) offset to the owning person's centre,
    applied on ground-truth person rays only

Run from the repo root:
    python -m perception.train_perception
"""

import os
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization

jax.config.update("jax_compilation_cache_dir",
                  os.path.expanduser("~/.cache/jax_comp_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

from legnav import paths
from perception.config import PerceptionConfig as PC
from perception.perception_env import reset_perception, step_perception
from perception.perception_network import PerceptionNet

NUM_ENVS      = PC.NUM_ENVS
ROLLOUT_STEPS = PC.ROLLOUT_STEPS

CKPT_DIR  = paths.checkpoint("perception")
CKPT_BEST = paths.checkpoint("perception", "perception_best.msgpack")
CKPT_LAST = paths.checkpoint("perception", "perception_last.msgpack")


def _check_devices():
    try:
        gpus = jax.devices("cuda")
        print(f"GPU verified: {gpus}")
    except RuntimeError:
        print(f"WARNING: no CUDA GPU found, running on {jax.devices()} (slow).")


# ── Data collection ───────────────────────────────────────────────────────────

_vmap_reset = jax.jit(jax.vmap(reset_perception))
_vmap_step  = jax.jit(jax.vmap(step_perception))


def collect_chunk(rng, states):
    """ROLLOUT_STEPS × NUM_ENVS samples: (stacks, person_mask, offsets)."""

    def _one_step(carry, _):
        st, key = carry
        key, sub = jax.random.split(key)
        step_keys = jax.random.split(sub, NUM_ENVS)
        new_st, gt = _vmap_step(step_keys, st)
        sample = (new_st.lidar_stack, gt["person_mask"],
                  gt["offsets"], gt["leg_offsets"])
        return (new_st, key), sample

    (states, _), (stacks, masks, offsets, leg_offsets) = jax.lax.scan(
        _one_step, (states, rng), None, length=ROLLOUT_STEPS)
    return states, stacks, masks, offsets, leg_offsets


collect_chunk = jax.jit(collect_chunk)


# ── Loss / update ─────────────────────────────────────────────────────────────

def loss_fn(params, apply_fn, stacks, masks, offsets, leg_offsets):
    logits, pred_off, pred_leg = apply_fn({"params": params}, stacks)

    labels = masks.astype(jnp.float32)
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    bce = jnp.where(masks, PC.POS_WEIGHT * bce, bce)
    cls_loss = bce.mean()

    n_pos = jnp.maximum(masks.sum(), 1.0)
    huber = optax.huber_loss(pred_off, offsets).sum(-1)        # (..., num_rays)
    off_loss = jnp.where(masks, huber, 0.0).sum() / n_pos
    huber_leg = optax.huber_loss(pred_leg, leg_offsets).sum(-1)
    leg_loss = jnp.where(masks, huber_leg, 0.0).sum() / n_pos

    total = cls_loss + PC.OFFSET_LOSS_W * (off_loss + leg_loss)

    # Monitoring metrics
    pred_pos  = logits > 0.0
    tp        = (pred_pos & masks).sum()
    precision = tp / jnp.maximum(pred_pos.sum(), 1)
    recall    = tp / jnp.maximum(masks.sum(), 1)
    off_mae   = (jnp.where(masks[..., None], jnp.abs(pred_off - offsets), 0.0).sum()
                 / (2.0 * n_pos)) * PC.OFFSET_SCALE
    leg_mae   = (jnp.where(masks[..., None], jnp.abs(pred_leg - leg_offsets), 0.0).sum()
                 / (2.0 * n_pos)) * PC.LEG_OFFSET_SCALE
    metrics = {"loss": total, "cls": cls_loss, "off": off_loss, "leg": leg_loss,
               "precision": precision, "recall": recall,
               "off_mae_m": off_mae, "leg_mae_m": leg_mae}
    return total, metrics


@functools.partial(jax.jit, static_argnums=(0, 1), donate_argnums=(2, 3))
def update_chunk(apply_fn, tx, params, opt_state, rng, stacks, masks,
                 offsets, leg_offsets):
    """Shuffle the chunk and run PC.MINIBATCHES gradient steps."""
    n = ROLLOUT_STEPS * NUM_ENVS
    stacks      = stacks.reshape(n, PC.STACK_DIM, PC.NUM_RAYS)
    masks       = masks.reshape(n, PC.NUM_RAYS)
    offsets     = offsets.reshape(n, PC.NUM_RAYS, 2)
    leg_offsets = leg_offsets.reshape(n, PC.NUM_RAYS, 2)

    perm = jax.random.permutation(rng, n)
    mb = n // PC.MINIBATCHES

    def _mb_step(carry, i):
        params, opt_state = carry
        idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, apply_fn, stacks[idx], masks[idx], offsets[idx],
            leg_offsets[idx])
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), metrics

    (params, opt_state), metrics = jax.lax.scan(
        _mb_step, (params, opt_state), jnp.arange(PC.MINIBATCHES))
    return params, opt_state, jax.tree_util.tree_map(jnp.mean, metrics)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(params, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes({"params": params}))


def load_checkpoint(params_template, filepath):
    with open(filepath, "rb") as f:
        bundle = flax.serialization.from_bytes({"params": params_template}, f.read())
    return bundle["params"]


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train the perception module")
    ap.add_argument("--resume", default="", nargs="?", const=CKPT_BEST,
                    help="Fine-tune from a checkpoint (default when given "
                         "without a path: perception_best.msgpack).")
    args = ap.parse_args()

    _check_devices()
    rng = jax.random.PRNGKey(0)

    net = PerceptionNet()
    rng, k_init = jax.random.split(rng)
    params = net.init(k_init, jnp.zeros((1, PC.STACK_DIM, PC.NUM_RAYS)))["params"]
    if args.resume:
        params = load_checkpoint(params, args.resume)
        print(f"Resumed weights from {args.resume}")
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"PerceptionNet: {n_params:,} parameters | "
          f"{NUM_ENVS} envs × {ROLLOUT_STEPS} steps per chunk")

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(PC.LR))
    opt_state = tx.init(params)

    rng, k_reset = jax.random.split(rng)
    states, _ = _vmap_reset(jax.random.split(k_reset, NUM_ENVS))

    # Warm-up so the 8-frame stack holds real history, not tiled resets
    for _ in range(PC.STACK_DIM):
        rng, k = jax.random.split(rng)
        states, _ = _vmap_step(jax.random.split(k, NUM_ENVS), states)

    chunks_per_episode = max(PC.EPISODE_STEPS // ROLLOUT_STEPS, 1)
    best_loss = float("inf")

    for update in range(1, PC.NUM_UPDATES + 1):
        # Resample all scenes periodically (people drift toward equilibrium)
        if update % chunks_per_episode == 0:
            rng, k_reset = jax.random.split(rng)
            states, _ = _vmap_reset(jax.random.split(k_reset, NUM_ENVS))
            for _ in range(PC.STACK_DIM):
                rng, k = jax.random.split(rng)
                states, _ = _vmap_step(jax.random.split(k, NUM_ENVS), states)

        rng, k_collect, k_update = jax.random.split(rng, 3)
        states, stacks, masks, offsets, leg_offsets = collect_chunk(k_collect, states)
        params, opt_state, m = update_chunk(
            net.apply, tx, params, opt_state, k_update, stacks, masks,
            offsets, leg_offsets)

        if update % PC.LOG_EVERY == 0:
            m = {k: float(v) for k, v in m.items()}
            print(f"[{update:5d}/{PC.NUM_UPDATES}] "
                  f"loss {m['loss']:.4f} (cls {m['cls']:.4f}, off {m['off']:.4f}, "
                  f"leg {m['leg']:.4f}) | "
                  f"P {m['precision']:.3f} R {m['recall']:.3f} | "
                  f"body MAE {m['off_mae_m']:.3f} m  leg MAE {m['leg_mae_m']:.3f} m")
            if m["loss"] < best_loss:
                best_loss = m["loss"]
                save_checkpoint(params, CKPT_BEST)

        if update % PC.CKPT_EVERY == 0:
            save_checkpoint(params, CKPT_LAST)

    save_checkpoint(params, CKPT_LAST)
    print(f"Done. Best checkpoint: {CKPT_BEST}\n      Last checkpoint: {CKPT_LAST}")


if __name__ == "__main__":
    main()
