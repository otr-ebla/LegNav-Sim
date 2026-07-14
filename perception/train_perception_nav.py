"""
perception/train_perception_nav.py — On-the-go perception training under PPO
==============================================================================
Trains PerceptionNet on data gathered while the trained PPO policy drives the
robot through the same episodes as jax_eval_multi (random scenarios 0-6,
24 HSFM pedestrians with legs, sensor noise on): NUM_ENVS parallel envs run
the policy in the background, and every step contributes the robot's own
8-frame LiDAR stack + geometric ground truth (perception_env's labels) to the
supervised loss. Unlike train_perception's stationary robot, the stack here
carries real ego-motion, matching deployment.

Scenes keep the leg-like static decoy clutter (hard negatives) and episodes
auto-reset on goal/collision/timeout.

Run from the repo root:
    python -m perception.train_perception_nav
    python -m perception.train_perception_nav --resume   # warm-start from perception_best
"""

import argparse
import functools
import os

import jax
import jax.numpy as jnp
import optax
import flax.serialization

jax.config.update("jax_compilation_cache_dir",
                  os.path.expanduser("~/.cache/jax_comp_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

from legnav import paths
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env
from legnav.core.jax_network import EndToEndActorCritic, scale_actions_batched
from perception.config import PerceptionConfig as PC
from perception.perception_env import compute_ground_truth, _inject_decoy_clutter
from perception.perception_network import PerceptionNet
import perception.train_perception as T

NUM_ENVS      = 128            # nav env (24 humans + legs) is heavier than stationary
ROLLOUT_STEPS = PC.ROLLOUT_STEPS
MAX_GOAL_DIST = 9.0            # same episode setting as live_eval / jax_eval_multi
MIN_GOAL_DIST = 4.0
ACTION_DIM    = 2

CKPT_PPO  = paths.checkpoint("ppo", "ppo_tanh_inside_final.msgpack")
CKPT_BEST = paths.checkpoint("perception", "perception_nav_best.msgpack")
CKPT_LAST = paths.checkpoint("perception", "perception_nav_last.msgpack")


def _parse_args():
    p = argparse.ArgumentParser(description="On-the-go perception training under PPO")
    p.add_argument("--ppo-ckpt", default=CKPT_PPO,
                   help="PPO checkpoint that drives the robot.")
    p.add_argument("--resume", default="", nargs="?", const=T.CKPT_BEST,
                   help="Warm-start perception weights from a checkpoint "
                        "(default when given without a path: perception_best.msgpack).")
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--updates", type=int, default=PC.NUM_UPDATES)
    return p.parse_args()


# ── PPO driver (batched, deterministic mean action) ───────────────────────────

def _load_ppo(path):
    net = EndToEndActorCritic(action_dim=ACTION_DIM)
    with open(path, "rb") as f:
        bundle = flax.serialization.msgpack_restore(f.read())
    params = bundle.get("params", bundle)

    def infer_batch(p, obs_b, max_v_b):
        mean, _, _ = net.apply({"params": p}, obs_b)
        return scale_actions_batched(mean, max_v_b)

    return params, infer_batch


# ── Batched env with auto-reset + rolling 8-frame perception buffer ───────────

_rs, _ss = make_stacked_env(reset_env, step_env, stack_dim=3)


def _reset_one(key):
    """Reset one env (random scenario 0-6) and inject decoy clutter."""
    obs, ss = _rs(key, MAX_GOAL_DIST, ghost_prob=0.0, min_goal_dist=MIN_GOAL_DIST)
    es = _inject_decoy_clutter(ss.env_state, jax.random.fold_in(key, 7))
    ss = ss.replace(env_state=es)
    perc = jnp.tile(ss.lidar_stack[-1][None, :], (PC.STACK_DIM, 1))
    return obs, ss, perc


def make_collect(infer_batch, num_envs):
    """ROLLOUT_STEPS × num_envs samples gathered while PPO drives."""

    def collect(ppo_params, rng, obs_b, ss_b, perc_b):

        def _one_step(carry, _):
            obs_b, ss_b, perc_b, key = carry
            key, k_step, k_reset = jax.random.split(key, 3)

            act = infer_batch(ppo_params, obs_b, ss_b.env_state.max_v)
            obs2, ss2, _, done, _ = jax.vmap(_ss)(
                jax.random.split(k_step, num_envs), ss_b, act)

            frame  = ss2.lidar_stack[:, -1]
            perc2  = jnp.concatenate([perc_b[:, 1:], frame[:, None, :]], axis=1)
            gt     = jax.vmap(compute_ground_truth)(ss2.env_state)
            sample = (perc2, gt["person_mask"], gt["offsets"], gt["leg_offsets"])

            # Auto-reset finished episodes (fresh scene, refilled perception buffer)
            obs_r, ss_r, perc_r = jax.vmap(_reset_one)(
                jax.random.split(k_reset, num_envs))

            def sel(a, b):
                d = done.reshape(done.shape + (1,) * (a.ndim - 1))
                return jnp.where(d, a, b)

            obs3  = sel(obs_r, obs2)
            perc3 = sel(perc_r, perc2)
            ss3   = jax.tree_util.tree_map(sel, ss_r, ss2)
            return (obs3, ss3, perc3, key), sample

        (obs_b, ss_b, perc_b, _), (stacks, masks, offsets, leg_offsets) = \
            jax.lax.scan(_one_step, (obs_b, ss_b, perc_b, rng),
                         None, length=ROLLOUT_STEPS)
        return obs_b, ss_b, perc_b, stacks, masks, offsets, leg_offsets

    return jax.jit(collect)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    num_envs = args.num_envs
    T.NUM_ENVS = num_envs   # update_chunk reshapes with train_perception globals

    T._check_devices()
    rng = jax.random.PRNGKey(0)

    ppo_params, infer_batch = _load_ppo(args.ppo_ckpt)
    print(f"PPO driver: {args.ppo_ckpt}")

    net = PerceptionNet()
    rng, k_init = jax.random.split(rng)
    params = net.init(k_init, jnp.zeros((1, PC.STACK_DIM, PC.NUM_RAYS)))["params"]
    if args.resume:
        params = T.load_checkpoint(params, args.resume)
        print(f"Resumed perception weights from {args.resume}")
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"PerceptionNet: {n_params:,} parameters | "
          f"{num_envs} PPO-driven envs × {ROLLOUT_STEPS} steps per chunk")

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(PC.LR))
    opt_state = tx.init(params)

    collect = make_collect(infer_batch, num_envs)

    rng, k_reset = jax.random.split(rng)
    obs_b, ss_b, perc_b = jax.vmap(_reset_one)(
        jax.random.split(k_reset, num_envs))

    # Warm-up so the 8-frame stack holds real driven history, not tiled resets
    rng, k_warm = jax.random.split(rng)
    obs_b, ss_b, perc_b, *_ = collect(ppo_params, k_warm, obs_b, ss_b, perc_b)

    best_loss = float("inf")
    for update in range(1, args.updates + 1):
        rng, k_collect, k_update = jax.random.split(rng, 3)
        obs_b, ss_b, perc_b, stacks, masks, offsets, leg_offsets = collect(
            ppo_params, k_collect, obs_b, ss_b, perc_b)
        params, opt_state, m = T.update_chunk(
            net.apply, tx, params, opt_state, k_update, stacks, masks,
            offsets, leg_offsets)

        if update % PC.LOG_EVERY == 0:
            m = {k: float(v) for k, v in m.items()}
            print(f"[{update:5d}/{args.updates}] "
                  f"loss {m['loss']:.4f} (cls {m['cls']:.4f}, off {m['off']:.4f}, "
                  f"leg {m['leg']:.4f}) | "
                  f"P {m['precision']:.3f} R {m['recall']:.3f} | "
                  f"body MAE {m['off_mae_m']:.3f} m  leg MAE {m['leg_mae_m']:.3f} m")
            if m["loss"] < best_loss:
                best_loss = m["loss"]
                T.save_checkpoint(params, CKPT_BEST)

        if update % PC.CKPT_EVERY == 0:
            T.save_checkpoint(params, CKPT_LAST)

    T.save_checkpoint(params, CKPT_LAST)
    print(f"Done. Best checkpoint: {CKPT_BEST}\n      Last checkpoint: {CKPT_LAST}")


if __name__ == "__main__":
    main()
