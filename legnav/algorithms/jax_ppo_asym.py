"""
jax_ppo_asym.py — PPO with an asymmetric (privileged) critic.

Same PPO machinery, curriculum and hyperparameters as jax_ppo.py. The only
difference: the critic is fed the ground-truth human state from
jax_privileged.py (shoe centres + body velocities within 5 m, 3-frame stack)
on top of its own LiDAR trunk, while the actor stays exactly the CALF policy of
jax_ppo.py. The saved actor is therefore deployable unchanged — the privileged
branch is simply never called at inference time.
"""

import os
import csv
from legnav import paths

os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

# jaxlib 0.9's experimental fusion autotuner stalls/crashes on this workload —
# see SACjax.py. Must be set before `import jax`.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_gpu_experimental_enable_fusion_autotuner=false").strip()

import time
import warnings
import jax
import jax.numpy as jnp

jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from legnav.core.jax_network import squash_corrected_log_prob
from legnav.core.jax_network_asym import AsymmetricActorCritic
from legnav.core.jax_privileged import PRIV_OBS_SIZE
from legnav.core.precision import bf16_apply, PRECISION_STR
import legnav.core.jax_env as jax_env
from legnav.algorithms.jax_train_asym import (
    collect_rollouts_asym, init_env_state_asym,
    NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE,
)
from legnav.algorithms.jax_ppo import (
    GAMMA, GAE_LAMBDA, CLIP_EPS, VF_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
    PPO_EPOCHS, LR_START, LR_END, LR_MIN, WARMUP_UPDATES,
    DEFAULT_TOTAL_ENV_STEPS, BATCH_SIZE, N_MINIBATCHES, MINI_BATCH_SIZE,
    _OPT_STEPS_PER_UPDATE, _WARMUP_OPT_STEPS, _SUC_ANCHORS,
    RunningMeanStd, normalize_batch_rewards, collect_episode_outcomes,
    compute_gae, get_continuous_curriculum,
)

network   = AsymmetricActorCritic(action_dim=2)
net_apply = bf16_apply(network.apply)   # bf16 forward, fp32 outputs/params

scheduler = None
optimizer = None


# ── PPO loss — flat forward pass over (T*N) samples ───────────────────────────

@jax.jit
def ppo_loss_fn(
    params,
    obs_mb,         # (MB, OBS_SIZE)
    priv_mb,        # (MB, PRIV_OBS_SIZE)  — critic only
    actions_mb,     # (MB, 2)
    advantages_mb,  # (MB,)
    returns_mb,     # (MB,)
    old_log_probs,  # (MB,)
    max_v_mb,       # (MB,)
    entropy_coef,   # ()
):
    mean, logstd, values = net_apply({"params": params}, obs_mb, priv_mb)

    log_prob    = squash_corrected_log_prob(actions_mb, mean, logstd, max_v_mb)
    ratio       = jnp.exp(log_prob - old_log_probs)
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages_mb,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_mb,
    ))
    value_loss   = VF_COEF * jnp.mean((returns_mb - values) ** 2)
    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -entropy_coef * entropy
    total_loss   = policy_loss + value_loss + entropy_loss

    kl_div    = jnp.mean(old_log_probs - log_prob)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32))

    return total_loss, (policy_loss, value_loss, entropy, kl_div, clip_frac)


@jax.jit
def ppo_update_epoch(carry, perm):
    (params, opt_state, obs_flat, priv_flat, actions_flat,
     adv_flat, ret_flat, old_lp_flat, max_v_flat, entropy_coef) = carry

    obs_p     = obs_flat[perm]
    priv_p    = priv_flat[perm]
    actions_p = actions_flat[perm]
    adv_p     = adv_flat[perm]
    ret_p     = ret_flat[perm]
    old_lp_p  = old_lp_flat[perm]
    max_v_p   = max_v_flat[perm]

    def _mb_step(mb_carry, mb_i):
        p, os_ = mb_carry
        s = mb_i * MINI_BATCH_SIZE
        mb_obs     = jax.lax.dynamic_slice_in_dim(obs_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_priv    = jax.lax.dynamic_slice_in_dim(priv_p,    s, MINI_BATCH_SIZE, axis=0)
        mb_actions = jax.lax.dynamic_slice_in_dim(actions_p, s, MINI_BATCH_SIZE, axis=0)
        mb_adv     = jax.lax.dynamic_slice_in_dim(adv_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_ret     = jax.lax.dynamic_slice_in_dim(ret_p,     s, MINI_BATCH_SIZE, axis=0)
        mb_old_lp  = jax.lax.dynamic_slice_in_dim(old_lp_p,  s, MINI_BATCH_SIZE, axis=0)
        mb_max_v   = jax.lax.dynamic_slice_in_dim(max_v_p,   s, MINI_BATCH_SIZE, axis=0)

        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, mb_obs, mb_priv, mb_actions, mb_adv, mb_ret, mb_old_lp, mb_max_v, entropy_coef
        )
        updates, new_os = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step, (params, opt_state), jnp.arange(N_MINIBATCHES),
    )

    new_carry = (new_p, new_os, obs_p, priv_p, actions_p,
                 adv_p, ret_p, old_lp_p, max_v_p, entropy_coef)
    return new_carry, (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_seq, priv_seq, actions_seq, adv_seq, ret_seq,
                    old_lp_seq, max_v_seq, rng_key, entropy_coef):
    params, opt_state = train_state

    TN = BATCH_SIZE
    obs_flat     = obs_seq.reshape(TN, OBS_SIZE)
    priv_flat    = priv_seq.reshape(TN, PRIV_OBS_SIZE)
    actions_flat = actions_seq.reshape(TN, -1)
    max_v_flat   = max_v_seq.reshape(TN)
    old_lp_flat  = old_lp_seq.reshape(TN)

    adv_flat = adv_seq.reshape(TN)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = ret_seq.reshape(TN)

    perms = jax.vmap(lambda k: jax.random.permutation(k, TN))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )

    carry = (params, opt_state, obs_flat, priv_flat, actions_flat,
             adv_flat, ret_flat, old_lp_flat, max_v_flat, entropy_coef)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


# ── Checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(params, opt_state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {"params": jax.device_get(params), "opt_state": jax.device_get(opt_state)}
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Checkpoint -> {filepath}")


def load_checkpoint(dummy_params, dummy_opt_state, filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes(
        {"params": dummy_params, "opt_state": dummy_opt_state}, raw
    )
    return bundle["params"], bundle["opt_state"]


def actor_apply(params, obs):
    """Deployment / eval path: (mean, logstd) with no privileged input."""
    return net_apply({"params": params}, obs, method=AsymmetricActorCritic.actor)


# ── Main training loop ────────────────────────────────────────────────────────

def train(total_env_steps: int = DEFAULT_TOTAL_ENV_STEPS):
    global optimizer, scheduler

    total_updates   = max(1, int(total_env_steps) // BATCH_SIZE)
    total_opt_steps = total_updates * _OPT_STEPS_PER_UPDATE

    warmup_sched = optax.linear_schedule(
        init_value=LR_MIN, end_value=LR_START, transition_steps=_WARMUP_OPT_STEPS,
    )
    decay_sched = optax.linear_schedule(
        init_value=LR_START, end_value=LR_END,
        transition_steps=max(1, total_opt_steps - _WARMUP_OPT_STEPS),
    )
    scheduler = optax.join_schedules(
        schedules=[warmup_sched, decay_sched], boundaries=[_WARMUP_OPT_STEPS],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(learning_rate=scheduler, eps=1e-5),
    )

    print(f"PPO Training  [ASYMMETRIC CRITIC — privileged human state]")
    print(f"  Envs        : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches : {N_MINIBATCHES} x {MINI_BATCH_SIZE} sample  (flat T*N)")
    print(f"  Budget      : {total_env_steps:,} env steps  →  {total_updates} updates")
    print(f"  Actor obs   : {OBS_SIZE}   Critic priv obs: {PRIV_OBS_SIZE} (extra)")
    print(f"  Precision   : {PRECISION_STR} (GPU compute)")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}")
    print(f"  Continuous curriculum anchors: suc={list(_SUC_ANCHORS)}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs  = jnp.zeros((1, OBS_SIZE))
    dummy_priv = jnp.zeros((1, PRIV_OBS_SIZE))
    params     = network.init(init_rng, dummy_obs, dummy_priv)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    tag = "legs" if jax_env.USE_LEGS else "circles"
    ckpt_path       = paths.checkpoint("ppo_asym", f"ppo_asym_{tag}_best.msgpack")
    final_ckpt_path = paths.checkpoint("ppo_asym", f"ppo_asym_{tag}_final.msgpack")
    _LOG_PATH       = paths.checkpoint("ppo_asym", f"ppo_asym_{tag}_log.csv")

    cur_max_dist, cur_ghost, cur_ent, cur_max_scen = get_continuous_curriculum(0.0)
    rolling_suc = 0.0
    highest_rolling_suc = 0.0

    print(f"Curriculum: max_goal_dist={cur_max_dist:.1f} m, "
          f"ghost_prob={cur_ghost:.1f}, max_scenario={cur_max_scen}")

    print("Initialising environments...")
    env_obs, env_state, priv_stack, vmap_step = init_env_state_asym(
        env_rng, ghost_prob=cur_ghost
    )

    rms_state   = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)

    print(f"Ready. obs={env_obs.shape}  priv={priv_stack.shape}\n")

    best_suc = 55.0  # NEVER TOUCH THIS LINE

    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    _log_file = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "acol_pct",
                          "pcol_pct", "tmo_pct", "sigma_v", "sigma_w"])

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} {'KL':>6} {'ClpF':>5} | {'FPS':>7} {'#Ep':>6} {'LR':>8} | "
           f"{'MaxDist':>7} {'Ghost':>6} {'Ent':>7} {'ScenMax':>7} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    try:
        for update in range(total_updates):
            t0 = time.time()

            rng, rollout_rng, update_rng = jax.random.split(rng, 3)

            rollout_history, env_state, env_obs, priv_stack, last_val = collect_rollouts_asym(
                rollout_rng, train_state[0], net_apply, vmap_step,
                env_state, env_obs, priv_stack, cur_max_dist, jnp.int32(-1), cur_ghost,
                jnp.int32(cur_max_scen)
            )

            raw_rewards = rollout_history["rewards"]
            values      = rollout_history["values"]
            dones       = rollout_history["dones"]

            rewards, running_ret, rms_state = normalize_batch_rewards(
                raw_rewards, dones, running_ret, rms_state, GAMMA
            )

            obs_seq   = rollout_history["obs"]
            priv_seq  = rollout_history["priv"]
            acts_seq  = rollout_history["actions"]
            lp_seq    = rollout_history["log_probs"]
            max_v_seq = rollout_history["max_v"]

            ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(
                raw_rewards, dones,
                rollout_history["goal_reached"], rollout_history["collision"],
                rollout_history["passive_col"], rollout_history["active_col"],
            )

            n_ep = int(ep_msk.sum())
            if n_ep > 0:
                mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
                suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
                obs_pct  = float((ep_obs  * ep_msk).sum() / n_ep) * 100.0
                acol_pct = float((ep_acol * ep_msk).sum() / n_ep) * 100.0
                pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
                tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0
            else:
                mean_ret, suc_pct, obs_pct, acol_pct, pcol_pct, tmo_pct = 0., 0., 0., 0., 0., 0.

            # ── Monotonic curriculum (identical to jax_ppo) ───────────────
            if n_ep > 0:
                rolling_suc         = 0.9 * rolling_suc + 0.1 * suc_pct
                highest_rolling_suc = max(highest_rolling_suc, rolling_suc)

            new_max_dist, new_ghost, new_ent, new_max_scen = get_continuous_curriculum(highest_rolling_suc)

            old_print_dist  = round(cur_max_dist, 1)
            old_print_ghost = round(cur_ghost, 2)
            old_print_scen  = cur_max_scen

            if new_max_dist > cur_max_dist:
                cur_max_dist = min(cur_max_dist + 0.2, new_max_dist)
            if new_ghost > cur_ghost:
                cur_ghost = new_ghost
            if new_max_scen > cur_max_scen:
                cur_max_scen = new_max_scen

            if (round(cur_max_dist, 1) > old_print_dist or
                round(cur_ghost, 2) > old_print_ghost or
                cur_max_scen > old_print_scen):
                print(f"  -> Curriculum advanced: dist={cur_max_dist:.1f}m, "
                      f"ghost_prob={cur_ghost:.2f}, unlocked_scenarios=0-{cur_max_scen}")

            entropy_coef = jnp.array(new_ent)

            advantages, returns = compute_gae(rewards, values, dones, last_val)

            train_state, mean_loss, aux = run_ppo_updates(
                train_state, obs_seq, priv_seq, acts_seq, advantages, returns,
                lp_seq, max_v_seq, update_rng, entropy_coef,
            )

            fps = BATCH_SIZE / (time.time() - t0)

            if update % 5 == 0:
                p_loss, v_loss, entropy, kl_div, clip_frac = aux
                lr_now       = float(scheduler(update * _OPT_STEPS_PER_UPDATE))
                ent_coef_now = float(entropy_coef)
                elapsedtime  = (time.time() - t_start) / 60.0
                print(
                    f"{update:>5d} | {mean_ret:>7.1f} | "
                    f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                    f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
                    f"{float(v_loss):>6.2f} {float(entropy):>6.2f} "
                    f"{float(kl_div):>6.4f} {float(clip_frac):>4.2f} | "
                    f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
                    f"{cur_max_dist:>6.1f}m {cur_ghost:>5.2f}g "
                    f"{ent_coef_now:>6.4f}e scen<=={cur_max_scen} {elapsedtime:>5.1f}min"
                )

            if n_ep > 0:
                _, curr_logstd = actor_apply(train_state[0], jnp.zeros((1, OBS_SIZE)))
                curr_sigma = jax.device_get(jnp.exp(curr_logstd[0]))
                _log_writer.writerow([update * BATCH_SIZE, round(mean_ret, 4),
                                      round(suc_pct, 2), round(acol_pct, 2),
                                      round(pcol_pct, 2), round(tmo_pct, 2),
                                      round(float(curr_sigma[0]), 5), round(float(curr_sigma[1]), 5)])
                _log_file.flush()

            curriculum_mature = cur_max_scen >= 1 and cur_max_dist >= 5.0
            if suc_pct > best_suc and n_ep > 0 and curriculum_mature:
                best_suc = suc_pct
                save_checkpoint(train_state[0], train_state[1], ckpt_path)
    finally:
        _log_file.close()

    print(f"Training log saved -> {_LOG_PATH}")

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    save_checkpoint(train_state[0], train_state[1], final_ckpt_path)

    _, final_logstd = actor_apply(train_state[0], jnp.zeros((1, OBS_SIZE)))
    final_sigma = jax.device_get(jnp.exp(final_logstd[0]))

    print("\n" + "=" * 55)
    print(" 🔍 FINAL EXPLORATORY NOISE (SIGMA) ANALYSIS ")
    print("=" * 55)
    print(f" Linear Velocity Sigma (v)  : {final_sigma[0]:.5f}")
    print(f" Angular Velocity Sigma (w) : {final_sigma[1]:.5f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train()
