"""
jax_ppo.py — Single-GPU PPO Training  (GPU 0)
===============================================
CHANGES vs previous version (which plateaued at ~18% success from update 90):

  1. ENTROPY SATURATION FIX (root cause of plateau):
     The global log_std scalar was pushed by entropy_coef=0.05 into its +0.5
     ceiling from update ~90, locking H=3.838 forever. The policy was maximally
     noisy; advantage signals washed out completely.
     Fix: log_std is now a state-dependent Dense head in jax_network.py.
     Here: ENTROPY_COEF reduced from 0.05->0.002 (constant, no schedule needed).
     Exploration now comes from the per-state adaptive std, not a blunt
     global entropy push.

  2. ROLLOUT_STEPS 64->256  /  NUM_ENVS 16384->4096  (same 1M batch):
     With ROLLOUT_STEPS=64 the GAE bootstrap bias was severe — the critic
     had to estimate V(s') at artificial 64-step truncations while still
     underfitting (V_loss ~2.0). Longer rollouts reduce this bootstrap bias
     and give each advantage estimate more episode context.

  3. VF_COEF 0.5->1.0:
     Critic was clearly underfitting (V_loss stuck at ~2.0 after 220 updates).
     Doubling the value loss coefficient drives faster critic convergence.

  4. N_MINIBATCHES 16->8  (same minibatch size 131k, 2x fewer PPO steps):
     Halves the PPO update wall time with no loss of effective coverage.

  5. LR linear decay over training for stability in later updates.
"""

import os

os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # PPO on GPU 0

import time
import warnings
import jax
import jax.numpy as jnp

jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_network import EndToEndActorCritic
from jax_train import collect_rollouts, init_env_state, NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE

# Hyperparameters
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 1.0      # was 0.5 — critic was underfitting
ENTROPY_COEF   = 0.002    # was 0.05 — was the cause of log_std saturation
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 4
LR_START       = 3e-4
LR_END         = 5e-5     # linear decay for late-training stability
TOTAL_UPDATES  = 600

# Batch config
BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS   # 4096 * 256 = 1,048,576
N_MINIBATCHES   = 8                          # was 16 — 2x faster update
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES  # 131,072

network   = EndToEndActorCritic(action_dim=2)


# Create a global schedule and optimizer
scheduler = optax.linear_schedule(
    init_value=LR_START,
    end_value=LR_END,
    transition_steps=TOTAL_UPDATES
)

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(learning_rate=scheduler, eps=1e-5),
)


# Checkpoint
def save_checkpoint(params, opt_state, filepath="checkpoints/ppo_model_best.msgpack"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {"params": jax.device_get(params), "opt_state": jax.device_get(opt_state)}
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Checkpoint -> {filepath}")


def load_checkpoint(dummy_params, dummy_opt_state,
                    filepath="checkpoints/ppo_model_best.msgpack"):
    with open(filepath, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes(
        {"params": dummy_params, "opt_state": dummy_opt_state}, raw
    )
    return bundle["params"], bundle["opt_state"]


# GAE (no reward normalisation — advantages normalised once in run_ppo_updates)
@jax.jit
def compute_gae(rewards, values, dones, last_val):
    """
    Standard GAE with reverse scan.
    last_done = zeros because autoreset already handled all episode boundaries.
    rewards, values, dones: (ROLLOUT_STEPS, NUM_ENVS)
    last_val: (NUM_ENVS,)
    """
    def _step(carry, t):
        gae, nv = carry
        r, v, d = t
        nd    = 1.0 - d
        delta = r + GAMMA * nv * nd - v
        gae   = delta + GAMMA * GAE_LAMBDA * nd * gae
        return (gae, v), gae

    _, adv = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_val), last_val),
        (rewards, values, dones.astype(jnp.float32)),
        reverse=True,
    )
    returns = adv + values
    return adv, returns


# PPO loss
@jax.jit
def ppo_loss_fn(params, obs, actions, advantages, returns, old_log_probs):
    mean, logstd, values = network.apply({"params": params}, obs)
    std = jnp.exp(logstd)

    # Log-prob in raw action space — consistent with collection
    z        = (actions - mean) / (std + 1e-8)
    log_prob = jnp.sum(-0.5 * (z ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)

    ratio       = jnp.exp(jnp.clip(log_prob - old_log_probs, -5.0, 5.0))
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages,
    ))

    value_loss = VF_COEF * jnp.mean((returns - values) ** 2)

    # Entropy: H = 0.5*log(2*pi*e) + logstd per dim, averaged over batch AND dims
    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -ENTROPY_COEF * entropy

    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss, (policy_loss, value_loss, entropy)


# On-device PPO epoch
# On-device PPO epoch
@jax.jit
def ppo_update_epoch(carry, perm):
    # REMOVED optimizer from the end of carry
    params, opt_state, obs, actions, adv, ret, old_lp = carry

    def _mb_step(mb_carry, mb_idx):
        p, os_ = mb_carry
        idx = jax.lax.dynamic_slice(perm, (mb_idx * MINI_BATCH_SIZE,), (MINI_BATCH_SIZE,))
        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, obs[idx], actions[idx], adv[idx], ret[idx], old_lp[idx]
        )
        updates, new_os = optimizer.update(grads, os_, p) # Uses global optimizer
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step, (params, opt_state), jnp.arange(N_MINIBATCHES)
    )
    # REMOVED optimizer from the return
    return (new_p, new_os, obs, actions, adv, ret, old_lp), (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_flat, actions_flat, adv_flat, ret_flat,
                    old_lp_flat, rng_key):  # REMOVED optimizer argument
    params, opt_state = train_state

    # Normalise advantages once (standard PPO)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    perms = jax.vmap(lambda k: jax.random.permutation(k, BATCH_SIZE))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )
    # REMOVED optimizer from carry
    carry = (params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


# Episode statistics
@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision):
    N = rewards.shape[1]

    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c = t
        ep_ret  = ep_ret + r
        timeout = d & ~g & ~c
        out_ret = jnp.where(d, ep_ret, 0.0)
        out_suc = jnp.where(d, g.astype(jnp.float32), 0.0)
        out_col = jnp.where(d, c.astype(jnp.float32), 0.0)
        out_tmo = jnp.where(d, timeout.astype(jnp.float32), 0.0)
        out_msk = d.astype(jnp.float32)
        ep_ret  = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_col, out_tmo, out_msk)

    _, (ep_rets, ep_suc, ep_col, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N),
        (rewards, dones, goal_reached, collision)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(), ep_tmo.ravel(), ep_msk.ravel()


# def update_rolling_window(buf, new_rets, new_suc, new_col, new_tmo, new_msk, write_ptr):
#     ep_rets = np.array(new_rets);   ep_suc = np.array(new_suc)
#     ep_col  = np.array(new_col);    ep_tmo = np.array(new_tmo)
#     ep_msk  = np.array(new_msk).astype(bool)
#     ep_data = np.stack([ep_rets, ep_suc, ep_col, ep_tmo], axis=1)[ep_msk]
#     for row in ep_data:
#         buf[write_ptr % STATS_WINDOW] = row
#         write_ptr += 1
#     return buf, write_ptr


# def compute_window_stats(buf, write_ptr):
#     n = min(write_ptr, STATS_WINDOW)
#     if n == 0:
#         return 0.0, 0.0, 0.0, 0.0
#     v = buf[:n]
#     return float(v[:, 0].mean()), float(v[:, 1].mean()) * 100.0, \
#            float(v[:, 2].mean()) * 100.0, float(v[:, 3].mean()) * 100.0


if __name__ == "__main__":

    print("PPO Training — GPU 0")
    print(f"  Envs       : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches: {N_MINIBATCHES} x {MINI_BATCH_SIZE} | epochs {PPO_EPOCHS}")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}  LR {LR_START}->{LR_END}")
    print(f"  log_std: state-dependent Dense head, bias=-1.0, clamp [{-4.0},{0.5}]\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    opt_state    = optimizer.init(params)
    train_state  = (params, opt_state)

    ckpt_path       = "checkpoints/ppo_model_best.msgpack"
    LOAD_CHECKPOINT = False
    if LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        try:
            params, opt_state = load_checkpoint(params, opt_state, ckpt_path)
            train_state = (params, opt_state)
            print("Resumed from checkpoint.")
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh.")
    else:
        print("Starting fresh.")

    print("Initialising environments...")
    env_obs, env_state = init_env_state(env_rng)
    print(f"Ready. obs={env_obs.shape}\n")

    # stat_buf  = np.zeros((STATS_WINDOW, 4), dtype=np.float32)
    # write_ptr = 0
    best_suc  = 25.0

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Col%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} | {'FPS':>7} {'#Ep':>6}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        # Linear LR decay
        # frac      = update / max(TOTAL_UPDATES - 1, 1)
        # lr_now    = LR_START + frac * (LR_END - LR_START)
        # optimizer = make_optimizer(lr_now)

        # 1. Rollout
        rng, rollout_rng, update_rng = jax.random.split(rng, 3)
        rollout_history, env_state, env_obs = collect_rollouts(
            rollout_rng, train_state[0], network.apply, env_state, env_obs
        )

        rewards      = rollout_history["rewards"]
        values       = rollout_history["values"]
        dones        = rollout_history["dones"]
        obs_all      = rollout_history["obs"]
        acts_all     = rollout_history["actions"]
        lp_all       = rollout_history["log_probs"]
        goal_reached = rollout_history["goal_reached"]
        collision    = rollout_history["collision"]

        # 2. Episode stats
        # 2. Episode stats (Instantaneous batch window)
        ep_rets, ep_suc, ep_col, ep_tmo, ep_msk = collect_episode_outcomes(
            rewards, dones, goal_reached, collision
        )
        
        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            # Vectorized average of the ~19,000 episodes completed in THIS update
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret, suc_pct, col_pct, tmo_pct = 0.0, 0.0, 0.0, 0.0

        # 3. GAE (last_done=zeros: autoreset already handled boundaries)
        _, _, last_val = network.apply({"params": train_state[0]}, env_obs)
        last_done      = jnp.zeros(NUM_ENVS, dtype=jnp.float32)
        advantages, returns = compute_gae(rewards, values, dones, last_val)

        # 4. PPO update
        train_state, mean_loss, aux = run_ppo_updates(
            train_state,
            obs_all.reshape(-1, OBS_SIZE),
            acts_all.reshape(-1, 2),
            advantages.reshape(-1),
            returns.reshape(-1),
            lp_all.reshape(-1),
            update_rng
        )

        fps = BATCH_SIZE / (time.time() - t0)

        # 5. Log
        if update % 10 == 0:
            p_loss, v_loss, entropy = aux
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.4f} {float(p_loss):>6.3f} "
                f"{float(v_loss):>6.3f} {float(entropy):>6.3f} | "
                f"{fps:>7,.0f} {n_ep:>6d}"
            )

        if suc_pct > best_suc:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")