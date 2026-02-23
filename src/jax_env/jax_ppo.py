"""
jax_ppo.py — Single-GPU PPO Training  (GPU 0)
===============================================
FIXES & IMPROVEMENTS vs previous version:

  FIX (carried) — removed dead-code last_done variable.

  FIX (carried) — BATCH_SIZE % N_MINIBATCHES assertion.

  IMPROVEMENT A (carried) — LR warmup schedule.

  IMPROVEMENT B (carried) — ENTROPY_COEF shared constant.

  IMPROVEMENT C (carried) — OBS_SIZE = 342.

  30-MIN TRAINING OVERHAUL (NEW):
    Hyperparameters retuned for ~30-minute wall-clock convergence at 110k FPS.

    Key changes:
      ROLLOUT_STEPS : 256  → 64    (4× more frequent updates)
      TOTAL_UPDATES : 600  → 400   (compensated by richer per-update signal)
      N_MINIBATCHES : 8   → 4     (larger minibatches, less overhead)
      PPO_EPOCHS    : 4   → 6     (reuse each batch more aggressively)
      WARMUP_UPDATES: 20  → 5     (fast ramp-up)
      LR_START      : 3e-4 → 5e-4  (more aggressive early learning)
      LR_END        : 5e-5 → 1e-4  (keep learning rate meaningful at end)
      LR_MIN        : 1e-6 → 1e-5  (warmup floor)
      ENTROPY_COEF  : 0.002 → 0.005 (more exploration — critical for curriculum)
      CLIP_EPS      : 0.2  → 0.25  (slightly more permissive updates)
      best_suc init : 25.0 → 10.0  (save checkpoint earlier)

  CURRICULUM (NEW):
    Goal distance starts small and grows with the rolling success rate.
    Controlled by curriculum_min_goal_dist() which maps suc_pct to a distance.
    Passed as a static arg to a specialised reset wrapper in jax_train.py.

      Stage 0  suc <  30% : min_dist = 1.0 m   (trivial warm-up)
      Stage 1  suc <  55% : min_dist = 2.5 m   (medium range)
      Stage 2  suc <  70% : min_dist = 4.5 m   (near full range)
      Stage 3  suc >= 70% : min_dist = 6.0 m   (full difficulty)

    Because min_goal_dist changes the JIT signature of reset_env (static float),
    we use a small Python-level dispatch table and retrace only when the stage
    changes (at most 3 retraces total across the whole run).
"""

import os

os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.25    # was 0.2
VF_COEF        = 0.25#1.0
ENTROPY_COEF   = 0.01   # was 0.002 — more exploration for curriculum
MAX_GRAD_NORM  = 0.3
PPO_EPOCHS     = 4       # was 4 — reuse data more
LR_START       = 3e-4    # was 3e-4
LR_END         = 1e-4    # was 5e-5
LR_MIN         = 1e-5    # was 1e-6
WARMUP_UPDATES = 10       # was 20
TOTAL_UPDATES  = 400     # was 600

# Batch config
BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS
N_MINIBATCHES   = 8      # was 8 — larger minibatches
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES

assert BATCH_SIZE % N_MINIBATCHES == 0, (
    f"BATCH_SIZE={BATCH_SIZE} not divisible by N_MINIBATCHES={N_MINIBATCHES}."
)

network = EndToEndActorCritic(action_dim=2)

# ── Curriculum ────────────────────────────────────────────────────────────────
# Maps rolling success rate → minimum goal distance for reset_env.
# Stages advance monotonically as the agent improves.
CURRICULUM_STAGES = [
    (15.0, 1.0),   # avanza già a 15% — non aspettare 30%
    (40.0, 2.5),
    (60.0, 4.5),
    (101., 6.0),
]

def curriculum_min_goal_dist(suc_pct: float) -> float:
    """Return min_goal_dist for the current success rate."""
    for threshold, dist in CURRICULUM_STAGES:
        if suc_pct < threshold:
            return dist
    return CURRICULUM_STAGES[-1][1]

# IMPROVEMENT A: piecewise warmup then linear decay
_warmup_schedule = optax.linear_schedule(
    init_value=LR_MIN,
    end_value=LR_START,
    transition_steps=WARMUP_UPDATES,
)
_decay_schedule = optax.linear_schedule(
    init_value=LR_START,
    end_value=LR_END,
    transition_steps=TOTAL_UPDATES - WARMUP_UPDATES,
)
scheduler = optax.join_schedules(
    schedules=[_warmup_schedule, _decay_schedule],
    boundaries=[WARMUP_UPDATES],
)

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(learning_rate=scheduler, eps=1e-5),
)


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


@jax.jit
def compute_gae(rewards, values, dones, last_val):
    """
    Standard GAE with reverse scan.
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


@jax.jit
def ppo_loss_fn(params, obs, actions, advantages, returns, old_log_probs):
    mean, logstd, values = network.apply({"params": params}, obs)
    std = jnp.exp(logstd)

    z        = (actions - mean) / (std + 1e-8)
    log_prob = jnp.sum(-0.5 * (z ** 2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)

    ratio       = jnp.exp(jnp.clip(log_prob - old_log_probs, -5.0, 5.0))
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages,
    ))

    value_loss = VF_COEF * jnp.mean((returns - values) ** 2)

    entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
    entropy_loss = -ENTROPY_COEF * entropy

    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss, (policy_loss, value_loss, entropy)


@jax.jit
def ppo_update_epoch(carry, perm):
    params, opt_state, obs, actions, adv, ret, old_lp = carry

    def _mb_step(mb_carry, mb_idx):
        p, os_ = mb_carry
        idx = jax.lax.dynamic_slice(perm, (mb_idx * MINI_BATCH_SIZE,), (MINI_BATCH_SIZE,))
        (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
            p, obs[idx], actions[idx], adv[idx], ret[idx], old_lp[idx]
        )
        updates, new_os = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, updates), new_os), (loss, aux)

    (new_p, new_os), (losses, auxes) = jax.lax.scan(
        _mb_step, (params, opt_state), jnp.arange(N_MINIBATCHES)
    )
    return (new_p, new_os, obs, actions, adv, ret, old_lp), (losses, auxes)


@jax.jit
def run_ppo_updates(train_state, obs_flat, actions_flat, adv_flat, ret_flat,
                    old_lp_flat, rng_key):
    params, opt_state = train_state

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    perms = jax.vmap(lambda k: jax.random.permutation(k, BATCH_SIZE))(
        jax.random.split(rng_key, PPO_EPOCHS)
    )
    carry = (params, opt_state, obs_flat, actions_flat, adv_flat, ret_flat, old_lp_flat)
    carry, (all_losses, all_auxes) = jax.lax.scan(ppo_update_epoch, carry, perms)
    last_aux = jax.tree_util.tree_map(lambda x: x[-1, -1], all_auxes)
    return (carry[0], carry[1]), all_losses.mean(), last_aux


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


if __name__ == "__main__":

    print("PPO Training — GPU 0  [30-min mode]")
    print(f"  Envs       : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches: {N_MINIBATCHES} x {MINI_BATCH_SIZE} | epochs {PPO_EPOCHS}")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}  LR warmup {LR_MIN}->{LR_START} then decay ->{LR_END}")
    print(f"  OBS_SIZE={OBS_SIZE}  log_std: state-dependent, bias=-1.0, clamp [{-4.0},{0.5}]")
    print(f"  Curriculum stages: {CURRICULUM_STAGES}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

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

    # ── Curriculum state ──────────────────────────────────────────────────────
    cur_min_dist  = curriculum_min_goal_dist(0.0)   # start at easiest stage
    cur_stage     = 0
    rolling_suc   = 0.0   # exponential moving average of success rate

    print(f"Curriculum: starting at min_goal_dist={cur_min_dist:.1f} m")

    print("Initialising environments...")
    env_obs, env_state = init_env_state(env_rng, min_goal_dist=cur_min_dist)
    print(f"Ready. obs={env_obs.shape}\n")

    best_suc = 60.0   # was 25.0 — save checkpoint as soon as we reach 10%

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Col%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} | {'FPS':>7} {'#Ep':>6} | {'MinDist':>7}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

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

        ep_rets, ep_suc, ep_col, ep_tmo, ep_msk = collect_episode_outcomes(
            rewards, dones, goal_reached, collision
        )

        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret, suc_pct, col_pct, tmo_pct = 0.0, 0.0, 0.0, 0.0

        # ── Curriculum update ─────────────────────────────────────────────────
        # Exponential moving average: alpha=0.1 → ~10-update lag → stable transitions
        if n_ep > 0:
            rolling_suc = 0.97 * rolling_suc + 0.03 * suc_pct

        new_min_dist = curriculum_min_goal_dist(rolling_suc)
        if new_min_dist > cur_min_dist:
            cur_min_dist = new_min_dist
            new_stage    = next(i for i, (t, _) in enumerate(CURRICULUM_STAGES) if rolling_suc < t)
            
            rng, reinit_rng = jax.random.split(rng)
            env_obs, env_state = init_env_state(reinit_rng, min_goal_dist=cur_min_dist)

        _, _, last_val = network.apply({"params": train_state[0]}, env_obs)
        advantages, returns = compute_gae(rewards, values, dones, last_val)

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

        if update % 10 == 0:
            p_loss, v_loss, entropy = aux
            lr_now = float(scheduler(update))
            print(
                f"{update:>5d} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(mean_loss):>7.4f} {float(p_loss):>6.3f} "
                f"{float(v_loss):>6.3f} {float(entropy):>6.3f} | "
                f"{fps:>7,.0f} {n_ep:>6d}  lr={lr_now:.2e} | {cur_min_dist:>5.1f}m"
            )

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")