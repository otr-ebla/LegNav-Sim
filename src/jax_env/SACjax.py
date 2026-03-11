"""
SACjax.py — Soft Actor-Critic
=========================================
Pinned to GPU 1 via CUDA_VISIBLE_DEVICES=1 so it runs alongside PPO on GPU 0.

SAC (Haarnoja et al. 2018) with:
  - Twin critics + target networks (soft update tau=0.005)
  - Squashed Gaussian policy with exact tanh Jacobian log-prob correction
  - Automatic temperature (alpha) tuning to maintain target entropy = -|A|
  - On-GPU circular replay buffer (500k transitions)
  - Vectorised collection across N_ENVS parallel environments

Action space:
  v in [0, max_v]: a_v = (tanh(u_v) + 1) / 2 * max_v
  w in [-1,   1 ]: a_w = tanh(u_w)
  max_v varies per episode; recovered from obs[..., MAX_V_OBS_IDX].

ARCHITECTURE — Fully-fused GPU loop (inspired by TQC implementation):

  The entire inner loop — env step, buf_add, buf_sample, sac_update — runs
  inside a single jax.lax.scan of length LOG_EVERY. One Python call dispatches
  LOG_EVERY iterations with zero host syncs. This matches TQC's train_chunk
  pattern and is the primary reason TQC reached 250k+ FPS while the previous
  SAC architecture (separate collect / Python buf_add loop / update_n) was
  bottlenecked at ~40k FPS despite equivalent hardware.

  Key changes vs previous version:

  FUSE 1 — train_chunk: single scan over collect + buf_add + buf_sample + update.
    The replay buffer is now part of the scan carry so buf_add and buf_sample
    happen on-GPU with no host roundtrip. The previous code called buf_add in
    a Python for-loop (16 host syncs per iter) and ran collect and update_n as
    separate JIT dispatches (2 more syncs). All replaced by 1 dispatch.

  FUSE 2 — collect_step: single-step collection helper (no STEPS_PER_ITER scan).
    TQC's approach: collect 1 step per scan iteration rather than batching N
    steps and returning them all. This avoids materialising a large
    (STEPS_PER_ITER, N_ENVS, OBS_SIZE) output tensor that caused the OOM.

  FUSE 3 — on-GPU episode stats via collect_episode_outcomes (ported from TQC).
    Replaces update_stats_batch which pulled (T, N_ENVS) arrays to CPU via
    np.array() every iteration — a ~45 MB device->host transfer per iter.
    Stats now accumulate and reduce entirely on-GPU; only 5 scalars cross
    the PCIe bus per LOG_EVERY updates.

  RETAINED — sample_action_sac_batched, batched critic/actor loss,
    gradient clipping, reward normalisation, checkpoint format.

  TUNING — N_ENVS=2048, BUFFER=500k, BATCH=1024 matching TQC defaults.
    These are proven to fit in VRAM alongside the fused scan without OOM.
"""

import os
import csv
os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import time
import functools
import warnings
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import flax.serialization

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_SIZE       = 342        # 9 (pose) + 9 (state_vec) + 324 (lidar)
ACTION_DIM     = 2
# N_ENVS/BUFFER/BATCH match TQC proven defaults — fit in VRAM with fused scan.
N_ENVS         = 2048
BUFFER_CAP     = 500_000
BATCH_SIZE     = 1024
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)   # -2.0
TOTAL_UPDATES  = 100_000
LOG_EVERY      = 500        # scan length inside train_chunk; also log interval
SAVE_EVERY     = 5000

# Reward normalisation: env emits rewards in [-70, +200].
REWARD_SCALE   = 50.0

# Gradient clipping
MAX_GRAD_NORM  = 10.0

# pose_stack indices 0-8; state_vec[2] = max_v/2 at flat index 9+2 = 11
MAX_V_OBS_IDX  = 11
LOG_STD_EPS    = 1e-6

CKPT_DIR  = "checkpoints_sac"
CKPT_PATH = f"{CKPT_DIR}/sac_best.msgpack"


# ── GPU check ─────────────────────────────────────────────────────────────────
def _check_gpu():
    num_devices = jax.device_count(backend='gpu')
    if num_devices >= 2:
        target_gpu = jax.devices('gpu')[0]
        print(f"Found {num_devices} GPU(s). Using CudaDevice(0)")
    elif num_devices == 1:
        target_gpu = jax.devices('gpu')[0]
        print(f"Found {num_devices} GPU(s). Using CudaDevice(0)")
    else:
        raise RuntimeError("No CUDA devices found.")
    return target_gpu

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)


# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

def init_env_state(rng_key, min_goal_dist: float = 3.0):
    """Build vmapped step/reset, initialise all envs. Returns (obs, state, vmap_step)."""
    step_auto  = make_autoreset_env(reset_stacked, step_stacked, min_goal_dist=min_goal_dist)
    vmap_step  = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))
    def _reset(key): return reset_stacked(key, min_goal_dist=min_goal_dist)
    vmap_reset = jax.jit(jax.vmap(_reset))
    env_obs, env_state = vmap_reset(jax.random.split(rng_key, N_ENVS))
    return env_obs, env_state, vmap_step


# ── Shared obs encoder ────────────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x):
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 9

        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]

        batch_shape = lidar_flat.shape[:-1]
        lidar_cnn   = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim))
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME')(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME')(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME')(cnn))
        cnn_feat = nn.LayerNorm()(cnn.reshape((*batch_shape, -1)))

        global_in   = jnp.concatenate([pose_stack, state_vec], axis=-1)
        global_feat = nn.relu(nn.Dense(128)(global_in))
        global_feat = nn.relu(nn.Dense(64)(global_feat))

        fused  = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256)(fused))
        return nn.relu(nn.Dense(128)(shared))


# ── Actor ─────────────────────────────────────────────────────────────────────
class SACActorNetwork(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0

    @nn.compact
    def __call__(self, obs):
        feat    = ObsEncoder()(obs)
        mean    = nn.Dense(self.action_dim)(feat)
        log_std = nn.Dense(self.action_dim)(feat)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std


# ── Twin critic ───────────────────────────────────────────────────────────────
class SACCriticNetwork(nn.Module):
    action_dim: int = ACTION_DIM

    @nn.compact
    def __call__(self, obs, action):
        # Separate encoders — no shared weights between Q1 and Q2 (standard SAC)
        feat1 = ObsEncoder(name='enc_q1')(obs)
        feat2 = ObsEncoder(name='enc_q2')(obs)

        q1_in = jnp.concatenate([feat1, action], axis=-1)
        q2_in = jnp.concatenate([feat2, action], axis=-1)

        q1 = nn.relu(nn.Dense(256, name='q1_l1')(q1_in))
        q1 = nn.relu(nn.Dense(128, name='q1_l2')(q1))
        q1 = jnp.squeeze(nn.Dense(1, name='q1_out')(q1), axis=-1)

        q2 = nn.relu(nn.Dense(256, name='q2_l1')(q2_in))
        q2 = nn.relu(nn.Dense(128, name='q2_l2')(q2))
        q2 = jnp.squeeze(nn.Dense(1, name='q2_out')(q2), axis=-1)

        return q1, q2


actor_net  = SACActorNetwork()
critic_net = SACCriticNetwork()

actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
alpha_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))


# ── Action squashing + exact log-prob ─────────────────────────────────────────

def _tanh_log_prob_correction(tanh_u, max_v):
    corr_v = (
        jnp.log(max_v * 0.5 + LOG_STD_EPS)
        + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    )
    corr_w = jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    return -(corr_v + corr_w)


def sample_action_sac_batched(rng_key, mean, log_std, max_v):
    """
    Fully batched reparameterised sample — one random.normal call, no vmap.
    mean, log_std : (N, 2)   max_v : (N,)
    Returns: env_action (N, 2), log_pi (N,)
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise**2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)
    tanh_u = jnp.tanh(u)
    a_v = (tanh_u[:, 0] + 1.0) * 0.5 * max_v
    a_w = tanh_u[:, 1]
    return jnp.stack([a_v, a_w], axis=-1), lp_gauss + _tanh_log_prob_correction(tanh_u, max_v)


@jax.jit
def extract_max_v(obs):
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular) ───────────────────────────────────────────
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),             jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),   jnp.float32),
        "done":     jnp.zeros((capacity,),             jnp.float32),
        "ptr":      jnp.int32(0),
        "size":     jnp.int32(0),
    }


@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done):
    cap  = buf["obs"].shape[0]
    N    = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs":      buf["obs"].at[idxs].set(obs),
        "action":   buf["action"].at[idxs].set(action),
        "reward":   buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs),
        "done":     buf["done"].at[idxs].set(done),
        "ptr":      jnp.int32((buf["ptr"] + N) % cap),
        "size":     jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    capacity = buf["obs"].shape[0]
    idxs = jax.random.randint(rng_key, (batch_size,), 0, capacity)
    idxs = jnp.where(idxs < buf["size"], idxs, jnp.int32(0))
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs],
            buf["next_obs"][idxs], buf["done"][idxs])


# ── Soft target update ────────────────────────────────────────────────────────
@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(
        lambda t, o: TAU * o + (1.0 - TAU) * t, target, online
    )


# ── SAC update step ───────────────────────────────────────────────────────────
@jax.jit
def sac_update(ap, aos, cp, cos, tp, la, alo,
               obs, action, reward, next_obs, done, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    # 1. Critic — Bellman backup with clipped double-Q
    def _critic_loss(critic_params):
        alpha      = jnp.exp(la)
        next_max_v = extract_max_v(next_obs)
        mean_n, lgs_n = actor_net.apply({"params": ap}, next_obs)
        next_act, next_lp = sample_action_sac_batched(rng_c, mean_n, lgs_n, next_max_v)
        q1_t, q2_t = critic_net.apply({"params": tp}, next_obs, next_act)
        v_next  = jnp.minimum(q1_t, q2_t) - alpha * next_lp
        backup  = jax.lax.stop_gradient(
            reward / REWARD_SCALE + GAMMA * (1.0 - done) * v_next
        )
        q1, q2  = critic_net.apply({"params": critic_params}, obs, action)
        loss    = jnp.mean((q1 - backup)**2) + jnp.mean((q2 - backup)**2)
        return loss, jnp.mean(backup) * REWARD_SCALE

    (c_loss, q_mean), c_grads = jax.value_and_grad(_critic_loss, has_aux=True)(cp)
    c_upd, new_cos = critic_opt.update(c_grads, cos, cp)
    new_cp = optax.apply_updates(cp, c_upd)

    # 2. Actor — minimise E[alpha*log_pi - min(Q1,Q2)]
    def _actor_loss(actor_params):
        alpha = jnp.exp(la)
        max_v = extract_max_v(obs)
        mean, log_std = actor_net.apply({"params": actor_params}, obs)
        action_new, log_pi = sample_action_sac_batched(rng_a, mean, log_std, max_v)
        q1, q2 = critic_net.apply({"params": new_cp}, obs, action_new)
        return jnp.mean(alpha * log_pi - jnp.minimum(q1, q2)), jnp.mean(log_pi)

    (a_loss, log_pi_mean), a_grads = jax.value_and_grad(_actor_loss, has_aux=True)(ap)
    a_upd, new_aos = actor_opt.update(a_grads, aos, ap)
    new_ap = optax.apply_updates(ap, a_upd)

    # 3. Alpha — tune temperature toward target entropy
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_grad   = jax.grad(lambda a: -a * (log_pi_sg + TARGET_ENTROPY))(la)
    al_upd, new_alo = alpha_opt.update(al_grad, alo)
    new_la    = optax.apply_updates(la, al_upd)

    # 4. Soft target update
    new_tp = soft_update(tp, new_cp)

    metrics = {
        "critic_loss": c_loss,
        "actor_loss":  a_loss,
        "alpha":       jnp.exp(new_la),
        "log_pi":      log_pi_mean,
        "q_mean":      q_mean,
    }
    return new_ap, new_aos, new_cp, new_cos, new_tp, new_la, new_alo, metrics


# ── Single collection step ────────────────────────────────────────────────────
# FUSE 2: one step per scan iteration — no large output tensor materialised.
@functools.partial(jax.jit, static_argnums=(4,))
def collect_step(actor_params, env_state, env_obs, rng_key, vmap_step):
    mean, log_std = actor_net.apply({"params": actor_params}, env_obs)
    env_action, _ = sample_action_sac_batched(
        rng_key, mean, log_std, extract_max_v(env_obs)
    )
    # fold_in avoids an extra split call; gives a distinct key for env stepping
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 1), N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(step_keys, env_state, env_action)
    return new_obs, new_state, env_obs, env_action, reward, done, info


# ── Fused GPU train chunk ─────────────────────────────────────────────────────
# FUSE 1: collect + buf_add + buf_sample + sac_update inside one lax.scan.
# The replay buffer lives in the carry — no host transfers, zero Python syncs.
@functools.partial(jax.jit, static_argnums=(8,))
def train_chunk(ap, aos, cp, cos, tp, la, alo, buf, vmap_step, es, eo, key):
    """
    Execute LOG_EVERY steps of:
      collect 1 env step -> add to buffer -> sample batch -> SAC gradient update.
    Everything runs on-GPU inside a single jax.lax.scan — one Python dispatch,
    no host syncs until the scan completes.
    """
    def _loop_body(carry, _):
        ap, aos, cp, cos, tp, la, alo, buf, es, eo, key = carry
        key, k_col, k_samp, k_upd = jax.random.split(key, 4)
        # Collect one step
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(
            ap, es, eo, k_col, vmap_step
        )
        
        # Calculate true terminal state (ignore timeouts for Bellman backup)
        terminal = done & ~info["timeout"]
        
        # Add to replay buffer using 'terminal' instead of 'done'
        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        # Sample a training batch
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(new_buf, k_samp, BATCH_SIZE)
        # SAC gradient update
        new_ap, new_aos, new_cp, new_cos, new_tp, new_la, new_alo, metrics = sac_update(
            ap, aos, cp, cos, tp, la, alo,
            b_obs, b_act, b_rew, b_next, b_done, k_upd
        )

        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_ap, new_aos, new_cp, new_cos, new_tp, new_la, new_alo,
                new_buf, new_es, new_eo, key), (step_data, metrics)

    carry = (ap, aos, cp, cos, tp, la, alo, buf, es, eo, key)
    new_carry, (all_step_data, all_metrics) = jax.lax.scan(
        _loop_body, carry, None, length=LOG_EVERY
    )
    return new_carry, all_step_data, all_metrics


# ── On-GPU episode stats ───────────────────────────────────────────────────────
# FUSE 3: accumulation runs entirely on-GPU. Only 5 scalars cross PCIe per chunk.
@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    """
    Accumulate per-env episode returns over (LOG_EVERY, N_ENVS) step data on-GPU.
    Returns flat arrays over all completed episodes in the chunk.
    """
    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p = t
        ep_ret  = ep_ret + r
        act_col = c & ~p
        is_suc  = g
        is_acol = act_col & ~g
        is_pcol = p & ~g
        is_tmo  = d & ~g & ~act_col & ~p
        out_ret  = jnp.where(d, ep_ret,                       0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_col  = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        return jnp.where(d, 0.0, ep_ret), (out_ret, out_suc, out_col, out_pcol, out_tmo,
                                            d.astype(jnp.float32))

    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan,
        jnp.zeros(rewards.shape[1]),
        (rewards, dones, goal_reached, collision, passive_col),
    )
    return (ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(),
            ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel())


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(ap, cp, tp, aos, cos, la, alo, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "actor_params":     jax.device_get(ap),
        "critic_params":    jax.device_get(cp),
        "target_params":    jax.device_get(tp),
        "actor_opt_state":  jax.device_get(aos),
        "critic_opt_state": jax.device_get(cos),
        "log_alpha":        jax.device_get(la),
        "alpha_opt_state":  jax.device_get(alo),
        "step":             int(step),
    }
    with open(CKPT_PATH, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("SAC Training")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}\n")

    rng = jax.random.PRNGKey(7)
    rng, ka, kc = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((2, OBS_SIZE))
    dummy_act = jnp.zeros((2, ACTION_DIM))

    ap  = actor_net.init(ka, dummy_obs)["params"]
    cp  = critic_net.init(kc, dummy_obs, dummy_act)["params"]
    tp  = jax.tree_util.tree_map(jnp.array, cp)
    aos = actor_opt.init(ap)
    cos = critic_opt.init(cp)
    la  = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    alo = alpha_opt.init(la)

    print("Initialising environments...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state, vmap_step = init_env_state(env_rng)
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    n_updates   = 0
    best_suc    = 20.0

    # ── Warmup: fill buffer with random actions before training ───────────────
    print("Warming up buffer...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        rng, c_rng = jax.random.split(rng)
        new_obs, env_state, obs_before, env_action, reward, done, info = \
            collect_step(ap, env_state, env_obs, c_rng, vmap_step)
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             new_obs, done.astype(jnp.float32))
        env_obs     = new_obs
        total_steps += N_ENVS
    print("Warmup done. JIT compiling train_chunk (this may take ~1 min)...")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | "
           f"{'FPS':>7} | {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()

    # ── Training log (CSV) ───────────────────────────────────────────────────
    _LOG_PATH = "checkpoints_sac/sac_training_log.csv"
    os.makedirs("checkpoints_sac", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()

        # ── Single fused GPU dispatch: LOG_EVERY steps of everything ──────────
        new_carry, all_step_data, all_metrics = train_chunk(
            ap, aos, cp, cos, tp, la, alo,
            replay_buf, vmap_step, env_state, env_obs, rng
        )
        ap, aos, cp, cos, tp, la, alo, replay_buf, env_state, env_obs, rng = new_carry

        n_updates   += LOG_EVERY
        total_steps += N_ENVS * LOG_EVERY

        # ── On-GPU stats reduction (FUSE 3) — only scalars cross PCIe ─────────
        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = \
            collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())

        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col  * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        fps     = (N_ENVS * LOG_EVERY) / (time.time() - t0)
        elapsed = time.time() - t_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:d}m{s:02d}s"

        print(
            f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
            f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
            f"{float(all_metrics['critic_loss'].mean()):>7.4f} "
            f"{float(all_metrics['actor_loss'].mean()):>7.4f} "
            f"{float(all_metrics['alpha'].mean()):>6.4f} "
            f"{float(all_metrics['log_pi'].mean()):>6.3f} "
            f"{float(all_metrics['q_mean'].mean()):>7.3f} | "
            f"{fps:>7,.0f} | "
            f"{elapsed_str:>8}"
        )
        # ── CSV log row ──────────────────────────────────────────────────────
        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        if n_updates % SAVE_EVERY == 0 and suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(ap, cp, tp, aos, cos, la, alo, n_updates)

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")