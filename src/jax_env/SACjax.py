"""
jax_sac.py — Soft Actor-Critic  (GPU 1)
=========================================
Pinned to GPU 1 via CUDA_VISIBLE_DEVICES=1 so it runs alongside PPO on GPU 0.

SAC (Haarnoja et al. 2018) with:
  - Twin critics + target networks (soft update tau=0.005)
  - Squashed Gaussian policy with exact tanh Jacobian log-prob correction
  - Automatic temperature (alpha) tuning to maintain target entropy = -|A|
  - On-GPU circular replay buffer (500k transitions, ~1.4 GB)
  - Vectorised collection across N_ENVS parallel environments

Action space:
  v in [0, max_v]: a_v = (tanh(u_v) + 1) / 2 * max_v
  w in [-1,   1 ]: a_w = tanh(u_w)
  max_v varies per episode; recovered from obs[11] = max_v/2 (state_vec[2]).

Log-prob correction (change of variables for tanh squashing):
  log pi(a|s) = log N(u; mean, std)
               - log(max_v/2 + eps)
               - log(1 - tanh^2(u_v) + eps)   [v-dim Jacobian]
               - log(1 - tanh^2(u_w) + eps)   [w-dim Jacobian]
"""

import os

# ── PIN TO GPU 1 ──────────────────────────────────────────────────────────────
os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"

import time
import warnings
import jax
import jax.numpy as jnp

#jax.config.update("jax_default_device", jax.devices("cuda")[0])  # device 0 inside visible set

import optax
import flax
import flax.linen as nn
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_env import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_SIZE       = 339
ACTION_DIM     = 2
N_ENVS         = 2048
BUFFER_CAP     = 500_000
BATCH_SIZE     = 1024
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)   # -2.0
TOTAL_UPDATES  = 100_000
LOG_EVERY      = 500
SAVE_EVERY     = 5000
STATS_WINDOW   = 300
# obs[11] = max_v / 2  (pose_stack=9 bytes, then state_vec[2])
MAX_V_OBS_IDX  = 11
LOG_STD_EPS    = 1e-6

CKPT_DIR  = "checkpoints_sac"
CKPT_PATH = f"{CKPT_DIR}/sac_best.msgpack"

# ── GPU check ─────────────────────────────────────────────────────────────────
def _check_gpu():
    try:
        devs = jax.devices("cuda")
    except RuntimeError:
        devs = []
    
    if len(devs) < 2:
        raise RuntimeError("Less than 2 CUDA devices found. SAC requires CudaDevice(1).")
        
    target_device = devs[1] # Explicitly grab the second GPU
    print(f"SAC pinned to: {target_device}  (physical GPU 1)")
    return target_device

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)

# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto   = make_autoreset_env(reset_stacked, step_stacked)
_vmap_reset = jax.jit(jax.vmap(reset_stacked))
_vmap_step  = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))


# ── Shared obs encoder ────────────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x):
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 6

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
        shared = nn.relu(nn.Dense(128)(shared))
        return shared


# ── Actor ─────────────────────────────────────────────────────────────────────
class SACActorNetwork(nn.Module):
    action_dim: int = ACTION_DIM
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
        # Separate encoders for Q1 and Q2 (no shared weights between twins)
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

actor_opt  = optax.adam(LR, eps=1e-5)
critic_opt = optax.adam(LR, eps=1e-5)
alpha_opt  = optax.adam(LR, eps=1e-5)


# ── Action squashing + exact log-prob ─────────────────────────────────────────
def sample_action_sac(rng_key, mean, log_std, max_v):
    """
    Reparameterised sample with exact tanh Jacobian log-prob correction.
    Returns: env_action (2,), log_pi scalar, pre-squash u (2,)
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std

    # Gaussian log-prob via noise (numerically cleaner)
    lp_gauss = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)

    tanh_u = jnp.tanh(u)

    # v: shift tanh [-1,1] to [0, max_v]
    a_v = (tanh_u[..., 0] + 1.0) * 0.5 * max_v
    a_w = tanh_u[..., 1]
    env_action = jnp.stack([a_v, a_w], axis=-1)

    # Jacobian correction:
    #   v-dim: d/du [(tanh+1)/2 * max_v] = max_v/2 * (1-tanh^2)
    #   w-dim: d/du [tanh(u)]             = 1 - tanh^2
    corr_v = (-jnp.log(max_v * 0.5 + LOG_STD_EPS)
               - jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS))
    corr_w = -jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    log_pi = lp_gauss + corr_v + corr_w

    return env_action, log_pi, u


def get_det_action(mean, max_v):
    """Deterministic eval action (no noise)."""
    tanh_mean = jnp.tanh(mean)
    a_v = (tanh_mean[..., 0] + 1.0) * 0.5 * max_v
    a_w = tanh_mean[..., 1]
    return jnp.stack([a_v, a_w], axis=-1)


@jax.jit
def extract_max_v(obs):
    """obs[..., 11] = max_v / 2 (from state_vec[2] in stacked obs)."""
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Replay buffer (on-GPU circular) ───────────────────────────────────────────
def make_buffer(capacity):
    return {
        "obs":      jnp.zeros((capacity, OBS_SIZE),  jnp.float32),
        "action":   jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward":   jnp.zeros((capacity,),            jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE),  jnp.float32),
        "done":     jnp.zeros((capacity,),            jnp.float32),
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
    idxs = jax.random.randint(rng_key, (batch_size,), 0, buf["size"])
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs],
            buf["next_obs"][idxs], buf["done"][idxs])


# ── Soft target update ────────────────────────────────────────────────────────
@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(
        lambda t, o: TAU * o + (1.0 - TAU) * t, target, online
    )


# ── Critic loss ───────────────────────────────────────────────────────────────
@jax.jit
def critic_loss_fn(critic_params, target_params, actor_params, log_alpha,
                   obs, action, reward, next_obs, done, rng_key):
    """
    Bellman backup with clipped double-Q:
      y = r + gamma*(1-d)*[min(Q1_t,Q2_t)(s',a') - alpha*log_pi(a'|s')]
    a' sampled fresh from current actor (not from replay buffer).
    backup is stop_gradient'd — no gradient through target or actor here.
    """
    alpha      = jnp.exp(log_alpha)
    next_max_v = extract_max_v(next_obs)

    mean_n, lgs_n = actor_net.apply({"params": actor_params}, next_obs)
    rng_acts = jax.random.split(rng_key, next_obs.shape[0])
    next_act, next_lp, _ = jax.vmap(sample_action_sac)(rng_acts, mean_n, lgs_n, next_max_v)

    q1_t, q2_t = critic_net.apply({"params": target_params}, next_obs, next_act)
    v_next   = jnp.minimum(q1_t, q2_t) - alpha * next_lp
    backup   = jax.lax.stop_gradient(reward + GAMMA * (1.0 - done) * v_next)

    q1, q2   = critic_net.apply({"params": critic_params}, obs, action)
    loss_q1  = jnp.mean((q1 - backup) ** 2)
    loss_q2  = jnp.mean((q2 - backup) ** 2)
    return loss_q1 + loss_q2, (loss_q1, loss_q2, jnp.mean(backup))


# ── Actor loss ────────────────────────────────────────────────────────────────
@jax.jit
def actor_loss_fn(actor_params, critic_params, log_alpha, obs, rng_key):
    """
    Minimise E[alpha*log_pi(a|s) - min(Q1,Q2)(s,a)].
    Action is reparameterised -> gradient flows through actor_params.
    critic_params not in argnums=0 -> no critic gradient here.
    """
    alpha = jnp.exp(log_alpha)
    max_v = extract_max_v(obs)

    mean, log_std = actor_net.apply({"params": actor_params}, obs)
    rng_acts = jax.random.split(rng_key, obs.shape[0])
    action, log_pi, _ = jax.vmap(sample_action_sac)(rng_acts, mean, log_std, max_v)

    q1, q2 = critic_net.apply({"params": critic_params}, obs, action)
    loss   = jnp.mean(alpha * log_pi - jnp.minimum(q1, q2))
    return loss, jnp.mean(log_pi)


# ── Alpha loss ────────────────────────────────────────────────────────────────
@jax.jit
def alpha_loss_fn(log_alpha, log_pi_mean):
    """L(alpha) = -alpha * (log_pi + H_target). log_pi treated as constant."""
    return -log_alpha * (log_pi_mean + TARGET_ENTROPY)


# ── Full SAC update step ──────────────────────────────────────────────────────
@jax.jit
def sac_update(actor_params, actor_opt_state,
               critic_params, critic_opt_state,
               target_params,
               log_alpha, alpha_opt_state,
               obs, action, reward, next_obs, done,
               rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    # 1. Critic
    (c_loss, (c1, c2, q_mean)), c_grads = jax.value_and_grad(
        critic_loss_fn, argnums=0, has_aux=True
    )(critic_params, target_params, actor_params,
      log_alpha, obs, action, reward, next_obs, done, rng_c)
    c_upd, new_copt = critic_opt.update(c_grads, critic_opt_state, critic_params)
    new_critic      = optax.apply_updates(critic_params, c_upd)

    # 2. Actor
    (a_loss, log_pi_mean), a_grads = jax.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(actor_params, new_critic, log_alpha, obs, rng_a)
    a_upd, new_aopt = actor_opt.update(a_grads, actor_opt_state, actor_params)
    new_actor       = optax.apply_updates(actor_params, a_upd)

    # 3. Alpha
    log_pi_sg = jax.lax.stop_gradient(log_pi_mean)
    al_loss, al_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha, log_pi_sg)
    al_upd, new_alopt = alpha_opt.update(al_grad, alpha_opt_state)
    new_log_alpha     = optax.apply_updates(log_alpha, al_upd)

    # 4. Soft target update
    new_target = soft_update(target_params, new_critic)

    metrics = {
        "critic_loss": c_loss,
        "actor_loss":  a_loss,
        "alpha":       jnp.exp(new_log_alpha),
        "log_pi":      log_pi_mean,
        "q_mean":      q_mean,
    }
    return (new_actor, new_aopt,
            new_critic, new_copt,
            new_target,
            new_log_alpha, new_alopt,
            metrics)


# ── Collection step ───────────────────────────────────────────────────────────
@jax.jit
def collect_step(actor_params, env_state, env_obs, rng_key):
    """One parallel step across N_ENVS envs. Actor called with full batch."""
    rng_act, rng_step = jax.random.split(rng_key)

    mean, log_std = actor_net.apply({"params": actor_params}, env_obs)
    max_v         = extract_max_v(env_obs)

    rng_acts = jax.random.split(rng_act, N_ENVS)
    env_action, _lp, _ = jax.vmap(sample_action_sac)(rng_acts, mean, log_std, max_v)

    step_keys = jax.random.split(rng_step, N_ENVS)
    new_obs, new_state, reward, done, info = _vmap_step(step_keys, env_state, env_action)

    return new_obs, new_state, env_obs, env_action, reward, done, info


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(actor_params, critic_params, target_params,
                    actor_opt_state, critic_opt_state,
                    log_alpha, alpha_opt_state, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "actor_params":     jax.device_get(actor_params),
        "critic_params":    jax.device_get(critic_params),
        "target_params":    jax.device_get(target_params),
        "actor_opt_state":  jax.device_get(actor_opt_state),
        "critic_opt_state": jax.device_get(critic_opt_state),
        "log_alpha":        jax.device_get(log_alpha),
        "alpha_opt_state":  jax.device_get(alpha_opt_state),
        "step":             int(step),
    }
    with open(CKPT_PATH, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SAC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Stats ─────────────────────────────────────────────────────────────────────
def update_stats(buf, ptr, rewards, dones, goals, cols, ep_acc):
    ep_acc += np.array(rewards)
    dones_  = np.array(dones).astype(bool)
    goals_  = np.array(goals).astype(bool)
    cols_   = np.array(cols).astype(bool)
    for i in range(len(dones_)):
        if dones_[i]:
            buf[ptr % STATS_WINDOW] = [ep_acc[i], float(goals_[i]),
                                        float(cols_[i]),
                                        float(not goals_[i] and not cols_[i])]
            ptr += 1
            ep_acc[i] = 0.0
    return buf, ptr, ep_acc


def window_stats(buf, ptr):
    n = min(ptr, STATS_WINDOW)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    v = buf[:n]
    return float(v[:,0].mean()), float(v[:,1].mean())*100, \
           float(v[:,2].mean())*100, float(v[:,3].mean())*100


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("SAC Training — GPU 1 (CUDA_VISIBLE_DEVICES=1)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}\n")

    rng = jax.random.PRNGKey(7)

    # Initialise networks
    rng, ka, kc = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((2, OBS_SIZE))   # batch of 2 for Conv init
    dummy_act = jnp.zeros((2, ACTION_DIM))

    actor_params  = actor_net.init(ka, dummy_obs)["params"]
    critic_params = critic_net.init(kc, dummy_obs, dummy_act)["params"]
    target_params = jax.tree_util.tree_map(jnp.array, critic_params)

    actor_opt_state  = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    log_alpha        = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    alpha_opt_state  = alpha_opt.init(log_alpha)

    # Init envs
    print("Initialising environments on GPU 1...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state = _vmap_reset(jax.random.split(env_rng, N_ENVS))
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    stat_buf    = np.zeros((STATS_WINDOW, 4), dtype=np.float32)
    write_ptr   = 0
    ep_accum    = np.zeros(N_ENVS, dtype=np.float32)
    best_suc    = 20.0
    n_updates   = 0

    hdr = (f"{'Upd':>7} | {'Steps':>8} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'Col%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} | {'FPS':>7}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()
    t_log   = time.time()

    while n_updates < TOTAL_UPDATES:
        # Collect one step from all envs
        rng, collect_rng = jax.random.split(rng)
        new_obs, env_state, obs_before, env_action, reward, done, info = \
            collect_step(actor_params, env_state, env_obs, collect_rng)

        # Add to replay buffer
        replay_buf = buf_add(replay_buf,
                             obs_before,
                             env_action,
                             reward,
                             new_obs,
                             done.astype(jnp.float32))
        env_obs     = new_obs
        total_steps += N_ENVS

        # Track episode stats
        stat_buf, write_ptr, ep_accum = update_stats(
            stat_buf, write_ptr,
            reward, done, info["goal_reached"], info["collision"],
            ep_accum
        )

        # Warmup before gradient updates
        if total_steps < WARMUP_STEPS:
            if total_steps % (N_ENVS * 20) == 0:
                print(f"  warmup {total_steps:,}/{WARMUP_STEPS:,}")
            continue

        # One gradient update per collection step
        rng, samp_rng, upd_rng = jax.random.split(rng, 3)
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(replay_buf, samp_rng, BATCH_SIZE)

        (actor_params, actor_opt_state,
         critic_params, critic_opt_state,
         target_params,
         log_alpha, alpha_opt_state,
         metrics) = sac_update(
            actor_params, actor_opt_state,
            critic_params, critic_opt_state,
            target_params,
            log_alpha, alpha_opt_state,
            b_obs, b_act, b_rew, b_next, b_done,
            upd_rng,
        )
        n_updates += 1

        if n_updates % LOG_EVERY == 0:
            t_now = time.time()
            fps   = N_ENVS * LOG_EVERY / (t_now - t_log + 1e-8)
            t_log = t_now
            mean_ret, suc_pct, col_pct, tmo_pct = window_stats(stat_buf, write_ptr)
            print(
                f"{n_updates:>7d} | {total_steps:>8,} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(metrics['critic_loss']):>7.4f} "
                f"{float(metrics['actor_loss']):>7.4f} "
                f"{float(metrics['alpha']):>6.4f} "
                f"{float(metrics['log_pi']):>6.3f} | "
                f"{fps:>7,.0f}"
            )

        if n_updates % SAVE_EVERY == 0:
            mean_ret, suc_pct, col_pct, tmo_pct = window_stats(stat_buf, write_ptr)
            if suc_pct > best_suc and write_ptr >= STATS_WINDOW // 4:
                best_suc = suc_pct
                save_checkpoint(actor_params, critic_params, target_params,
                                actor_opt_state, critic_opt_state,
                                log_alpha, alpha_opt_state, n_updates)

    elapsed = time.time() - t_start
    print(f"\nSAC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")