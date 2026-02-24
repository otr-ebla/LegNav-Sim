"""
TQCjax.py — Truncated Quantile Critics  (GPU 1)
================================================
Pinned to GPU 1 via CUDA_VISIBLE_DEVICES=1 so it runs alongside PPO on GPU 0.

TQC (Kuznetsov et al. 2020, https://arxiv.org/abs/2005.04269) extends SAC with:
  - N_CRITICS separate critic networks, each outputting N_ATOMS quantile atoms
  - Critic loss: quantile Huber regression (not MSE)
  - Target backup: sort the N_CRITICS * N_ATOMS target atoms per sample,
    DROP the top N_TOP_ATOMS_DROP to suppress overestimation, regress the
    remaining (N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP) atoms
  - Actor loss: maximise mean over ALL atoms across ALL critics
  - Automatic alpha tuning: identical to SAC

Key shapes (defaults: N_CRITICS=5, N_ATOMS=25, N_TOP_ATOMS_DROP=5):
  critic output per network  : (batch, N_ATOMS)
  all critics stacked         : (batch, N_CRITICS, N_ATOMS)
  total target atoms          : N_CRITICS * N_ATOMS = 125
  after dropping top-5        : N_TARGET_ATOMS = 120
  quantile midpoints τ        : (N_ATOMS,)  τ_i = (2i-1)/(2*N_ATOMS)

Action space (same as SAC):
  v in [0, max_v]: a_v = (tanh(u_v) + 1) / 2 * max_v
  w in [-1,   1 ]: a_w = tanh(u_w)
  max_v recovered from obs[..., MAX_V_OBS_IDX] * 2.

Quantile Huber loss (κ=1):
  u     = target_j - atom_i
  L_κ   = 0.5*u²            if |u| <= κ, else  κ*(|u| - 0.5*κ)
  ρ_τ_i = |τ_i - (u < 0)| * L_κ / κ
  loss  = mean_over_batch[ mean_over_atoms[ mean_over_targets[ρ] ] ]
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
import optax
import flax
import flax.linen as nn
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_env import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# ── Hyperparameters ───────────────────────────────────────────────────────────
OBS_SIZE       = 342        # pose_stack(9) + state_vec(9) + lidar_stack(324)
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

# TQC-specific
N_CRITICS        = 5     # number of independent critic networks
N_ATOMS          = 25    # quantile atoms per critic
N_TOP_ATOMS_DROP = 5     # drop this many largest target atoms per sample
N_TARGET_ATOMS   = N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP   # 120
HUBER_KAPPA      = 1.0   # Huber loss threshold κ

# obs[..., 11] = max_v/2  (pose_stack=9, state_vec[2] at flat index 9+2=11)
MAX_V_OBS_IDX = 11
LOG_STD_EPS   = 1e-6

CKPT_DIR  = "checkpoints_tqc"
CKPT_PATH = f"{CKPT_DIR}/tqc_best.msgpack"


# ── GPU check ─────────────────────────────────────────────────────────────────
def _check_gpu():
    try:
        devs = jax.devices("cuda")
    except RuntimeError:
        devs = []
    if len(devs) < 2:
        raise RuntimeError("Less than 2 CUDA devices found. TQC requires CudaDevice(1).")
    target_device = devs[1]
    print(f"TQC pinned to: {target_device}  (physical GPU 1)")
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
    """
    Encodes stacked obs → 128-dim feature vector.
    Obs layout: pose_stack(9) | state_vec(9) | lidar_stack(324) = 342
    """
    stack_dim: int = 3
    num_rays:  int = 108

    @nn.compact
    def __call__(self, x):
        pose_size  = 3 * self.stack_dim   # 9
        state_size = 9                    # STATE_VEC_SIZE from jax_env.py

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
        return shared   # (batch, 128)


# ── Actor ─────────────────────────────────────────────────────────────────────
class TQCActorNetwork(nn.Module):
    """Squashed Gaussian actor — same architecture as SAC actor."""
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


# ── Single quantile critic ────────────────────────────────────────────────────
class QuantileCriticNetwork(nn.Module):
    """
    One critic that outputs N_ATOMS quantile values for Q(s,a).
    Input:  obs (batch, OBS_SIZE), action (batch, ACTION_DIM)
    Output: atoms (batch, N_ATOMS)
    """
    n_atoms:    int = N_ATOMS
    action_dim: int = ACTION_DIM

    @nn.compact
    def __call__(self, obs, action):
        feat  = ObsEncoder()(obs)
        x     = jnp.concatenate([feat, action], axis=-1)
        x     = nn.relu(nn.Dense(256)(x))
        x     = nn.relu(nn.Dense(128)(x))
        atoms = nn.Dense(self.n_atoms)(x)   # (batch, N_ATOMS) — no final activation
        return atoms


# ── Ensemble of N_CRITICS quantile critics ────────────────────────────────────
class TQCCriticEnsemble(nn.Module):
    """
    N_CRITICS independent QuantileCriticNetwork instances.
    Each critic has fully separate weights (own ObsEncoder + heads).
    Output: (batch, N_CRITICS, N_ATOMS)
    """
    n_critics:  int = N_CRITICS
    n_atoms:    int = N_ATOMS
    action_dim: int = ACTION_DIM

    @nn.compact
    def __call__(self, obs, action):
        all_atoms = []
        for i in range(self.n_critics):
            atoms = QuantileCriticNetwork(
                n_atoms=self.n_atoms,
                action_dim=self.action_dim,
                name=f'critic_{i}'
            )(obs, action)   # (batch, N_ATOMS)
            all_atoms.append(atoms)
        return jnp.stack(all_atoms, axis=1)   # (batch, N_CRITICS, N_ATOMS)


# ── Quantile midpoints τ (precomputed at module level) ────────────────────────
# τ_i = (2i - 1) / (2 * N_ATOMS)  for i = 1 … N_ATOMS
# Shape (N_ATOMS,) — used in the Huber loss below.
_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)


# ── Network + optimiser instances ─────────────────────────────────────────────
actor_net  = TQCActorNetwork()
critic_net = TQCCriticEnsemble()

actor_opt  = optax.adam(LR, eps=1e-5)
critic_opt = optax.adam(LR, eps=1e-5)
alpha_opt  = optax.adam(LR, eps=1e-5)


# ── Action squashing helpers ──────────────────────────────────────────────────

def _tanh_log_prob_correction(tanh_u, max_v):
    """
    Negative log-det-Jacobian of the squashing transform (identical to SAC).
      v: a_v = (tanh(u_v)+1)/2 * max_v  →  log|da/du| = log(max_v/2) + log(1-tanh²)
      w: a_w = tanh(u_w)                 →  log|da/du| = log(1-tanh²)
    Returns the NEGATIVE sum (add to Gaussian log-prob to get log π).
    """
    corr_v = (jnp.log(max_v * 0.5 + LOG_STD_EPS)
              + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS))
    corr_w = jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS)
    return -(corr_v + corr_w)


def sample_action(rng_key, mean, log_std, max_v):
    """
    Reparameterised sample from squashed Gaussian.
    Returns: env_action (2,), log_pi scalar.
    """
    std   = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u     = mean + noise * std

    lp_gauss = jnp.sum(
        -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std,
        axis=-1
    )
    tanh_u     = jnp.tanh(u)
    a_v        = (tanh_u[..., 0] + 1.0) * 0.5 * max_v
    a_w        = tanh_u[..., 1]
    env_action = jnp.stack([a_v, a_w], axis=-1)
    log_pi     = lp_gauss + _tanh_log_prob_correction(tanh_u, max_v)
    return env_action, log_pi


@jax.jit
def extract_max_v(obs):
    """obs[..., 11] = max_v/2  →  return max_v."""
    return obs[..., MAX_V_OBS_IDX] * 2.0


# ── Quantile Huber loss ───────────────────────────────────────────────────────

def quantile_huber_loss(atoms, targets, taus, kappa=HUBER_KAPPA):
    """
    Quantile Huber regression loss for one critic.

    Args:
      atoms   : (batch, N_ATOMS)         — predicted quantile values
      targets : (batch, N_TARGET_ATOMS)  — regression targets (stop-gradient'd)
      taus    : (N_ATOMS,)               — quantile midpoints for the predictions
      kappa   : float                    — Huber threshold (1.0)

    Returns scalar loss.

    For each (b, i, j):
      u     = targets[b,j] - atoms[b,i]      target minus prediction
      L_κ   = 0.5*u²          if |u| <= κ
              κ*(|u| - 0.5*κ) otherwise
      ρ     = |τ_i - 1(u<0)| * L_κ / κ

    Loss = mean_b [ (1/N_ATOMS) * sum_i [ (1/N_TARGET_ATOMS) * sum_j ρ ] ]
    """
    # Expand dims for broadcasting:
    #   atoms  : (batch, N_ATOMS, 1)
    #   targets: (batch, 1, N_TARGET_ATOMS)
    #   u      : (batch, N_ATOMS, N_TARGET_ATOMS)
    u = targets[:, None, :] - atoms[:, :, None]

    # Huber loss
    abs_u = jnp.abs(u)
    huber = jnp.where(abs_u <= kappa,
                      0.5 * u ** 2,
                      kappa * (abs_u - 0.5 * kappa))

    # Asymmetric quantile weighting: |τ_i - 1(u < 0)|
    # taus: (N_ATOMS,) → (1, N_ATOMS, 1)
    indicator = (u < 0.0).astype(jnp.float32)
    weight    = jnp.abs(taus[None, :, None] - indicator)

    rho = weight * huber / kappa   # (batch, N_ATOMS, N_TARGET_ATOMS)

    # Mean over targets axis, then mean over atoms and batch
    return jnp.mean(jnp.mean(rho, axis=2))   # scalar


# ── Replay buffer ─────────────────────────────────────────────────────────────

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
    """
    Sample from full capacity (concrete int → JIT-safe).
    Clamp out-of-range indices to slot 0 during warmup.
    """
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


# ── Critic loss ───────────────────────────────────────────────────────────────
@jax.jit
def critic_loss_fn(critic_params, target_params, actor_params, log_alpha,
                   obs, action, reward, next_obs, done, rng_key):
    """
    TQC critic loss.

    Step-by-step target construction:
      1. Sample a' ~ π(·|s') from the CURRENT actor.
      2. Compute target atoms via the TARGET critic: (batch, N_CRITICS, N_ATOMS).
      3. Flatten critics dim → (batch, N_CRITICS*N_ATOMS), sort ascending.
      4. Keep only the bottom N_TARGET_ATOMS (drop top N_TOP_ATOMS_DROP).
         This is the truncation that suppresses overestimation.
      5. Subtract entropy: backup_j = r + γ*(1-d)*(z*_j - α*log_π(a'|s'))
         The entropy term is subtracted from each kept atom individually.
         stop_gradient applied to the entire backup.

    Online loss:
      For each of the N_CRITICS online critics, compute quantile_huber_loss
      between its (batch, N_ATOMS) outputs and the (batch, N_TARGET_ATOMS)
      backup. Sum losses across critics.
    """
    alpha      = jnp.exp(log_alpha)
    next_max_v = extract_max_v(next_obs)
    batch_size = obs.shape[0]

    # Next action from current actor
    mean_n, lgs_n = actor_net.apply({"params": actor_params}, next_obs)
    rng_acts      = jax.random.split(rng_key, batch_size)
    next_act, next_lp = jax.vmap(sample_action)(rng_acts, mean_n, lgs_n, next_max_v)

    # Target atoms: (batch, N_CRITICS, N_ATOMS) → flatten → sort → truncate
    target_atoms  = critic_net.apply({"params": target_params}, next_obs, next_act)
    target_flat   = target_atoms.reshape(batch_size, N_CRITICS * N_ATOMS)
    target_sorted = jnp.sort(target_flat, axis=1)            # ascending along atom axis
    target_kept   = target_sorted[:, :N_TARGET_ATOMS]        # (batch, N_TARGET_ATOMS)

    # Bellman backup with entropy regularisation
    # next_lp: (batch,) → (batch, 1) to broadcast across N_TARGET_ATOMS
    backup = jax.lax.stop_gradient(
        reward[:, None]
        + GAMMA * (1.0 - done[:, None]) * (target_kept - alpha * next_lp[:, None])
    )   # (batch, N_TARGET_ATOMS)

    # Online atoms: (batch, N_CRITICS, N_ATOMS)
    online_atoms = critic_net.apply({"params": critic_params}, obs, action)

    # Sum quantile Huber loss across all critics
    total_loss = jnp.float32(0.0)
    q_sum      = jnp.float32(0.0)
    for i in range(N_CRITICS):
        atoms_i     = online_atoms[:, i, :]          # (batch, N_ATOMS)
        total_loss  = total_loss + quantile_huber_loss(atoms_i, backup, _TAUS)
        q_sum       = q_sum + jnp.mean(atoms_i)

    mean_q = q_sum / N_CRITICS
    return total_loss, mean_q


# ── Actor loss ────────────────────────────────────────────────────────────────
@jax.jit
def actor_loss_fn(actor_params, critic_params, log_alpha, obs, rng_key):
    """
    TQC actor loss:
      L(π) = E_s [ α*log_π(a|s) - mean_{c,i}[Z_c_i(s,a)] ]

    Maximising the mean over ALL critics and ALL atoms is an unbiased
    estimate of the full distributional value E_τ[Z(s,a)].
    No critic gradient: critic_params not in argnums=0.
    """
    alpha = jnp.exp(log_alpha)
    max_v = extract_max_v(obs)

    mean, log_std = actor_net.apply({"params": actor_params}, obs)
    rng_acts      = jax.random.split(rng_key, obs.shape[0])
    action, log_pi = jax.vmap(sample_action)(rng_acts, mean, log_std, max_v)

    # (batch, N_CRITICS, N_ATOMS) → mean over all critics and atoms per sample
    all_atoms = critic_net.apply({"params": critic_params}, obs, action)
    q_mean    = jnp.mean(all_atoms)   # scalar: mean over batch, critics, atoms

    loss = jnp.mean(alpha * log_pi) - q_mean
    return loss, jnp.mean(log_pi)


# ── Alpha loss ────────────────────────────────────────────────────────────────
@jax.jit
def alpha_loss_fn(log_alpha, log_pi_mean):
    """Identical to SAC: L(α) = -α*(log_π + H*)."""
    return -log_alpha * (log_pi_mean + TARGET_ENTROPY)


# ── Full TQC update step ──────────────────────────────────────────────────────
@jax.jit
def tqc_update(actor_params, actor_opt_state,
               critic_params, critic_opt_state,
               target_params,
               log_alpha, alpha_opt_state,
               obs, action, reward, next_obs, done,
               rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    # 1. Critic
    (c_loss, q_mean), c_grads = jax.value_and_grad(
        critic_loss_fn, argnums=0, has_aux=True
    )(critic_params, target_params, actor_params,
      log_alpha, obs, action, reward, next_obs, done, rng_c)
    c_upd, new_copt = critic_opt.update(c_grads, critic_opt_state, critic_params)
    new_critic      = optax.apply_updates(critic_params, c_upd)

    # 2. Actor (uses freshly updated critic)
    (a_loss, log_pi_mean), a_grads = jax.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(actor_params, new_critic, log_alpha, obs, rng_a)
    a_upd, new_aopt = actor_opt.update(a_grads, actor_opt_state, actor_params)
    new_actor       = optax.apply_updates(actor_params, a_upd)

    # 3. Alpha
    log_pi_sg         = jax.lax.stop_gradient(log_pi_mean)
    al_loss, al_grad  = jax.value_and_grad(alpha_loss_fn)(log_alpha, log_pi_sg)
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
    """One parallel step across all N_ENVS environments."""
    rng_act, rng_step = jax.random.split(rng_key)

    mean, log_std = actor_net.apply({"params": actor_params}, env_obs)
    max_v         = extract_max_v(env_obs)

    rng_acts           = jax.random.split(rng_act, N_ENVS)
    env_action, _lp    = jax.vmap(sample_action)(rng_acts, mean, log_std, max_v)

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
    print(f"  TQC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Episode stats (identical logic to SACjax.py) ─────────────────────────────
def update_stats(buf, ptr, rewards, dones, goals, cols, pcols, ep_acc):
    """
    Mutually exclusive outcomes (always sum to 100%):
      [0] ep_return
      [1] success    : goal_reached
      [2] active_col : collision & ~passive_col
      [3] passive_col: human walked into stopped robot
      [4] timeout    : ~goal & ~collision
    """
    ep_acc += np.array(rewards)
    dones_  = np.array(dones).astype(bool)
    goals_  = np.array(goals).astype(bool)
    cols_   = np.array(cols).astype(bool)
    pcols_  = np.array(pcols).astype(bool)
    for i in range(len(dones_)):
        if dones_[i]:
            goal    = goals_[i]
            col     = cols_[i]
            pcol    = pcols_[i]
            act_col = col and not pcol
            timeout = not goal and not col
            buf[ptr % STATS_WINDOW] = [ep_acc[i], float(goal), float(act_col),
                                        float(pcol), float(timeout)]
            ptr += 1
            ep_acc[i] = 0.0
    return buf, ptr, ep_acc


def window_stats(buf, ptr):
    n = min(ptr, STATS_WINDOW)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    v = buf[:n]
    return (float(v[:, 0].mean()), float(v[:, 1].mean()) * 100,
            float(v[:, 2].mean()) * 100, float(v[:, 3].mean()) * 100,
            float(v[:, 4].mean()) * 100)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("TQC Training — GPU 1 (CUDA_VISIBLE_DEVICES=1)")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  "
          f"N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")
    print(f"  gamma={GAMMA}  tau={TAU}  lr={LR}  H*={TARGET_ENTROPY}\n")

    rng = jax.random.PRNGKey(7)

    # ── Initialise networks ───────────────────────────────────────────────────
    rng, ka, kc = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((2, OBS_SIZE))    # batch ≥ 2 for Conv shape inference
    dummy_act = jnp.zeros((2, ACTION_DIM))

    actor_params  = actor_net.init(ka, dummy_obs)["params"]
    critic_params = critic_net.init(kc, dummy_obs, dummy_act)["params"]
    target_params = jax.tree_util.tree_map(jnp.array, critic_params)

    actor_opt_state  = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    log_alpha        = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    alpha_opt_state  = alpha_opt.init(log_alpha)

    # ── Initialise environments ───────────────────────────────────────────────
    print("Initialising environments on GPU 1...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state = _vmap_reset(jax.random.split(env_rng, N_ENVS))
    print(f"Ready. obs shape: {env_obs.shape}\n")

    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps = 0
    stat_buf    = np.zeros((STATS_WINDOW, 5), dtype=np.float32)
    write_ptr   = 0
    ep_accum    = np.zeros(N_ENVS, dtype=np.float32)
    best_suc    = 20.0
    n_updates   = 0

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | {'FPS':>7}")
    print(hdr)
    print("─" * len(hdr))

    t_start = time.time()
    t_log   = time.time()

    while n_updates < TOTAL_UPDATES:

        # ── Collect ───────────────────────────────────────────────────────────
        rng, collect_rng = jax.random.split(rng)
        new_obs, env_state, obs_before, env_action, reward, done, info = \
            collect_step(actor_params, env_state, env_obs, collect_rng)

        replay_buf = buf_add(replay_buf, obs_before, env_action,
                             reward, new_obs, done.astype(jnp.float32))
        env_obs      = new_obs
        total_steps += N_ENVS

        stat_buf, write_ptr, ep_accum = update_stats(
            stat_buf, write_ptr,
            reward, done,
            info["goal_reached"], info["collision"], info["passive_col"],
            ep_accum
        )

        # ── Warmup ────────────────────────────────────────────────────────────
        if total_steps < WARMUP_STEPS:
            if total_steps % (N_ENVS * 20) == 0:
                print(f"  warmup {total_steps:,}/{WARMUP_STEPS:,}")
            continue

        # ── Update ────────────────────────────────────────────────────────────
        rng, samp_rng, upd_rng = jax.random.split(rng, 3)
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(replay_buf, samp_rng, BATCH_SIZE)

        (actor_params, actor_opt_state,
         critic_params, critic_opt_state,
         target_params,
         log_alpha, alpha_opt_state,
         metrics) = tqc_update(
            actor_params, actor_opt_state,
            critic_params, critic_opt_state,
            target_params,
            log_alpha, alpha_opt_state,
            b_obs, b_act, b_rew, b_next, b_done,
            upd_rng,
        )
        n_updates += 1

        # ── Log ───────────────────────────────────────────────────────────────
        if n_updates % LOG_EVERY == 0:
            t_now = time.time()
            fps   = N_ENVS * LOG_EVERY / (t_now - t_log + 1e-8)
            t_log = t_now
            mean_ret, suc_pct, acol_pct, pcol_pct, tmo_pct = window_stats(stat_buf, write_ptr)
            print(
                f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                f"{suc_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                f"{float(metrics['critic_loss']):>7.4f} "
                f"{float(metrics['actor_loss']):>7.4f} "
                f"{float(metrics['alpha']):>6.4f} "
                f"{float(metrics['log_pi']):>6.3f} "
                f"{float(metrics['q_mean']):>7.3f} | "
                f"{fps:>7,.0f}"
            )

        # ── Save ──────────────────────────────────────────────────────────────
        if n_updates % SAVE_EVERY == 0:
            mean_ret, suc_pct, acol_pct, pcol_pct, tmo_pct = window_stats(stat_buf, write_ptr)
            if suc_pct > best_suc and write_ptr >= STATS_WINDOW // 4:
                best_suc = suc_pct
                save_checkpoint(actor_params, critic_params, target_params,
                                actor_opt_state, critic_opt_state,
                                log_alpha, alpha_opt_state, n_updates)

    elapsed = time.time() - t_start
    print(f"\nTQC done! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")