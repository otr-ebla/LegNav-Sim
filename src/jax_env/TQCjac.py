"""
TQCjax.py — Truncated Quantile Critics
================================================
FIXES for 110k+ FPS & Flexibility:
  1. ARGPARSE: Added support for --gpu (0 or 1) and --bfloat16.
  2. MIXED PRECISION: When --bfloat16 is active, the CNN and MLPs execute in 
     bfloat16 on the GPU Tensor Cores, while the physics engine and the Huber 
     Loss remain safely in float32.
  3. FULL JAX LOOP FUSION: Wraps the collection, buffer insertion, sampling, 
     and gradient update steps inside a `jax.lax.scan`.
  4. JAX EPISODE STATS: Vectorized GPU statistics tracker.
  5. CRITIC ENSEMBLE REDUCTION: N_CRITICS=3 to drastically cut memory bandwidth.
"""

import os
import csv
import argparse

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="JAX TQC Training")
parser.add_argument("--gpu", type=str, default="1", choices=["0", "1"], help="Target GPU ID (0 or 1)")
parser.add_argument("--bfloat16", action="store_true", help="Enable bfloat16 mixed precision for neural networks")
args, _ = parser.parse_known_args()

os.environ["JAX_PLATFORMS"]               = "cuda"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"]        = args.gpu

import time
import warnings
import functools
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import flax.serialization

warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_env_multi import reset_env, step_env
from jax_wrappers import make_stacked_env, make_autoreset_env

# ── Hyperparameters ───────────────────────────────────────────────────────────
OBS_SIZE       = 342
ACTION_DIM     = 2
N_ENVS         = 2048
BUFFER_CAP     = 500_000
BATCH_SIZE     = 1024
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.005
LR             = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)   
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 500
SAVE_EVERY     = 5000

# TQC-specific (Optimized for execution speed)
N_CRITICS        = 3     
N_ATOMS          = 25    
N_TOP_ATOMS_DROP = 3     
N_TARGET_ATOMS   = N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP   # 72
HUBER_KAPPA      = 1.0   

MAX_V_OBS_IDX = 11
LOG_STD_EPS   = 1e-6

CKPT_DIR  = "checkpoints_tqc"
CKPT_PATH = f"{CKPT_DIR}/tqc_best.msgpack"

NET_DTYPE = jnp.bfloat16 if args.bfloat16 else jnp.float32

# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM_STAGES = [
    (20.0, 1.5),
    (35.0, 3.0),
    (50.0, 5.5),
    (65.0, 7.0),
    (101., 8.0),
]

def curriculum_min_goal_dist(suc_pct: float) -> float:
    for threshold, dist in CURRICULUM_STAGES:
        if suc_pct < threshold: return dist
    return CURRICULUM_STAGES[-1][1]

def _curriculum_stage(suc_pct: float) -> int:
    for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
        if suc_pct < threshold: return i
    return len(CURRICULUM_STAGES) - 1

# ── GPU check ─────────────────────────────────────────────────────────────────
def _check_gpu():
    try: devs = jax.devices("cuda")
    except RuntimeError: devs = []
    if not devs:
        raise RuntimeError(f"No CUDA devices found for GPU {args.gpu}.")
    target_device = devs[0] # CUDA_VISIBLE_DEVICES isolates the selected GPU to index 0
    print(f"TQC pinned to: {target_device}  (physical GPU {args.gpu})")
    return target_device

target_gpu = _check_gpu()
jax.config.update("jax_default_device", target_gpu)

# ── Environment ───────────────────────────────────────────────────────────────
reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

def init_env_state(rng_key, min_goal_dist: float = 3.0):
    step_auto = make_autoreset_env(reset_stacked, step_stacked, min_goal_dist=min_goal_dist)
    vmap_step = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0)))
    def _reset_with_dist(key): return reset_stacked(key, min_goal_dist=min_goal_dist)
    vmap_reset = jax.jit(jax.vmap(_reset_with_dist))
    reset_keys = jax.random.split(rng_key, N_ENVS)
    env_obs, env_state = vmap_reset(reset_keys)
    return env_obs, env_state, vmap_step

# ── Shared obs encoder ────────────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    stack_dim: int = 3
    num_rays:  int = 108
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, x):
        pose_size = 3 * self.stack_dim
        state_size = 9                    
        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]
        batch_shape = lidar_flat.shape[:-1]
        
        # Cast inputs to network dtype
        lidar_cnn = lidar_flat.reshape((*batch_shape, self.num_rays, self.stack_dim)).astype(self.dtype)
        cnn = nn.relu(nn.Conv(features=32, kernel_size=(7,), strides=(2,), padding='SAME', dtype=self.dtype)(lidar_cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(5,), strides=(2,), padding='SAME', dtype=self.dtype)(cnn))
        cnn = nn.relu(nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='SAME', dtype=self.dtype)(cnn))
        cnn_feat = nn.LayerNorm(dtype=self.dtype)(cnn.reshape((*batch_shape, -1)))
        
        global_in = jnp.concatenate([pose_stack, state_vec], axis=-1).astype(self.dtype)
        global_feat = nn.relu(nn.Dense(128, dtype=self.dtype)(global_in))
        global_feat = nn.relu(nn.Dense(64, dtype=self.dtype)(global_feat))
        
        fused = jnp.concatenate([cnn_feat, global_feat], axis=-1)
        shared = nn.relu(nn.Dense(256, dtype=self.dtype)(fused))
        return nn.relu(nn.Dense(128, dtype=self.dtype)(shared))   

# ── Actor ─────────────────────────────────────────────────────────────────────
class TQCActorNetwork(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, obs):
        feat = ObsEncoder(dtype=self.dtype)(obs)
        mean = nn.Dense(self.action_dim, dtype=self.dtype)(feat)
        log_std = nn.Dense(self.action_dim, dtype=self.dtype)(feat)
        # Always cast outputs back to float32 for physics and loss stability
        return mean.astype(jnp.float32), jnp.clip(log_std.astype(jnp.float32), self.LOG_STD_MIN, self.LOG_STD_MAX)

# ── Quantile Critics ──────────────────────────────────────────────────────────
class QuantileCriticNetwork(nn.Module):
    n_atoms:    int = N_ATOMS
    action_dim: int = ACTION_DIM
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, obs, action):
        feat = ObsEncoder(dtype=self.dtype)(obs)
        x = nn.relu(nn.Dense(256, dtype=self.dtype)(jnp.concatenate([feat, action.astype(self.dtype)], axis=-1)))
        x = nn.relu(nn.Dense(128, dtype=self.dtype)(x))
        # Always cast output atoms back to float32 for stable Huber loss
        return nn.Dense(self.n_atoms, dtype=self.dtype)(x).astype(jnp.float32)

class TQCCriticEnsemble(nn.Module):
    n_critics:  int = N_CRITICS
    n_atoms:    int = N_ATOMS
    action_dim: int = ACTION_DIM
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, obs, action):
        all_atoms = [QuantileCriticNetwork(self.n_atoms, self.action_dim, dtype=self.dtype, name=f'critic_{i}')(obs, action) for i in range(self.n_critics)]
        return jnp.stack(all_atoms, axis=1)

_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)

actor_net  = TQCActorNetwork()
critic_net = TQCCriticEnsemble()
actor_opt  = optax.adam(LR, eps=1e-5)
critic_opt = optax.adam(LR, eps=1e-5)
alpha_opt  = optax.adam(LR, eps=1e-5)

# ── Action Squashing ──────────────────────────────────────────────────────────
def _tanh_log_prob_correction(tanh_u, max_v):
    corr_v = jnp.log(max_v * 0.5 + LOG_STD_EPS) + jnp.log(1.0 - tanh_u[..., 0] ** 2 + LOG_STD_EPS)
    return -(corr_v + jnp.log(1.0 - tanh_u[..., 1] ** 2 + LOG_STD_EPS))

def sample_action(rng_key, mean, log_std, max_v):
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, shape=mean.shape)
    u = mean + noise * std
    lp_gauss = jnp.sum(-0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1)
    tanh_u = jnp.tanh(u)
    env_action = jnp.stack([(tanh_u[..., 0] + 1.0) * 0.5 * max_v, tanh_u[..., 1]], axis=-1)
    return env_action, lp_gauss + _tanh_log_prob_correction(tanh_u, max_v)

@jax.jit
def extract_max_v(obs): return obs[..., MAX_V_OBS_IDX] * 2.0

def quantile_huber_loss(atoms, targets, taus, kappa=HUBER_KAPPA):
    u = targets[:, None, :] - atoms[:, :, None]
    abs_u = jnp.abs(u)
    huber = jnp.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    indicator = (u < 0.0).astype(jnp.float32)
    rho = jnp.abs(taus[None, :, None] - indicator) * huber / kappa
    return jnp.mean(jnp.mean(rho, axis=2))

# ── Replay buffer ─────────────────────────────────────────────────────────────
def make_buffer(capacity):
    return {
        "obs": jnp.zeros((capacity, OBS_SIZE), jnp.float32),
        "action": jnp.zeros((capacity, ACTION_DIM), jnp.float32),
        "reward": jnp.zeros((capacity,), jnp.float32),
        "next_obs": jnp.zeros((capacity, OBS_SIZE), jnp.float32),
        "done": jnp.zeros((capacity,), jnp.float32),
        "ptr": jnp.int32(0), "size": jnp.int32(0),
    }

@jax.jit
def buf_add(buf, obs, action, reward, next_obs, done):
    cap = buf["obs"].shape[0]; N = obs.shape[0]
    idxs = (buf["ptr"] + jnp.arange(N)) % cap
    return {
        "obs": buf["obs"].at[idxs].set(obs),
        "action": buf["action"].at[idxs].set(action),
        "reward": buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs),
        "done": buf["done"].at[idxs].set(done),
        "ptr": jnp.int32((buf["ptr"] + N) % cap),
        "size": jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }

@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    capacity = buf["obs"].shape[0]
    idxs = jnp.where(jax.random.randint(rng_key, (batch_size,), 0, capacity) < buf["size"], 
                     jax.random.randint(rng_key, (batch_size,), 0, capacity), jnp.int32(0))
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs], buf["next_obs"][idxs], buf["done"][idxs])

# ── Core TQC Update ───────────────────────────────────────────────────────────
@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, target, online)

@jax.jit
def critic_loss_fn(critic_params, target_params, actor_params, log_alpha, obs, action, reward, next_obs, done, rng_key):
    alpha = jnp.exp(log_alpha)
    mean_n, lgs_n = actor_net.apply({"params": actor_params}, next_obs)
    next_act, next_lp = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean_n, lgs_n, extract_max_v(next_obs))
    
    target_atoms = critic_net.apply({"params": target_params}, next_obs, next_act)
    target_kept = jnp.sort(target_atoms.reshape(obs.shape[0], N_CRITICS * N_ATOMS), axis=1)[:, :N_TARGET_ATOMS]        
    backup = jax.lax.stop_gradient(reward[:, None] + GAMMA * (1.0 - done[:, None]) * (target_kept - alpha * next_lp[:, None]))

    online_atoms = critic_net.apply({"params": critic_params}, obs, action)
    total_loss, q_sum = jnp.float32(0.0), jnp.float32(0.0)
    for i in range(N_CRITICS):
        total_loss += quantile_huber_loss(online_atoms[:, i, :], backup, _TAUS)
        q_sum += jnp.mean(online_atoms[:, i, :])
    return total_loss, q_sum / N_CRITICS

@jax.jit
def actor_loss_fn(actor_params, critic_params, log_alpha, obs, rng_key):
    mean, log_std = actor_net.apply({"params": actor_params}, obs)
    action, log_pi = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean, log_std, extract_max_v(obs))
    return jnp.mean(jnp.exp(log_alpha) * log_pi) - jnp.mean(critic_net.apply({"params": critic_params}, obs, action)), jnp.mean(log_pi)

@jax.jit
def tqc_update(ap, aos, cp, cos, tp, la, laos, obs, action, reward, next_obs, done, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)
    (c_loss, q_mean), c_grads = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)(cp, tp, ap, la, obs, action, reward, next_obs, done, rng_c)
    c_upd, new_cos = critic_opt.update(c_grads, cos, cp)
    new_cp = optax.apply_updates(cp, c_upd)

    (a_loss, log_pi_mean), a_grads = jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)(ap, new_cp, la, obs, rng_a)
    a_upd, new_aos = actor_opt.update(a_grads, aos, ap)
    new_ap = optax.apply_updates(ap, a_upd)

    al_loss, al_grad = jax.value_and_grad(lambda a, lp: -a * (lp + TARGET_ENTROPY))(la, jax.lax.stop_gradient(log_pi_mean))
    al_upd, new_laos = alpha_opt.update(al_grad, laos)
    
    metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": jnp.exp(optax.apply_updates(la, al_upd)), "log_pi": log_pi_mean, "q_mean": q_mean}
    return new_ap, new_aos, new_cp, new_cos, soft_update(tp, new_cp), optax.apply_updates(la, al_upd), new_laos, metrics

# ── Fused GPU Collection & Update Loop ────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(4,))
def collect_step(actor_params, env_state, env_obs, rng_key, vmap_step):
    mean, log_std = actor_net.apply({"params": actor_params}, env_obs)
    env_action, _ = jax.vmap(sample_action)(jax.random.split(rng_key, N_ENVS), mean, log_std, extract_max_v(env_obs))
    new_obs, new_state, reward, done, info = vmap_step(jax.random.split(jax.random.split(rng_key)[1], N_ENVS), env_state, env_action)
    return new_obs, new_state, env_obs, env_action, reward, done, info

@functools.partial(jax.jit, static_argnums=(8,))
def train_chunk(ap, aos, cp, cos, tp, la, laos, buf, vmap_step, es, eo, key):
    """Executes LOG_EVERY steps of collection, buffer insertion, and gradient updates natively on GPU."""
    def _loop_body(carry, _):
        ap, aos, cp, cos, tp, la, laos, buf, es, eo, key = carry
        key, k_col, k_samp, k_upd = jax.random.split(key, 4)
        
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(ap, es, eo, k_col, vmap_step)
        
        # Calculate true terminal state (ignore timeouts for Bellman backup)
        terminal = done & ~info["timeout"]
        
        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(new_buf, k_samp, BATCH_SIZE)
        
        new_ap, new_aos, new_cp, new_cos, new_tp, new_la, new_laos, metrics = \
            tqc_update(ap, aos, cp, cos, tp, la, laos, b_obs, b_act, b_rew, b_next, b_done, k_upd)
                       
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_ap, new_aos, new_cp, new_cos, new_tp, new_la, new_laos, new_buf, new_es, new_eo, key), (step_data, metrics)
        
    carry = (ap, aos, cp, cos, tp, la, laos, buf, es, eo, key)
    new_carry, (all_step_data, all_metrics) = jax.lax.scan(_loop_body, carry, None, length=LOG_EVERY)
    return new_carry, all_step_data, all_metrics

@jax.jit
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col):
    def _scan(carry, t):
        ep_ret = carry; r, d, g, c, p = t; ep_ret = ep_ret + r
        act_col = c & ~p
        is_suc, is_acol, is_pcol, is_tmo = g, act_col & ~g, p & ~g, d & ~g & ~act_col & ~p
        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_col  = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        return jnp.where(d, 0.0, ep_ret), (out_ret, out_suc, out_col, out_pcol, out_tmo, d.astype(jnp.float32))
    _, (ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(rewards.shape[1]), (rewards, dones, goal_reached, collision, passive_col)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_col.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()

def save_checkpoint(ap, cp, tp, aos, cos, la, laos, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {"actor_params": jax.device_get(ap), "critic_params": jax.device_get(cp), "target_params": jax.device_get(tp), "actor_opt_state": jax.device_get(aos), "critic_opt_state": jax.device_get(cos), "log_alpha": jax.device_get(la), "alpha_opt_state": jax.device_get(laos), "step": int(step)}
    with open(CKPT_PATH, "wb") as f: f.write(flax.serialization.to_bytes(bundle))
    print(f"  TQC checkpoint -> {CKPT_PATH}  (step {step})")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    precision_str = "bfloat16" if args.bfloat16 else "float32"
    print(f"TQC Training — GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu}) | Precision: {precision_str}")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")

    rng = jax.random.PRNGKey(7)
    rng, ka, kc = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((2, OBS_SIZE), dtype=jnp.float32); dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    ap  = actor_net.init(ka, dummy_obs)["params"]
    cp = critic_net.init(kc, dummy_obs, dummy_act)["params"]
    tp = jax.tree_util.tree_map(jnp.array, cp)
    aos  = actor_opt.init(ap)
    cos = critic_opt.init(cp)
    la        = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    laos  = alpha_opt.init(la)

    cur_min_dist = curriculum_min_goal_dist(0.0)
    cur_stage    = _curriculum_stage(0.0)
    rolling_suc  = 0.0

    print(f"Curriculum: starting stage {cur_stage}, min_goal_dist={cur_min_dist:.1f} m")
    print(f"Initialising environments on GPU {args.gpu}...")
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state, vmap_step = init_env_state(env_rng, min_goal_dist=cur_min_dist)
    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps, n_updates = 0, 0
    best_suc = 44.0

    print("Warming up buffer (filling with random actions)...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        rng, c_rng = jax.random.split(rng)
        new_obs, env_state, obs_before, env_action, reward, done, info = collect_step(ap, env_state, env_obs, c_rng, vmap_step)
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward, new_obs, done.astype(jnp.float32))
        env_obs = new_obs
        total_steps += N_ENVS

    print("Warmup done. JIT compiling train chunk (this may take up to a minute)...")
    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | {'FPS':>7} | "
           f"{'Time':>8} | {'Stage':>5} {'MinDist':>7}")
    print(hdr); print("─" * len(hdr))

    t_start = time.time()

    # ── Training log (CSV) ───────────────────────────────────────────────────
    _LOG_PATH = "checkpoints_tqc/tqc_training_log.csv"
    os.makedirs("checkpoints_tqc", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()
        
        # ── 1. Execute Fused GPU Chunk ──
        new_carry, all_step_data, all_metrics = train_chunk(
            ap, aos, cp, cos, tp, la, laos, 
            replay_buf, vmap_step, env_state, env_obs, rng
        )
        
        ap, aos, cp, cos, tp, la, laos, replay_buf, env_state, env_obs, rng = new_carry
        
        n_updates += LOG_EVERY
        total_steps += N_ENVS * LOG_EVERY
        
        # ── 2. Vectorized Statistics ──
        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(*all_step_data)
        n_ep = int(ep_msk.sum())
        
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else: mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        m_crit = float(all_metrics["critic_loss"].mean())
        m_act  = float(all_metrics["actor_loss"].mean())
        m_alph = float(all_metrics["alpha"].mean())
        m_lpi  = float(all_metrics["log_pi"].mean())
        m_qm   = float(all_metrics["q_mean"].mean())
        
        fps = (N_ENVS * LOG_EVERY) / (time.time() - t0)
        
        # Calculate formatted elapsed time
        elapsed = time.time() - t_start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        
        print(f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
              f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
              f"{m_crit:>7.4f} {m_act:>7.4f} {m_alph:>6.4f} {m_lpi:>6.3f} {m_qm:>7.3f} | "
              f"{fps:>7,.0f} | {time_str:>8} | {cur_stage:>5d} {cur_min_dist:>5.1f}m")
        # ── CSV log row ──────────────────────────────────────────────────────
        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        # ── 3. Curriculum Update ──
        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            new_min_dist = curriculum_min_goal_dist(rolling_suc)
            new_stage    = _curriculum_stage(rolling_suc)
            
            if new_min_dist > cur_min_dist:
                cur_min_dist = new_min_dist
                cur_stage    = new_stage
                rng, reinit_rng = jax.random.split(rng)
                env_obs, env_state, vmap_step = init_env_state(reinit_rng, min_goal_dist=cur_min_dist)

        if suc_pct > best_suc:
            best_suc = suc_pct
            save_checkpoint(ap, cp, tp, aos, cos, la, laos, n_updates)

    print(f"\nTQC done! {(time.time() - t_start)/3600:.2f}h | Best success: {best_suc:.1f}%")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")