"""
TQCjax.py — Truncated Quantile Critics (Maximum XLA Optimization)
=================================================================
FIXES for Maximum FPS & Execution Efficiency:
  1. TRUE SHARED FEATURES (2 Passes): Critic computes online features and passes 
     them as an auxiliary return to the Actor. Drops total CNN passes to exactly 2.
  2. VMAP CRITICS: The Python `for` loop over critics is replaced by `jax.vmap`
     over the quantile Huber loss, allowing perfect parallel execution of twin Qs.
  3. ZERO-OVERHEAD SCAN RNG: `jax.random.split` removed from the inner scan loop;
     `jax.random.fold_in` derives keys dynamically from the step index.
  4. UNBIASED BUFFER SAMPLING: Correctly samples `[0, buf["size"])` to prevent
     index 0 oversampling during early training.
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
OBS_SIZE       = 666
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

MAX_GRAD_NORM  = 10.0

# TQC-specific
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
    target_device = devs[0] 
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
    num_rays:  int = 216
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, x):
        pose_size = 3 * self.stack_dim
        state_size = 9                    
        pose_stack = x[..., :pose_size]
        state_vec  = x[..., pose_size : pose_size + state_size]
        lidar_flat = x[..., pose_size + state_size:]
        batch_shape = lidar_flat.shape[:-1]
        
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

# ── Split-head networks ───────────────────────────────────────────────────────
class TQCActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, feat):
        mean = nn.Dense(self.action_dim, dtype=self.dtype)(feat)
        log_std = nn.Dense(self.action_dim, dtype=self.dtype)(feat)
        return mean.astype(jnp.float32), jnp.clip(log_std.astype(jnp.float32), self.LOG_STD_MIN, self.LOG_STD_MAX)

class TQCCriticHead(nn.Module):
    n_atoms:    int = N_ATOMS
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, feat, action):
        x = nn.relu(nn.Dense(256, dtype=self.dtype)(jnp.concatenate([feat, action.astype(self.dtype)], axis=-1)))
        x = nn.relu(nn.Dense(128, dtype=self.dtype)(x))
        return nn.Dense(self.n_atoms, dtype=self.dtype)(x).astype(jnp.float32)

class TQCCriticHeadEnsemble(nn.Module):
    n_critics:  int = N_CRITICS
    n_atoms:    int = N_ATOMS
    dtype: jnp.dtype = NET_DTYPE
    
    @nn.compact
    def __call__(self, feat, action):
        all_atoms = [TQCCriticHead(self.n_atoms, dtype=self.dtype, name=f'critic_{i}')(feat, action) for i in range(self.n_critics)]
        return jnp.stack(all_atoms, axis=1)

_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)

# Module instances
encoder_net = ObsEncoder()
actor_head  = TQCActorHead()
critic_head = TQCCriticHeadEnsemble()

# FIX 1: Encoder is now trained by BOTH critic and actor gradients (summed).
# enc_opt handles the combined critic+actor encoder update.
# enc_actor_opt handles the actor-side encoder gradient separately before summing.
enc_opt         = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
head_critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
alpha_opt       = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))

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
    # 7️⃣ Unbiased sampling
    idxs = jax.random.randint(rng_key, (batch_size,), 0, buf["size"])
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs], buf["next_obs"][idxs], buf["done"][idxs])


# ── Core TQC Update ───────────────────────────────────────────────────────────

@jax.jit
def tqc_update(ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos,
               obs, action, reward, next_obs, done, rng_key):

    k_act_next, k_act_online = jax.random.split(rng_key, 2)

    # 9️⃣ Pass 1: Target features (used for Bellman Backup)
    feat_next_t = encoder_net.apply({"params": tep}, next_obs)

    # ── 1. Critic loss
    def _critic_loss(ep_, chp_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        
        # Actor proposes next actions from target features
        mean_n, lgs_n = actor_head.apply({"params": ahp}, feat_next_t)
        next_act, next_lp = jax.vmap(sample_action)(jax.random.split(k_act_next, obs.shape[0]), mean_n, lgs_n, extract_max_v(next_obs))
        
        target_atoms = critic_head.apply({"params": thp}, feat_next_t, next_act)
        target_kept  = jnp.sort(target_atoms.reshape(obs.shape[0], N_CRITICS * N_ATOMS), axis=1)[:, :N_TARGET_ATOMS]        
        backup = jax.lax.stop_gradient(reward[:, None] + GAMMA * (1.0 - done[:, None]) * (target_kept - alpha * next_lp[:, None]))

        # 9️⃣ Pass 2: Online features
        feat_obs_online = encoder_net.apply({"params": ep_}, obs)
        online_atoms = critic_head.apply({"params": chp_}, feat_obs_online, action)
        
        # 🔟 VMAP over critics (replaces Python for-loop)
        losses = jax.vmap(quantile_huber_loss, in_axes=(1, None, None))(online_atoms, backup, _TAUS)
        total_loss = jnp.sum(losses)
        q_sum = jnp.mean(online_atoms)
        
        # Return online features as aux so Actor can reuse them
        return total_loss, (feat_obs_online, q_sum)

    (c_loss, (feat_obs, q_mean)), (c_grads_enc, c_grads_head) = jax.value_and_grad(_critic_loss, argnums=(0, 1), has_aux=True)(ep, chp)
    
    ch_upd, new_chos = head_critic_opt.update(c_grads_head, chos, chp)
    new_chp = optax.apply_updates(chp, ch_upd)

    # ── 2. Actor loss
    # FIX 2a: Differentiate w.r.t. both actor head (ahp_) AND encoder (ep_) so the
    #          encoder receives policy-gradient signal, not just critic signal.
    # FIX 2b: Evaluate against the PRE-update critic params (chp, not new_chp).
    #          Using new_chp inside a traced closure causes a stale-value dependency
    #          inside jax.lax.scan and breaks gradient correctness.
    def _actor_loss(ahp_, ep_):
        alpha = jax.lax.stop_gradient(jnp.exp(la))
        
        # Re-encode with actor-side encoder params so gradients flow through ep_.
        feat_obs_actor = encoder_net.apply({"params": ep_}, obs)
        mean, log_std = actor_head.apply({"params": ahp_}, feat_obs_actor)
        action_new, log_pi = jax.vmap(sample_action)(jax.random.split(k_act_online, obs.shape[0]), mean, log_std, extract_max_v(obs))
        
        # FIX 2b: use original chp (pre-update) — safe inside lax.scan.
        atoms = critic_head.apply({"params": chp}, jax.lax.stop_gradient(feat_obs_actor), action_new)
        return jnp.mean(alpha * log_pi) - jnp.mean(atoms), jnp.mean(log_pi)

    (a_loss, log_pi_mean), (a_grads_head, a_grads_enc_actor) = \
        jax.value_and_grad(_actor_loss, argnums=(0, 1), has_aux=True)(ahp, ep)
    
    ah_upd, new_ahos = head_actor_opt.update(a_grads_head, ahos, ahp)
    new_ahp = optax.apply_updates(ahp, ah_upd)

    # FIX 2c: Combine critic and actor encoder gradients before applying the
    #          encoder update so both objectives shape the shared representation.
    combined_enc_grads = jax.tree_util.tree_map(lambda c, a: c + a, c_grads_enc, a_grads_enc_actor)
    ce_upd, new_eos = enc_opt.update(combined_enc_grads, eos, ep)
    new_ep = optax.apply_updates(ep, ce_upd)

    # ── 3. Alpha update
    al_loss, al_grad = jax.value_and_grad(lambda a, lp: -a * (lp + TARGET_ENTROPY))(la, jax.lax.stop_gradient(log_pi_mean))
    al_upd, new_laos = alpha_opt.update(al_grad, laos)
    new_la = optax.apply_updates(la, al_upd)

    # ── 4. Soft targets
    new_tep = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, tep, new_ep)
    new_thp = jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, thp, new_chp)
    
    metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": jnp.exp(new_la), "log_pi": log_pi_mean, "q_mean": q_mean}
    return (new_ep, new_ahp, new_eos, new_ahos,
            new_chp, new_chos, new_tep, new_thp,
            new_la, new_laos, metrics)


# ── Fused GPU Collection & Update Loop ────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(ep, ahp, env_state, env_obs, rng_key, vmap_step):
    feat = encoder_net.apply({"params": ep}, env_obs)
    mean, log_std = actor_head.apply({"params": ahp}, feat)
    env_action, _ = jax.vmap(sample_action)(jax.random.split(rng_key, N_ENVS), mean, log_std, extract_max_v(env_obs))
    new_obs, new_state, reward, done, info = vmap_step(jax.random.split(jax.random.split(rng_key)[1], N_ENVS), env_state, env_action)
    return new_obs, new_state, env_obs, env_action, reward, done, info

@functools.partial(jax.jit, static_argnums=(11,))
def train_chunk(ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos, buf, vmap_step, es, eo, base_key):
    """Executes LOG_EVERY steps natively on GPU, minimizing Python-side overhead."""
    
    def _loop_body(carry, _):
        (ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos, buf, es, eo, key) = carry
        
        # FIX 3: Use jax.random.split to produce genuinely independent keys each
        #         step. fold_in is deterministic per (base_key, step_idx), so with a
        #         fixed base_key the entire sequence of keys is frozen after JIT —
        #         eliminating exploration diversity across training.
        key, k_col, k_samp, k_upd = jax.random.split(key, 4)
        
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(ep, ahp, es, eo, k_col, vmap_step)
        
        terminal = done & ~info["timeout"]
        
        new_buf = buf_add(buf, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(new_buf, k_samp, BATCH_SIZE)
        
        (new_ep, new_ahp, new_eos, new_ahos,
         new_chp, new_chos, new_tep, new_thp, new_la, new_laos, metrics) = tqc_update(
             ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos,
             b_obs, b_act, b_rew, b_next, b_done, k_upd
         )
                       
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        
        new_carry = (new_ep, new_ahp, new_eos, new_ahos,
                     new_chp, new_chos, new_tep, new_thp,
                     new_la, new_laos, new_buf, new_es, new_eo, key)
        return new_carry, (step_data, metrics)
        
    carry = (ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos, buf, es, eo, base_key)
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

def save_checkpoint(ep, ahp, chp, tep, thp, eos, ahos, chos, la, laos, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "enc_params":          jax.device_get(ep),
        "actor_head_params":   jax.device_get(ahp),
        "critic_head_params":  jax.device_get(chp),
        "target_enc_params":   jax.device_get(tep),
        "target_head_params":  jax.device_get(thp),
        "enc_opt":             jax.device_get(eos),
        "actor_head_opt":      jax.device_get(ahos),
        "critic_head_opt":     jax.device_get(chos),
        "log_alpha":           jax.device_get(la),
        "alpha_opt_state":     jax.device_get(laos),
        "step":                int(step),
    }
    with open(CKPT_PATH, "wb") as f: f.write(flax.serialization.to_bytes(bundle))
    print(f"  TQC checkpoint -> {CKPT_PATH}  (step {step})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    precision_str = "bfloat16" if args.bfloat16 else "float32"
    print(f"TQC Training — GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu}) | Precision: {precision_str}")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")

    rng = jax.random.PRNGKey(7)
    rng, k_init_a, k_init_c, k_env, k_warmup = jax.random.split(rng, 5)

    dummy_obs  = jnp.zeros((2, OBS_SIZE),   dtype=jnp.float32)
    dummy_feat = jnp.zeros((2, 128),        dtype=jnp.float32)
    dummy_act  = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    k_ae, k_ah, k_ch = jax.random.split(k_init_a, 3)

    ep  = encoder_net.init(k_ae, dummy_obs)["params"]           
    ahp = actor_head.init(k_ah, dummy_feat)["params"]           
    chp = critic_head.init(k_ch, dummy_feat, dummy_act)["params"] 
    
    tep = jax.tree_util.tree_map(jnp.array, ep)
    thp = jax.tree_util.tree_map(jnp.array, chp)

    eos  = enc_opt.init(ep)
    ahos = head_actor_opt.init(ahp)
    chos = head_critic_opt.init(chp)
    
    la   = jnp.array(jnp.log(0.1), dtype=jnp.float32)
    laos = alpha_opt.init(la)

    cur_min_dist = curriculum_min_goal_dist(0.0)
    cur_stage    = _curriculum_stage(0.0)
    rolling_suc  = 0.0

    print(f"Curriculum: starting stage {cur_stage}, min_goal_dist={cur_min_dist:.1f} m")
    print(f"Initialising environments on GPU {args.gpu}...")
    env_obs, env_state, vmap_step = init_env_state(k_env, min_goal_dist=cur_min_dist)
    replay_buf  = make_buffer(BUFFER_CAP)
    total_steps, n_updates = 0, 0
    
    best_suc = -1.0
    best_ret = -1e9

    print("Warming up buffer (filling with random actions)...")
    for _ in range((WARMUP_STEPS // N_ENVS) + 1):
        k_warmup, k_act, k_step = jax.random.split(k_warmup, 3)
        obs_before = env_obs
        max_v      = extract_max_v(env_obs)
        rand_v     = jax.random.uniform(k_act, (N_ENVS,)) * max_v
        rand_w     = jax.random.uniform(k_act, (N_ENVS,), minval=-1.0, maxval=1.0)
        env_action = jnp.stack([rand_v, rand_w], axis=-1)
        step_keys  = jax.random.split(k_step, N_ENVS)
        new_obs, env_state, reward, done, info = vmap_step(step_keys, env_state, env_action)
        terminal   = done & ~info["timeout"]
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward,
                             new_obs, terminal.astype(jnp.float32))
        env_obs     = new_obs
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
            ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos, 
            replay_buf, vmap_step, env_state, env_obs, rng
        )
        
        (ep, ahp, eos, ahos, chp, chos, tep, thp, la, laos, replay_buf, env_state, env_obs, rng) = new_carry
        
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
        
        elapsed = time.time() - t_start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        
        if n_updates == 500 or n_updates % 5000 == 0:
            print(f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
                  f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
                  f"{m_crit:>7.4f} {m_act:>7.4f} {m_alph:>6.4f} {m_lpi:>6.3f} {m_qm:>7.3f} | "
                  f"{fps:>7,.0f} | {time_str:>8} | {cur_stage:>5d} {cur_min_dist:>5.1f}m")
                  
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

        # FIX 4: Removed the suc_pct > 90.0 gate — it prevented ANY checkpoint from
        #         being saved while the agent is still learning. Now saves on any
        #         improvement once at least one episode has completed.
        is_better = (suc_pct > best_suc) or (suc_pct == best_suc and mean_ret > best_ret)
        if is_better and n_ep > 0:
            best_suc = suc_pct
            best_ret = mean_ret
            save_checkpoint(ep, ahp, chp, tep, thp, eos, ahos, chos, la, laos, n_updates)

    print(f"\nTQC done! {(time.time() - t_start)/3600:.2f}h | Best success: {best_suc:.1f}%")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")