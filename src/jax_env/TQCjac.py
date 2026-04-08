import os
import csv
import argparse

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

OBS_SIZE       = 662
ACTION_DIM     = 2
N_ENVS         = 256
BUFFER_CAP     = 500_000
BATCH_SIZE     = 512
G_UPDATES      = 10
WARMUP_STEPS   = 10_000
GAMMA          = 0.99
TAU            = 0.00025
LR             = 3e-4
MAX_GRAD_NORM  = 10.0
TARGET_ENTROPY = -2.0
ACTOR_ENC_GRAD_SCALE = 0.1
TOTAL_UPDATES  = 200_000
LOG_EVERY      = 10
SAVE_EVERY     = 5000

N_CRITICS        = 5
N_ATOMS          = 25
N_TOP_ATOMS_DROP = 3
N_TARGET_ATOMS   = N_CRITICS * N_ATOMS - N_TOP_ATOMS_DROP
HUBER_KAPPA      = 1.0

MAX_V_OBS_IDX = 13
LOG_STD_EPS   = 1e-6

CKPT_DIR  = "checkpoints_tqc"
CKPT_PATH = f"{CKPT_DIR}/tqc_best.msgpack"

NET_DTYPE = jnp.bfloat16 if args.bfloat16 else jnp.float32

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

reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)

_step_auto = make_autoreset_env(reset_stacked, step_stacked)
vmap_step  = jax.jit(jax.vmap(_step_auto, in_axes=(0, 0, 0, None, None, None)))

@jax.jit
def _vmap_reset(reset_keys, min_goal_dist, ghost_prob, scenario_idx):
    def _single(key):
        return reset_stacked(key, max_goal_dist=min_goal_dist, ghost_prob=ghost_prob, scenario_idx=scenario_idx)
    return jax.vmap(_single)(reset_keys)

def init_env_state(rng_key, min_goal_dist: float = 1.5, ghost_prob: float = 0.0, scenario_idx: int = -1):
    reset_keys = jax.random.split(rng_key, N_ENVS)
    env_obs, env_state = _vmap_reset(reset_keys, jnp.float32(min_goal_dist), jnp.float32(ghost_prob), jnp.int32(scenario_idx))
    return env_obs, env_state, vmap_step

from jax_network import SharedEncoder

@jax.custom_vjp
def scale_gradient(x, scale): return x
def _scale_gradient_fwd(x, scale): return x, scale
def _scale_gradient_bwd(scale, g): return g * scale, None
scale_gradient.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)

class TQCActorHead(nn.Module):
    action_dim:  int   = ACTION_DIM
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float =  2.0

    @nn.compact
    def __call__(self, feat):
        mean    = nn.Dense(self.action_dim)(feat)
        log_std = nn.Dense(self.action_dim)(feat)
        return mean.astype(jnp.float32), jnp.clip(log_std.astype(jnp.float32), self.LOG_STD_MIN, self.LOG_STD_MAX)

class QuantileCriticBranch(nn.Module):
    n_atoms: int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        x = nn.relu(nn.Dense(256)(jnp.concatenate([feat, action], axis=-1)))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(self.n_atoms)(x).astype(jnp.float32)

class TQCCriticEnsemble(nn.Module):
    n_critics: int = N_CRITICS
    n_atoms:   int = N_ATOMS

    @nn.compact
    def __call__(self, feat, action):
        vmap_critic = nn.vmap(
            QuantileCriticBranch,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_critics
        )
        return vmap_critic(n_atoms=self.n_atoms, name='critic')(feat, action)

_TAUS = (2.0 * jnp.arange(1, N_ATOMS + 1) - 1.0) / (2.0 * N_ATOMS)

shared_enc = SharedEncoder()
actor_head = TQCActorHead()
critic_net = TQCCriticEnsemble()

enc_opt    = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
actor_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR, eps=1e-5))
ALPHA_LR   = LR
alpha_opt  = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(ALPHA_LR, eps=1e-5))

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
        "obs": buf["obs"].at[idxs].set(obs.astype(jnp.float32)),
        "action": buf["action"].at[idxs].set(action),
        "reward": buf["reward"].at[idxs].set(reward),
        "next_obs": buf["next_obs"].at[idxs].set(next_obs.astype(jnp.float32)),
        "done": buf["done"].at[idxs].set(done),
        "ptr": jnp.int32((buf["ptr"] + N) % cap),
        "size": jnp.minimum(jnp.int32(buf["size"] + N), jnp.int32(cap)),
    }


@jax.jit(static_argnames=["batch_size"])
def buf_sample(buf, rng_key, batch_size: int):
    max_idx = jnp.maximum(1, buf["size"])
    idxs = jax.random.randint(rng_key, (batch_size,), 0, max_idx)
    return (buf["obs"][idxs], buf["action"][idxs], buf["reward"][idxs], buf["next_obs"][idxs], buf["done"][idxs])

@jax.jit
def soft_update(target, online):
    return jax.tree_util.tree_map(lambda t, o: TAU * o + (1.0 - TAU) * t, target, online)

@jax.jit
def critic_loss_fn(cp, tp, ap, sep, tsep, log_alpha, obs, action, reward, next_obs, done, rng_key):
    alpha = jnp.exp(log_alpha)

    feat_next_critic = jax.lax.stop_gradient(shared_enc.apply({"params": tsep}, next_obs.astype(NET_DTYPE)))
    feat_next_actor = jax.lax.stop_gradient(shared_enc.apply({"params": sep}, next_obs.astype(NET_DTYPE)))

    mean_n, lgs_n = actor_head.apply({"params": ap}, feat_next_actor)
    next_act, next_lp = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean_n, lgs_n, extract_max_v(next_obs))

    target_atoms = critic_net.apply({"params": tp}, feat_next_critic, next_act)
    target_kept = jnp.sort(target_atoms.reshape(obs.shape[0], N_CRITICS * N_ATOMS), axis=1)[:, :N_TARGET_ATOMS]
    backup = jax.lax.stop_gradient(reward[:, None] + GAMMA * (1.0 - done[:, None]) * (target_kept - alpha * next_lp[:, None]))

    feat_obs = shared_enc.apply({"params": sep}, obs.astype(NET_DTYPE))
    online_atoms = critic_net.apply({"params": cp}, feat_obs, action.astype(NET_DTYPE))

    total_loss, q_sum = jnp.float32(0.0), jnp.float32(0.0)
    for i in range(N_CRITICS):
        total_loss += quantile_huber_loss(online_atoms[:, i, :], backup, _TAUS)
        q_sum += jnp.mean(online_atoms[:, i, :])
    return total_loss, q_sum / N_CRITICS

@jax.jit
def actor_loss_fn(ap, cp, sep, log_alpha, obs, rng_key):
    feat_raw = shared_enc.apply({"params": sep}, obs.astype(NET_DTYPE))
    feat_f   = scale_gradient(feat_raw, ACTOR_ENC_GRAD_SCALE)

    mean, log_std = actor_head.apply({"params": ap}, feat_f)
    action_new, log_pi = jax.vmap(sample_action)(jax.random.split(rng_key, obs.shape[0]), mean, log_std, extract_max_v(obs))

    q_atoms = critic_net.apply({"params": jax.lax.stop_gradient(cp)}, feat_f, action_new.astype(NET_DTYPE))
    return jnp.mean(jnp.exp(log_alpha) * log_pi) - jnp.mean(q_atoms), jnp.mean(log_pi)

@jax.jit
def tqc_update(ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, obs, action, reward, next_obs, done, rng_key):
    rng_c, rng_a = jax.random.split(rng_key)

    (c_loss, q_mean), (c_grads_cp, c_grads_sep) = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 3), has_aux=True
    )(cp, tp, ap, sep, tsep, la, obs, action, reward, next_obs, done, rng_c)
    c_upd, new_cos = critic_opt.update(c_grads_cp, cos, cp)
    new_cp = optax.apply_updates(cp, c_upd)

    (a_loss, log_pi_mean), (a_grads_ap, a_grads_sep) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 2), has_aux=True
    )(ap, new_cp, sep, la, obs, rng_a)
    a_upd, new_aos = actor_opt.update(a_grads_ap, aos, ap)
    new_ap = optax.apply_updates(ap, a_upd)

    combined_enc_grads = jax.tree_util.tree_map(lambda cg, ag: cg + ag, c_grads_sep, a_grads_sep)
    e_upd, new_eos = enc_opt.update(combined_enc_grads, eos, sep)
    new_sep = optax.apply_updates(sep, e_upd)

    al_loss, al_grad = jax.value_and_grad(lambda a, lp: -jnp.exp(a) * (lp + TARGET_ENTROPY))(la, jax.lax.stop_gradient(log_pi_mean))
    al_upd, new_laos = alpha_opt.update(al_grad, laos)

    metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": jnp.exp(optax.apply_updates(la, al_upd)), "log_pi": log_pi_mean, "q_mean": q_mean}
    return new_ap, new_aos, new_cp, new_cos, soft_update(tp, new_cp), new_sep, new_eos, soft_update(tsep, new_sep), optax.apply_updates(la, al_upd), new_laos, metrics

@functools.partial(jax.jit, static_argnums=(5,))
def collect_step(sep, ap, env_state, env_obs, rng_key, vmap_step, min_goal_dist, scenario_idx, ghost_prob):
    feat = shared_enc.apply({"params": sep}, env_obs.astype(NET_DTYPE))
    mean, log_std = actor_head.apply({"params": ap}, feat)
    env_action, _ = jax.vmap(sample_action)(jax.random.split(rng_key, N_ENVS), mean, log_std, extract_max_v(env_obs))
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 99), N_ENVS)
    new_obs, new_state, reward, done, info = vmap_step(step_keys, env_state, env_action, min_goal_dist, scenario_idx, ghost_prob)
    return new_obs, new_state, env_obs, env_action, reward, done, info

@functools.partial(jax.jit, static_argnums=(11,), donate_argnums=(10, 15, 16))
def train_chunk(ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, buf, vmap_step, min_goal_dist, scenario_idx, ghost_prob, es, eo, key):
    def _collect_body(carry, _):
        es_, eo_, buf_, key_ = carry
        key_, k_col = jax.random.split(key_)
        new_eo, new_es, obs_b, env_a, rew, done, info = collect_step(sep, ap, es_, eo_, k_col, vmap_step, min_goal_dist, scenario_idx, ghost_prob)
        terminal = done
        new_buf = buf_add(buf_, obs_b, env_a, rew, new_eo, terminal.astype(jnp.float32))
        step_data = (rew, done, info["goal_reached"], info["collision"], info["passive_col"])
        return (new_es, new_eo, new_buf, key_), step_data

    (new_es, new_eo, new_buf, key), all_step_data = jax.lax.scan(
        _collect_body, (es, eo, buf, key), None, length=LOG_EVERY
    )

    def _update_step(update_carry, upd_idx):
        ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_, la_, laos_, key_ = update_carry
        k_samp = jax.random.fold_in(key_, upd_idx * 2)
        k_upd  = jax.random.fold_in(key_, upd_idx * 2 + 1)
        b_obs, b_act, b_rew, b_next, b_done = buf_sample(new_buf, k_samp, BATCH_SIZE)
        new_ap_, new_aos_, new_cp_, new_cos_, new_tp_, new_sep_, new_eos_, new_tsep_, new_la_, new_laos_, metrics_ = \
            tqc_update(ap_, aos_, cp_, cos_, tp_, sep_, eos_, tsep_, la_, laos_, b_obs, b_act, b_rew, b_next, b_done, k_upd)
        return (new_ap_, new_aos_, new_cp_, new_cos_, new_tp_, new_sep_, new_eos_, new_tsep_, new_la_, new_laos_, key_), metrics_

    update_carry_init = (ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, key)
    (new_ap, new_aos, new_cp, new_cos, new_tp, new_sep, new_eos, new_tsep, new_la, new_laos, key), all_metrics = jax.lax.scan(
        _update_step, update_carry_init, jnp.arange(G_UPDATES * LOG_EVERY)
    )

    new_carry = (new_ap, new_aos, new_cp, new_cos, new_tp, new_sep, new_eos, new_tsep, new_la, new_laos, new_buf, new_es, new_eo, key)
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

def save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, la, laos, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    bundle = {
        "enc_params":        jax.device_get(sep),
        "target_enc_params": jax.device_get(tsep),
        "actor_params":      jax.device_get(ap),
        "critic_params":     jax.device_get(cp),
        "target_params":     jax.device_get(tp),
        "enc_opt_state":     jax.device_get(eos),
        "actor_opt_state":   jax.device_get(aos),
        "critic_opt_state":  jax.device_get(cos),
        "log_alpha":         jax.device_get(la),
        "alpha_opt_state":   jax.device_get(laos),
        "step":              int(step),
    }
    with open(CKPT_PATH, "wb") as f: f.write(flax.serialization.to_bytes(bundle))
    print(f"  TQC checkpoint -> {CKPT_PATH}  (step {step})")

if __name__ == "__main__":
    precision_str = "bfloat16" if args.bfloat16 else "float32"
    print(f"TQC Training — GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu}) | Precision: {precision_str}")
    print(f"  N_ENVS={N_ENVS}  BUFFER={BUFFER_CAP:,}  BATCH={BATCH_SIZE}")
    print(f"  N_CRITICS={N_CRITICS}  N_ATOMS={N_ATOMS}  N_TOP_DROP={N_TOP_ATOMS_DROP}  N_TARGET_ATOMS={N_TARGET_ATOMS}")

    rng = jax.random.PRNGKey(7)
    rng, k_se, ka, kc = jax.random.split(rng, 4)
    dummy_obs = jnp.zeros((2, OBS_SIZE), dtype=jnp.float32)
    dummy_act = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)

    sep  = shared_enc.init(k_se, dummy_obs)["params"]
    tsep = jax.tree_util.tree_map(jnp.array, sep)

    dummy_feat = shared_enc.apply({"params": sep}, dummy_obs)
    ap  = actor_head.init(ka, dummy_feat)["params"]
    cp  = critic_net.init(kc, dummy_feat, dummy_act)["params"]
    tp  = jax.tree_util.tree_map(jnp.array, cp)

    eos  = enc_opt.init(sep)
    aos  = actor_opt.init(ap)
    cos  = critic_opt.init(cp)
    la   = jnp.array(0.0, dtype=jnp.float32)
    laos = alpha_opt.init(la)

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
        new_obs, env_state, obs_before, env_action, reward, done, info = collect_step(
            sep, ap, env_state, env_obs, c_rng, vmap_step,
            jnp.float32(cur_min_dist), jnp.int32(-1), jnp.float32(0.0)
        )
        replay_buf = buf_add(replay_buf, obs_before, env_action, reward, new_obs, done.astype(jnp.float32))
        env_obs = new_obs
        total_steps += N_ENVS

    print("Warmup done. JIT compiling train chunk (this may take up to a minute)...")
    _c, _sd, _m = train_chunk(
        ap, aos, cp, cos, tp, sep, eos, tsep, la, laos,
        replay_buf, vmap_step, jnp.float32(cur_min_dist), jnp.int32(-1), jnp.float32(0.0), env_state, env_obs, rng
    )
    jax.block_until_ready(_c)
    ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, replay_buf, env_state, env_obs, rng = _c
    n_updates   += LOG_EVERY
    total_steps += N_ENVS * LOG_EVERY
    print("Compilation done.")

    hdr = (f"{'Upd':>7} | {'Steps':>10} | {'EpRet':>7} | "
           f"{'Suc%':>5} {'ACo%':>5} {'PCo%':>5} {'Tmo%':>5} | "
           f"{'CritL':>7} {'ActL':>7} {'Alpha':>6} {'LogPi':>6} {'Qmean':>7} | {'FPS':>7} | "
           f"{'Time':>8} | {'Stage':>5} {'MinDist':>7}")
    print(hdr); print("─" * len(hdr))

    t_start = time.time()

    _LOG_PATH = "checkpoints_tqc/tqc_training_log.csv"
    os.makedirs("checkpoints_tqc", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "col_pct",
                           "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    while n_updates < TOTAL_UPDATES:
        t0 = time.time()

        new_carry, all_step_data, all_metrics = train_chunk(
            ap, aos, cp, cos, tp, sep, eos, tsep, la, laos,
            replay_buf, vmap_step, jnp.float32(cur_min_dist), jnp.int32(-1), jnp.float32(0.0), env_state, env_obs, rng
        )

        ap, aos, cp, cos, tp, sep, eos, tsep, la, laos, replay_buf, env_state, env_obs, rng = new_carry

        n_updates += LOG_EVERY
        total_steps += N_ENVS * LOG_EVERY

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

        print(f"{n_updates:>7d} | {total_steps:>10,} | {mean_ret:>7.1f} | "
              f"{suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
              f"{m_crit:>7.4f} {m_act:>7.4f} {m_alph:>6.4f} {m_lpi:>6.3f} {m_qm:>7.3f} | "
              f"{fps:>7,.0f} | {time_str:>8} | {cur_stage:>5d} {cur_min_dist:>5.1f}m")
        _log_writer.writerow([total_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(col_pct, 4),
                               round(pcol_pct, 4), round(tmo_pct, 4), n_ep])
        _log_file.flush()

        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct
            new_min_dist = curriculum_min_goal_dist(rolling_suc)
            new_stage    = _curriculum_stage(rolling_suc)

            if new_min_dist > cur_min_dist:
                print(f"*** Curriculum advance: stage {new_stage}, min_dist={new_min_dist:.1f}. Keeping buffer intact. ***")
                cur_min_dist = new_min_dist
                cur_stage    = new_stage
                rng, reinit_rng = jax.random.split(rng)
                reset_keys = jax.random.split(reinit_rng, N_ENVS)
                env_obs, env_state = _vmap_reset(reset_keys, jnp.float32(cur_min_dist), jnp.float32(0.0), jnp.int32(-1))

        if suc_pct > best_suc:
            best_suc = suc_pct
            save_checkpoint(sep, tsep, ap, cp, tp, eos, aos, cos, la, laos, n_updates)

    print(f"\nTQC done! {(time.time() - t_start)/3600:.2f}h | Best success: {best_suc:.1f}%")
    _log_file.close()
    print(f"Training log saved -> {_LOG_PATH}")
