"""
jax_ppo.py — Single-GPU PPO Training  (GPU 0), must train always with GPU, no CPU, never CPU

"""

import os
import csv
import argparse

# Parse arguments BEFORE setting environment variables and importing JAX
parser = argparse.ArgumentParser(description="JAX PPO Training")
# Keep type=str because os.environ requires string values
parser.add_argument("--gpu", type=str, default="0", choices=["0", "1"], help="Target GPU ID (0 or 1)")
args, _ = parser.parse_known_args()

# ── GPU memory configuration ─────────────────────────────────────────────────
# Use the BFC allocator (default JAX behaviour) — NOT "platform".
# "platform" allocates per-buffer from the OS, bypasses JAX's memory pool,
# and causes fragmentation + CUDA_ILLEGAL_ADDRESS when the pool is exhausted.
# BFC pre-reserves a fraction of VRAM and manages it internally.
#
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 reserves 88% of VRAM (~8.8 GB on a
# 10 GB card), leaving ~1.2 GB for the CUDA driver, cuDNN workspace, and OS.
# Do NOT set XLA_PYTHON_CLIENT_ALLOCATOR — absence means BFC (the safe default).
os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
# cuda_malloc_async reduces driver-level fragmentation without changing the
# JAX allocator — safe to keep.
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import warnings
import jax
import jax.numpy as jnp
import functools

# ALWAYS index 0, regardless of which physical GPU you chose
jax.config.update("jax_default_device", jax.devices("cuda")[0])

import optax
import flax.serialization
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

#from jax_network import EndToEndActorCritic
from jax_network import DecoupledActorCritic
from jax_train import collect_rollouts, init_env_state, NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE



# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.25
# FIX: ENTROPY_COEF abbassato 0.01→0.003.
# Con reward scale -50/-10 e valore assoluto episodio ~50, il value loss domina.
# ENTROPY_COEF=0.01 era troppo alto: il gradient dell'entropia spingeva sempre
# verso logstd_max, bloccando l'entropia a 2.838 (il massimo con LOG_STD_MAX=0.0).
# Con 0.003 l'entropia può scendere naturalmente quando la policy trova azioni
# che massimizzano il reward, senza essere soffocata dall'entropy bonus.
ENTROPY_COEF   = 0.01
MAX_GRAD_NORM  = 0.5
PPO_EPOCHS     = 6
LR_START       = 5e-4
LR_END         = 1e-5
LR_MIN         = 1e-5
WARMUP_UPDATES = 5
# TOTAL_UPDATES: 128 × 16384 × 96 = 201,326,592 ≈ 200M env steps
TOTAL_UPDATES  = 800

BATCH_SIZE      = NUM_ENVS * ROLLOUT_STEPS          # 8192 × 64 = 524,288
# N_MINIBATCHES=64: MBS = 524288/64 = 8192 — matches GPU L2 cache working set
# well on 10 GB cards. Previous 128 minibatches × 16384 envs = 12288 MBS,
# but 16384 envs was already OOM before minibatching.
N_MINIBATCHES   = 64
MINI_BATCH_SIZE = BATCH_SIZE // N_MINIBATCHES       # 8192

assert BATCH_SIZE % N_MINIBATCHES == 0, (
    f"BATCH_SIZE={BATCH_SIZE} not divisible by N_MINIBATCHES={N_MINIBATCHES}."
)

# Each outer update runs PPO_EPOCHS × N_MINIBATCHES optimizer steps inside
# jax.lax.scan, so the optax step counter advances by that factor per update.
# The scheduler must be expressed in true optimizer-step units, otherwise
# warmup exhausts in update 0 and LR collapses to LR_END by update 1-2.
_OPT_STEPS_PER_UPDATE = PPO_EPOCHS * N_MINIBATCHES   # 6 × 64 = 384
_WARMUP_OPT_STEPS     = WARMUP_UPDATES * _OPT_STEPS_PER_UPDATE   # 1_920
_TOTAL_OPT_STEPS      = TOTAL_UPDATES  * _OPT_STEPS_PER_UPDATE   # 49_152

#network = EndToEndActorCritic(action_dim=2)
network = DecoupledActorCritic(action_dim=2)

# ── Curriculum ────────────────────────────────────────────────────────────────
# Each entry: (suc_pct_threshold, max_goal_dist)
# The curriculum controls the MAXIMUM distance the goal can be from the robot.
# Stage 0: goal within 1.5m — robot learns basic goal-seeking.
# As success rate rises, max_goal_dist increases — robot must navigate farther.
CURRICULUM_STAGES = [
    (25.0, 1.5),
    (38.0, 2.5),
    (50.0, 4.0),
    (60.0, 5.0),
    (70.0, 6.5),
    (80.0, 8.0),
    (101., 9.0),
]

GHOST_PROB_STAGES = [
    (50.0, 1.0),
    (65.0, 0.8),
    (78.0, 0.6),
    (101., 0.4),
]

from flax import struct

@struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls):
        return cls(mean=jnp.array(0.0), var=jnp.array(1.0), count=jnp.array(1e-4))

    def update(self, x: jnp.ndarray):
        batch_mean = jnp.mean(x)
        batch_var = jnp.var(x)
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        return self.replace(mean=new_mean, var=new_var, count=tot_count)

@jax.jit
def normalize_batch_rewards(rewards, dones, running_ret, rms_state, gamma):
    """
    Calculates discounted returns, updates the running statistics,
    and scales the rewards by the running standard deviation.
    """
    def _step(ret, t):
        r, d = t
        ret = r + gamma * ret * (1.0 - d)
        return ret, ret

    # Scan over the ROLLOUT_STEPS axis
    running_ret, returns = jax.lax.scan(_step, running_ret, (rewards, dones))

    # Update running stats with the flattened returns
    new_rms_state = rms_state.update(returns.flatten())

    # Normalize rewards and clip to [-10, 10] to prevent extreme gradient spikes
    normalized_rewards = rewards / jnp.sqrt(new_rms_state.var + 1e-8)
    normalized_rewards = jnp.clip(normalized_rewards, -10.0, 10.0)

    return normalized_rewards, running_ret, new_rms_state

def curriculum_ghost_prob(suc_pct: float) -> float:
    for threshold, prob in GHOST_PROB_STAGES:
        if suc_pct < threshold:
            return prob
    return GHOST_PROB_STAGES[-1][1]

def curriculum_max_goal_dist(suc_pct: float) -> float:
    for threshold, dist in CURRICULUM_STAGES:
        if suc_pct < threshold:
            return dist
    return CURRICULUM_STAGES[-1][1]

def _curriculum_stage(suc_pct: float) -> int:
    for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
        if suc_pct < threshold:
            return i
    return len(CURRICULUM_STAGES) - 1

_warmup_schedule = optax.linear_schedule(
    init_value=LR_MIN,
    end_value=LR_START,
    transition_steps=_WARMUP_OPT_STEPS,
)
_decay_schedule = optax.linear_schedule(
    init_value=LR_START,
    end_value=LR_END,
    transition_steps=_TOTAL_OPT_STEPS - _WARMUP_OPT_STEPS,
)
scheduler = optax.join_schedules(
    schedules=[_warmup_schedule, _decay_schedule],
    boundaries=[_WARMUP_OPT_STEPS],
)

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(learning_rate=scheduler, eps=1e-5),
)


@functools.partial(jax.jit, static_argnums=(4, 7, 8))
def ppo_train_chunk(train_state, env_state, env_obs, rms_state, vmap_step, running_ret, rng_key, max_goal_dist, scenario_idx):
    """
    Esegue LOG_EVERY update completi di PPO interamente su GPU.
    Nessuna sincronizzazione con il processore host fino alla fine del blocco.
    """
    def _update_step(carry, _):
        (ts, es, eo, rms, r_ret, key) = carry
        key, k_roll, k_upd = jax.random.split(key, 3)

        # 1. Raccolta dati (fusa direttamente nell'update)
        rollout_history, next_es, next_eo, last_val = collect_rollouts(
            k_roll, ts[0], network.apply, vmap_step, es, eo, max_goal_dist, scenario_idx
        )

        # 2. Normalizzazione dinamica dei reward
        rewards, new_r_ret, new_rms = normalize_batch_rewards(
            rollout_history["rewards"], rollout_history["dones"], r_ret, rms, GAMMA
        )

        # 3. Calcolo GAE
        adv, ret = compute_gae(
            rewards, rollout_history["values"], rollout_history["dones"], last_val
        )

        # 4. Aggiornamento Pesi (Actor e Critic)
        new_ts, mean_loss, aux = run_ppo_updates(
            ts,
            rollout_history["obs"].reshape(-1, OBS_SIZE),
            rollout_history["actions"].reshape(-1, 2),
            adv.reshape(-1),
            ret.reshape(-1),
            rollout_history["log_probs"].reshape(-1),
            k_upd
        )

        # Dati da restituire per il logging a fine chunk
        step_data = (
            rollout_history["rewards"], rollout_history["dones"],
            rollout_history["goal_reached"], rollout_history["collision"],
            rollout_history["passive_col"], rollout_history["active_col"]
        )
        
        new_carry = (new_ts, next_es, next_eo, new_rms, new_r_ret, key)
        return new_carry, (step_data, mean_loss, aux)

    # Scansiona per LOG_EVERY update
    carry = (train_state, env_state, env_obs, rms_state, running_ret, rng_key)
    new_carry, (all_step_data, all_losses, all_aux) = jax.lax.scan(
        _update_step, carry, None, length=LOG_EVERY
    )
    
    return new_carry, all_step_data, all_losses, all_aux


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



# ── Fused Episode Outcomes ──────────────────────────────────────────────────
@jax.jit
def collect_episode_outcomes_chunked(rewards, dones, goal_reached, collision, passive_col, active_col):
    """
    Accumulates episode outcomes across the entire chunk (LOG_EVERY, ROLLOUT_STEPS, NUM_ENVS).
    This matches the SAC implementation to avoid pulling massive tensors back to the CPU.
    """
    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p, a = t
        ep_ret = ep_ret + r

        is_suc  = g
        is_acol = a & ~is_suc
        is_pcol = p & ~is_suc
        is_obs  = c & ~a & ~p & ~is_suc
        is_tmo  = d & ~is_suc & ~c

        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_obs  = jnp.where(d, is_obs.astype(jnp.float32),  0.0)
        out_acol = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        out_msk  = d.astype(jnp.float32)

        ep_ret = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_obs, out_acol, out_pcol, out_tmo, out_msk)

    # Flatten the first two dimensions (LOG_EVERY * ROLLOUT_STEPS) so we can scan over time
    N_ENVS = rewards.shape[-1]
    flat_len = rewards.shape[0] * rewards.shape[1]
    
    r_flat = rewards.reshape(flat_len, N_ENVS)
    d_flat = dones.reshape(flat_len, N_ENVS)
    g_flat = goal_reached.reshape(flat_len, N_ENVS)
    c_flat = collision.reshape(flat_len, N_ENVS)
    p_flat = passive_col.reshape(flat_len, N_ENVS)
    a_flat = active_col.reshape(flat_len, N_ENVS)

    _, (ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N_ENVS),
        (r_flat, d_flat, g_flat, c_flat, p_flat, a_flat)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_obs.ravel(), ep_acol.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()


@jax.jit
def compute_gae(rewards, values, dones, last_val):
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

    value_loss   = VF_COEF * jnp.mean((returns - values) ** 2)
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
def collect_episode_outcomes(rewards, dones, goal_reached, collision, passive_col, active_col):
    """
    Build mutually exclusive episode outcome categories that sum to 100%.
    
    Categories:
      1. success : goal_reached
      2. obs_col : wall or static obstacle collision
      3. acol    : active human collision
      4. pcol    : passive human collision
      5. timeout : done without success or collision
    """
    N = rewards.shape[1]

    def _scan(carry, t):
        ep_ret = carry
        r, d, g, c, p, a = t
        ep_ret = ep_ret + r

        # Strict priority masks (each episode falls in exactly one)
        is_suc  = g
        is_acol = a & ~is_suc
        is_pcol = p & ~is_suc
        is_obs  = c & ~a & ~p & ~is_suc
        is_tmo  = d & ~is_suc & ~c

        out_ret  = jnp.where(d, ep_ret, 0.0)
        out_suc  = jnp.where(d, is_suc.astype(jnp.float32),  0.0)
        out_obs  = jnp.where(d, is_obs.astype(jnp.float32),  0.0)
        out_acol = jnp.where(d, is_acol.astype(jnp.float32), 0.0)
        out_pcol = jnp.where(d, is_pcol.astype(jnp.float32), 0.0)
        out_tmo  = jnp.where(d, is_tmo.astype(jnp.float32),  0.0)
        out_msk  = d.astype(jnp.float32)

        ep_ret = jnp.where(d, 0.0, ep_ret)
        return ep_ret, (out_ret, out_suc, out_obs, out_acol, out_pcol, out_tmo, out_msk)

    _, (ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk) = jax.lax.scan(
        _scan, jnp.zeros(N),
        (rewards, dones, goal_reached, collision, passive_col, active_col)
    )
    return ep_rets.ravel(), ep_suc.ravel(), ep_obs.ravel(), ep_acol.ravel(), ep_pcol.ravel(), ep_tmo.ravel(), ep_msk.ravel()


























if __name__ == "__main__":
    # We define LOG_EVERY globally so the JIT compiler can use it
    global LOG_EVERY
    LOG_EVERY = 10 
    
    print(f"PPO Training — GPU {args.gpu}  [Fused Chunk Mode]")
    print(f"  Envs       : {NUM_ENVS}  x  steps {ROLLOUT_STEPS}  =  {BATCH_SIZE:,} batch")
    print(f"  Minibatches: {N_MINIBATCHES} x {MINI_BATCH_SIZE} | epochs {PPO_EPOCHS}")
    print(f"  VF_COEF={VF_COEF}  ENTROPY_COEF={ENTROPY_COEF}  LR warmup {LR_MIN}->{LR_START} then decay ->{LR_END}")
    print(f"  OBS_SIZE={OBS_SIZE}  log_std: global param, bias=-1.0, clamp [{-4.0},{0.0}]\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]

    opt_state   = optimizer.init(params)
    train_state = (params, opt_state)

    ckpt_path = "checkpoints/ppo_model_best.msgpack"
    
    # ── Fixed Difficulty Curriculum (Like SAC) ──
    cur_max_dist = 4.0   
    cur_scenario = -1    
    cur_ghost    = 1.0

    print(f"Starting Fixed Curriculum: max_goal_dist={cur_max_dist:.1f}m, ghost_prob={cur_ghost:.1f}, scenario={cur_scenario}")
    print("Initialising environments...")
    env_obs, env_state, vmap_step = init_env_state(env_rng, ghost_prob=cur_ghost)

    rms_state = RunningMeanStd.create()
    running_ret = jnp.zeros(NUM_ENVS)                                
                            
    print(f"Ready. obs={env_obs.shape}\n")
    print("JIT compiling train_chunk (this may take ~1 min)...")

    best_suc = 99.0

    hdr = (f"{'Upd':>5} | {'EpRet':>7} | {'Suc%':>5} {'Obs%':>5} {'Acol%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
           f" {'Loss':>7} {'pi':>6} {'V':>6} {'H':>6} | {'FPS':>7} {'#Ep':>6} {'LR':>6}  | "
           f"{'Time':>6}")
    print(hdr)
    print("─" * len(hdr))

    _LOG_PATH = "checkpoints/ppo_training_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["step", "mean_ep_reward", "suc_pct", "obs_pct",
                           "acol_pct", "pcol_pct", "tmo_pct", "n_ep"])
    _log_file.flush()

    t_start = time.time()
    n_updates = 0

    while n_updates < TOTAL_UPDATES: 
        t0 = time.time()

        # Execute 10 PPO updates fused on the GPU
        new_carry, all_step_data, all_losses, all_aux = ppo_train_chunk(
            train_state, env_state, env_obs, rms_state, vmap_step, running_ret, rng, cur_max_dist, cur_scenario
        )
        
        train_state, env_state, env_obs, rms_state, running_ret, rng = new_carry
        
        n_updates += LOG_EVERY

        # Reduce chunk metrics on GPU
        ep_rets, ep_suc, ep_obs, ep_acol, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes_chunked(*all_step_data)

        # Transfer only scalars to CPU
        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc * ep_msk).sum() / n_ep) * 100.0
            obs_pct  = float((ep_obs * ep_msk).sum() / n_ep) * 100.0
            acol_pct = float((ep_acol * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret, suc_pct, obs_pct, acol_pct, pcol_pct, tmo_pct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        p_loss, v_loss, entropy = jax.tree_util.tree_map(lambda x: x[-1], all_aux)
        mean_loss = all_losses[-1]
        
        fps = (BATCH_SIZE * LOG_EVERY) / (time.time() - t0)
        lr_now = float(scheduler(n_updates * _OPT_STEPS_PER_UPDATE))
        elapsedtime = (time.time() - t_start)/60.0

        print(
            f"{n_updates:>5d} | {mean_ret:>7.1f} | "
            f"{suc_pct:>4.1f}% {obs_pct:>4.1f}% {acol_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% | "
            f"{float(mean_loss):>7.2f} {float(p_loss):>6.2f} "
            f"{float(v_loss):>6.2f} {float(entropy):>6.2f} | "
            f"{fps:>7,.0f} {n_ep:>6d} {lr_now:.2e} | "
            f"{elapsedtime:>5.1f}min"
        )

        total_env_steps = n_updates * NUM_ENVS * ROLLOUT_STEPS
        _log_writer.writerow([total_env_steps, round(mean_ret, 4),
                               round(suc_pct, 4), round(obs_pct, 4),
                               round(acol_pct, 4), round(pcol_pct, 4),
                               round(tmo_pct, 4), n_ep])
        _log_file.flush()

        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(train_state[0], train_state[1], ckpt_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")

    _log_file.close()
    save_checkpoint(train_state[0], train_state[1], "checkpoints/ppo_model_final.msgpack")