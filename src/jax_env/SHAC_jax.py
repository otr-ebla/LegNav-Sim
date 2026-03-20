"""
shac_train.py — SHAC Pure Training (no PPO)
============================================

Addestra la policy usando SOLO il gradiente analitico SHAC (BPTT attraverso
il simulatore differenziabile). Nessun PPO, nessun rollout Monte Carlo.

PERCHÉ SHAC PURO È SUFFICIENTE
--------------------------------

SHAC calcola il gradiente esatto di J(θ) rispetto ai parametri della policy:

    J(θ) = E[ Σ_{t=0}^{H-1} γ^t · r(s_t, π(s_t; θ)) + γ^H · V_φ(s_H) ]

tramite backpropagation attraverso il simulatore (BPTT). Il gradiente ha
varianza ZERO (deterministico dato lo stato iniziale) — PPO serve per ridurre
la varianza del gradiente MC, qui non è necessario.

Vantaggi rispetto a PPO+SHAC ibrido:
  - Un solo ottimizzatore, nessun conflitto di momentum
  - LR e grad clip calibrati su un solo algoritmo
  - Nessuna sincronizzazione di parametri tra due loop
  - Codice ~3× più semplice
  - Convergenza tipicamente più rapida nelle prime fasi

ARCHITETTURA
------------

Per ogni outer update:
  1. Campiona N_ENVS stati iniziali
  2. Esegue rollout differenziabile di H step in parallelo via vmap
  3. Calcola J(θ) = mean returns + bootstrap V(s_H)
  4. Aggiorna actor con jax.value_and_grad(J)(θ)
  5. Aggiorna critico con TD(λ) sugli stessi rollout
  6. Soft update del target network critico (τ=0.005)

CURRICULUM
----------
Stesso schema curriculum di prima ma basato sul reward medio rolling invece
della success rate (non disponibile senza rollout completi a episodio):
  stage 0: dist=1.5m  (target reward > -30 rolling)
  stage 1: dist=2.5m  (target reward > 0 rolling)
  stage 2: dist=4.0m  (target reward > 30 rolling)
  ...

Il curriculum usa una EMA del reward medio normalizzato invece della success
rate (non disponibile senza rollout completi a episodio). Un reward medio
che supera la soglia indica che la policy ha imparato a navigare stabily.

IPERPARAMETRI
-------------
N_ENVS       = 2048   — ambienti paralleli per update (VRAM ~400 MB)
H_INIT       = 8      — orizzonte iniziale (passi)
H_MAX        = 32     — orizzonte massimo
H_GROWTH     = 30     — update tra ogni incremento di orizzonte
LR_ACTOR     = 3e-4   — Adam per l'actor
LR_CRITIC    = 1e-3   — Adam per il critico (più alto: critico imparata veloce)
GRAD_CLIP    = 1.0    — clip globale gradienti actor
GAMMA        = 0.99
LAMBDA       = 0.95   — GAE λ per TD(λ) critico
TAU          = 0.005  — soft update target network
TOTAL_UPDATES= 1000
"""

import os
import csv
import time
import argparse

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SHAC Pure Training")
parser.add_argument("--gpu",    type=str, default="0", choices=["0", "1"])
parser.add_argument("--envs",   type=int, default=2048,
                    help="Numero di env paralleli per update SHAC")
parser.add_argument("--h-init", type=int, default=8,
                    help="Orizzonte iniziale SHAC")
parser.add_argument("--h-max",  type=int, default=32,
                    help="Orizzonte massimo SHAC")
parser.add_argument("--updates",type=int, default=1000,
                    help="Numero totale di outer update")
parser.add_argument("--load",   type=str, default="",
                    help="Percorso checkpoint da caricare")
parser.add_argument("--no-ghost", action="store_true",
                    help="Disabilita ghost robot (robot visibile agli umani)")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.serialization
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

jax.config.update("jax_default_device", jax.devices("cuda")[0])

from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_env_multi import (
    reset_env, step_env,
    ROBOT_RADIUS, PEOPLE_RADIUS, ROOM_W, ROOM_H, GOAL_RADIUS,
    _R_GOAL, _R_WALL_COL, _R_ACTIVE_COL,
    _PROGRESS_COEF, _STEP_PEN, _JERK_WEIGHT, _CF_CENTER,
)
from jax_wrappers import make_stacked_env, StackedEnvState
from jax_env import NUM_RAYS, STATE_VEC_SIZE

# ── Costanti ──────────────────────────────────────────────────────────────────
OBS_SIZE   = 3 * 3 + 9 + 108 * 3   # 342
_POSE_SIZE = 3

# ── Iperparametri ─────────────────────────────────────────────────────────────
N_ENVS          = args.envs
H_INIT          = args.h_init
H_MAX           = args.h_max
H_GROWTH        = 30       # update tra ogni +1 orizzonte
TOTAL_UPDATES   = args.updates

LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3
GRAD_CLIP_ACTOR = 1.0
GRAD_CLIP_CRITIC= 2.0

GAMMA           = 0.99
LAM             = 0.95
TAU             = 0.005    # soft update target network
VF_COEF         = 0.5

# Ricampiona stati iniziali ogni N update (evita overfitting su pochi stati)
RESAMPLE_INTERVAL = 3

# ── Curriculum ────────────────────────────────────────────────────────────────
# Soglie su reward medio rolling EMA (alpha=0.05, ~20 update lag).
# Valori approssimati: reward -60 = random, 0 = spesso raggiunge goal,
# +50 = navigazione efficiente e sociale.
CURRICULUM = [
    # (rolling_reward_threshold, min_goal_dist)
    (-20.0, 1.5),
    ( 10.0, 2.5),
    ( 30.0, 4.0),
    ( 50.0, 5.5),
    ( 70.0, 7.0),
    (101.0, 9.0),
]

def curriculum_dist(rolling_ret: float) -> float:
    for thresh, dist in CURRICULUM:
        if rolling_ret < thresh:
            return dist
    return CURRICULUM[-1][1]

def curriculum_stage(rolling_ret: float) -> int:
    for i, (thresh, _) in enumerate(CURRICULUM):
        if rolling_ret < thresh:
            return i
    return len(CURRICULUM) - 1

# ── Reward differenziabile per BPTT ──────────────────────────────────────────
# step_env usa jnp.where(bool, CONSTANT) → ∂reward/∂action = 0 sui terminali.
# Usiamo approssimazioni smooth per mantenere il gradiente non-zero ovunque.
_CF_SLOPE_DIFF = 0.3

def _diff_reward(new_state, prev_x, prev_y, prev_w, action):
    """Reward completamente differenziabile per BPTT."""
    new_x, new_y = new_state.x, new_state.y

    prev_dist = jnp.sqrt((prev_x - new_state.goal_x)**2 +
                          (prev_y - new_state.goal_y)**2 + 1e-8)
    new_dist  = jnp.sqrt((new_x  - new_state.goal_x)**2 +
                          (new_y  - new_state.goal_y)**2 + 1e-8)
    progress  = prev_dist - new_dist

    active    = new_state.people[:, 10] >= 0.0
    dx_p      = new_state.people[:, 0] - new_x
    dy_p      = new_state.people[:, 1] - new_y
    dists_p   = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)
    dists_act = jnp.where(active, dists_p, jnp.inf)
    closest   = jnp.min(dists_act)

    edge      = jnp.maximum(0.0, closest - PEOPLE_RADIUS - ROBOT_RADIUS)
    cf        = jax.nn.sigmoid((edge - _CF_CENTER) / _CF_SLOPE_DIFF)

    social_progress = _PROGRESS_COEF * progress * cf
    jerk_pen        = -_JERK_WEIGHT * (action[1] - prev_w) ** 2
    r               = social_progress + _STEP_PEN + jerk_pen

    # Terminali smooth
    goal_s  = jax.nn.sigmoid(20.0 * (GOAL_RADIUS - new_dist))
    wall_d  = jnp.minimum(
        jnp.minimum(new_x - ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS - new_x),
        jnp.minimum(new_y - ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS - new_y),
    )
    wall_s  = jax.nn.sigmoid(-20.0 * wall_d)
    human_s = jax.nn.sigmoid(20.0 * (ROBOT_RADIUS + PEOPLE_RADIUS - closest))

    r = r + _R_GOAL * goal_s + _R_WALL_COL * wall_s + _R_ACTIVE_COL * human_s
    return jnp.where(jnp.isfinite(r), r, jnp.zeros_like(r))


# ── Critic network ────────────────────────────────────────────────────────────

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(64)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


# ── Rollout singolo (differenziabile rispetto ad actor_params) ─────────────────

def _rollout_single(actor_params, critic_params, actor_apply, critic_apply,
                    init_obs, init_state, rng_key, horizon_mask, ghost_robot):
    """
    Rollout differenziabile di H_MAX step per un singolo env.
    Ritorna: (return_scalar, (obs_seq, rewards, dones, final_obs))
    """
    def _step(carry, t):
        obs, state, key, cum_disc = carry
        key, step_key = jax.random.split(key)

        # Azione deterministica — necessario per BPTT (no noise nel grafo)
        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean       = mean[0]
        action     = scale_action_to_env(mean, state.env_state.max_v)

        # Step simulatore
        base_obs, new_base, _reward, done, _info = step_env(
            step_key, state.env_state, action, ghost_robot=ghost_robot
        )

        # Aggiorna obs stack
        new_pose  = base_obs[0:_POSE_SIZE]
        new_sv    = base_obs[_POSE_SIZE : _POSE_SIZE + STATE_VEC_SIZE]
        new_lidar = base_obs[_POSE_SIZE + STATE_VEC_SIZE:]

        new_ls = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], 0)
        new_ps = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  0)
        new_state = state.replace(
            env_state=new_base, lidar_stack=new_ls, pose_stack=new_ps
        )
        new_obs = jnp.concatenate([new_ps.flatten(), new_sv, new_ls.flatten()])

        # Reward differenziabile
        diff_r     = _diff_reward(new_base,
                                  state.env_state.x, state.env_state.y,
                                  state.env_state.w, action)
        done_f     = jax.lax.stop_gradient(done.astype(jnp.float32))
        disc_r     = cum_disc * diff_r * horizon_mask[t].astype(jnp.float32)
        next_disc  = cum_disc * GAMMA * (1.0 - done_f)

        return (new_obs, new_state, key, next_disc), (disc_r, new_obs, done_f)

    # Gradient checkpointing: salva VRAM sul backward pass BPTT
    _step_ckpt = jax.remat(_step, policy=None)

    (final_obs, _, _, final_disc), (rewards, obs_seq, dones) = jax.lax.scan(
        _step_ckpt,
        (init_obs, init_state, rng_key, jnp.ones(())),
        jnp.arange(H_MAX),
        length=H_MAX,
        unroll=1,
    )

    # Bootstrap con il critico (stop_gradient — critico non backprop nell'actor)
    bootstrap = jax.lax.stop_gradient(
        critic_apply({"params": critic_params}, final_obs[None])[0]
    )
    total_return = jnp.sum(rewards) + final_disc * bootstrap

    return total_return, (obs_seq, rewards, dones, final_obs)


# vmap su N_ENVS env in parallelo
_rollout_batched = jax.vmap(
    _rollout_single,
    in_axes=(None, None, None, None, 0, 0, 0, None, None)
)


# ── Loss functions ─────────────────────────────────────────────────────────────

def _actor_loss(actor_params, critic_params, actor_apply, critic_apply,
                obs_batch, state_batch, keys, horizon_mask, ghost_robot):
    returns, aux = _rollout_batched(
        actor_params, critic_params, actor_apply, critic_apply,
        obs_batch, state_batch, keys, horizon_mask, ghost_robot,
    )
    # Baseline con stop_gradient: riduce varianza senza azzerare la loss
    baseline = jax.lax.stop_gradient(jnp.mean(returns))
    loss = -jnp.mean(returns - baseline)
    return loss, (returns, aux)


def _critic_loss(critic_params, critic_apply,
                 obs_seq, rewards, dones, final_obs, target_params):
    """TD(λ) con target network EMA — elimina deadly triad."""
    N, H = rewards.shape

    v_final = jax.lax.stop_gradient(
        critic_apply({"params": target_params}, final_obs)
    )

    def _td_scan(carry, t):
        gae, next_v = carry
        r   = rewards[:, t]
        d   = dones[:, t]
        obs = obs_seq[:, t, :]
        v_t = jax.lax.stop_gradient(critic_apply({"params": target_params}, obs))
        delta  = r + GAMMA * next_v * (1.0 - d) - v_t
        gae    = delta + GAMMA * LAM * (1.0 - d) * gae
        target = jax.lax.stop_gradient(gae + v_t)
        return (gae, v_t), target

    (_, _), targets = jax.lax.scan(
        _td_scan, (jnp.zeros(N), v_final), jnp.arange(H), reverse=True
    )
    targets = targets.T   # (H,N) → (N,H)

    obs_flat  = obs_seq.reshape(N * H, OBS_SIZE)
    v_preds   = critic_apply({"params": critic_params}, obs_flat).reshape(N, H)
    tgt_sg    = jax.lax.stop_gradient(targets)
    return VF_COEF * jnp.mean((v_preds - tgt_sg) ** 2)


# ── Compiled update step ───────────────────────────────────────────────────────

def make_update_step(actor_apply, critic_apply,
                     actor_optimizer, critic_optimizer,
                     ghost_robot: bool):
    """
    Costruisce e compila (JIT) il kernel di update completo.
    actor_apply, critic_apply, actor_optimizer, critic_optimizer e ghost_robot
    sono valori Python statici — vengono "baked in" via closure invece di
    essere passati come argomenti. Questo evita il TypeError di JAX che non
    sa come trattare metodi Python come array astratti.

    Ritorna una funzione JIT-compilata con firma:
        update_fn(actor_params, critic_params, critic_target_params,
                  actor_opt_state, critic_opt_state,
                  obs_batch, state_batch, keys, horizon_mask)
    """
    @jax.jit
    def _update_step(actor_params, critic_params, critic_target_params,
                     actor_opt_state, critic_opt_state,
                     obs_batch, state_batch, keys, horizon_mask):
        """
        Un outer update completo:
          1. jax.value_and_grad su J(θ) → aggiorna actor
          2. _critic_loss su stessi rollout → aggiorna critico
          3. Soft update target network
        """
        # ── 1. Actor update ───────────────────────────────────────────────────
        (a_loss, (returns, (obs_seq, rewards, dones, final_obs))), a_grads = \
            jax.value_and_grad(_actor_loss, has_aux=True)(
                actor_params, critic_params, actor_apply, critic_apply,
                obs_batch, state_batch, keys, horizon_mask, ghost_robot,
            )

        a_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), a_grads
        )
        a_grad_norm = optax.global_norm(a_grads)
        a_updates, new_a_opt = actor_optimizer.update(
            a_grads, actor_opt_state, actor_params
        )
        new_actor_params = optax.apply_updates(actor_params, a_updates)

        # ── 2. Critic update ──────────────────────────────────────────────────
        c_loss, c_grads = jax.value_and_grad(_critic_loss)(
            critic_params, critic_apply,
            obs_seq, rewards, dones, final_obs, critic_target_params,
        )
        c_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), c_grads
        )
        c_grad_norm = optax.global_norm(c_grads)
        c_updates, new_c_opt = critic_optimizer.update(
            c_grads, critic_opt_state, critic_params
        )
        new_critic_params = optax.apply_updates(critic_params, c_updates)

        # ── 3. Soft update target ─────────────────────────────────────────────
        new_target_params = jax.tree_util.tree_map(
            lambda tgt, online: (1.0 - TAU) * tgt + TAU * online,
            critic_target_params, new_critic_params
        )

        metrics = {
            "actor_loss":       a_loss,
            "critic_loss":      c_loss,
            "mean_return":      jnp.mean(returns),
            "actor_gn":         a_grad_norm,
            "critic_gn":        c_grad_norm,
            "mean_reward_step": jnp.mean(rewards),
        }
        return (new_actor_params, new_critic_params, new_target_params,
                new_a_opt, new_c_opt, metrics)

    return _update_step


# ── Reset cache ───────────────────────────────────────────────────────────────
_RESET_CACHE: dict = {}

def _sample_init_states(rng_key, min_goal_dist, ghost_robot):
    """Campiona N_ENVS stati iniziali. Cached per (ghost_robot, min_goal_dist)."""
    cache_key = (ghost_robot, min_goal_dist)
    if cache_key not in _RESET_CACHE:
        _RESET_CACHE.clear()
        reset_stacked, _ = make_stacked_env(
            reset_env, step_env, stack_dim=3, ghost_robot=ghost_robot
        )
        def _reset_one(key):
            return reset_stacked(key, min_goal_dist=min_goal_dist)
        _RESET_CACHE[cache_key] = jax.jit(jax.vmap(_reset_one))

    keys = jax.random.split(rng_key, N_ENVS)
    return _RESET_CACHE[cache_key](keys)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(actor_params, critic_params, critic_target_params,
                    actor_opt, critic_opt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "actor_params":         jax.device_get(actor_params),
        "critic_params":        jax.device_get(critic_params),
        "critic_target_params": jax.device_get(critic_target_params),
        "actor_opt":            jax.device_get(actor_opt),
        "critic_opt":           jax.device_get(critic_opt),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  Checkpoint → {path}")


def load_checkpoint(actor_params, critic_params, critic_target_params,
                    actor_opt, critic_opt, path):
    with open(path, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.from_bytes({
        "actor_params":         actor_params,
        "critic_params":        critic_params,
        "critic_target_params": critic_target_params,
        "actor_opt":            actor_opt,
        "critic_opt":           critic_opt,
    }, raw)
    return (bundle["actor_params"], bundle["critic_params"],
            bundle["critic_target_params"],
            bundle["actor_opt"],    bundle["critic_opt"])


# ── Horizon mask ──────────────────────────────────────────────────────────────

def make_horizon_mask(h: int) -> jnp.ndarray:
    """(H_MAX,) bool: True per t < h. JAX traced → zero ricompilazioni."""
    return jnp.arange(H_MAX) < h

def get_horizon(update: int) -> int:
    h = H_INIT + update // H_GROWTH
    return min(h, H_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  SHAC Pure Training  (GPU {args.gpu})")
    print(f"  Envs={N_ENVS}  H: {H_INIT}→{H_MAX} (ogni {H_GROWTH} upd)")
    print(f"  LR actor={LR_ACTOR}  LR critic={LR_CRITIC}")
    print(f"  Total updates={TOTAL_UPDATES}")
    print(f"  Ghost robot={'OFF' if args.no_ghost else 'ON'}")
    print(f"{'='*60}\n")

    ghost_robot = not args.no_ghost

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # ── Inizializza reti ──────────────────────────────────────────────────────
    actor_net  = EndToEndActorCritic(action_dim=2)
    critic_net = Critic()

    dummy_obs      = jnp.zeros((1, OBS_SIZE))
    actor_params   = actor_net.init(init_rng, dummy_obs)["params"]

    rng, ck = jax.random.split(rng)
    critic_params        = critic_net.init(ck, dummy_obs)["params"]
    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)

    # LR schedule: warmup 50 update → decay fino a LR/10
    _WARMUP  = 50
    _sched_a = optax.join_schedules(
        [optax.linear_schedule(LR_ACTOR * 0.1, LR_ACTOR, _WARMUP),
         optax.cosine_decay_schedule(LR_ACTOR, TOTAL_UPDATES - _WARMUP, alpha=0.1)],
        [_WARMUP]
    )
    _sched_c = optax.join_schedules(
        [optax.linear_schedule(LR_CRITIC * 0.1, LR_CRITIC, _WARMUP),
         optax.cosine_decay_schedule(LR_CRITIC, TOTAL_UPDATES - _WARMUP, alpha=0.1)],
        [_WARMUP]
    )

    actor_optimizer  = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_ACTOR),
        optax.adam(_sched_a, b1=0.9, b2=0.999, eps=1e-5),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_CRITIC),
        optax.adam(_sched_c, eps=1e-5),
    )

    actor_opt_state  = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)

    # Compila il kernel di update (bake-in apply/optimizer/ghost_robot via closure)
    update_step = make_update_step(
        actor_net.apply, critic_net.apply,
        actor_optimizer, critic_optimizer,
        ghost_robot,
    )

    # ── Carica checkpoint se richiesto ────────────────────────────────────────
    if args.load and os.path.exists(args.load):
        try:
            (actor_params, critic_params, critic_target_params,
             actor_opt_state, critic_opt_state) = load_checkpoint(
                actor_params, critic_params, critic_target_params,
                actor_opt_state, critic_opt_state, args.load
            )
            # Ricostruisci update_step con i params caricati
            # (gli optimizer state sono già aggiornati, il kernel rimane lo stesso)
            print(f"Checkpoint caricato da {args.load}")
        except Exception as e:
            print(f"Caricamento fallito ({e}), partenza da zero.")

    # ── Curriculum iniziale ───────────────────────────────────────────────────
    rolling_ret  = -80.0   # stima iniziale pessimistica
    cur_dist     = curriculum_dist(rolling_ret)
    cur_stage    = curriculum_stage(rolling_ret)

    print(f"Curriculum: stage={cur_stage}, min_goal_dist={cur_dist:.1f}m")

    # ── Campiona stati iniziali ───────────────────────────────────────────────
    print(f"Campionamento {N_ENVS} stati iniziali...")
    rng, sample_rng = jax.random.split(rng)
    obs_batch, state_batch = _sample_init_states(sample_rng, cur_dist, ghost_robot)
    print(f"Pronti. obs shape={obs_batch.shape}")

    # ── Verifica gradient flow ────────────────────────────────────────────────
    print("Verifica gradient flow...")
    _mask_test = make_horizon_mask(H_INIT)
    _keys_test = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
    def _test_loss(p):
        returns, _ = _rollout_batched(
            p, critic_params, actor_net.apply, critic_net.apply,
            obs_batch, state_batch, _keys_test, _mask_test, ghost_robot,
        )
        return -jnp.mean(returns)
    _grads = jax.grad(_test_loss)(actor_params)
    _gn    = float(optax.global_norm(_grads))
    print(f"  ‖∇θ J‖ = {_gn:.4e}  "
          f"({'OK' if _gn > 1e-8 else 'PROBLEMA — gradiente zero!'})")
    if _gn < 1e-8:
        raise RuntimeError("Gradiente zero — controlla step_env e hsfm_diff.")

    # ── CSV log ───────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    _LOG_PATH   = "checkpoints/shac_training_log.csv"
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow([
        "update", "total_env_steps",
        "actor_loss", "critic_loss", "mean_return",
        "actor_gn", "critic_gn",
        "horizon", "stage", "dist", "rolling_ret", "elapsed_min"
    ])
    _log_file.flush()

    best_ret = -999.0
    t_start  = time.time()

    hdr = (
        f"{'Upd':>5} | {'Return':>8} {'Roll':>7} | "
        f"{'Actor-L':>8} {'Critic-L':>8} {'AGN':>6} {'CGN':>6} | "
        f"{'H':>3} {'Stage':>5} {'Dist':>5} | "
        f"{'FPS':>7} {'Time':>6}"
    )
    print(hdr)
    print("─" * len(hdr))

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════════

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        rng, step_rng, sample_rng2 = jax.random.split(rng, 3)

        # Horizon curriculum
        horizon      = get_horizon(update)
        horizon_mask = make_horizon_mask(horizon)

        # Ricampiona stati iniziali ogni RESAMPLE_INTERVAL update
        if update % RESAMPLE_INTERVAL == 0:
            obs_batch, state_batch = _sample_init_states(
                sample_rng2, cur_dist, ghost_robot
            )

        # Keys per i rollout
        step_keys = jax.random.split(step_rng, N_ENVS)

        # ── Update ────────────────────────────────────────────────────────────
        (actor_params, critic_params, critic_target_params,
         actor_opt_state, critic_opt_state, metrics) = update_step(
            actor_params, critic_params, critic_target_params,
            actor_opt_state, critic_opt_state,
            obs_batch, state_batch, step_keys, horizon_mask,
        )

        mean_ret    = float(metrics["mean_return"])
        actor_loss  = float(metrics["actor_loss"])
        critic_loss = float(metrics["critic_loss"])
        a_gn        = float(metrics["actor_gn"])
        c_gn        = float(metrics["critic_gn"])

        # FPS calcolato sul rollout: N_ENVS × H step per update
        env_steps_per_update = N_ENVS * horizon
        fps = env_steps_per_update / (time.time() - t0)

        # EMA reward rolling (lag ~20 update con alpha=0.05)
        rolling_ret = 0.95 * rolling_ret + 0.05 * mean_ret

        # ── Curriculum ────────────────────────────────────────────────────────
        new_dist  = curriculum_dist(rolling_ret)
        new_stage = curriculum_stage(rolling_ret)
        if new_dist > cur_dist:
            cur_dist  = new_dist
            cur_stage = new_stage
            rng, reinit_rng = jax.random.split(rng)
            obs_batch, state_batch = _sample_init_states(
                reinit_rng, cur_dist, ghost_robot
            )
            print(f"  → Curriculum stage={cur_stage}, dist={cur_dist:.1f}m "
                  f"(rolling_ret={rolling_ret:.1f})")

        # ── Logging ───────────────────────────────────────────────────────────
        if update % 5 == 0:
            elapsed = (time.time() - t_start) / 60.0
            print(
                f"{update:>5d} | {mean_ret:>8.2f} {rolling_ret:>7.2f} | "
                f"{actor_loss:>8.4f} {critic_loss:>8.4f} {a_gn:>6.3f} {c_gn:>6.3f} | "
                f"{horizon:>3d} {cur_stage:>5d} {cur_dist:>4.1f}m | "
                f"{fps:>7,.0f} {elapsed:>5.1f}min"
            )
            total_env_steps = (update + 1) * env_steps_per_update
            _log_writer.writerow([
                update, total_env_steps,
                round(actor_loss, 5), round(critic_loss, 5),
                round(mean_ret, 3), round(a_gn, 4), round(c_gn, 4),
                horizon, cur_stage, round(cur_dist, 1),
                round(rolling_ret, 3), round(elapsed, 2),
            ])
            _log_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if mean_ret > best_ret:
            best_ret = mean_ret
            save_checkpoint(
                actor_params, critic_params, critic_target_params,
                actor_opt_state, critic_opt_state,
                "checkpoints/shac_best.msgpack"
            )

    # ── Fine ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best return: {best_ret:.2f}")

    save_checkpoint(
        actor_params, critic_params, critic_target_params,
        actor_opt_state, critic_opt_state,
        "checkpoints/shac_final.msgpack"
    )
    _log_file.close()
    print(f"Log → {_LOG_PATH}")