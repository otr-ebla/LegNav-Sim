"""
SHAC_jax.py — SHAC Pure Training (no PPO)
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

# Memory budget per BPTT:
#   BPTT stores activations for every step in the scan for backprop.
#   Peak VRAM ≈ N_ENVS × H_MAX × OBS_SIZE × bytes_per_activation × ~8 (CNN)
#   1024 × 32 × 342 × 4 × 8 ≈ 360 MB for obs alone, but CNN activations
#   (32 feature maps × 54 spatial × 3 layers) add ~10× → ~3.6 GB just for
#   the forward pass activations. With gradient buffers: ~2× = ~7 GB.
#   Fix: use jax.checkpoint on the scan body to recompute activations during
#   backward instead of storing them. This trades ~2× compute for ~10× less
#   VRAM — acceptable for BPTT where compute >> memory bandwidth anyway.
TOTAL_UPDATES   = args.updates

LR_ACTOR        = 1e-4    # abbassato: con AGN esplosivo (>100) serve lr basso
LR_CRITIC       = 3e-4    # abbassato proporzionalmente
GRAD_CLIP_ACTOR = 0.5     # più conservativo: BPTT su cinematica unicycle
                           # produce gradienti O(H) — clip basso è essenziale
GRAD_CLIP_CRITIC= 1.0

GAMMA           = 0.99
LAM             = 0.95
TAU             = 0.02    # was 0.005: too slow, target ≈ online after ~200 updates
VF_COEF         = 0.5

# Multiple critic updates per actor step: critic must keep up with shifting
# state distribution from BPTT. 4 is standard in SAC/TD3.
N_CRITIC_UPDATES = 4

# Critic warmup: pre-train critic before first actor BPTT update so the
# bootstrap V(s_H) is calibrated, not near-zero.
CRITIC_WARMUP_UPDATES = 30

# Ricampiona stati iniziali ogni N update
RESAMPLE_INTERVAL = 3

# ── Curriculum ────────────────────────────────────────────────────────────────
# Il curriculum usa il return PER STEP (return/H), non il return totale.
# Questo normalizza rispetto all'orizzonte crescente:
#   return_totale = H × reward_per_step  →  reward_per_step è invariante a H.
#
# Calibrazione:
#   reward/step ≈ -5   : policy random (collide sempre)
#   reward/step ≈ -2   : inizia a evitare le collisioni
#   reward/step ≈ -0.5 : raggiunge il goal con qualche collisione
#   reward/step ≈  0   : navigazione base competente
#   reward/step ≈ +1   : navigazione sociale buona
CURRICULUM = [
    # (rolling_reward_per_step_threshold, min_goal_dist)
    (-1.5, 1.5),   # policy random → inizia qui
    (-0.5, 2.5),   # evita collisioni, spesso raggiunge goal a 1.5m
    ( 0.0, 4.0),   # navigazione base a 2.5m
    ( 0.5, 5.5),
    ( 1.0, 7.0),
    ( 2.0, 9.0),
]

def curriculum_dist(rolling_ret_per_step: float) -> float:
    for thresh, dist in CURRICULUM:
        if rolling_ret_per_step < thresh:
            return dist
    return CURRICULUM[-1][1]

def curriculum_stage(rolling_ret_per_step: float) -> int:
    for i, (thresh, _) in enumerate(CURRICULUM):
        if rolling_ret_per_step < thresh:
            return i
    return len(CURRICULUM) - 1

# ── Reward differenziabile per BPTT ──────────────────────────────────────────
# step_env usa jnp.where(bool, CONSTANT) → ∂reward/∂action = 0 sui terminali.
# Usiamo approssimazioni smooth per mantenere il gradiente non-zero ovunque.
_CF_SLOPE_DIFF = 0.3

def _diff_reward(new_state_sg, prev_x, prev_y, prev_theta, prev_w,
                  action, goal_x, goal_y, dt=0.15):
    """
    Reward differenziabile per BPTT con cinematica esplicita.

    Poiche' new_state_sg e' stop_gradient, non possiamo usare new_state_sg.x/y
    per calcolare il progresso (grad = 0). Ricostruiamo la posizione del robot
    con la cinematica unicycle differenziabile:
        mid_theta  = prev_theta + 0.5 * w * dt
        new_x_diff = prev_x + v * cos(mid_theta) * dt
        new_y_diff = prev_y + v * sin(mid_theta) * dt
    La dipendenza da actor_params e' theta->mean->action->(v,w)->new_x/y->reward.

    Fix NaN (validi per qualsiasi batch size):
    - sqrt: jnp.maximum(..., 1e-6) invece di +1e-8
    - closest: 30.0 invece di jnp.inf per i dummy
    - sigmoid: input clippati a [-20, 20]
    """
    v = action[0]
    w = action[1]
    mid_theta  = prev_theta + 0.5 * w * dt
    new_x_diff = prev_x + v * jnp.cos(mid_theta) * dt
    new_y_diff = prev_y + v * jnp.sin(mid_theta) * dt
    new_x_diff = jnp.clip(new_x_diff, ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS)
    new_y_diff = jnp.clip(new_y_diff, ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS)

    def _sd(ax, ay, bx, by):
        return jnp.sqrt(jnp.maximum((ax-bx)**2 + (ay-by)**2, 1e-6))

    prev_dist = _sd(prev_x, prev_y, goal_x, goal_y)
    new_dist  = _sd(new_x_diff, new_y_diff, goal_x, goal_y)
    progress  = prev_dist - new_dist

    active    = new_state_sg.people[:, 10] >= 0.0
    dx_p      = new_state_sg.people[:, 0] - new_x_diff
    dy_p      = new_state_sg.people[:, 1] - new_y_diff
    dists_p   = jnp.sqrt(jnp.maximum(dx_p**2 + dy_p**2, 1e-6))
    dists_act = jnp.where(active, dists_p, 30.0)
    closest   = jnp.min(dists_act)

    edge = jnp.maximum(0.0, closest - PEOPLE_RADIUS - ROBOT_RADIUS)
    cf   = jax.nn.sigmoid((edge - _CF_CENTER) / _CF_SLOPE_DIFF)

    social_progress = _PROGRESS_COEF * progress * cf
    jerk_pen        = -_JERK_WEIGHT * (w - prev_w) ** 2
    # Rimuovi _STEP_PEN dal reward differenziabile:
    # Con _STEP_PEN=-0.02 e H=32, il robot impara v=0 per minimizzare
    # i passi (ogni passo "costa" -0.02, stare fermo ne evita molti).
    # Il penalty per step è utile nell'env reale per efficienza, ma nel
    # grafo BPTT crea un attractor a v=0 che impedisce l'apprendimento.
    r               = social_progress + jerk_pen

    wall_d  = jnp.minimum(
        jnp.minimum(new_x_diff - ROBOT_RADIUS, ROOM_W - ROBOT_RADIUS - new_x_diff),
        jnp.minimum(new_y_diff - ROBOT_RADIUS, ROOM_H - ROBOT_RADIUS - new_y_diff),
    )
    goal_s  = jax.nn.sigmoid(jnp.clip(20.0*(GOAL_RADIUS  - new_dist),          -20., 20.))
    wall_s  = jax.nn.sigmoid(jnp.clip(-20.0*wall_d,                             -20., 20.))
    human_s = jax.nn.sigmoid(jnp.clip(20.0*(ROBOT_RADIUS+PEOPLE_RADIUS-closest),-20., 20.))

    r = r + _R_GOAL*goal_s + _R_WALL_COL*wall_s + _R_ACTIVE_COL*human_s
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

    DESIGN: differenziare ATTRAVERSO step_env (non stop_gradient).

    hsfm_diff.py è già completamente differenziabile (fix A-G nel suo docstring).
    Il NaN con batch grandi non viene da hsfm ma dalla _diff_reward con
    cinematica ricostruita che non corrisponde all'env reale — causando
    divergenza del critico (Critic-L 17→254 in 1000 update).

    Usiamo il reward di step_env direttamente. Per evitare NaN nel backward:
      - nan_to_num sui gradienti prima dell'optimizer (già presente)
      - jax.checkpoint sul scan body riduce il picco VRAM del backward

    Il gradiente fluisce: theta->mean->action->step_env->(x,y,reward)->J(theta)
    Questo è il vero gradiente SHAC — esatto rispetto alla dinamica reale.
    """
    # ── VRAM fix: remat on entire scan body ───────────────────────────────────
    def _step(carry, t):
        obs, state, key, cum_disc, already_done = carry
        key, step_key = jax.random.split(key)

        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean   = mean[0]
        action = scale_action_to_env(mean, state.env_state.max_v)

        base_obs, new_base, diff_r, done, _info = step_env(
            step_key, state.env_state, action, ghost_robot=ghost_robot
        )

        new_pose  = base_obs[0:_POSE_SIZE]
        new_sv    = base_obs[_POSE_SIZE : _POSE_SIZE + STATE_VEC_SIZE]
        new_lidar = base_obs[_POSE_SIZE + STATE_VEC_SIZE:]
        new_ls    = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], 0)
        new_ps    = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  0)
        new_state = state.replace(
            env_state=new_base, lidar_stack=new_ls, pose_stack=new_ps
        )
        new_obs = jnp.concatenate([new_ps.flatten(), new_sv, new_ls.flatten()])

        done_f = jax.lax.stop_gradient(done.astype(jnp.float32))

        # active = step is within curriculum horizon AND episode not yet ended.
        # Post-terminal steps are zeroed so -90/+200 terminal rewards from dead
        # episodes don't corrupt TD targets or the actor return. [Bug B fix]
        active     = horizon_mask[t] & ~already_done
        act_f      = active.astype(jnp.float32)
        raw_r_safe = jnp.where(jnp.isfinite(diff_r), diff_r, jnp.zeros_like(diff_r))
        masked_r   = raw_r_safe * act_f

        # Discount zeroed after done/horizon so bootstrap is only γ^H when all
        # H steps ran to completion.
        next_disc        = cum_disc * GAMMA * (1.0 - done_f) * act_f
        new_already_done = already_done | done

        # Bug A fix: output obs (= s_t, start of step t), NOT new_obs (= s_{t+1}).
        # The critic TD scan needs V(s_t) aligned with r_t:
        #   delta_t = r_t + γ·V(s_{t+1}) − V(s_t)  ← correct
        # Previously new_obs was stored → delta used V(s_{t+2}) − V(s_{t+1}).
        return (new_obs, new_state, key, next_disc, new_already_done), \
               (masked_r, obs, done_f, cum_disc, act_f)

    _step_remat = jax.checkpoint(_step)

    (final_obs, _, _, final_disc, _), \
        (raw_rewards, obs_seq, dones, disc_weights, active_mask) = jax.lax.scan(
        _step_remat,
        (init_obs, init_state, rng_key, jnp.ones(()), jnp.array(False)),
        jnp.arange(H_MAX),
        length=H_MAX,
        unroll=1,
    )
    # obs_seq[t]     = s_t (start of step t).   Shape: (H_MAX, OBS_SIZE)
    # raw_rewards[t] = r_t zeroed post-done / beyond horizon
    # active_mask[t] = 1.0 if step executed, 0.0 otherwise
    # disc_weights[t]= γ^t (cum_disc entering step t)
    # final_obs      = s_H (for bootstrap)

    discounted_rewards = raw_rewards * disc_weights   # γ^t · r_t
    bootstrap = jax.lax.stop_gradient(
        critic_apply({"params": critic_params}, final_obs[None])[0]
    )
    total_return = jnp.sum(discounted_rewards) + final_disc * bootstrap
    return total_return, (obs_seq, raw_rewards, dones, final_obs, active_mask)


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
    loss = -jnp.mean(returns)
    return loss, (returns, aux)


def _critic_loss(critic_params, critic_apply,
                 obs_seq, rewards, dones, final_obs, active_mask, target_params):
    """
    TD(λ) critic loss — corrected obs alignment + Huber loss + active masking.

    Fixes vs previous:
      1. obs_seq[t] = s_t (not s_{t+1}): correct Bellman alignment
         delta_t = r_t + γ·V(s_{t+1})·(1-d) − V(s_t)
      2. Huber loss (δ=10) instead of MSE: robust to terminal rewards ±90/+200.
         MSE amplifies a -90 collision to 8100 gradient; Huber clamps it to ~90.
         δ=10 is quadratic for |error|<10 (covers normal V range), linear beyond.
      3. NO reward normalisation: normalising rewards creates a unit mismatch —
         the critic predicts V in normalised units while the actor sums raw rewards,
         causing the bootstrap γ^H·V to have the wrong scale and grow with H.
         Huber alone is sufficient to handle the reward scale spread.
      4. active_mask: MSE/Huber only over real executed steps, not padding.

    TD(λ) reverse scan (corrected):
        obs_seq[:, t, :] = s_t  →  v_t = V(s_t)  from target network
        next_v in carry        = V(s_{t+1})
        delta_t = r_t + γ·next_v·(1-d_t) − v_t   ← correct Bellman error
        gae_t   = delta_t + γ·λ·(1-d_t)·gae_{t+1}
        target_t = gae_t + v_t                     ← TD(λ) target for V(s_t)
    """
    N, H = rewards.shape

    v_final = jax.lax.stop_gradient(
        critic_apply({"params": target_params}, final_obs)
    )  # (N,) — V(s_H) for bootstrap at the end of the horizon

    def _td_scan(carry, t):
        gae, next_v = carry   # next_v = V(s_{t+1})
        r   = rewards[:, t]
        d   = dones[:, t]
        act = active_mask[:, t]
        # obs_seq[:, t, :] = s_t → v_t = V(s_t) from target network
        v_t = jax.lax.stop_gradient(
            critic_apply({"params": target_params}, obs_seq[:, t, :])
        )
        delta = r + GAMMA * next_v * (1.0 - d) - v_t   # correct Bellman error
        gae   = delta + GAMMA * LAM * (1.0 - d) * gae
        gae   = gae * act   # zero gae for inactive steps
        target = jax.lax.stop_gradient(gae + v_t)
        # next iteration: next_v = v_t = V(s_t), which is V(s_{t+1}) for step t-1
        return (gae, v_t), (target, act)

    (_, _), (targets, acts) = jax.lax.scan(
        _td_scan, (jnp.zeros(N), v_final), jnp.arange(H), reverse=True
    )
    targets = targets.T   # (H,N) → (N,H)
    acts    = acts.T      # (H,N) → (N,H)

    obs_flat = obs_seq.reshape(N * H, OBS_SIZE)
    v_preds  = critic_apply({"params": critic_params}, obs_flat).reshape(N, H)
    tgt_sg   = jax.lax.stop_gradient(targets)

    # Huber loss (δ=10): quadratic for |v_pred - target| < 10, linear beyond.
    # δ=10 covers the full range of V values for this reward scale (~[-50, +10]).
    # Quadratic regime gives proper gradient for normal TD errors; the linear
    # regime caps the gradient from rare large-magnitude terminal targets.
    err   = v_preds - tgt_sg
    huber = jnp.where(jnp.abs(err) < 10.0,
                      0.5 * err ** 2,
                      10.0 * jnp.abs(err) - 50.0)
    n_act = jnp.maximum(jnp.sum(acts), 1.0)
    return VF_COEF * jnp.sum(acts * huber) / n_act


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
          2. N_CRITIC_UPDATES × _critic_loss → aggiorna critico
          3. Soft update target network
        """
        # ── 1. Actor update ───────────────────────────────────────────────────
        (a_loss, (returns, (obs_seq, rewards, dones, final_obs, active_mask))), a_grads = \
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

        # ── 2. Critic — N_CRITIC_UPDATES steps on the same rollout data ───────
        # Multiple steps let the critic converge before the next BPTT update.
        # Target network is soft-updated after each critic step (TAU=0.02).
        def _one_critic_step(carry, _):
            cp, co, ct = carry
            cl, cg = jax.value_and_grad(_critic_loss)(
                cp, critic_apply,
                obs_seq, rewards, dones, final_obs, active_mask, ct,
            )
            cg = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), cg
            )
            cgn = optax.global_norm(cg)
            cu, new_co = critic_optimizer.update(cg, co, cp)
            new_cp = optax.apply_updates(cp, cu)
            new_ct = jax.tree_util.tree_map(
                lambda tgt, online: (1.0 - TAU) * tgt + TAU * online,
                ct, new_cp
            )
            return (new_cp, new_co, new_ct), (cl, cgn)

        (new_critic_params, new_c_opt, new_target_params), (c_losses, c_gns) = \
            jax.lax.scan(
                _one_critic_step,
                (critic_params, critic_opt_state, critic_target_params),
                None, length=N_CRITIC_UPDATES
            )
        c_loss      = jnp.mean(c_losses)
        c_grad_norm = c_gns[-1]

        # mean_raw_reward: true reward/step in raw units, H-invariant, for curriculum.
        n_active = jnp.maximum(jnp.sum(active_mask), 1.0)
        metrics = {
            "actor_loss":      a_loss,
            "critic_loss":     c_loss,
            "mean_return":     jnp.mean(returns),
            "actor_gn":        a_grad_norm,
            "critic_gn":       c_grad_norm,
            "mean_raw_reward": jnp.sum(rewards * active_mask) / n_active,
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
    _WARMUP  = 100   # warmup più lungo: BPTT ha alta varianza nelle prime iterazioni
    _sched_a = optax.join_schedules(
        [optax.linear_schedule(LR_ACTOR * 0.01, LR_ACTOR, _WARMUP),
         optax.cosine_decay_schedule(LR_ACTOR, TOTAL_UPDATES - _WARMUP, alpha=0.05)],
        [_WARMUP]
    )
    _sched_c = optax.join_schedules(
        [optax.linear_schedule(LR_CRITIC * 0.01, LR_CRITIC, _WARMUP),
         optax.cosine_decay_schedule(LR_CRITIC, TOTAL_UPDATES - _WARMUP, alpha=0.05)],
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
    rolling_ret  = -5.0    # stima iniziale in unità reward/step
    cur_dist     = curriculum_dist(rolling_ret)
    cur_stage    = curriculum_stage(rolling_ret)

    print(f"Curriculum: stage={cur_stage}, min_goal_dist={cur_dist:.1f}m")

    # ── Campiona stati iniziali ───────────────────────────────────────────────
    print(f"Campionamento {N_ENVS} stati iniziali...")
    rng, sample_rng = jax.random.split(rng)
    obs_batch, state_batch = _sample_init_states(sample_rng, cur_dist, ghost_robot)
    print(f"Pronti. obs shape={obs_batch.shape}")

    # ── Verifica gradient flow ────────────────────────────────────────────────
    # IMPORTANT: run on a tiny slice (4 envs) — only tests that gradients are
    # non-zero/finite. Using all N_ENVS here forces XLA to plan the full graph
    # (N_ENVS × H_MAX activations) before JIT compilation, causing OOM on
    # 10 GB cards. The first real update_step call will JIT-compile the full
    # kernel and will use jax.checkpoint to stay within budget.
    _N_TEST = 4
    print(f"Verifica gradient flow (su {_N_TEST} env per evitare OOM)...")
    _mask_test = make_horizon_mask(H_INIT)
    _keys_test = jax.random.split(jax.random.PRNGKey(0), _N_TEST)
    _obs_test   = obs_batch[:_N_TEST]
    _state_test = jax.tree_util.tree_map(lambda x: x[:_N_TEST], state_batch)
    def _test_loss(p):
        returns, _ = _rollout_batched(
            p, critic_params, actor_net.apply, critic_net.apply,
            _obs_test, _state_test, _keys_test, _mask_test, ghost_robot,
        )
        return -jnp.mean(returns)
    _grads = jax.grad(_test_loss)(actor_params)
    _leaves  = jax.tree_util.tree_leaves(_grads)
    _has_nan = any(bool(jnp.any(jnp.isnan(g) | jnp.isinf(g))) for g in _leaves)
    if _has_nan:
        _grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), _grads
        )
        print("  ‖∇θ J‖ = NaN/Inf (azzerato, procedo comunque)")
    _gn = float(optax.global_norm(_grads))
    print(f"  ‖∇θ J‖ = {_gn:.4e}  "
          f"({'OK' if _gn > 1e-8 else 'ATTENZIONE gradiente molto piccolo'})")

    # ── Critic warmup ─────────────────────────────────────────────────────────
    # Pre-train the critic for CRITIC_WARMUP_UPDATES steps with the actor frozen.
    # Without warmup V(s_H) ≈ 0, so the bootstrap in the actor return is wrong
    # from step 0 and the actor gradient points in a misleading direction.
    print(f"Critic warmup ({CRITIC_WARMUP_UPDATES} steps, critic-only)...")

    @jax.jit
    def _critic_warmup_step(ap_sg, cp, ct, co,
                             obs_b, state_b, keys_b, hmask):
        # _actor_loss returns (loss, (returns, (obs_seq, rewards, dones, final_obs, active_mask)))
        _, (_, rollout) = _actor_loss(
            ap_sg, cp, actor_net.apply, critic_net.apply,
            obs_b, state_b, keys_b, hmask, ghost_robot,
        )
        obs_seq, rewards, dones, final_obs, active_mask = rollout

        def _one(carry, _):
            cp_i, co_i, ct_i = carry
            cl, cg = jax.value_and_grad(_critic_loss)(
                cp_i, critic_net.apply,
                obs_seq, rewards, dones, final_obs, active_mask, ct_i,
            )
            cg = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), cg
            )
            cu, new_co_i = critic_optimizer.update(cg, co_i, cp_i)
            new_cp_i = optax.apply_updates(cp_i, cu)
            new_ct_i = jax.tree_util.tree_map(
                lambda t, o: (1.0 - TAU) * t + TAU * o, ct_i, new_cp_i
            )
            return (new_cp_i, new_co_i, new_ct_i), cl

        (new_cp, new_co, new_ct), losses = jax.lax.scan(
            _one, (cp, co, ct), None, length=N_CRITIC_UPDATES
        )
        return new_cp, new_ct, new_co, jnp.mean(losses)

    _wu_mask = make_horizon_mask(H_INIT)
    # stop_gradient so warmup never computes BPTT gradients through actor
    _ap_sg = jax.lax.stop_gradient(actor_params)
    for _wu in range(CRITIC_WARMUP_UPDATES):
        rng, _wu_rng, _wu_smp = jax.random.split(rng, 3)
        if _wu % RESAMPLE_INTERVAL == 0:
            obs_batch, state_batch = _sample_init_states(_wu_smp, cur_dist, ghost_robot)
        _wu_keys = jax.random.split(_wu_rng, N_ENVS)
        critic_params, critic_target_params, critic_opt_state, _wu_loss = \
            _critic_warmup_step(_ap_sg, critic_params, critic_target_params,
                                critic_opt_state, obs_batch, state_batch,
                                _wu_keys, _wu_mask)
        if (_wu + 1) % 10 == 0:
            print(f"  warmup {_wu+1:3d}/{CRITIC_WARMUP_UPDATES}  "
                  f"critic_loss={float(_wu_loss):.4f}")
    print("Critic warmup done.")

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

    best_r_step = -999.0
    t_start     = time.time()

    hdr = (
        f"{'Upd':>5} | {'R/step':>7} {'Roll':>7} | "
        f"{'Critic-L':>8} {'AGN':>6} {'CGN':>6} | "
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
        # Bug #2 fix: curriculum EMA must use reward/step, not the discounted
        # return. mean_return = Σ γ^t·r_t grows with H → thresholds triggered
        # too early as H increased in curriculum. mean_raw_reward = mean r_t
        # over active steps, invariant to H, matching the calibrated thresholds.
        mean_r_step = float(metrics["mean_raw_reward"])

        # FPS calcolato sul rollout: N_ENVS × H step per update
        env_steps_per_update = N_ENVS * horizon
        fps = env_steps_per_update / (time.time() - t0)

        # EMA on reward/step — invariant to horizon length H
        rolling_ret = 0.95 * rolling_ret + 0.05 * mean_r_step

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
                f"{update:>5d} | {mean_r_step:>7.3f} {rolling_ret:>7.3f} | "
                f"{critic_loss:>8.4f} {a_gn:>6.3f} {c_gn:>6.3f} | "
                f"{horizon:>3d} {cur_stage:>5d} {cur_dist:>4.1f}m | "
                f"{fps:>7,.0f} {elapsed:>5.1f}min"
            )
            total_env_steps = (update + 1) * env_steps_per_update
            _log_writer.writerow([
                update, total_env_steps,
                round(actor_loss, 5), round(critic_loss, 5),
                round(mean_ret, 3), round(a_gn, 4), round(c_gn, 4),
                horizon, cur_stage, round(cur_dist, 1),
                round(rolling_ret, 3), round(mean_r_step, 4), round(elapsed, 2),
            ])
            _log_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────────────
        # Use mean_r_step (reward/step, H-invariant) as checkpoint criterion.
        # mean_ret (discounted sum) grows negatively with H and is not comparable
        # across curriculum stages — it's a misleading checkpoint signal.
        if mean_r_step > best_r_step:
            best_r_step = mean_r_step
            save_checkpoint(
                actor_params, critic_params, critic_target_params,
                actor_opt_state, critic_opt_state,
                "checkpoints/shac_best.msgpack"
            )

    # ── Fine ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best r/step: {best_r_step:.4f}")

    save_checkpoint(
        actor_params, critic_params, critic_target_params,
        actor_opt_state, critic_opt_state,
        "checkpoints/shac_final.msgpack"
    )
    _log_file.close()
    print(f"Log → {_LOG_PATH}")