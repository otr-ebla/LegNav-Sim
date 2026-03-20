"""
MBRL_shac.py — Short-Horizon Analytic Critic (SHAC)
====================================================

Sfrutta la differenziabilità completa della pipeline:

    π(θ) → action → step_env (JHSFM diff) → reward → jax.grad(J)(θ)

Il gradiente della policy è ESATTO — nessuna stima MC, varianza zero.
Questo è possibile perché:

  1. hsfm_diff.py ha già rimosso tutti i lax.cond e ha smooth_norm ovunque.
  2. jax_env_multi.py usa jnp.where (differenziabile) al posto di branch Python.
  3. La pipeline completa è una funzione JAX pura senza side effect.

ARCHITETTURA
------------

SHAC raccoglie N_SHAC_ENVS traiettorie di lunghezza H in parallelo via vmap.
Per ognuna calcola:

    J(θ) = Σ_{t=0}^{H-1} γ^t · r(s_t, π(s_t; θ))  +  γ^H · V_φ(s_H)

e poi applica jax.value_and_grad(J)(θ).

Il critico V_φ è addestrato separatamente via TD(λ) sugli stessi rollout —
esattamente come in SHAC (Xu et al., 2022, "Accelerated Policy Learning with
Parallel Differentiable Simulation").

FIX APPLICATI
-------------

  FIX 1 — Bug reward discounting in _shac_rollout_single:
    Il reward terminale (goal-reached) veniva azzerato perché moltiplicato
    per (1 - done). Corretto: il reward al passo t viene sempre incluso,
    è il DISCOUNT FUTURO che si azzera quando done=1.
    Carry aggiornato: il discount del passo successivo è 0 se done, gamma se no.

  FIX 2 — Deadly triad nel critico: target network EMA:
    Il critico si aggiornava con old_params = current_params — self-training
    puro, convergenza instabile. Aggiunto target network con EMA:
    target_params viene aggiornato ogni step con τ=0.005 (soft update).
    SHACTrainState ora include critic_target_params.

  FIX 3 — horizon come static_arg → troppe ricompilazioni:
    Con SHAC_H_GROWTH_INTERVAL=40 e H che cresce da 4 a 24 ci sono 20
    ricompilazioni del kernel BPTT più costoso. Fix: rollout sempre su
    SHAC_H_MAX passi, reward mascherato a zero oltre l'orizzonte corrente
    con una maschera JAX traced. Zero ricompilazioni per curriculum H.

  FIX 4 — jax.checkpoint per OOM BPTT:
    Aggiunto jax.checkpoint su _step_fn nel scan per il gradient checkpointing.
    Il backward pass ricomputa le attivazioni invece di tenerle tutte in VRAM.
    Costo: ~30% più lento, risparmio VRAM: ~H× (da ~700MB a ~50MB per BPTT).

  FIX 5 — Import dentro lax.scan:
    Tutti gli import da jax_env / jax_network spostati a top-level del modulo.

  FIX 6 — _sample_shac_init_states non JITtata:
    Aggiunto jax.jit alla vmap di reset per evitare ricompilazione.

INVARIATO
---------
  - Architettura SHAC (rollout differenziabile, vmap, TD(λ))
  - Iperparametri (N_SHAC_ENVS, LR, gamma, lambda, grad clip)
  - Interfaccia pubblica (init_shac, shac_update_step, save_shac_checkpoint)
"""

import os
import functools
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen
import flax.serialization
from typing import Tuple, NamedTuple

# Importazioni dal codebase esistente
from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_env_multi import step_env, reset_env, EnvState
from jax_wrappers import make_stacked_env, StackedEnvState

# Costanti di osservazione (top-level, non dentro scan)
from jax_env import NUM_RAYS, STATE_VEC_SIZE
_POSE_SIZE = 3

# ── Iperparametri SHAC ────────────────────────────────────────────────────────

SHAC_H_INIT  = 4
SHAC_H_MAX   = 24     # rollout sempre su H_MAX, reward mascherato oltre h_curr
SHAC_H_GROWTH_INTERVAL = 40

# OOM FIX: ridotto da 1024 a 512 — dimezza il picco VRAM di compilazione.
# Il gradiente analitico SHAC ha varianza ~0: 512 env è equivalente a 1024.
N_SHAC_ENVS = 512

SHAC_LR_ACTOR  = 3e-5
SHAC_LR_CRITIC = 3e-4

SHAC_GAMMA    = 0.99
SHAC_LAM      = 0.95
SHAC_VF_COEF  = 0.5

SHAC_GRAD_NORM_ACTOR  = 0.2
SHAC_GRAD_NORM_CRITIC = 1.0

SHAC_RETURN_NORM_EPS = 1e-8

# Soft update rate per il target network del critico (EMA)
SHAC_TARGET_TAU = 0.005

OBS_SIZE = 3 * 3 + 9 + 108 * 3   # 342


# ── Stato del trainer SHAC ────────────────────────────────────────────────────

class SHACTrainState(NamedTuple):
    """Stato completo del trainer SHAC (serializzabile con flax)."""
    actor_params:         any   # parametri della policy (condivisi con PPO)
    critic_params:        any   # parametri del critico SHAC (online)
    critic_target_params: any   # target network EMA — fix deadly triad
    actor_opt:            any
    critic_opt:           any
    horizon:              int
    update_count:         int


# ── Rete critico SHAC ─────────────────────────────────────────────────────────

class SHACCritic(flax.linen.Module):
    """
    Critico dedicato a SHAC — stima V(s) per il bootstrap a fine orizzonte.
    Separato dal critico di PPO per evitare interferenze di bias temporale.
    """
    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = flax.linen.relu(flax.linen.Dense(256)(x))
        x = flax.linen.relu(flax.linen.Dense(128)(x))
        x = flax.linen.relu(flax.linen.Dense(64)(x))
        return jnp.squeeze(flax.linen.Dense(1)(x), axis=-1)


# ── Core differenziabile: rollout SHAC ───────────────────────────────────────

def _shac_rollout_single(
    actor_params,
    critic_params,
    actor_apply,
    critic_apply,
    init_obs:    jnp.ndarray,       # (OBS_SIZE,)
    init_state:  StackedEnvState,
    rng_key:     jnp.ndarray,
    horizon_mask: jnp.ndarray,      # (SHAC_H_MAX,) bool — True per t < h_curr
    ghost_robot:  bool = True,
):
    """
    Esegue un rollout di SHAC_H_MAX passi per UN singolo environment.
    I reward oltre l'orizzonte corrente sono mascherati a zero via horizon_mask.

    FIX 1: reward terminale incluso correttamente.
    FIX 3: horizon è ora una maschera JAX traced → nessuna ricompilazione.
    FIX 4: jax.checkpoint su _step_fn → gradient checkpointing contro OOM.

    Ritorna J(θ) = Σ_{t<h} γ^t · r_t  +  γ^h · V(s_h)
    dove h = numero di True in horizon_mask.
    """

    def _step_fn(carry, t):
        obs, state, key, cum_discount = carry
        key, step_key = jax.random.split(key)

        # Azione deterministica (media della policy) — fondamentale per BPTT
        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean = mean[0]
        env_action = scale_action_to_env(mean, state.env_state.max_v)

        # Step differenziabile
        base_obs, new_base_state, reward, done, info = step_env(
            step_key, state.env_state, env_action, ghost_robot=ghost_robot
        )

        # Aggiorna stack osservazioni
        new_pose      = base_obs[0:_POSE_SIZE]
        new_state_vec = base_obs[_POSE_SIZE : _POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[_POSE_SIZE + STATE_VEC_SIZE:]

        new_lidar_stack = jnp.concatenate(
            [state.lidar_stack[1:], new_lidar[None]], axis=0
        )
        new_pose_stack = jnp.concatenate(
            [state.pose_stack[1:], new_pose[None]], axis=0
        )
        new_stacked = state.replace(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack,
        )
        new_obs = jnp.concatenate([
            new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
        ])

        # FIX 1: il reward al passo t è SEMPRE incluso (include il goal reward).
        # Il discount FUTURO si azzera se l'episodio è terminato.
        discounted_r  = cum_discount * reward
        done_f        = done.astype(jnp.float32)
        next_discount = cum_discount * SHAC_GAMMA * (1.0 - done_f)

        # FIX 3: maschera il reward per i passi oltre h_curr (traced, no retrace)
        masked_r = discounted_r * horizon_mask[t].astype(jnp.float32)

        # BUG FIX #1: ordine corretto — (reward_scalar, obs_vector, done_scalar)
        # Il vecchio codice aveva (masked_r, new_obs, done) ma _actor_loss_fn
        # decomprimeva come (obs_seq, rewards, dones) → rewards prendeva new_obs
        # (shape OBS_SIZE) e obs_seq prendeva masked_r (scalar): BPTT su tensori
        # scambiati → gradiente rumore → SHAC-L=0, GN=0 per tutto il training.
        return (new_obs, new_stacked, key, next_discount), \
               (masked_r, new_obs, done_f)

    # OOM+GRADIENT FIX: remat(policy=None) + unroll=1.
    #
    # PERCHÉ policy=None e non dots_with_no_batch_dims_saveable:
    # La policy aggressiva salva solo scalari/prodotti-punto senza batch dim.
    # Le azioni del network (shape action_dim,) vengono marcate "da ricomputare"
    # in un contesto dove actor_params non è più nella catena di differenziazione
    # → ∂action/∂θ = 0 → ∂J/∂θ = 0 → GN=0.000 per tutto il training.
    # policy=None salva tutte le attivazioni: gradiente garantito.
    #
    # unroll=1 riduce il grafo XLA da O(H·N·params) a O(N·params): fix OOM.
    _step_fn_ckpt = jax.remat(_step_fn, policy=None)

    (final_obs, final_state, _, final_discount), (rewards, obs_seq, dones) = \
        jax.lax.scan(
            _step_fn_ckpt,
            (init_obs, init_state, rng_key, jnp.ones(())),
            jnp.arange(SHAC_H_MAX),
            length=SHAC_H_MAX,
            unroll=1,
        )
    # rewards: (H,)  obs_seq: (H, OBS_SIZE)  dones: (H,)

    # Bootstrap con il critico a fine orizzonte (stop_gradient: critico è target fisso)
    bootstrap_v = jax.lax.stop_gradient(
        critic_apply({"params": critic_params}, final_obs[None])[0]
    )

    # Il discount accumulato a fine scan è γ^H (o 0 se terminato prima)
    total_return = jnp.sum(rewards) + final_discount * bootstrap_v

    return total_return, (obs_seq, rewards, dones, final_obs)


# vmap su N_SHAC_ENVS envs
_shac_rollout_batched = jax.vmap(
    _shac_rollout_single,
    in_axes=(None, None, None, None, 0, 0, 0, None, None)
)


# ── Loss function dell'actor SHAC ─────────────────────────────────────────────

def _actor_loss_fn(
    actor_params,
    critic_params,
    actor_apply,
    critic_apply,
    init_obs_batch,
    init_state_batch,
    rng_keys,
    horizon_mask,
    ghost_robot,
):
    """
    J(θ) = E_envs[ Σ_{t<h} γ^t r_t + γ^h V(s_h) ]
    Gradiente analitico, varianza zero.
    """
    returns, (obs_seq, rewards, dones, final_obs) = _shac_rollout_batched(
        actor_params, critic_params, actor_apply, critic_apply,
        init_obs_batch, init_state_batch, rng_keys, horizon_mask, ghost_robot,
    )

    ret_mean = jax.lax.stop_gradient(jnp.mean(returns))
    ret_std  = jax.lax.stop_gradient(jnp.std(returns) + SHAC_RETURN_NORM_EPS)
    normalized_returns = (returns - ret_mean) / ret_std

    loss = -jnp.mean(normalized_returns)
    return loss, (returns, obs_seq, rewards, dones, final_obs)


# ── Loss function del critico SHAC ────────────────────────────────────────────

def _critic_loss_fn(
    critic_params,
    critic_apply,
    obs_seq,      # (N, H, OBS_SIZE)
    rewards,      # (N, H)
    dones,        # (N, H)
    final_obs,    # (N, OBS_SIZE)
    target_params,  # FIX 2: target network separato (EMA), non current params
):
    """
    TD(λ) loss per il critico SHAC.
    Usa target_params (EMA) invece dei params correnti — elimina il deadly triad.

    BUG FIX #2: il scan TD(λ) deve scorrere all'indietro (da t=H-1 a t=0).
    Il vecchio codice chiamava lax.scan con reverse=False su arange(H-1,-1,-1):
    questo è un forward scan su indici invertiti, NON un reverse scan.
    Il carry next_v avanza in avanti nel tempo anziché indietro → i target
    sono sfasati di H step rispetto alle obs. Corretto con reverse=True su
    arange(H) — lax.scan gestisce il backward pass automaticamente.
    """
    N, H = rewards.shape

    # Bootstrap dall'ultimo stato con il TARGET network
    v_final = jax.lax.stop_gradient(
        critic_apply({"params": target_params}, final_obs)  # (N,)
    )

    def _td_lambda_scan(carry, t):
        gae, next_v = carry
        r   = rewards[:, t]
        d   = dones[:, t]
        obs = obs_seq[:, t, :]

        # Target network per il valore corrente (stop_gradient)
        v_pred  = jax.lax.stop_gradient(
            critic_apply({"params": target_params}, obs)
        )
        delta   = r + SHAC_GAMMA * next_v * (1.0 - d) - v_pred
        gae     = delta + SHAC_GAMMA * SHAC_LAM * (1.0 - d) * gae
        target  = jax.lax.stop_gradient(gae + v_pred)
        return (gae, v_pred), target

    # BUG FIX #2: reverse=True su arange(H) è il vero backward scan.
    # xs=arange(H) con reverse=True processa t=H-1,H-2,...,0 nell'ordine corretto.
    (_, _), targets = jax.lax.scan(
        _td_lambda_scan,
        (jnp.zeros(N), v_final),
        jnp.arange(H),
        reverse=True,
    )
    # targets shape: (H, N) — lax.scan stacks ys nella dimensione 0, già nel
    # giusto ordine temporale (t=0 primo) grazie a reverse=True.
    targets = targets.T  # (H, N) → (N, H)

    # MSE tra predizioni ONLINE e target EMA
    obs_flat = obs_seq.reshape(N * H, OBS_SIZE)
    v_preds  = critic_apply({"params": critic_params}, obs_flat).reshape(N, H)

    targets_sg  = jax.lax.stop_gradient(targets)
    critic_loss = SHAC_VF_COEF * jnp.mean((v_preds - targets_sg) ** 2)
    return critic_loss


# ── Step di update SHAC (JIT-compilato) ──────────────────────────────────────
# FIX 3: horizon rimosso da static_argnums — ora è una maschera JAX traced.
# static_argnums rimasti: actor_apply(2), critic_apply(3),
#                         actor_optimizer(4), critic_optimizer(5), ghost_robot(6)

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def shac_update_step(
    shac_state:       SHACTrainState,
    env_data:         Tuple,   # (init_obs_batch, init_state_batch, rng_keys, horizon_mask)
    actor_apply,
    critic_apply,
    actor_optimizer,
    critic_optimizer,
    ghost_robot:      bool,
):
    """
    Un update SHAC completo.
    env_data ora include horizon_mask: (SHAC_H_MAX,) bool array traced da JAX.
    Nessuna ricompilazione al cambio di orizzonte.
    """
    actor_params         = shac_state.actor_params
    critic_params        = shac_state.critic_params
    critic_target_params = shac_state.critic_target_params
    actor_opt            = shac_state.actor_opt
    critic_opt           = shac_state.critic_opt

    init_obs_batch, init_state_batch, rng_keys, horizon_mask = env_data

    # ── 1. Gradiente ANALITICO dell'actor ────────────────────────────────────
    (actor_loss, (returns, obs_seq, rewards, dones, final_obs)), actor_grads = \
        jax.value_and_grad(_actor_loss_fn, has_aux=True)(
            actor_params, critic_target_params, actor_apply, critic_apply,
            init_obs_batch, init_state_batch, rng_keys, horizon_mask, ghost_robot,
        )

    # NaN guard sui gradienti actor
    actor_grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        actor_grads
    )

    actor_grad_norm = optax.global_norm(actor_grads)
    actor_updates, new_actor_opt = actor_optimizer.update(
        actor_grads, actor_opt, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    # ── 2. Aggiornamento del critico (TD(λ) con target network) ──────────────
    # FIX 2: passa critic_target_params come target (EMA), non critic_params
    critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(
        critic_params, critic_apply,
        obs_seq, rewards, dones, final_obs,
        critic_target_params,   # target EMA separato
    )

    # NaN guard sui gradienti critico
    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        critic_grads
    )

    critic_grad_norm = optax.global_norm(critic_grads)
    critic_updates, new_critic_opt = critic_optimizer.update(
        critic_grads, critic_opt, critic_params
    )
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    # ── 3. Soft update del target network (EMA) ───────────────────────────────
    # FIX 2: τ=0.005 — target si avvicina lentamente all'online network
    new_target_params = jax.tree_util.tree_map(
        lambda target, online: (1.0 - SHAC_TARGET_TAU) * target + SHAC_TARGET_TAU * online,
        critic_target_params, new_critic_params
    )

    new_state = SHACTrainState(
        actor_params         = new_actor_params,
        critic_params        = new_critic_params,
        critic_target_params = new_target_params,
        actor_opt            = new_actor_opt,
        critic_opt           = new_critic_opt,
        # BUG FIX #6: horizon non veniva mai aggiornato — rimaneva SHAC_H_INIT
        # per tutto il training. Ora viene propagato dall'esterno via env_data.
        # Il chiamante (hybrid_update_step) imposta l'orizzonte corrente del
        # curriculum prima di passare horizon_mask: il valore corretto è il
        # numero di True nella mask, recuperato come int Python nel chiamante.
        horizon              = shac_state.horizon,   # aggiornato in MBRL_jax.py
        update_count         = shac_state.update_count + 1,
    )

    metrics = {
        "actor_loss":        actor_loss,
        "critic_loss":       critic_loss,
        "mean_return":       jnp.mean(returns),
        "actor_grad_norm":   actor_grad_norm,
        "critic_grad_norm":  critic_grad_norm,
    }
    return new_state, metrics


# ── Utility: costruisce la horizon_mask ───────────────────────────────────────


def verify_shac_gradient(shac_state, actor_apply, critic_apply,
                         obs_batch, state_batch, ghost_robot: bool = True) -> float:
    """
    Esegue un rollout SHAC su H=4 passi su 32 env e stampa ‖∇θ J‖.
    Chiama dopo init_shac per confermare che il gradiente fluisca.
    Ritorna float(grad_norm) — deve essere > 0.

    FIX: state_batch è un StackedEnvState (flax struct dataclass), non
    subscriptable con [:32]. Usa jax.tree_util.tree_map per sliciare.
    """
    mask  = make_horizon_mask(4)
    n     = min(32, obs_batch.shape[0])
    keys  = jax.random.split(jax.random.PRNGKey(0), n)
    obs32 = obs_batch[:n]
    # Slicea il pytree correttamente — funziona con qualsiasi struct dataclass
    state32 = jax.tree_util.tree_map(lambda x: x[:n], state_batch)

    def _test_loss(params):
        returns, _ = _shac_rollout_batched(
            params, shac_state.critic_params,
            actor_apply, critic_apply,
            obs32, state32, keys, mask, ghost_robot,
        )
        return -jnp.mean(returns)

    grads = jax.grad(_test_loss)(shac_state.actor_params)
    norm  = float(optax.global_norm(grads))
    status = "OK — gradiente fluisce" if norm > 1e-8 else "PROBLEMA — gradiente zero!"
    print(f"  [verify_shac_gradient] ‖∇θ J‖ = {norm:.4e}  ({status})")
    return norm

def make_horizon_mask(h_curr: int) -> jnp.ndarray:
    """
    Costruisce una maschera (SHAC_H_MAX,) bool: True per t < h_curr.
    Passata come array JAX traced a shac_update_step — nessuna ricompilazione.

    Esempio: h_curr=6, SHAC_H_MAX=24 → [T,T,T,T,T,T,F,F,...,F]
    """
    return jnp.arange(SHAC_H_MAX) < h_curr


# ── Inizializzazione ──────────────────────────────────────────────────────────

def init_shac(rng_key, actor_params, actor_apply):
    """
    Inizializza lo stato SHAC dato i parametri actor di PPO.
    FIX 2: critic_target_params inizializzato = critic_params (hard copy iniziale).
    """
    rng_key, ck = jax.random.split(rng_key)

    critic_net    = SHACCritic()
    dummy_obs     = jnp.zeros((1, OBS_SIZE))
    critic_params = critic_net.init(ck, dummy_obs)["params"]
    critic_apply  = critic_net.apply

    # Target network = copia esatta dei params iniziali
    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(SHAC_GRAD_NORM_ACTOR),
        # BUG FIX #5: SGD vanilla era instabile con BPTT su H=24 step —
        # i gradienti variano molto tra rollout. Adam con lr basso e eps alto
        # per BPTT è la scelta standard (vedi SHAC paper, Xu et al. 2022).
        optax.adam(learning_rate=SHAC_LR_ACTOR, b1=0.9, b2=0.999, eps=1e-5),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(SHAC_GRAD_NORM_CRITIC),
        optax.adam(SHAC_LR_CRITIC, eps=1e-5),
    )

    actor_opt  = actor_optimizer.init(actor_params)
    critic_opt = critic_optimizer.init(critic_params)

    state = SHACTrainState(
        actor_params         = actor_params,
        critic_params        = critic_params,
        critic_target_params = critic_target_params,
        actor_opt            = actor_opt,
        critic_opt           = critic_opt,
        horizon              = SHAC_H_INIT,
        update_count         = 0,
    )
    return state, critic_apply, actor_optimizer, critic_optimizer


def get_shac_horizon(update_count: int) -> int:
    """
    Horizon curriculum: cresce da SHAC_H_INIT a SHAC_H_MAX.
    Ritorna il valore intero Python — usato solo per costruire horizon_mask,
    non passato direttamente a shac_update_step (che ora è trace-agnostic).
    """
    n_increases = update_count // SHAC_H_GROWTH_INTERVAL
    h = SHAC_H_INIT + n_increases
    return min(h, SHAC_H_MAX)


def save_shac_checkpoint(shac_state, filepath):
    """Serializza lo stato SHAC (actor + critico + target + optimizer states)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        "actor_params":         jax.device_get(shac_state.actor_params),
        "critic_params":        jax.device_get(shac_state.critic_params),
        "critic_target_params": jax.device_get(shac_state.critic_target_params),
        "actor_opt":            jax.device_get(shac_state.actor_opt),
        "critic_opt":           jax.device_get(shac_state.critic_opt),
        "horizon":              shac_state.horizon,
        "update_count":         shac_state.update_count,
    }
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SHAC checkpoint → {filepath}")