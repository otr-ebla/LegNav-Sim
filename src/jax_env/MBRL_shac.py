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

INTEGRAZIONE CON PPO
--------------------

jax_mbrl.py coordina PPO e SHAC in un ibrido:

  • PPO  → esplorazione globale, robustezza, stabilità
  • SHAC → fine-tuning locale con gradiente esatto, convergenza veloce

Il mixing ratio α cresce con il success rate del curriculum:
  α=0.0 all'inizio (solo PPO, quando la policy è rumorosa)
  α→0.8 a convergenza (SHAC domina, PPO mantiene l'entropia)

INSTABILITÀ DEL GRADIENTE
--------------------------

BPTT su horizon H accumula il prodotto Jacobiano ∏ ∂s_{t+1}/∂s_t.
Le discontinuità residue (jnp.clip) producono gradienti zero in saturazione,
non NaN — quindi non fanno esplodere ma possono causare vanishing.
Mitigazioni implementate:

  1. Gradient clipping su norma globale (separato da PPO clip).
  2. Horizon warmup: H parte da H_MIN, cresce ogni SHAC_H_GROWTH_INTERVAL steps.
  3. Stop-gradient sulla V_φ bootstrap (previene feedback loop critico→policy).
  4. Normalizzazione del ritorno per scale invariance.

NOTA SULLA DIFFERENZIABILITÀ DI lax.scan
-----------------------------------------

jax.lax.scan è completamente differenziabile via BPTT.
Il suo gradiente è calcolato efficacemente con l'algoritmo di Pearlmutter
(forward accumulation degli Jacobiani + backward sweep) che XLA fonde in
un singolo kernel — non ci sono copie intermedie per step.
Su H=16 passi e 4096 envs paralleli questo è ~O(H · cost_forward),
lo stesso ordine di complessità di PPO sulla stessa batch.
"""

import os
import functools
import jax
import jax.numpy as jnp
import optax
import flax.serialization
from typing import Tuple, NamedTuple

# Importazioni dal codebase esistente
from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_env_multi import step_env, reset_env, EnvState
from jax_wrappers import make_stacked_env, StackedEnvState

# ── Iperparametri SHAC ────────────────────────────────────────────────────────

# Horizon: numero di passi su cui fare BPTT.
# Inizia corto (gradienti stabili), cresce con il training.
SHAC_H_INIT  = 4      # passi iniziali — gradienti sicuramente stabili
SHAC_H_MAX   = 24     # massimo — ~3.6s di simulazione a dt=0.15
SHAC_H_GROWTH_INTERVAL = 40   # outer updates tra ogni aumento di H

# Numero di envs paralleli per SHAC — separato da NUM_ENVS di PPO.
# Più piccolo perché la BPTT per env è più costosa della forward-only di PPO.
N_SHAC_ENVS = 1024

# Learning rates SHAC (separati da PPO — il gradiente esatto richiede lr più piccolo)
SHAC_LR_ACTOR  = 3e-5   # lr per la policy — gradiente esatto, no noise → lr basso
SHAC_LR_CRITIC = 3e-4   # lr per il critico TD — come PPO ma separato

# Discount e lambda per il critico SHAC
SHAC_GAMMA    = 0.99
SHAC_LAM      = 0.95   # TD(λ) per il critico
SHAC_VF_COEF  = 0.5

# Gradient clipping SHAC (separato e più stretto di PPO per stabilità BPTT)
SHAC_GRAD_NORM_ACTOR  = 0.2
SHAC_GRAD_NORM_CRITIC = 1.0

# Normalizzazione ritorni (stabilizza il gradiente analitico attraverso l'horizon)
SHAC_RETURN_NORM_EPS = 1e-8

# Mixing ratio PPO↔SHAC in jax_mbrl.py:
# alpha = 0 → solo PPO, alpha = 1 → solo SHAC
# Viene calcolato esternamente; qui esponiamo solo la funzione di update.

OBS_SIZE = 3 * 3 + 9 + 108 * 3   # 342 — deve coincidere con jax_train.py


# ── Stato del trainer SHAC ────────────────────────────────────────────────────

class SHACTrainState(NamedTuple):
    """Stato completo del trainer SHAC (serializzabile con flax)."""
    actor_params:  any   # parametri della policy (condivisi con PPO)
    critic_params: any   # parametri del critico SHAC (separati)
    actor_opt:     any   # stato optimizer actor
    critic_opt:    any   # stato optimizer critico
    horizon:       int   # H corrente (cresce con il training)
    update_count:  int   # outer updates completati


# ── Rete critico SHAC ─────────────────────────────────────────────────────────
# Critico separato dalla rete actor-critic di PPO.
# Architettura deliberatamente più semplice: il critico SHAC lavora su
# traiettorie brevi (H piccolo), la rappresentazione non deve essere profonda.

class SHACCritic(flax.linen.Module):
    """
    Critico dedicato a SHAC — stima V(s) per il bootstrap a fine orizzonte.

    Separato dal critico di PPO per evitare interferenze:
      - PPO addestra il critico con GAE su ROLLOUT_STEPS=64 passi
      - SHAC usa un critico che vede SHAC_H_MAX=24 passi di ritorno discounted
      Il bias temporale è diverso → due critici separati è più pulito.
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
    init_obs:   jnp.ndarray,    # (OBS_SIZE,)
    init_state: StackedEnvState,
    rng_key:    jnp.ndarray,
    horizon:    int,
    ghost_robot: bool = True,
):
    """
    Esegue un rollout di `horizon` passi per UN singolo environment.
    Ritorna J(θ) = Σ γ^t r_t + γ^H V(s_H).

    Questa funzione è differenziabile rispetto ad actor_params e critic_params.
    Viene vmappata su N_SHAC_ENVS envs in parallelo.

    NOTA: horizon è un argomento Python (non traced) perché lax.scan richiede
    length statica. Il chiamante passa il valore corretto per ogni update.

    NOTA 2: lo stop_gradient su V(s_H) è deliberato: non vogliamo che il
    gradiente dell'actor scorra attraverso il critico durante l'ottimizzazione
    dell'actor. Il critico è addestrato separatamente.
    """

    def _step_fn(carry, _):
        obs, state, key, cum_discount = carry
        key, action_key, step_key, obs_key = jax.random.split(key, 4)

        # ── Azione DETERMINISTICA dalla policy (nessun campionamento) ──────────
        # Usare la media è fondamentale per la differenziabilità:
        # il campionamento gaussiano introduce noise non differenziabile
        # rispetto ai parametri (è differenziabile via reparametrizzazione,
        # ma introduce varianza nel gradiente che annulla il vantaggio di SHAC).
        # La media è la policy "più probabile" — per SHAC vogliamo il gradiente
        # del comportamento deterministico, non del comportamento stocastico.
        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean = mean[0]   # rimuovi batch dim
        raw_action = mean  # deterministic

        # Scala l'azione all'intervallo dell'environment
        env_action = scale_action_to_env(raw_action, state.env_state.max_v)

        # ── Step dell'environment ──────────────────────────────────────────────
        # step_env è differenziabile (hsfm_diff, jnp.where ovunque)
        # La differenziabilità si propaga: obs_new e reward dipendono
        # in modo smooth da env_action che dipende da actor_params.
        base_obs, new_base_state, reward, done, info = step_env(
            step_key, state.env_state, env_action, ghost_robot=ghost_robot
        )

        # Aggiorna lo stack di osservazioni (stesso meccanismo di jax_wrappers)
        from jax_env import NUM_RAYS, STATE_VEC_SIZE
        POSE_SIZE = 3
        new_pose      = base_obs[0:POSE_SIZE]
        new_state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[POSE_SIZE + STATE_VEC_SIZE:]

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

        # Accumula reward scontato (nota: done=1 azzera il discount futuro
        # per fermare il gradiente agli episodi terminati — comportamento corretto)
        discounted_r = cum_discount * reward * (1.0 - done.astype(jnp.float32))

        return (new_obs, new_stacked, key, cum_discount * SHAC_GAMMA), \
               (discounted_r, new_obs, done)

    # Esegui il rollout completo — completamente differenziabile
    (final_obs, final_state, _, _), (rewards, obs_seq, dones) = jax.lax.scan(
        _step_fn,
        (init_obs, init_state, rng_key, 1.0),
        None,
        length=horizon,
    )

    # ── Bootstrap con il critico a fine orizzonte ─────────────────────────────
    # Stop-gradient: il critico è un target fisso per l'actor.
    # Addestriamo il critico separatamente sotto.
    bootstrap_v = jax.lax.stop_gradient(
        critic_apply({"params": critic_params}, final_obs[None])[0]
    )

    # Ritorno totale = Σ reward scontati + bootstrap
    # I reward sono già scontati cumulativamente nel scan (cum_discount * r)
    total_return = jnp.sum(rewards) + SHAC_GAMMA ** horizon * bootstrap_v

    # Normalizza per scale invariance (stabile attraverso horizon diversi)
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
    horizon,
    ghost_robot,
):
    """
    J(θ) = E_envs[ Σ γ^t r_t + γ^H V(s_H) ]

    jax.grad(actor_loss_fn)(actor_params) è il gradiente analitico della policy.
    Varianza zero rispetto al campionamento — il vantaggio principale di SHAC.
    """
    returns, (obs_seq, rewards, dones, final_obs) = _shac_rollout_batched(
        actor_params, critic_params, actor_apply, critic_apply,
        init_obs_batch, init_state_batch, rng_keys, horizon, ghost_robot,
    )

    # Normalizza i ritorni per stabilità numerica
    ret_mean = jnp.mean(returns)
    ret_std  = jnp.std(returns) + SHAC_RETURN_NORM_EPS
    normalized_returns = (returns - ret_mean) / ret_std

    # Loss: massimizza il ritorno atteso → minimizza il negativo
    loss = -jnp.mean(normalized_returns)
    return loss, (returns, obs_seq, rewards, dones, final_obs)


# ── Loss function del critico SHAC ────────────────────────────────────────────

def _critic_loss_fn(
    critic_params,
    critic_apply,
    obs_seq,      # (N_SHAC_ENVS, H, OBS_SIZE)
    rewards,      # (N_SHAC_ENVS, H)
    dones,        # (N_SHAC_ENVS, H)
    final_obs,    # (N_SHAC_ENVS, OBS_SIZE)
    old_critic_params,  # per calcolare il target — stop_gradient
):
    """
    Addestra il critico con TD(λ) sui rollout SHAC.

    Il target usa old_critic_params (stop_gradient) per evitare il
    deadly triad: critic aggiorna sia il target che la predizione
    con gli stessi parametri → divergenza.
    """
    N, H = rewards.shape

    # ── Calcola il ritorno TD(λ) come target ─────────────────────────────────
    # bootstrap dall'ultimo stato con il target network (old params)
    v_final = jax.lax.stop_gradient(
        critic_apply({"params": old_critic_params}, final_obs)  # (N,)
    )

    def _td_lambda_scan(carry, t):
        # carry: (gae, next_v)
        gae, next_v = carry
        r   = rewards[:, t]
        d   = dones[:, t].astype(jnp.float32)
        obs = obs_seq[:, t, :]

        v_pred  = critic_apply({"params": old_critic_params}, obs)
        delta   = r + SHAC_GAMMA * next_v * (1.0 - d) - v_pred
        gae     = delta + SHAC_GAMMA * SHAC_LAM * (1.0 - d) * gae
        target  = jax.lax.stop_gradient(gae + v_pred)
        return (gae, v_pred), target

    (_, _), targets = jax.lax.scan(
        _td_lambda_scan,
        (jnp.zeros(N), v_final),
        jnp.arange(H - 1, -1, -1),  # scan al contrario (da H a 0)
        reverse=False,
    )
    # targets shape: (H, N) — trasponi in (N, H)
    targets = targets.T  # (N, H)

    # ── MSE tra predizioni e target ───────────────────────────────────────────
    # Calcola V(s) per tutti gli step
    obs_flat    = obs_seq.reshape(N * H, OBS_SIZE)
    v_preds     = critic_apply({"params": critic_params}, obs_flat)  # (N*H,)
    v_preds     = v_preds.reshape(N, H)

    targets_sg  = jax.lax.stop_gradient(targets)
    critic_loss = SHAC_VF_COEF * jnp.mean((v_preds - targets_sg) ** 2)
    return critic_loss


# ── Step di update SHAC (JIT-compilato) ──────────────────────────────────────

@functools.partial(jax.jit, static_argnums=(2, 3, 6, 7))
def shac_update_step(
    shac_state:       SHACTrainState,
    env_data:         Tuple,   # (init_obs_batch, init_state_batch, rng_keys)
    actor_apply,               # static: funzione apply della rete actor
    critic_apply,              # static: funzione apply del critico
    actor_optimizer,           # ottimizzatore actor
    critic_optimizer,          # ottimizzatore critico
    horizon:          int,     # static: H corrente
    ghost_robot:      bool,    # static: modalità ghost
):
    """
    Un update SHAC completo.
    Ritorna (new_shac_state, metrics_dict).

    Questo è l'unico punto di ingresso JIT per tutto il training SHAC —
    actor e critico vengono aggiornati in sequenza nello stesso step.
    """
    actor_params  = shac_state.actor_params
    critic_params = shac_state.critic_params
    actor_opt     = shac_state.actor_opt
    critic_opt    = shac_state.critic_opt

    init_obs_batch, init_state_batch, rng_keys = env_data

    # ── 1. Gradiente ANALITICO dell'actor ────────────────────────────────────
    # value_and_grad calcola J e ∇_θ J in un solo forward+backward pass.
    # Il backward pass percorre la pipeline:
    #   ∇_θ J = ∂J/∂returns · ∂returns/∂rewards · ∂rewards/∂actions · ∂actions/∂θ
    # dove ogni ∂ è calcolato esattamente da XLA automatic differentiation.
    (actor_loss, (returns, obs_seq, rewards, dones, final_obs)), actor_grads = \
        jax.value_and_grad(_actor_loss_fn, has_aux=True)(
            actor_params, critic_params, actor_apply, critic_apply,
            init_obs_batch, init_state_batch, rng_keys, horizon, ghost_robot,
        )

    # Clipping del gradiente dell'actor — più stretto di PPO perché il gradiente
    # analitico è esatto ma può essere grande quando la policy è lontana dall'ottimo.
    actor_grad_norm = optax.global_norm(actor_grads)
    actor_updates, new_actor_opt = actor_optimizer.update(
        actor_grads, actor_opt, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    # ── 2. Aggiornamento del critico (TD(λ)) ─────────────────────────────────
    # Il critico usa gli stessi rollout generati dall'actor — nessun passo
    # di simulazione aggiuntivo. Costo totale = 1 rollout forward + 2 backward.
    critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(
        critic_params, critic_apply,
        obs_seq, rewards, dones, final_obs,
        critic_params,   # old params = current (soft target, non hard copy)
    )

    critic_grad_norm = optax.global_norm(critic_grads)
    critic_updates, new_critic_opt = critic_optimizer.update(
        critic_grads, critic_opt, critic_params
    )
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    # ── 3. Aggiorna lo stato ──────────────────────────────────────────────────
    new_state = SHACTrainState(
        actor_params  = new_actor_params,
        critic_params = new_critic_params,
        actor_opt     = new_actor_opt,
        critic_opt    = new_critic_opt,
        horizon       = shac_state.horizon,
        update_count  = shac_state.update_count + 1,
    )

    metrics = {
        "actor_loss":        actor_loss,
        "critic_loss":       critic_loss,
        "mean_return":       jnp.mean(returns),
        "actor_grad_norm":   actor_grad_norm,
        "critic_grad_norm":  critic_grad_norm,
        "horizon":           horizon,
    }
    return new_state, metrics


# ── Inizializzazione ──────────────────────────────────────────────────────────

def init_shac(rng_key, actor_params, actor_apply):
    """
    Inizializza lo stato SHAC dato i parametri actor di PPO.

    actor_params: condivisi con PPO — SHAC aggiorna la stessa rete.
    Il critico SHAC è una rete separata, inizializzata qui.
    """
    rng_key, ck = jax.random.split(rng_key)

    critic_net    = SHACCritic()
    dummy_obs     = jnp.zeros((1, OBS_SIZE))
    critic_params = critic_net.init(ck, dummy_obs)["params"]
    critic_apply  = critic_net.apply

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(SHAC_GRAD_NORM_ACTOR),
        optax.adam(SHAC_LR_ACTOR, eps=1e-5),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(SHAC_GRAD_NORM_CRITIC),
        optax.adam(SHAC_LR_CRITIC, eps=1e-5),
    )

    actor_opt  = actor_optimizer.init(actor_params)
    critic_opt = critic_optimizer.init(critic_params)

    state = SHACTrainState(
        actor_params  = actor_params,
        critic_params = critic_params,
        actor_opt     = actor_opt,
        critic_opt    = critic_opt,
        horizon       = SHAC_H_INIT,
        update_count  = 0,
    )
    return state, critic_apply, actor_optimizer, critic_optimizer


def get_shac_horizon(update_count: int) -> int:
    """
    Horizon curriculum: cresce linearmente da SHAC_H_INIT a SHAC_H_MAX.
    Iniziare con H piccolo è essenziale: con H=24 i gradienti di BPTT
    all'inizio del training (policy casuale) sono rumorosissimi.
    Con H=4 i gradienti locali sono significativi già dal primo update.
    """
    n_increases = update_count // SHAC_H_GROWTH_INTERVAL
    h = SHAC_H_INIT + n_increases
    return min(h, SHAC_H_MAX)


def save_shac_checkpoint(shac_state, filepath):
    """Serializza lo stato SHAC (actor + critico + optimizer states)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        "actor_params":  jax.device_get(shac_state.actor_params),
        "critic_params": jax.device_get(shac_state.critic_params),
        "actor_opt":     jax.device_get(shac_state.actor_opt),
        "critic_opt":    jax.device_get(shac_state.critic_opt),
        "horizon":       shac_state.horizon,
        "update_count":  shac_state.update_count,
    }
    with open(filepath, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    print(f"  SHAC checkpoint → {filepath}")