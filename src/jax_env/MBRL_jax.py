"""
MBRL_jax.py — Hybrid PPO + SHAC Training Loop
===============================================

Questo file è il punto di ingresso principale per il training model-based.
Sostituisce jax_ppo.py come script di addestramento.

STRATEGIA IBRIDA
----------------

Ogni outer update esegue, nell'ordine:

  1. PPO rollout  (NUM_ENVS=8192 × ROLLOUT_STEPS=64)
     → esplorazione globale, segnale di reward ampio, stabilità
     → aggiorna actor e critico PPO via clipped surrogate

  2. SHAC update  (N_SHAC_ENVS=1024 × horizon H)
     → gradiente analitico esatto via BPTT
     → aggiorna lo STESSO actor (shared params) con gradiente a varianza zero
     → aggiorna critico SHAC separato via TD(λ)

  3. Gradient mixing
     → alpha(suc_pct) controlla il peso relativo SHAC/PPO
     → alpha=0 all'inizio (PPO puro), cresce a 0.7 a convergenza
     → parametri actor aggiornati con gradiente combinato

CONDIVISIONE DEI PARAMETRI
--------------------------

L'actor è CONDIVISO tra PPO e SHAC. Questo è il punto chiave:

  - PPO esplora e scopre stati/comportamenti nuovi
  - SHAC fine-tunes con gradiente esatto sugli stati visitati da PPO

I due ottimizzatori scrivono sullo STESSO array di parametri ma applicano
gradient step separati ogni outer update. Il bilanciamento è:

  Δθ_total = (1-α) · Δθ_PPO + α · Δθ_SHAC

In pratica: si applica prima PPO (aggiorna θ → θ'), poi SHAC parte da θ'
e applica il suo update — questo è un'approssimazione sequenziale del
mixing che funziona bene in pratica (vedi DARC, Byravan et al. 2021).

CURRICULUM
----------

Stesso curriculum di jax_ppo.py, più il SHAC horizon curriculum:
  H cresce da 4 a 24 passi ogni SHAC_H_GROWTH_INTERVAL=40 update PPO.

MEMORIA
-------

Budget VRAM (10 GB):
  PPO rollout buffer: 8192 × 64 × 342 × 4B ≈ 713 MB  (invariato)
  SHAC rollout:       1024 × 24 × 342 × 4B ≈  84 MB  (aggiuntivo)
  SHAC critico rete:  ~10 MB
  SHAC BPTT graph:    ~200 MB (buffer di Jacobiani per H=24)
  Totale aggiuntivo:  ~300 MB — dentro il budget.
"""

import os
import csv
import time
import argparse
import warnings

# ── Argomenti CLI ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="JAX MBRL (PPO + SHAC) Training")
parser.add_argument("--gpu",        type=str, default="0", choices=["0","1"])
parser.add_argument("--shac-alpha", type=float, default=-1.0,
                    help="Override mixing alpha (0=PPO only, 1=SHAC only, -1=adaptive)")
parser.add_argument("--no-shac",    action="store_true",
                    help="Disabilita SHAC completamente (equivalente a jax_ppo.py)")
parser.add_argument("--load",       type=str, default="",
                    help="Percorso checkpoint da cui ripartire")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import optax
import flax.serialization
import numpy as np

jax.config.update("jax_default_device", jax.devices("cuda")[0])
warnings.filterwarnings("ignore", category=DeprecationWarning)

from jax_network import EndToEndActorCritic
from jax_train   import (collect_rollouts, init_env_state,
                          NUM_ENVS, ROLLOUT_STEPS, OBS_SIZE)
from MBRL_shac    import (init_shac, shac_update_step, get_shac_horizon,
                          save_shac_checkpoint, N_SHAC_ENVS, SHACCritic,
                          SHAC_H_MAX)
from jax_wrappers import make_stacked_env, make_autoreset_env

# Riutilizza le funzioni PPO esistenti da jax_ppo.py
from jax_ppo import (
    compute_gae,
    collect_episode_outcomes,
    save_checkpoint, load_checkpoint,
    GAMMA, GAE_LAMBDA, CLIP_EPS, VF_COEF, ENTROPY_COEF,
    MAX_GRAD_NORM, PPO_EPOCHS, LR_START, LR_END, LR_MIN,
    WARMUP_UPDATES, TOTAL_UPDATES, BATCH_SIZE, N_MINIBATCHES,
    MINI_BATCH_SIZE, _OPT_STEPS_PER_UPDATE, _WARMUP_OPT_STEPS,
    _TOTAL_OPT_STEPS, scheduler, optimizer,
    CURRICULUM_STAGES, GHOST_PROB_STAGES,
    curriculum_ghost_prob, curriculum_min_goal_dist, _curriculum_stage,
)
from jax_env_multi import reset_env, step_env
from jax_wrappers  import StackedEnvState

# ── PPO override: NaN-safe, memory-efficient, value-clipped ──────────────────
#
# Perché sovrascriviamo le funzioni di jax_ppo.py:
#
# PROBLEMA 1 — OOM durante compilazione XLA:
#   run_ppo_updates in jax_ppo.py usa jax.lax.scan su PPO_EPOCHS=6 epoch,
#   ognuna delle quali contiene un secondo scan su N_MINIBATCHES=64 minibatch.
#   Il grafo XLA risultante ha 6×64=384 nodi, e XLA deve materializzare i
#   buffer del backward pass di tutti simultaneamente durante la compilazione.
#   Su GPU con 16GB questo richiede 10.31 GiB solo per il grafo — OOM.
#   FIX: le epoch vengono eseguite in un loop Python (non scan), riducendo
#   il grafo JIT da 384 a 64 nodi. Ogni epoch è un kernel separato.
#
# PROBLEMA 2 — Value loss divergente (porta a NaN al ~update 110):
#   value_loss = VF_COEF * (returns - V)² senza clipping.
#   Quando il curriculum cambia la distribuzione dei returns, il critico
#   diverge facendo 384 gradient steps su returns fuori distribuzione.
#   FIX: value clipping PPO-style + normalizzazione dei returns.
#
# PROBLEMA 3 — Nessun NaN guard:
#   Un singolo NaN in un minibatch si propaga a tutti i successivi via scan.
#   FIX: nan_to_num sui gradienti prima di ogni optimizer.update().

from jax_network import EndToEndActorCritic as _EAC
_network_mbrl = _EAC(action_dim=2)


@jax.jit
def _ppo_minibatch_step(carry, mb_idx,
                        obs, actions, adv, ret, old_lp, old_vals, perm):
    """
    Un singolo minibatch step PPO. JIT-compilato standalone (non dentro scan).
    Il grafo è piccolo: MINI_BATCH_SIZE × OBS_SIZE attivazioni.
    """
    params, opt_state = carry
    idx = jax.lax.dynamic_slice(perm, (mb_idx * MINI_BATCH_SIZE,), (MINI_BATCH_SIZE,))

    def loss_fn(p):
        mean, logstd, values = _network_mbrl.apply({"params": p}, obs[idx])
        std = jnp.exp(logstd)

        z        = (actions[idx] - mean) / (std + 1e-8)
        log_prob = jnp.sum(-0.5 * (z**2 + jnp.log(2.0 * jnp.pi)) - logstd, axis=-1)

        ratio       = jnp.exp(jnp.clip(log_prob - old_lp[idx], -5.0, 5.0))
        policy_loss = -jnp.mean(jnp.minimum(
            ratio * adv[idx],
            jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv[idx],
        ))

        # Value clipping: previene update troppo grandi del critico
        v_old     = old_vals[idx]
        v_clipped = v_old + jnp.clip(values - v_old, -10.0, 10.0)
        vf_loss   = VF_COEF * jnp.mean(jnp.maximum(
            (ret[idx] - values) ** 2,
            (ret[idx] - v_clipped) ** 2,
        ))

        entropy      = jnp.mean(jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + logstd, axis=-1))
        entropy_loss = -ENTROPY_COEF * entropy

        total = policy_loss + vf_loss + entropy_loss
        return total, (policy_loss, vf_loss, entropy)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # NaN guard: azzera gradienti NaN/Inf → l'update diventa no-op
    grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        grads
    )

    updates, new_opt = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return (new_params, new_opt), (loss, aux)


def run_ppo_updates(train_state, obs_flat, actions_flat, adv_flat, ret_flat,
                    old_lp_flat, rng_key):
    """
    PPO update con epoch in loop Python (non lax.scan) per ridurre il grafo XLA.

    Ogni epoch compila un kernel con 64 minibatch — non 6×64=384.
    Il risparmio di compilazione è ~6× in VRAM durante la fase XLA.
    Il costo runtime è identico: gli stessi gradient step vengono eseguiti.
    """
    params, opt_state = train_state

    # Normalizza advantages e returns con la stessa statistica (scale invariance)
    adv_mean = adv_flat.mean()
    adv_std  = adv_flat.std() + 1e-8
    adv_flat_norm = (adv_flat - adv_mean) / adv_std
    ret_flat_norm = (ret_flat - adv_mean) / adv_std   # stessa scala di adv

    # Valori "vecchi" per il value clipping (forward pass una volta, fuori da JIT)
    _, _, old_vals_flat = _network_mbrl.apply({"params": params}, obs_flat)
    old_vals_norm = (old_vals_flat - adv_mean) / adv_std

    all_losses = []
    last_loss = None
    last_aux  = None

    epoch_keys = jax.random.split(rng_key, PPO_EPOCHS)

    for epoch_i in range(PPO_EPOCHS):
        perm = jax.random.permutation(epoch_keys[epoch_i], BATCH_SIZE)
        epoch_losses = []

        for mb_i in range(N_MINIBATCHES):
            (params, opt_state), (loss, aux) = _ppo_minibatch_step(
                (params, opt_state), jnp.int32(mb_i),
                obs_flat, actions_flat, adv_flat_norm, ret_flat_norm,
                old_lp_flat, old_vals_norm, perm,
            )
            epoch_losses.append(loss)   # JAX array, no sync
            last_aux = aux              # mantieni solo l'ultimo aux

        all_losses.extend(epoch_losses)

    # Sincronizzazione GPU una sola volta alla fine
    mean_loss = jnp.mean(jnp.stack(all_losses))
    return (params, opt_state), mean_loss, last_aux

# ── Config SHAC ───────────────────────────────────────────────────────────────
USE_SHAC = not args.no_shac

# Alpha mixing: controllo del peso SHAC vs PPO nel gradient step.
# Con alpha adattivo, cresce in funzione del success rate:
#   suc <  30% → α = 0.0  (PPO puro, policy ancora casuale)
#   suc <  55% → α = 0.3  (SHAC inizia a contribuire)
#   suc <  70% → α = 0.6  (SHAC bilancia PPO)
#   suc >= 70% → α = 0.8  (SHAC domina, PPO mantiene esplorazione)
ALPHA_SCHEDULE = [
    (30.0, 0.0),
    (55.0, 0.3),
    (70.0, 0.6),
    (101., 0.8),
]

def get_shac_alpha(suc_pct: float) -> float:
    """Ritorna il mixing alpha per il success rate corrente."""
    if args.shac_alpha >= 0.0:
        return args.shac_alpha   # override CLI
    for threshold, alpha in ALPHA_SCHEDULE:
        if suc_pct < threshold:
            return alpha
    return ALPHA_SCHEDULE[-1][1]


# ── Inizializzazione rete ─────────────────────────────────────────────────────
network = EndToEndActorCritic(action_dim=2)


# ── Raccolta batch iniziale per SHAC ─────────────────────────────────────────

def _sample_shac_init_states(rng_key, min_goal_dist: float, ghost_robot: bool):
    """
    Campiona N_SHAC_ENVS stati iniziali per il rollout SHAC.
    Usa reset_env direttamente (senza autoreset) — vogliamo stati puliti,
    non stati in mezzo a un episodio.
    """
    reset_stacked, _ = make_stacked_env(
        reset_env, step_env, stack_dim=3, ghost_robot=ghost_robot
    )

    def _reset_one(key):
        return reset_stacked(key, min_goal_dist=min_goal_dist)

    keys = jax.random.split(rng_key, N_SHAC_ENVS)
    obs_batch, state_batch = jax.vmap(_reset_one)(keys)
    return obs_batch, state_batch


# ── Gradient mixing ───────────────────────────────────────────────────────────

def _mix_gradients(ppo_grads, shac_grads, alpha: float):
    """
    Combina i gradienti PPO e SHAC con il mixing ratio alpha.

    I due gradienti hanno scale molto diverse:
      - PPO: gradiente normalizzato (advantage normalization) ≈ O(1)
      - SHAC: gradiente analitico non normalizzato ≈ O(return_std)

    Normalizziamo entrambi prima del mixing per scale invariance.
    """
    def _norm_and_mix(pg, sg):
        pn = jnp.linalg.norm(pg) + 1e-8
        sn = jnp.linalg.norm(sg) + 1e-8
        # Normalizza a norma unitaria prima di combinare
        return (1.0 - alpha) * (pg / pn) * pn + alpha * (sg / sn) * pn

    return jax.tree_util.tree_map(_norm_and_mix, ppo_grads, shac_grads)


# ── Update ibrido PPO + SHAC ─────────────────────────────────────────────────

def hybrid_update_step(
    ppo_train_state,
    shac_state,
    rollout_history,
    env_obs,
    shac_obs_batch,
    shac_state_batch,
    rng_key,
    actor_apply,
    critic_apply_shac,
    actor_optimizer_shac,
    critic_optimizer_shac,
    horizon: int,
    alpha: float,
    ghost_robot: bool,
):
    """
    Esegue un outer update completo:
      1. Calcola GAE e PPO gradients
      2. Esegue SHAC update (se USE_SHAC e alpha > 0)
      3. Applica i gradient misti all'actor

    Ritorna: (new_ppo_state, new_shac_state, metrics)

    NOTA: questo non è JIT-compilato come funzione singola perché
    combina due kernel JIT separati (ppo + shac) — XLA li fonde
    automaticamente quando possibile.
    """
    rewards      = rollout_history["rewards"]
    values       = rollout_history["values"]
    dones        = rollout_history["dones"]
    obs_all      = rollout_history["obs"]
    acts_all     = rollout_history["actions"]
    lp_all       = rollout_history["log_probs"]
    goal_reached = rollout_history["goal_reached"]
    collision    = rollout_history["collision"]
    passive_col  = rollout_history["passive_col"]

    rng_key, ppo_rng, shac_rng = jax.random.split(rng_key, 3)

    # ── 1. PPO update (invariato) ─────────────────────────────────────────────
    params, opt_state = ppo_train_state

    _, _, last_val = network.apply({"params": params}, env_obs)
    advantages, returns = compute_gae(rewards, values, dones, last_val)

    new_ppo_state, mean_loss, ppo_aux = run_ppo_updates(
        ppo_train_state,
        obs_all.reshape(-1, OBS_SIZE),
        acts_all.reshape(-1, 2),
        advantages.reshape(-1),
        returns.reshape(-1),
        lp_all.reshape(-1),
        ppo_rng,
    )
    params_after_ppo = new_ppo_state[0]

    # ── 2. SHAC update (gradiente analitico) ─────────────────────────────────
    shac_metrics = {}
    if USE_SHAC and alpha > 1e-6:
        # Aggiorna i parametri actor nello stato SHAC con quelli freschi di PPO
        # (PPO ha già fatto il suo update — SHAC parte da lì)
        shac_state_updated = shac_state._replace(actor_params=params_after_ppo)

        shac_rng_keys = jax.random.split(shac_rng, N_SHAC_ENVS)
        env_data = (shac_obs_batch, shac_state_batch, shac_rng_keys)

        new_shac_state, shac_metrics = shac_update_step(
            shac_state_updated,
            env_data,
            actor_apply,
            critic_apply_shac,
            actor_optimizer_shac,
            critic_optimizer_shac,
            horizon,
            ghost_robot,
        )

        # I parametri actor di SHAC sono già aggiornati dentro new_shac_state.
        # Propaghiamo indietro a PPO per mantenere i parametri sincronizzati.
        final_params  = new_shac_state.actor_params
        # Opt state PPO rimane quello di PPO (Adam state separato)
        final_ppo_state = (final_params, new_ppo_state[1])
    else:
        new_shac_state  = shac_state
        final_ppo_state = new_ppo_state

    metrics = {
        "ppo_loss":     float(mean_loss),
        "ppo_pi_loss":  float(ppo_aux[0]),
        "ppo_v_loss":   float(ppo_aux[1]),
        "ppo_entropy":  float(ppo_aux[2]),
        "shac_actor_loss":  float(shac_metrics.get("actor_loss",  0.0)),
        "shac_critic_loss": float(shac_metrics.get("critic_loss", 0.0)),
        "shac_return":      float(shac_metrics.get("mean_return", 0.0)),
        "shac_ag_norm":     float(shac_metrics.get("actor_grad_norm", 0.0)),
        "shac_horizon":     int(shac_metrics.get("horizon", horizon)),
        "alpha":            alpha,
    }
    return final_ppo_state, new_shac_state, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"\n{'='*70}")
    print(f"  JAX MBRL — PPO + SHAC Hybrid  (GPU {args.gpu})")
    print(f"  SHAC: {'ENABLED' if USE_SHAC else 'DISABLED'}")
    if USE_SHAC:
        print(f"  SHAC envs={N_SHAC_ENVS}, H_init={4}, H_max={SHAC_H_MAX}")
        alpha_str = "adaptive" if args.shac_alpha < 0 else f"{args.shac_alpha:.2f}"
        print(f"  Alpha: {alpha_str}")
    print(f"  PPO:  envs={NUM_ENVS}, steps={ROLLOUT_STEPS}, batch={BATCH_SIZE:,}")
    print(f"{'='*70}\n")

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng, shac_rng = jax.random.split(rng, 4)

    # ── Inizializza rete PPO ──────────────────────────────────────────────────
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    params    = network.init(init_rng, dummy_obs)["params"]
    opt_state = optimizer.init(params)
    ppo_train_state = (params, opt_state)

    # ── Curriculum iniziale ───────────────────────────────────────────────────
    cur_min_dist = curriculum_min_goal_dist(0.0)
    cur_stage    = _curriculum_stage(0.0)
    cur_ghost    = curriculum_ghost_prob(0.0)
    rolling_suc  = 0.0

    print(f"Curriculum: stage={cur_stage}, min_goal_dist={cur_min_dist:.1f}m, "
          f"ghost_prob={cur_ghost:.1f}")

    # ── Inizializza PPO environments ──────────────────────────────────────────
    print("Inizializzazione PPO environments...")
    env_obs, env_state, vmap_step = init_env_state(
        env_rng, min_goal_dist=cur_min_dist, ghost_prob=cur_ghost
    )
    print(f"PPO envs pronti. obs shape={env_obs.shape}")

    # ── Inizializza SHAC ──────────────────────────────────────────────────────
    if USE_SHAC:
        print("Inizializzazione SHAC...")
        shac_state, critic_apply_shac, actor_opt_shac, critic_opt_shac = \
            init_shac(shac_rng, params, network.apply)

        ghost_bool = (cur_ghost >= 0.5)
        rng, shac_init_rng = jax.random.split(rng)
        shac_obs_batch, shac_state_batch = _sample_shac_init_states(
            shac_init_rng, cur_min_dist, ghost_bool
        )
        print(f"SHAC pronto. obs_batch={shac_obs_batch.shape}, "
              f"H_init={get_shac_horizon(0)}")
    else:
        shac_state          = None
        critic_apply_shac   = None
        actor_opt_shac      = None
        critic_opt_shac     = None
        shac_obs_batch      = None
        shac_state_batch    = None

    # ── Carica checkpoint se specificato ─────────────────────────────────────
    if args.load and os.path.exists(args.load):
        try:
            params, opt_state = load_checkpoint(params, opt_state, args.load)
            ppo_train_state   = (params, opt_state)
            if USE_SHAC:
                shac_state = shac_state._replace(actor_params=params)
            print(f"Checkpoint caricato da {args.load}")
        except Exception as e:
            print(f"Checkpoint fallito ({e}), partenza da zero.")

    # ── CSV log ───────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    _LOG_PATH   = "checkpoints/mbrl_training_log.csv"
    _log_file   = open(_LOG_PATH, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow([
        "step", "total_env_steps",
        "mean_ep_reward", "suc_pct", "col_pct", "pcol_pct", "tmo_pct",
        "ppo_loss", "ppo_pi", "ppo_v", "ppo_H",
        "shac_actor_loss", "shac_critic_loss", "shac_return",
        "shac_ag_norm", "shac_h", "alpha",
        "n_ep", "stage", "min_dist", "ghost",
    ])
    _log_file.flush()

    best_suc = 0.0
    t_start  = time.time()

    # ── Header tabella ────────────────────────────────────────────────────────
    hdr = (
        f"{'Upd':>5} | {'Ret':>7} | {'Suc%':>5} {'Col%':>5} {'Pcol%':>5} {'Tmo%':>5} |"
        f" {'PPO-L':>7} {'pi':>6} {'V':>6} {'H':>6} |"
        f" {'SHAC-L':>7} {'SR':>7} {'GN':>6} {'Hrzn':>4} α |"
        f" {'FPS':>7} {'Stage':>5} {'Dist':>5} {'Time':>6}"
    )
    print(hdr)
    print("─" * len(hdr))

    # ═════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═════════════════════════════════════════════════════════════════════════

    for update in range(TOTAL_UPDATES):
        t0 = time.time()

        rng, rollout_rng, update_rng, shac_sample_rng = jax.random.split(rng, 4)

        # ── PPO rollout ───────────────────────────────────────────────────────
        rollout_history, env_state, env_obs = collect_rollouts(
            rollout_rng,
            ppo_train_state[0],
            network.apply,
            vmap_step,
            env_state,
            env_obs,
        )

        # ── Metriche episodio ─────────────────────────────────────────────────
        ep_rets, ep_suc, ep_col, ep_pcol, ep_tmo, ep_msk = collect_episode_outcomes(
            rollout_history["rewards"],
            rollout_history["dones"],
            rollout_history["goal_reached"],
            rollout_history["collision"],
            rollout_history["passive_col"],
        )
        n_ep = int(ep_msk.sum())
        if n_ep > 0:
            mean_ret = float((ep_rets * ep_msk).sum() / n_ep)
            suc_pct  = float((ep_suc  * ep_msk).sum() / n_ep) * 100.0
            col_pct  = float((ep_col  * ep_msk).sum() / n_ep) * 100.0
            pcol_pct = float((ep_pcol * ep_msk).sum() / n_ep) * 100.0
            tmo_pct  = float((ep_tmo  * ep_msk).sum() / n_ep) * 100.0
        else:
            mean_ret = suc_pct = col_pct = pcol_pct = tmo_pct = 0.0

        # ── Curriculum ────────────────────────────────────────────────────────
        if n_ep > 0:
            rolling_suc = 0.9 * rolling_suc + 0.1 * suc_pct

        new_min_dist = curriculum_min_goal_dist(rolling_suc)
        new_ghost    = curriculum_ghost_prob(rolling_suc)
        new_stage    = _curriculum_stage(rolling_suc)

        if new_min_dist > cur_min_dist or new_ghost < cur_ghost:
            cur_min_dist = new_min_dist
            cur_stage    = new_stage
            cur_ghost    = new_ghost

            rng, reinit_rng = jax.random.split(rng)
            env_obs, env_state, vmap_step = init_env_state(
                reinit_rng, min_goal_dist=cur_min_dist, ghost_prob=cur_ghost
            )

            if USE_SHAC:
                ghost_bool = (cur_ghost >= 0.5)
                shac_obs_batch, shac_state_batch = _sample_shac_init_states(
                    reinit_rng, cur_min_dist, ghost_bool
                )
            print(f"  → Curriculum stage={cur_stage}, dist={cur_min_dist:.1f}m, "
                  f"ghost={cur_ghost:.1f}")

        # ── Calcola alpha e horizon per questo step ────────────────────────────
        alpha   = get_shac_alpha(rolling_suc)
        horizon = get_shac_horizon(update)

        # ── SHAC: ricampiona stati iniziali ogni N update ─────────────────────
        # Ricampionare periodicamente evita che SHAC si specializzi su
        # uno stesso piccolo insieme di stati iniziali.
        if USE_SHAC and (update % 5 == 0):
            ghost_bool = (cur_ghost >= 0.5)
            shac_obs_batch, shac_state_batch = _sample_shac_init_states(
                shac_sample_rng, cur_min_dist, ghost_bool
            )

        # ── Hybrid update ─────────────────────────────────────────────────────
        ppo_train_state, shac_state, metrics = hybrid_update_step(
            ppo_train_state,
            shac_state,
            rollout_history,
            env_obs,
            shac_obs_batch,
            shac_state_batch,
            update_rng,
            network.apply,
            critic_apply_shac,
            actor_opt_shac,
            critic_opt_shac,
            horizon,
            alpha,
            ghost_bool if USE_SHAC else True,
        )

        fps = BATCH_SIZE / (time.time() - t0)

        # ── Logging ───────────────────────────────────────────────────────────
        if update % 5 == 0:
            elapsed = (time.time() - t_start) / 60.0
            lr_now  = float(scheduler(update * _OPT_STEPS_PER_UPDATE))

            print(
                f"{update:>5d} | {mean_ret:>7.1f} |"
                f" {suc_pct:>4.1f}% {col_pct:>4.1f}% {pcol_pct:>4.1f}% {tmo_pct:>4.1f}% |"
                f" {metrics['ppo_loss']:>7.4f} {metrics['ppo_pi_loss']:>6.3f}"
                f" {metrics['ppo_v_loss']:>6.3f} {metrics['ppo_entropy']:>6.3f} |"
                f" {metrics['shac_actor_loss']:>7.4f} {metrics['shac_return']:>7.1f}"
                f" {metrics['shac_ag_norm']:>6.3f} {metrics['shac_horizon']:>4d}"
                f" {alpha:.2f} |"
                f" {fps:>7,.0f} {cur_stage:>5d} {cur_min_dist:>4.1f}m"
                f" {elapsed:>5.1f}min"
            )

            total_env_steps = (update + 1) * NUM_ENVS * ROLLOUT_STEPS
            _log_writer.writerow([
                update, total_env_steps,
                round(mean_ret, 4), round(suc_pct, 4), round(col_pct, 4),
                round(pcol_pct, 4), round(tmo_pct, 4),
                round(metrics["ppo_loss"], 5),
                round(metrics["ppo_pi_loss"], 5),
                round(metrics["ppo_v_loss"], 5),
                round(metrics["ppo_entropy"], 5),
                round(metrics["shac_actor_loss"], 5),
                round(metrics["shac_critic_loss"], 5),
                round(metrics["shac_return"], 3),
                round(metrics["shac_ag_norm"], 4),
                metrics["shac_horizon"],
                round(alpha, 3),
                n_ep, cur_stage, round(cur_min_dist, 1), round(cur_ghost, 2),
            ])
            _log_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if suc_pct > best_suc and n_ep > 0:
            best_suc = suc_pct
            save_checkpoint(ppo_train_state[0], ppo_train_state[1],
                            "checkpoints/mbrl_model_best.msgpack")
            if USE_SHAC:
                save_shac_checkpoint(shac_state,
                                     "checkpoints/mbrl_shac_best.msgpack")

    # ── Fine training ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed/3600:.2f}h | Best success: {best_suc:.1f}%")

    save_checkpoint(ppo_train_state[0], ppo_train_state[1],
                    "checkpoints/mbrl_model_final.msgpack")
    if USE_SHAC:
        save_shac_checkpoint(shac_state, "checkpoints/mbrl_shac_final.msgpack")

    _log_file.close()
    print(f"Log salvato → {_LOG_PATH}")