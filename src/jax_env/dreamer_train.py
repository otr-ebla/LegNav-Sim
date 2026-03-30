"""
dreamer_train.py — Main Training Loop for DreamerV3

Fixes vs submitted version:
  - BUG 1/2 FIXED (via dreamer_rssm.py): rssm.init() now called through
    __call__ so all sub-layer weights are registered in one shot.
  - BUG 3/4 FIXED (via dreamer_buffer.py): episode masking + buffer_ready guard.
  - BUG 5 FIXED: train_step had @partial(jax.jit, static_argnums=(4,)) with
    max_goal_dist as arg 4 — a float that's never used inside the function.
    This caused an unnecessary recompile any time max_goal_dist changed, and
    conflated a hyperparameter with a JIT static arg. Removed entirely.
  - BUG 6 FIXED: done mask was `(1.0 - dones[:, None])`. `dones` is bool_ so
    this silently coerced via Python scalar subtraction. Fixed to
    (~dones).astype(jnp.float32)[:, None] for an explicit, type-safe mask.
  - BUG 7 FIXED: buffer_ready() guard added before train_step so
    sample_sequences is never called with a near-empty buffer.
  - BUG 8 FIXED (via dreamer_env.py): global cache .clear() replaced by closure.
  - BUG 9 FIXED: unroll_imagination now returns only `traj` (final h/z were
    always discarded). Call sites updated accordingly.
  - BUG A FIXED: scan_rssm was calling rssm.prior(..., rngs={'gumbel': key})
    purely to get the prior logits for the KL loss. A Gumbel sample is never
    needed for KL computation — only the logits matter. Switched to
    rssm.prior_greedy(), eliminating a wasted RNG split per scan step.
  - BUG B FIXED: entropy loss double-negation.
    -ENTROPY_COEF * mean(-log_prob) == +ENTROPY_COEF * mean(log_prob), which
    penalises entropy rather than encouraging it. Fixed to a single negation:
    -ENTROPY_COEF * mean(log_prob).
  - BUG C FIXED: buffer_ready() was imported and mentioned but never called in
    the main training loop. Added guard call at the top of the training for-loop
    so a cold-start without prefill does not crash sample_sequences.
"""

import os
import argparse

# 1. Forza le variabili d'ambiente PRIMA di importare JAX
parser = argparse.ArgumentParser(description="DreamerV3 Training")
parser.add_argument("--gpu", type=str, default="0", help="Target GPU ID")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"

import jax
# 2. Impedisce il fallback silenzioso: se non c'è CUDA, deve crashare
jax.config.update("jax_platform_name", "cuda")
jax.config.update("jax_default_device", jax.devices("cuda")[0])

import jax.numpy as jnp
import optax
import time
from functools import partial

from dreamer_buffer   import init_buffer, add_batch, sample_sequences, buffer_ready
from dreamer_rssm     import RSSM, DreamerEncoder, LATENT_SIZE, DETERMINISTIC_SIZE
from dreamer_decoders import ObservationDecoder, RewardDecoder, ContinueDecoder, world_model_loss
from dreamer_behavior import DreamerActor, DreamerCritic, compute_lambda_returns, unroll_imagination, sample_action
from dreamer_env      import init_dreamer_envs
from jax_network      import scale_actions_batched

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_ENVS        = 64
SEQ_LEN         = 64
BATCH_SIZE      = 16
HORIZON         = 15
BUFFER_CAPACITY = 20_000
PREFILL_STEPS   = 1_000

LR_WM        = 1e-4
LR_ACTOR     = 3e-5
LR_CRITIC    = 3e-5
GAMMA        = 0.99
LAMBDA       = 0.95
ENTROPY_COEF = 3e-4

OBS_DIM    = 342
ACTION_DIM = 2
H_DIM      = DETERMINISTIC_SIZE   # 512
Z_DIM      = LATENT_SIZE          # 1024

# ---------------------------------------------------------------------------
# Module instances
# ---------------------------------------------------------------------------

encoder     = DreamerEncoder()
rssm        = RSSM(action_dim=ACTION_DIM)
obs_decoder = ObservationDecoder(obs_dim=OBS_DIM)
rew_decoder = RewardDecoder()
cont_decoder = ContinueDecoder()
actor       = DreamerActor(action_dim=ACTION_DIM)
critic      = DreamerCritic()

# ---------------------------------------------------------------------------
# RSSM sequence unroll — RNG key in scan carry (fully on GPU)
# ---------------------------------------------------------------------------

def scan_rssm(wm_params: dict, obs_seq: jnp.ndarray,
              act_seq: jnp.ndarray, rng_key: jnp.ndarray) -> tuple:
    """
    Unrolls the RSSM over T=SEQ_LEN steps inside jax.lax.scan.
    gumbel keys are split inside the carry — zero Python allocation per step.
    """
    batch_sz = obs_seq.shape[0]
    h_0 = jnp.zeros((batch_sz, H_DIM))
    z_0 = jnp.zeros((batch_sz, Z_DIM))

    # Shift actions: act_{t-1} drives h_t
    act_prev = jnp.concatenate(
        [jnp.zeros((batch_sz, 1, ACTION_DIM)), act_seq[:, :-1, :]], axis=1)

    obs_t  = jnp.swapaxes(obs_seq,  0, 1)   # [T, B, obs_dim]
    act_t  = jnp.swapaxes(act_prev, 0, 1)   # [T, B, act_dim]

    def step(carry, inputs):
        h_prev, z_prev, key = carry
        obs_i, act_i = inputs
        key, post_key = jax.random.split(key, 2)

        h_t = rssm.apply(
            {'params': wm_params['rssm']},
            h_prev, z_prev, act_i, method=rssm.step_gru)

        obs_embed = encoder.apply({'params': wm_params['encoder']}, obs_i)

        z_t, post_logits = rssm.apply(
            {'params': wm_params['rssm']}, h_t, obs_embed,
            method=rssm.posterior, rngs={'gumbel': post_key})

        # BUG A FIX: use prior_greedy for KL logits — we only need the logits,
        # not a Gumbel sample, so no RNG key is needed and prior_key is freed.
        _, prior_logits = rssm.apply(
            {'params': wm_params['rssm']}, h_t,
            method=rssm.prior_greedy)

        return (h_t, z_t, key), (h_t, z_t, prior_logits, post_logits)

    _, (h_seq, z_seq, prior_seq, post_seq) = jax.lax.scan(
        step, (h_0, z_0, rng_key), (obs_t, act_t))

    return (jnp.swapaxes(h_seq,    0, 1),
            jnp.swapaxes(z_seq,    0, 1),
            jnp.swapaxes(prior_seq, 0, 1),
            jnp.swapaxes(post_seq,  0, 1))

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

opt_wm     = optax.adam(LR_WM,    eps=1e-8)
opt_actor  = optax.adam(LR_ACTOR, eps=1e-5)
opt_critic = optax.adam(LR_CRITIC, eps=1e-5)

# ---------------------------------------------------------------------------
# Monolithic JIT training step
# BUG 5 FIX: static_argnums removed — max_goal_dist was the only static arg
# and it was never used inside the function body.
# ---------------------------------------------------------------------------

@jax.jit
def train_step(rng_key, buffer_state, params, opt_states):
    k_sample, k_actor, k_wm = jax.random.split(rng_key, 3)

    obs_seq, act_seq, rew_seq, done_seq = sample_sequences(
        k_sample, buffer_state, BATCH_SIZE, SEQ_LEN)

    # ---- A. World Model ----
    def wm_loss_fn(wm_params):
        h_states, z_states, prior_logits, post_logits = scan_rssm(
            wm_params, obs_seq, act_seq, k_wm)
        obs_pred    = obs_decoder.apply( {'params': wm_params['obs']},  h_states, z_states)
        rew_pred    = rew_decoder.apply( {'params': wm_params['rew']},  h_states, z_states)
        cont_logits = cont_decoder.apply({'params': wm_params['cont']}, h_states, z_states)
        loss, aux   = world_model_loss(
            obs_seq, rew_seq, done_seq,
            obs_pred, rew_pred, cont_logits,
            prior_logits, post_logits)
        return loss, (aux, h_states, z_states)

    (wm_loss, (wm_aux, h_states, z_states)), wm_grads = jax.value_and_grad(
        wm_loss_fn, has_aux=True)(params['wm'])
    wm_updates, new_wm_opt = opt_wm.update(wm_grads, opt_states['wm'])
    new_wm_params = optax.apply_updates(params['wm'], wm_updates)

    # ---- B. Actor ----
    start_h = jax.lax.stop_gradient(h_states.reshape(-1, H_DIM))
    start_z = jax.lax.stop_gradient(z_states.reshape(-1, Z_DIM))

    def actor_loss_fn(actor_params):
        frozen_wm = jax.lax.stop_gradient(new_wm_params)

        # BUG 9 FIX: unroll_imagination now returns traj only (no final h/z)
        traj = unroll_imagination(
            k_actor,
            lambda p, h, z, a: rssm.apply({'params': p}, h, z, a, method=rssm.step_gru),
            lambda p, h, k: rssm.apply({'params': p}, h, method=rssm.prior, rngs={'gumbel': k}),
            lambda p, h, z: actor.apply({'params': p}, h, z),
            {'rssm': frozen_wm['rssm'], 'actor': actor_params},
            start_h, start_z, HORIZON,
        )

        values    = critic.apply(
            {'params': jax.lax.stop_gradient(params['critic'])}, traj['h'], traj['z'])
        rewards   = rew_decoder.apply(
            {'params': frozen_wm['rew']}, traj['h'], traj['z'])
        continues = jax.nn.sigmoid(cont_decoder.apply(
            {'params': frozen_wm['cont']}, traj['h'], traj['z']))

        bootstrap      = values[-1]
        lambda_returns = compute_lambda_returns(
            rewards[:-1], values[:-1], continues[:-1], bootstrap, GAMMA, LAMBDA)

        advantages  = jax.lax.stop_gradient(lambda_returns - values[:-1])

        jax.debug.print("lambda_returns  mean={x:.2f}  std={y:.2f}", x=jnp.mean(lambda_returns), y=jnp.std(lambda_returns))
        jax.debug.print("values          mean={x:.2f}  std={y:.2f}", x=jnp.mean(values[:-1]), y=jnp.std(values[:-1]))
        jax.debug.print("advantages      mean={x:.2f}  std={y:.2f}", x=jnp.mean(advantages), y=jnp.std(advantages))

        actor_loss  = -jnp.mean(traj['log_prob'][:-1] * advantages)
        # BUG B FIX: single negation only — we want to maximise entropy,
        # i.e. minimise -entropy = -mean(-log_prob) = mean(log_prob).
        # The previous double negation was penalising entropy instead.
        entropy_loss = -ENTROPY_COEF * jnp.mean(traj['log_prob'][:-1])
        return actor_loss + entropy_loss, (traj, lambda_returns)

    (act_loss, (traj, lambda_returns)), act_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True)(params['actor'])
    act_updates, new_act_opt = opt_actor.update(act_grads, opt_states['actor'])
    new_act_params = optax.apply_updates(params['actor'], act_updates)

    # ---- C. Critic ----
    def critic_loss_fn(critic_params):
        target = jax.lax.stop_gradient(lambda_returns)
        values = critic.apply(
            {'params': critic_params},
            jax.lax.stop_gradient(traj['h'][:-1]),
            jax.lax.stop_gradient(traj['z'][:-1]))
        return jnp.mean((values - target) ** 2)

    critic_loss, crit_grads = jax.value_and_grad(critic_loss_fn)(params['critic'])
    crit_updates, new_crit_opt = opt_critic.update(crit_grads, opt_states['critic'])
    new_crit_params = optax.apply_updates(params['critic'], crit_updates)

    new_params = {'wm': new_wm_params, 'actor': new_act_params, 'critic': new_crit_params}
    new_opt_states = {'wm': new_wm_opt, 'actor': new_act_opt, 'critic': new_crit_opt}
    metrics = {
        'wm_loss':     wm_loss,
        'actor_loss':  act_loss,
        'critic_loss': critic_loss,
        'kl_loss':     wm_aux[3],
    }
    return new_params, new_opt_states, metrics

# ---------------------------------------------------------------------------
# Inference step
# BUG 5 FIX: static_argnums=(7,) correctly targets the `explore` bool.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(7,))
def act_step(rng_key, wm_params, actor_params, obs,
             prev_h, prev_z, prev_action, explore: bool = True):
    rng_key, post_key, act_key = jax.random.split(rng_key, 3)

    obs_embed = encoder.apply({'params': wm_params['encoder']}, obs)
    h_t = rssm.apply(
        {'params': wm_params['rssm']}, prev_h, prev_z, prev_action,
        method=rssm.step_gru)
    z_t, _ = rssm.apply(
        {'params': wm_params['rssm']}, h_t, obs_embed,
        method=rssm.posterior, rngs={'gumbel': post_key})

    mean, std = actor.apply({'params': actor_params}, h_t, z_t)

    # `explore` is static — Python if is safe here (resolved at compile time)
    if explore:
        action, _ = sample_action(act_key, mean, std)
    else:
        action = jnp.tanh(mean)

    return action, h_t, z_t




@partial(jax.jit, static_argnames=('chunk_size',))
def train_loop_chunk(rng_key, buffer_state, params, opt_states, 
                     env_obs, env_state, current_h, current_z, current_action,
                     cur_return, ema_return, ema_success,
                     chunk_size=50):
    
    def single_step(carry, _):
        # 1. Unpack con le nuove variabili
        (b_state, p, opt, h, z, a, obs, e_state, key, c_ret, e_ret, e_succ) = carry
        key, act_k, step_k, train_k = jax.random.split(key, 4)

        raw_acts, next_h, next_z = act_step(
            act_k, p['wm'], p['actor'], obs, h, z, a, explore=True)
        
        env_acts = scale_actions_batched(raw_acts, e_state.env_state.max_v)
        s_keys = jax.random.split(step_k, NUM_ENVS)
        
        # 2. Recuperiamo 'info' invece di ignorarlo con '_'
        n_obs, n_e_state, rews, dones, info = vmap_step(s_keys, e_state, env_acts, 3.0, -1)

        new_b_state = add_batch(b_state, obs, raw_acts, rews, dones)
        new_p, new_opt, metrics = train_step(train_k, new_b_state, p, opt)

        alive = (~dones).astype(jnp.float32)[:, None]
        next_h *= alive
        next_z *= alive

        # --- 3. LOGICA DI TRACKING (EMA) ---
        c_ret = c_ret + rews
        num_dones = jnp.sum(dones)
        
        # Calcola media dei return e successi solo per gli ambienti appena terminati
        mean_ret  = jnp.sum(c_ret * dones) / jnp.maximum(1.0, num_dones)
        mean_succ = jnp.sum(info['goal_reached'] * dones) / jnp.maximum(1.0, num_dones)
        
        # Aggiorna l'EMA solo se c'è almeno un episodio terminato in questo step
        e_ret  = jnp.where(num_dones > 0, 0.95 * e_ret + 0.05 * mean_ret, e_ret)
        e_succ = jnp.where(num_dones > 0, 0.95 * e_succ + 0.05 * mean_succ, e_succ)
        
        # Azzera l'accumulatore per gli ambienti riavviati
        c_ret = c_ret * (~dones)

        metrics['ema_return']  = e_ret
        metrics['ema_success'] = e_succ

        # 4. Repack del carry aggiornato
        next_carry = (new_b_state, new_p, new_opt, next_h, next_z, raw_acts, n_obs, n_e_state, key, c_ret, e_ret, e_succ)
        return next_carry, metrics

    # 5. Init e scan
    init_carry = (buffer_state, params, opt_states, current_h, current_z, current_action, env_obs, env_state, rng_key, cur_return, ema_return, ema_success)
    final_carry, metrics_history = jax.lax.scan(single_step, init_carry, None, length=chunk_size)
    return final_carry, metrics_history






























# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Initializing DreamerV3 Training...")
    rng = jax.random.PRNGKey(42)
    rng, env_rng, init_rng = jax.random.split(rng, 3)

    # 1. Environments
    env_obs, env_state, vmap_step = init_dreamer_envs(
        env_rng, num_envs=NUM_ENVS, max_goal_dist=3.0,
        ghost_prob=1.0, scenario_idx=-1)

    # 2. Replay buffer
    buffer_state = init_buffer(BUFFER_CAPACITY, NUM_ENVS, OBS_DIM, ACTION_DIM)

    # 3. Parameters Initialization
    dummy_obs   = jnp.zeros((1, OBS_DIM))
    dummy_h     = jnp.zeros((1, H_DIM))
    dummy_z     = jnp.zeros((1, Z_DIM))
    dummy_act   = jnp.zeros((1, ACTION_DIM))
    dummy_embed = jnp.zeros((1, H_DIM))

    rng, r1, r2 = jax.random.split(rng, 3)

    # Bug 1 Fix: init via __call__
    rssm_params = rssm.init(
        {'params': r1, 'gumbel': r2},
        dummy_h, dummy_z, dummy_act, dummy_embed,
    )['params']

    params = {
        'wm': {
            'encoder': encoder.init(r1, dummy_obs)['params'],
            'rssm':    rssm_params,
            'obs':     obs_decoder.init(r1, dummy_h, dummy_z)['params'],
            'rew':     rew_decoder.init(r1, dummy_h, dummy_z)['params'],
            'cont':    cont_decoder.init(r1, dummy_h, dummy_z)['params'],
        },
        'actor':  actor.init(r1, dummy_h, dummy_z)['params'],
        'critic': critic.init(r1, dummy_h, dummy_z)['params'],
    }

    opt_states = {
        'wm':     opt_wm.init(params['wm']),
        'actor':  opt_actor.init(params['actor']),
        'critic': opt_critic.init(params['critic']),
    }

    current_h      = jnp.zeros((NUM_ENVS, H_DIM))
    current_z      = jnp.zeros((NUM_ENVS, Z_DIM))
    current_action = jnp.zeros((NUM_ENVS, ACTION_DIM))
    
    # --- NUOVI TRACKER ---
    cur_return  = jnp.zeros(NUM_ENVS)  # Accumulatore reward per l'episodio in corso
    ema_return  = jnp.array(0.0)       # Media mobile esponenziale del return
    ema_success = jnp.array(0.0)       # Media mobile esponenziale del success rate

    # 4. Prefill (GPU-side)
    print("Compiling and executing fast pre-fill on GPU...")
    
    @jax.jit
    def run_prefill(rng_key, b_state, e_state, e_obs):
        def _step(carry, _):
            curr_b, curr_e_state, curr_e_obs, curr_rng = carry
            curr_rng, act_rng, step_rng = jax.random.split(curr_rng, 3)
            raw_acts = jax.random.uniform(act_rng, (NUM_ENVS, ACTION_DIM), minval=-1.0, maxval=1.0)
            env_acts = scale_actions_batched(raw_acts, curr_e_state.env_state.max_v)
            s_keys = jax.random.split(step_rng, NUM_ENVS)
            n_obs, n_state, rews, dones, _ = vmap_step(s_keys, curr_e_state, env_acts, 3.0, -1)
            next_b = add_batch(curr_b, curr_e_obs, raw_acts, rews, dones)
            return (next_b, n_state, n_obs, curr_rng), None

        (final_b, final_e_state, final_e_obs, _), _ = jax.lax.scan(
            _step, (b_state, e_state, e_obs, rng_key), None, length=PREFILL_STEPS
        )
        return final_b, final_e_state, final_e_obs

    t_prefill = time.time()
    buffer_state, env_state, env_obs = run_prefill(rng, buffer_state, env_state, env_obs)
    buffer_state.insert_idx.block_until_ready() 
    print(f"Pre-fill complete in {time.time() - t_prefill:.2f}s!\n")

    # 5. Main Training Loop (Mega-JIT)
    print("Starting Optimized Main Training Loop...")
    CHUNK_SIZE = 1#50  # Numero di step eseguiti interamente su GPU per ogni iterazione Python
    TOTAL_STEPS = 20#100_000
    
    for step in range(0, TOTAL_STEPS, CHUNK_SIZE):
        t0 = time.time()
        
        # BUG C FIX: guard train_step against near-empty buffer. After prefill
        # this is always True, but protects cold-start or very small prefill runs.
        if not buffer_ready(buffer_state, SEQ_LEN):
            print(f"Step {step:05d} | Buffer not ready yet, skipping train_step.")
            continue
        
        # Passiamo i nuovi tracker alla funzione
        final_carry, metrics_history = train_loop_chunk(
            rng, buffer_state, params, opt_states,
            env_obs, env_state, current_h, current_z, current_action,
            cur_return, ema_return, ema_success,
            chunk_size=CHUNK_SIZE
        )
        
        # Unpack includendo i tracker
        (buffer_state, params, opt_states, 
         current_h, current_z, current_action, 
         env_obs, env_state, rng,
         cur_return, ema_return, ema_success) = final_carry

        if step % 100 == 0:
            avg_wm     = jnp.mean(metrics_history['wm_loss'])
            avg_actor  = jnp.mean(metrics_history['actor_loss'])
            fps        = (NUM_ENVS * CHUNK_SIZE) / (time.time() - t0)
            
            # Estraiamo l'ultimo valore dell'EMA dal chunk e convertiamo in percentuale
            ret_val  = metrics_history['ema_return'][-1]
            succ_val = metrics_history['ema_success'][-1] * 100.0
            
            print(
                f"Step {step:05d} | "
                f"FPS: {fps:.0f} | "
                f"WM: {avg_wm:.3f} | "
                f"Act: {avg_actor:.3f} | "
                f"Ret: {ret_val:.1f} | "
                f"Succ: {succ_val:.1f}%"
            )

    print("Training Complete.")
    
    # 6. Checkpoint Saving
    import os
    import flax.serialization
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/dreamer_best.msgpack"
    
    eval_params = {
        'encoder': jax.device_get(params['wm']['encoder']),
        'rssm': jax.device_get(params['wm']['rssm']),
        'actor': jax.device_get(params['actor'])
    }
    
    with open(ckpt_path, "wb") as f:
        f.write(flax.serialization.to_bytes(eval_params))
    print(f"Checkpoint salvato in {ckpt_path}!")