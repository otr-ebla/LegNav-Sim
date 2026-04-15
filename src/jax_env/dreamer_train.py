"""
dreamer_train.py — Main Training Loop for DreamerV3

"""

import os
import argparse

# Force env vars before importing JAX
parser = argparse.ArgumentParser(description="DreamerV3 Training")
parser.add_argument("--gpu", type=str, default="0", help="Target GPU ID")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]           = args.gpu
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"

import jax
jax.config.update("jax_platform_name", "cuda")
jax.config.update("jax_default_device", jax.devices("cuda")[0])

import jax.numpy as jnp
import optax
import time
from functools import partial

from dreamer_buffer   import init_buffer, add_batch, sample_sequences, buffer_ready
from dreamer_rssm     import RSSM, DreamerEncoder, LATENT_SIZE, DETERMINISTIC_SIZE
from dreamer_decoders import ObservationDecoder, RewardDecoder, ContinueDecoder, world_model_loss, two_hot_loss
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
PREFILL_STEPS   = 2_000      # increased: more data before WM training begins

LR_WM        = 3e-4          # increased from 1e-4 to accelerate WM convergence early
LR_ACTOR     = 3e-5
LR_CRITIC    = 3e-5
GAMMA        = 0.99
LAMBDA_      = 0.95           # renamed from LAMBDA to avoid shadowing Python builtin
ENTROPY_COEF = 3e-4

WM_WARMUP_STEPS = 20_000

OBS_DIM    = 662
ACTION_DIM = 2
H_DIM      = DETERMINISTIC_SIZE   # 512
Z_DIM      = LATENT_SIZE          # 1024

# ---------------------------------------------------------------------------
# Module instances
# ---------------------------------------------------------------------------

encoder      = DreamerEncoder()
rssm         = RSSM(action_dim=ACTION_DIM)
obs_decoder  = ObservationDecoder(obs_dim=OBS_DIM)
rew_decoder  = RewardDecoder()
cont_decoder = ContinueDecoder()
actor        = DreamerActor(action_dim=ACTION_DIM)
critic       = DreamerCritic()

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

    obs_t = jnp.swapaxes(obs_seq,  0, 1)   # [T, B, obs_dim]
    act_t = jnp.swapaxes(act_prev, 0, 1)   # [T, B, act_dim]

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

        # BUG A FIX: prior_greedy — only logits needed for KL, no gumbel key.
        _, prior_logits = rssm.apply(
            {'params': wm_params['rssm']}, h_t,
            method=rssm.prior_greedy)

        return (h_t, z_t, key), (h_t, z_t, prior_logits, post_logits)

    _, (h_seq, z_seq, prior_seq, post_seq) = jax.lax.scan(
        step, (h_0, z_0, rng_key), (obs_t, act_t))

    return (jnp.swapaxes(h_seq,     0, 1),
            jnp.swapaxes(z_seq,     0, 1),
            jnp.swapaxes(prior_seq, 0, 1),
            jnp.swapaxes(post_seq,  0, 1))

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def make_optimizer(lr):
    b1 = 0.9
    b2 = 0.99
    return optax.chain(
        optax.zero_nans(),
        optax.adaptive_grad_clip(0.3),
        # LaProp: scale by RMS first, then apply momentum (trace) to normalized gradients
        optax.scale_by_rms(decay=b2, eps=1e-20),
        optax.trace(decay=b1, nesterov=False),
        # Optax's trace accumulates as a sum, so we multiply by (1 - b1) to form an EMA
        optax.scale(-lr * (1.0 - b1))
    )

opt_wm     = make_optimizer(LR_WM)
opt_actor  = make_optimizer(LR_ACTOR)
opt_critic = make_optimizer(LR_CRITIC)


@jax.jit
def train_step(rng_key, buffer_state, params, opt_states, step_count, ema_s, slow_critic_params):
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
    wm_updates, new_wm_opt = opt_wm.update(wm_grads, opt_states['wm'], params['wm'])
    new_wm_params = optax.apply_updates(params['wm'], wm_updates)

    wm_warm_mask = (step_count >= WM_WARMUP_STEPS).astype(jnp.float32)

    # ---- B. Actor ----
    start_h = jax.lax.stop_gradient(h_states.reshape(-1, H_DIM))
    start_z = jax.lax.stop_gradient(z_states.reshape(-1, Z_DIM))

    def actor_loss_fn(actor_params):
        frozen_wm = jax.lax.stop_gradient(new_wm_params)

        traj = unroll_imagination(
            k_actor,
            lambda p, h, z, a: rssm.apply({'params': p}, h, z, a, method=rssm.step_gru),
            lambda p, h, k: rssm.apply({'params': p}, h, method=rssm.prior, rngs={'gumbel': k}),
            lambda p, h, z: actor.apply({'params': p}, h, z),
            {'rssm': frozen_wm['rssm'], 'actor': actor_params},
            start_h, start_z, HORIZON,
        )

        values_logits = critic.apply(
            {'params': jax.lax.stop_gradient(params['critic'])}, traj['h'], traj['z'])
        reward_logits = rew_decoder.apply(
            {'params': frozen_wm['rew']}, traj['h'], traj['z'])
        continues = jax.nn.sigmoid(cont_decoder.apply(
            {'params': frozen_wm['cont']}, traj['h'], traj['z']))

        # Decode two-hot logits to scalar expected values via symexp over bin centers
        from dreamer_rssm import symexp
        num_bins = values_logits.shape[-1]
        bin_centers = jnp.linspace(-20.0, 20.0, num_bins)
        values  = jnp.sum(jax.nn.softmax(values_logits,  axis=-1) * bin_centers, axis=-1)
        values  = symexp(values)
        rewards = jnp.sum(jax.nn.softmax(reward_logits, axis=-1) * bin_centers, axis=-1)
        rewards = symexp(rewards)

        bootstrap      = values[-1]
        lambda_returns = compute_lambda_returns(
            rewards[:-1], values[:-1], continues[:-1], bootstrap, GAMMA, LAMBDA_)

        adv_raw = lambda_returns - values[:-1]
        
        # V3 Percentile Return Normalization with EMA tracking
        batch_s = jnp.percentile(lambda_returns, 95) - jnp.percentile(lambda_returns, 5)
        new_ema_s = 0.99 * ema_s + 0.01 * batch_s
        advantages = jax.lax.stop_gradient(adv_raw / jnp.maximum(1.0, new_ema_s))

        actor_loss   = -jnp.mean(traj['log_prob'][:-1] * advantages)
        # BUG B FIX: single negation — maximise entropy = minimise mean(log_prob).
        entropy_loss = -ENTROPY_COEF * jnp.mean(traj['log_prob'][:-1])
        return actor_loss + entropy_loss, (traj, lambda_returns, new_ema_s)

    (act_loss, (traj, lambda_returns, new_ema_s)), act_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True)(params['actor'])

    # FIX 1: zero actor grads during WM warmup
    act_grads = jax.tree_util.tree_map(lambda g: g * wm_warm_mask, act_grads)
    act_updates, new_act_opt = opt_actor.update(act_grads, opt_states['actor'], params['actor'])
    new_act_params = optax.apply_updates(params['actor'], act_updates)

    def critic_loss_fn(critic_params):
        target = jax.lax.stop_gradient(lambda_returns)
        values_logits = critic.apply(
            {'params': critic_params},
            jax.lax.stop_gradient(traj['h'][:-1]),
            jax.lax.stop_gradient(traj['z'][:-1]))
            
        # V3 Critic EMA Regularizer
        slow_logits = critic.apply(
            {'params': jax.lax.stop_gradient(slow_critic_params)},
            jax.lax.stop_gradient(traj['h'][:-1]),
            jax.lax.stop_gradient(traj['z'][:-1]))
        
        from dreamer_rssm import symexp
        bin_centers = jnp.linspace(-20.0, 20.0, slow_logits.shape[-1])
        slow_values = jnp.sum(jax.nn.softmax(slow_logits, axis=-1) * bin_centers, axis=-1)
        slow_target = jax.lax.stop_gradient(symexp(slow_values))
        
        loss_crit = jnp.mean(two_hot_loss(values_logits, target))
        reg_crit  = jnp.mean(two_hot_loss(values_logits, slow_target))
        return loss_crit + reg_crit

    critic_loss, crit_grads = jax.value_and_grad(critic_loss_fn)(params['critic'])

    # FIX 1: zero critic grads during WM warmup
    crit_grads = jax.tree_util.tree_map(lambda g: g * wm_warm_mask, crit_grads)
    crit_updates, new_crit_opt = opt_critic.update(crit_grads, opt_states['critic'], params['critic'])
    new_crit_params = optax.apply_updates(params['critic'], crit_updates)
    
    # Update slow critic parameters with 0.98 EMA decay
    new_slow_critic = jax.tree_util.tree_map(
        lambda slow, fast: 0.98 * slow + 0.02 * fast,
        slow_critic_params, new_crit_params
    )

    new_params     = {'wm': new_wm_params, 'actor': new_act_params, 'critic': new_crit_params}
    new_opt_states = {'wm': new_wm_opt,    'actor': new_act_opt,    'critic': new_crit_opt}
    metrics = {
        'wm_loss':     wm_loss,
        'obs_loss':    wm_aux[0],
        'rew_loss':    wm_aux[1],
        'cont_loss':   wm_aux[2],
        'kl_loss':     wm_aux[3],
        'actor_loss':  act_loss,
        'critic_loss': critic_loss,
        'wm_warming':  1.0 - wm_warm_mask,   # 1.0 during warmup, 0.0 after
    }
    return new_params, new_opt_states, metrics, new_ema_s, new_slow_critic

# ---------------------------------------------------------------------------
# Inference step
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

    # explore is static — Python if resolved at compile time
    if explore:
        action, _ = sample_action(act_key, mean, std)
    else:
        action = jnp.tanh(mean)

    return action, h_t, z_t


@partial(jax.jit, static_argnames=('chunk_size',))
def train_loop_chunk(rng_key, buffer_state, params, opt_states,
                     env_obs, env_state, current_h, current_z, current_action,
                     cur_return, ema_return, ema_success,
                     step_count, ema_s, slow_critic_params,
                     chunk_size=50):

    def single_step(carry, _):
        (b_state, p, opt, h, z, a, obs, e_state,
         key, c_ret, e_ret, e_succ, s_count, cur_ema_s, cur_slow_crit) = carry

        key, act_k, step_k, train_k = jax.random.split(key, 4)

        raw_acts, next_h, next_z = act_step(
            act_k, p['wm'], p['actor'], obs, h, z, a, explore=True)

        env_acts = scale_actions_batched(raw_acts, e_state.env_state.max_v)
        s_keys   = jax.random.split(step_k, NUM_ENVS)

        n_obs, n_e_state, rews, dones, info = vmap_step(
            s_keys, e_state, env_acts, 3.0, -1)

        new_b_state = add_batch(b_state, obs, raw_acts, rews, dones)
        new_p, new_opt, metrics, new_ema_s, new_slow_crit = train_step(
            train_k, new_b_state, p, opt, s_count, cur_ema_s, cur_slow_crit)

        alive   = (~dones).astype(jnp.float32)[:, None]
        next_h *= alive
        next_z *= alive

        c_ret     = c_ret + rews
        num_dones = jnp.sum(dones)
        mean_ret  = jnp.sum(c_ret * dones)    / jnp.maximum(1.0, num_dones)
        mean_succ = jnp.sum(info['goal_reached'] * dones) / jnp.maximum(1.0, num_dones)
        e_ret  = jnp.where(num_dones > 0, 0.95 * e_ret  + 0.05 * mean_ret,  e_ret)
        e_succ = jnp.where(num_dones > 0, 0.95 * e_succ + 0.05 * mean_succ, e_succ)
        c_ret  = c_ret * (~dones)

        metrics['ema_return']  = e_ret
        metrics['ema_success'] = e_succ

        next_carry = (new_b_state, new_p, new_opt, next_h, next_z, raw_acts,
                      n_obs, n_e_state, key, c_ret, e_ret, e_succ,
                      s_count + 1, new_ema_s, new_slow_crit)
        return next_carry, metrics

    init_carry = (buffer_state, params, opt_states,
                  current_h, current_z, current_action,
                  env_obs, env_state, rng_key,
                  cur_return, ema_return, ema_success,
                  step_count, ema_s, slow_critic_params)

    final_carry, metrics_history = jax.lax.scan(
        single_step, init_carry, None, length=chunk_size)
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

    # 3. Parameter initialisation
    dummy_obs   = jnp.zeros((1, OBS_DIM))
    dummy_h     = jnp.zeros((1, H_DIM))
    dummy_z     = jnp.zeros((1, Z_DIM))
    dummy_act   = jnp.zeros((1, ACTION_DIM))
    dummy_embed = jnp.zeros((1, H_DIM))

    rng, r1, r2 = jax.random.split(rng, 3)

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

    cur_return  = jnp.zeros(NUM_ENVS)
    ema_return  = jnp.array(0.0)
    ema_success = jnp.array(0.0)

    # FIX 1: step counter as JAX int32 scalar, threaded through lax.scan carry
    step_count = jnp.int32(0)
    
    ema_s = jnp.array(1.0, dtype=jnp.float32)
    slow_critic_params = jax.tree_util.tree_map(lambda x: x, params['critic'])

    # 4. Prefill
    print("Compiling and executing fast pre-fill on GPU...")

    @jax.jit
    def run_prefill(rng_key, b_state, e_state, e_obs):
        def _step(carry, _):
            curr_b, curr_e_state, curr_e_obs, curr_rng = carry
            curr_rng, act_rng, step_rng = jax.random.split(curr_rng, 3)
            raw_acts = jax.random.uniform(
                act_rng, (NUM_ENVS, ACTION_DIM), minval=-1.0, maxval=1.0)
            env_acts = scale_actions_batched(raw_acts, curr_e_state.env_state.max_v)
            s_keys   = jax.random.split(step_rng, NUM_ENVS)
            n_obs, n_state, rews, dones, _ = vmap_step(
                s_keys, curr_e_state, env_acts, 3.0, -1)
            next_b = add_batch(curr_b, curr_e_obs, raw_acts, rews, dones)
            return (next_b, n_state, n_obs, curr_rng), None

        (final_b, final_e_state, final_e_obs, _), _ = jax.lax.scan(
            _step, (b_state, e_state, e_obs, rng_key), None, length=PREFILL_STEPS)
        return final_b, final_e_state, final_e_obs

    t_prefill = time.time()
    buffer_state, env_state, env_obs = run_prefill(
        rng, buffer_state, env_state, env_obs)
    buffer_state.insert_idx.block_until_ready()
    print(f"Pre-fill complete in {time.time() - t_prefill:.2f}s\n")

    # 5. Main training loop
    print("Starting Optimized Main Training Loop...")
    CHUNK_SIZE  = 50
    TOTAL_STEPS = 2_000_000

    for step in range(0, TOTAL_STEPS, CHUNK_SIZE):
        t0 = time.time()

        if not buffer_ready(buffer_state, SEQ_LEN):
            print(f"Step {step:05d} | Buffer not ready, skipping.")
            continue

        # Execute chunked training loop
        final_carry, metrics_history = train_loop_chunk(
            rng, buffer_state, params, opt_states,
            env_obs, env_state, current_h, current_z, current_action,
            cur_return, ema_return, ema_success,
            step_count, ema_s, slow_critic_params,
            chunk_size=CHUNK_SIZE,
        )

        # Unpack updated state
        (buffer_state, params, opt_states,
         current_h, current_z, current_action,
         env_obs, env_state, rng,
         cur_return, ema_return, ema_success,
         step_count, ema_s, slow_critic_params) = final_carry

        if step % 100 == 0:
            avg_wm     = float(jnp.mean(metrics_history['wm_loss']))
            avg_obs    = float(jnp.mean(metrics_history['obs_loss']))
            avg_rew    = float(jnp.mean(metrics_history['rew_loss']))
            avg_cont   = float(jnp.mean(metrics_history['cont_loss']))
            avg_kl     = float(jnp.mean(metrics_history['kl_loss']))
            avg_actor  = float(jnp.mean(metrics_history['actor_loss']))
            avg_critic = float(jnp.mean(metrics_history['critic_loss']))
            fps        = (NUM_ENVS * CHUNK_SIZE) / (time.time() - t0)
            ret_val    = float(metrics_history['ema_return'][-1])
            succ_val   = float(metrics_history['ema_success'][-1]) * 100.0
            warming    = float(metrics_history['wm_warming'][-1]) > 0.5

            print(
                f"Step {step:05d} | "
                f"FPS: {fps:.0f} | "
                f"WM: {avg_wm:.3f} "
                f"(obs={avg_obs:.3f} rew={avg_rew:.3f} cont={avg_cont:.3f} kl={avg_kl:.3f}) | "
                f"Act: {avg_actor:.4f} | "
                f"Crit: {avg_critic:.4f} | "
                f"Ret: {ret_val:.1f} | "
                f"Succ: {succ_val:.1f}%"
                + (" | [WM warmup]" if warming else "")
            )

    print("Training Complete.")

    # 6. Checkpoint saving
    import flax.serialization
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/dreamer_best.msgpack"

    eval_params = {
        'encoder': jax.device_get(params['wm']['encoder']),
        'rssm':    jax.device_get(params['wm']['rssm']),
        'actor':   jax.device_get(params['actor']),
    }

    with open(ckpt_path, "wb") as f:
        f.write(flax.serialization.to_bytes(eval_params))
    print(f"Checkpoint saved to {ckpt_path}.")