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
  - BUG 9 FIXED: unroll_imagination now returns only `traj` (final h/z were
    always discarded). Call sites updated accordingly.
"""

import jax
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
BUFFER_CAPACITY = 100_000
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
        key, post_key, prior_key = jax.random.split(key, 3)

        h_t = rssm.apply(
            {'params': wm_params['rssm']},
            h_prev, z_prev, act_i, method=rssm.step_gru)

        obs_embed = encoder.apply({'params': wm_params['encoder']}, obs_i)

        z_t, post_logits = rssm.apply(
            {'params': wm_params['rssm']}, h_t, obs_embed,
            method=rssm.posterior, rngs={'gumbel': post_key})

        _, prior_logits = rssm.apply(
            {'params': wm_params['rssm']}, h_t,
            method=rssm.prior, rngs={'gumbel': prior_key})

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
        actor_loss  = -jnp.mean(traj['log_prob'][:-1] * advantages)
        entropy_loss = -ENTROPY_COEF * jnp.mean(-traj['log_prob'][:-1])
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

    # 3. Parameters — BUG 1 FIX: init via __call__ to register ALL sub-layers
    dummy_obs   = jnp.zeros((1, OBS_DIM))
    dummy_h     = jnp.zeros((1, H_DIM))
    dummy_z     = jnp.zeros((1, Z_DIM))
    dummy_act   = jnp.zeros((1, ACTION_DIM))
    dummy_embed = jnp.zeros((1, H_DIM))

    rng, r1, r2, r3 = jax.random.split(rng, 4)

    # Use __call__ for rssm init so every sub-layer is touched
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

    # 4. Prefill
    print("Pre-filling buffer with random actions...")
    for _ in range(PREFILL_STEPS):
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        raw_actions = jax.random.uniform(act_rng, (NUM_ENVS, ACTION_DIM), minval=-1.0, maxval=1.0)
        env_actions = scale_actions_batched(raw_actions, env_state.env_state.max_v)
        step_keys   = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, _ = vmap_step(
            step_keys, env_state, env_actions, 3.0, -1)
        buffer_state = add_batch(buffer_state, env_obs, raw_actions, rewards, dones)
        env_obs   = next_obs
        env_state = next_state

    current_h      = jnp.zeros((NUM_ENVS, H_DIM))
    current_z      = jnp.zeros((NUM_ENVS, Z_DIM))
    current_action = jnp.zeros((NUM_ENVS, ACTION_DIM))

    # 5. Main loop
    print("Starting Main Training Loop...")
    for step in range(50_000):
        t0 = time.time()
        rng, act_rng, step_rng, train_rng = jax.random.split(rng, 4)

        raw_actions, current_h, current_z = act_step(
            act_rng, params['wm'], params['actor'],
            env_obs, current_h, current_z, current_action, explore=True)
        current_action = raw_actions

        env_actions = scale_actions_batched(raw_actions, env_state.env_state.max_v)
        step_keys   = jax.random.split(step_rng, NUM_ENVS)
        next_obs, next_state, rewards, dones, _ = vmap_step(
            step_keys, env_state, env_actions, 3.0, -1)

        buffer_state = add_batch(buffer_state, env_obs, raw_actions, rewards, dones)

        # BUG 6 FIX: explicit bool→float cast, not `1.0 - dones`
        alive          = (~dones).astype(jnp.float32)[:, None]
        current_h      = current_h      * alive
        current_z      = current_z      * alive
        current_action = current_action * alive

        env_obs   = next_obs
        env_state = next_state

        # BUG 7 FIX: only call train_step once the buffer has enough data
        if buffer_ready(buffer_state, SEQ_LEN):
            params, opt_states, metrics = train_step(
                train_rng, buffer_state, params, opt_states)

            if step % 100 == 0:
                fps = NUM_ENVS / (time.time() - t0)
                print(
                    f"Step {step:05d} | "
                    f"WM: {metrics['wm_loss']:.3f} | "
                    f"Actor: {metrics['actor_loss']:.3f} | "
                    f"Critic: {metrics['critic_loss']:.3f} | "
                    f"KL: {metrics['kl_loss']:.3f} | "
                    f"FPS: {fps:.0f}"
                )

    print("Training Complete.")