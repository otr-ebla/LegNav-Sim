"""
MBRL_shac.py — Short-Horizon Analytic Critic (SHAC)
====================================================
Integrates fully differentiable simulation for exact analytical gradients.
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

from jax_network import EndToEndActorCritic, scale_action_to_env
from jax_env_multi import step_env, reset_env, EnvState, ROOM_W, ROOM_H
from jax_wrappers import make_stacked_env, StackedEnvState
from jax_env import NUM_RAYS, STATE_VEC_SIZE

# Import differentiable metrics to provide exact gradients for boolean step functions
from MBRL_diff_utils import collision_loss, proxemics_loss, efficiency_loss, smoothness_loss

_POSE_SIZE = 3

# ── SHAC Hyperparameters ──────────────────────────────────────────────────────

SHAC_H_INIT  = 4
SHAC_H_MAX   = 24     
SHAC_H_GROWTH_INTERVAL = 40

N_SHAC_ENVS = 1024

SHAC_LR_ACTOR  = 3e-5
SHAC_LR_CRITIC = 3e-4

SHAC_GAMMA    = 0.99
SHAC_LAM      = 0.95
SHAC_VF_COEF  = 0.5

SHAC_GRAD_NORM_ACTOR  = 0.2
SHAC_GRAD_NORM_CRITIC = 1.0

SHAC_RETURN_NORM_EPS = 1e-8
SHAC_TARGET_TAU = 0.005

OBS_SIZE = 3 * 3 + 9 + 108 * 3   

# ── SHAC Trainer State ────────────────────────────────────────────────────────

class SHACTrainState(NamedTuple):
    actor_params:         any   
    critic_params:        any   
    critic_target_params: any   
    actor_opt:            any
    critic_opt:           any
    horizon:              int
    update_count:         int

# ── SHAC Critic Network ───────────────────────────────────────────────────────

class SHACCritic(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = flax.linen.relu(flax.linen.Dense(256)(x))
        x = flax.linen.relu(flax.linen.Dense(128)(x))
        x = flax.linen.relu(flax.linen.Dense(64)(x))
        return jnp.squeeze(flax.linen.Dense(1)(x), axis=-1)

# ── Differentiable Core: SHAC Rollout ───────────────────────────────────────

def _shac_rollout_single(
    actor_params,
    critic_params,
    actor_apply,
    critic_apply,
    init_obs:    jnp.ndarray,      
    init_state:  StackedEnvState,
    rng_key:     jnp.ndarray,
    horizon_mask: jnp.ndarray,     
    ghost_robot:  bool = True,
):
    def _step_fn(carry, t):
        obs, state, key, cum_discount = carry
        key, step_key = jax.random.split(key)

        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean = mean[0]
        env_action = scale_action_to_env(mean, state.env_state.max_v)

        base_obs, new_base_state, base_reward, done, info = step_env(
            step_key, state.env_state, env_action, ghost_robot=ghost_robot
        )

        # Inject Differentiable Loss functions into the analytical reward.
        # This replaces step-functions (boolean logic) with smooth gradients.
        s = new_base_state
        c_loss = collision_loss(s.x, s.y, s.people, s.obs_circles, s.obs_boxes, ROOM_W, ROOM_H)
        p_loss = proxemics_loss(s.x, s.y, s.people, s.v)
        e_loss = efficiency_loss(s.x, s.y, s.v, s.theta, s.goal_x, s.goal_y, s.max_v)
        s_loss = smoothness_loss(s.w, state.env_state.w, s.v)
        
        diff_cost = 2.0 * c_loss + 1.0 * p_loss + 0.5 * e_loss + 0.3 * s_loss
        analytical_reward = base_reward - diff_cost

        new_pose      = base_obs[0:_POSE_SIZE]
        new_state_vec = base_obs[_POSE_SIZE : _POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[_POSE_SIZE + STATE_VEC_SIZE:]

        new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
        new_pose_stack = jnp.concatenate([state.pose_stack[1:], new_pose[None]], axis=0)
        
        new_stacked = state.replace(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack,
        )
        new_obs = jnp.concatenate([
            new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
        ])

        discounted_r  = cum_discount * analytical_reward
        done_f        = done.astype(jnp.float32)
        next_discount = cum_discount * SHAC_GAMMA * (1.0 - done_f)

        masked_r = discounted_r * horizon_mask[t].astype(jnp.float32)

        return (new_obs, new_stacked, key, next_discount), (masked_r, new_obs, done)

    # Gradient checkpointing to prevent OOM
    _step_fn_ckpt = jax.checkpoint(_step_fn)

    (final_obs, final_state, _, final_discount), (rewards, obs_seq, dones) = \
        jax.lax.scan(
            _step_fn_ckpt,
            (init_obs, init_state, rng_key, jnp.ones(())),
            jnp.arange(SHAC_H_MAX),   
            length=SHAC_H_MAX,
        )

    bootstrap_v = jax.lax.stop_gradient(
        critic_apply({"params": critic_params}, final_obs[None])[0]
    )

    total_return = jnp.sum(rewards) + final_discount * bootstrap_v

    return total_return, (obs_seq, rewards, dones, final_obs)

_shac_rollout_batched = jax.vmap(
    _shac_rollout_single,
    in_axes=(None, None, None, None, 0, 0, 0, None, None)
)

# ── Actor & Critic Loss Functions ─────────────────────────────────────────────

def _actor_loss_fn(
    actor_params, critic_params, actor_apply, critic_apply,
    init_obs_batch, init_state_batch, rng_keys, horizon_mask, ghost_robot,
):
    returns, (obs_seq, rewards, dones, final_obs) = _shac_rollout_batched(
        actor_params, critic_params, actor_apply, critic_apply,
        init_obs_batch, init_state_batch, rng_keys, horizon_mask, ghost_robot,
    )

    ret_mean = jax.lax.stop_gradient(jnp.mean(returns))
    ret_std  = jax.lax.stop_gradient(jnp.std(returns) + SHAC_RETURN_NORM_EPS)
    normalized_returns = (returns - ret_mean) / ret_std

    loss = -jnp.mean(normalized_returns)
    return loss, (returns, obs_seq, rewards, dones, final_obs)

def _critic_loss_fn(
    critic_params, critic_apply, obs_seq, rewards, dones, final_obs, target_params,  
):
    N, H = rewards.shape
    v_final = jax.lax.stop_gradient(critic_apply({"params": target_params}, final_obs))

    def _td_lambda_scan(carry, t):
        gae, next_v = carry
        r   = rewards[:, t]
        d   = dones[:, t].astype(jnp.float32)
        obs = obs_seq[:, t, :]

        v_pred  = jax.lax.stop_gradient(critic_apply({"params": target_params}, obs))
        delta   = r + SHAC_GAMMA * next_v * (1.0 - d) - v_pred
        gae     = delta + SHAC_GAMMA * SHAC_LAM * (1.0 - d) * gae
        target  = jax.lax.stop_gradient(gae + v_pred)
        return (gae, v_pred), target

    (_, _), targets = jax.lax.scan(
        _td_lambda_scan,
        (jnp.zeros(N), v_final),
        jnp.arange(H - 1, -1, -1),
        reverse=False,
    )
    targets = targets.T  

    obs_flat = obs_seq.reshape(N * H, OBS_SIZE)
    v_preds  = critic_apply({"params": critic_params}, obs_flat).reshape(N, H)

    targets_sg  = jax.lax.stop_gradient(targets)
    critic_loss = SHAC_VF_COEF * jnp.mean((v_preds - targets_sg) ** 2)
    return critic_loss

# ── SHAC Update Step ──────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def shac_update_step(
    shac_state:       SHACTrainState,
    env_data:         Tuple,   
    actor_apply,
    critic_apply,
    actor_optimizer,
    critic_optimizer,
    ghost_robot:      bool,
):
    actor_params         = shac_state.actor_params
    critic_params        = shac_state.critic_params
    critic_target_params = shac_state.critic_target_params
    actor_opt            = shac_state.actor_opt
    critic_opt           = shac_state.critic_opt

    init_obs_batch, init_state_batch, rng_keys, horizon_mask = env_data

    # 1. Analytical Gradient for Actor
    (actor_loss, (returns, obs_seq, rewards, dones, final_obs)), actor_grads = \
        jax.value_and_grad(_actor_loss_fn, has_aux=True)(
            actor_params, critic_target_params, actor_apply, critic_apply,
            init_obs_batch, init_state_batch, rng_keys, horizon_mask, ghost_robot,
        )

    actor_grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        actor_grads
    )

    actor_grad_norm = optax.global_norm(actor_grads)
    actor_updates, new_actor_opt = actor_optimizer.update(actor_grads, actor_opt, actor_params)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    # 2. TD(lambda) update for Critic
    critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(
        critic_params, critic_apply, obs_seq, rewards, dones, final_obs, critic_target_params,   
    )

    critic_grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        critic_grads
    )

    critic_grad_norm = optax.global_norm(critic_grads)
    critic_updates, new_critic_opt = critic_optimizer.update(critic_grads, critic_opt, critic_params)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    # 3. Soft update for Target Network (EMA)
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
        horizon              = shac_state.horizon,
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

def make_horizon_mask(h_curr: int) -> jnp.ndarray:
    return jnp.arange(SHAC_H_MAX) < h_curr

# ── Initialization ────────────────────────────────────────────────────────────

def init_shac(rng_key, actor_params, actor_apply):
    rng_key, ck = jax.random.split(rng_key)

    critic_net    = SHACCritic()
    dummy_obs     = jnp.zeros((1, OBS_SIZE))
    critic_params = critic_net.init(ck, dummy_obs)["params"]
    critic_apply  = critic_net.apply

    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(SHAC_GRAD_NORM_ACTOR),
        optax.sgd(learning_rate=SHAC_LR_ACTOR)
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
    n_increases = update_count // SHAC_H_GROWTH_INTERVAL
    h = SHAC_H_INIT + n_increases
    return min(h, SHAC_H_MAX)

def save_shac_checkpoint(shac_state, filepath):
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
    print(f"  SHAC checkpoint -> {filepath}")