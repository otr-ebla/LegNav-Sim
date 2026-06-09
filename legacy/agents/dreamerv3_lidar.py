import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
import distrax 
import jax.lax as lax
from flax.training import train_state
import optax

RSSM_STOCHASTIC_SIZE = 32
RSSM_DISCRETE_CLASSES = 8

@struct.dataclass
class ReplayBuffer:
    obs: jnp.ndarray          # [capacity, obs_dim]
    goal: jnp.ndarray         # [capacity, goal_dim]
    act: jnp.ndarray          # [capacity, action_dim]
    rew: jnp.ndarray          # [capacity]
    done: jnp.ndarray         # [capacity]
    capacity: int
    ptr: int
    full: bool 

def create_buffer(capacity, obs_dim, goal_dim, act_dim):
    return ReplayBuffer(
        obs=jnp.zeros((capacity, obs_dim)),
        goal=jnp.zeros((capacity, goal_dim)),
        act=jnp.zeros((capacity, act_dim)),
        rew=jnp.zeros((capacity,)),
        done=jnp.zeros((capacity,)),
        capacity=capacity,
        ptr=0,
        full=False,
    )

def rb_add(buffer: ReplayBuffer, obs, goal, act, rew, done):
    idx = buffer.ptr

    new_buffer = buffer.replace(
        obs=buffer.obs.at[idx].set(obs),
        goal=buffer.goal.at[idx].set(goal),
        act=buffer.act.at[idx].set(act),
        rew=buffer.rew.at[idx].set(rew),
        done=buffer.done.at[idx].set(done),
        ptr=(idx + 1) % buffer.capacity,
        full=buffer.full | (idx + 1 >= buffer.capacity),
    )
    return new_buffer

def rb_sample(buffer: ReplayBuffer, batch_size, seq_len, key):
    # dove possiamo iniziare?
    max_start = buffer.capacity - seq_len if buffer.full else buffer.ptr - seq_len
    max_start = jnp.maximum(max_start, 1)

    # campiona gli indici
    key, subkey = jax.random.split(key)
    starts = jax.random.randint(subkey, (batch_size,), 0, max_start)

    # Costruisci gli indici temporali
    idx = starts[:, None] + jnp.arange(seq_len)[None, :]   # [batch, seq_len]

    batch = {
        "lidar": buffer.obs[idx],
        "goal": buffer.goal[idx],
        "actions": buffer.act[idx],
        "rewards": buffer.rew[idx],
        "done": buffer.done[idx],
    }
    return batch, key




# --- 1. ENCODER CON LAYERNORM ---
class SensorEncoder(nn.Module):
    @nn.compact
    def __call__(self, lidar_data, goal_polar):
        combined_input = jnp.concatenate([lidar_data, goal_polar], axis=-1)
        
        x = nn.Dense(features=256)(combined_input)
        x = nn.LayerNorm()(x) # <--- STABILIZZATORE
        x = nn.elu(x)
        
        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x) # <--- STABILIZZATORE
        x = nn.elu(x)
        
        return x

# --- 2. RSSM CON LAYERNORM ---
class RSSM(nn.Module):
    deterministic_size: int = 512
    stochastic_size: int = 32 
    discrete_classes: int = RSSM_DISCRETE_CLASSES

    def setup(self):
        self.fc_input = nn.Dense(512)
        self.ln_input = nn.LayerNorm() # <---
        self.rnn = nn.GRUCell(features=self.deterministic_size)
        
        self.fc_prior = nn.Dense(self.stochastic_size * self.discrete_classes)
        
        self.fc_post_embed = nn.Dense(512)
        self.ln_post_embed = nn.LayerNorm() # <---
        self.fc_post = nn.Dense(self.stochastic_size * self.discrete_classes)

    def __call__(self, prev_state, prev_action, embed_obs):
        prev_h, prev_z = prev_state
        
        # 1. Input Memoria
        x = jnp.concatenate([prev_z.reshape((prev_z.shape[0], -1)), prev_action], axis=-1)
        x = self.fc_input(x)
        x = self.ln_input(x) # Norm
        x = nn.elu(x)
        
        # 2. GRU
        h_next, _ = self.rnn(x, prev_h)
        
        # 3. Prior
        prior_out = self.fc_prior(h_next)
        prior_logits = prior_out.reshape(-1, self.stochastic_size, self.discrete_classes)
        
        # 4. Posterior
        x_post = jnp.concatenate([h_next, embed_obs], axis=-1)
        x_post = self.fc_post_embed(x_post)
        x_post = self.ln_post_embed(x_post) # Norm
        x_post = nn.elu(x_post)
        
        post_out = self.fc_post(x_post)
        post_logits = post_out.reshape(-1, self.stochastic_size, self.discrete_classes)
            
        return h_next, post_logits, prior_logits

    def img_step(self, prev_state, prev_action):
        prev_h, prev_z = prev_state
        x = jnp.concatenate([prev_z.reshape((prev_z.shape[0], -1)), prev_action], axis=-1)
        x = self.fc_input(x)
        x = self.ln_input(x) # Norm
        x = nn.elu(x)
        
        h_next, _ = self.rnn(x, prev_h)
        
        prior_out = self.fc_prior(h_next)
        prior_logits = prior_out.reshape(-1, self.stochastic_size, self.discrete_classes)
        prior_probs = nn.softmax(prior_logits)

        return (h_next, prior_probs), prior_logits

# --- 3. HEADS CON LAYERNORM ---
class WorldModelHeads(nn.Module):   
    input_shape: int

    @nn.compact
    def __call__(self, state_features):
        # Decoder Osservazioni
        x_obs = nn.Dense(512)(state_features)
        x_obs = nn.LayerNorm()(x_obs)
        x_obs = nn.elu(x_obs)
        x_obs = nn.Dense(512)(x_obs)
        x_obs = nn.LayerNorm()(x_obs)
        x_obs = nn.elu(x_obs)
        obs_pred = nn.Dense(self.input_shape)(x_obs) 

        # Decoder Reward
        x_rew = nn.Dense(256)(state_features)
        x_rew = nn.LayerNorm()(x_rew)
        x_rew = nn.elu(x_rew)
        reward_pred = nn.Dense(1)(x_rew) 

        # Decoder Continuità
        x_cont = nn.Dense(256)(state_features)
        x_cont = nn.LayerNorm()(x_cont)
        x_cont = nn.elu(x_cont)
        cont_pred = nn.Dense(1)(x_cont) 

        return obs_pred, reward_pred, cont_pred

# --- 4. ACTOR & CRITIC CON LAYERNORM ---
class Actor(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, state_features):
        x = nn.Dense(256)(state_features)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        out = nn.Dense(self.action_dim * 2)(x)
        mean, std_raw = jnp.split(out, 2, axis=-1)

        log_std = jnp.clip(std_raw, -5.0, 2.0)
        std = jnp.exp(log_std)

        base_dist = distrax.MultivariateNormalDiag(mean, std)
        tanh_dist = distrax.Transformed(distribution=base_dist,
                                        bijector=distrax.Tanh())
        return tanh_dist


class Critic(nn.Module):
    @nn.compact
    def __call__(self, state_features):
        x = nn.Dense(256)(state_features)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        value = nn.Dense(1)(x)
        return value

# --- 5. DREAMER (Invariato, ma include i moduli aggiornati) ---
class Dreamer(nn.Module):
    def setup(self):
        self.encoder = SensorEncoder()
        self.rssm = RSSM()
        self.heads = WorldModelHeads(input_shape=110) 
        self.actor = Actor()
        self.critic = Critic()

    def __call__(self, lidar, goal, actions):
        batch_size = lidar.shape[0]
        embed = self.encoder(lidar, goal)
        actions_T = jnp.swapaxes(actions, 0, 1)
        embed_T = jnp.swapaxes(embed, 0, 1)

        def scan_step(prev_state, inputs):
            action_t, embed_t = inputs
            h_next, post_logits, prior_logits = self.rssm(prev_state, action_t, embed_t)
            post_probs = nn.softmax(post_logits)
            return (h_next, post_probs), (h_next, post_logits, prior_logits)
        
        init_h = jnp.zeros((batch_size, self.rssm.deterministic_size))
        init_z = jnp.zeros((batch_size, RSSM_STOCHASTIC_SIZE, RSSM_DISCRETE_CLASSES))
        _ = self.rssm((init_h, init_z), actions_T[0], embed_T[0])
        
        _, (h_seq_T, post_logits_seq_T, prior_logits_seq_T) = lax.scan(
            scan_step, (init_h, init_z), (actions_T, embed_T)
        )
        
        h_seq = jnp.swapaxes(h_seq_T, 0, 1)
        post_logits_seq = jnp.swapaxes(post_logits_seq_T, 0, 1)
        prior_logits_seq = jnp.swapaxes(prior_logits_seq_T, 0, 1)
        
        z_seq = nn.softmax(post_logits_seq)
        features = jnp.concatenate([h_seq, z_seq.reshape(batch_size, -1, self.rssm.stochastic_size*RSSM_DISCRETE_CLASSES)], axis=-1)
        pred_obs, pred_rew, pred_cont = self.heads(features)

        _ = self.actor(features)
        _ = self.critic(features)
        
        return (pred_obs, pred_rew, pred_cont, post_logits_seq, prior_logits_seq, features, h_seq, z_seq)
    
    def imagination(self, start_h, start_z, key, horizon=15):
        # carry = ( (h, z), key )
        def dream_step(carry, _):
            (h, z), key = carry

            # 1) Split della key: da una key grande ne facciamo una nuova + una da usare ora
            key, subkey = jax.random.split(key)

            # 2) Costruiamo le feature dello stato
            features = jnp.concatenate([h, z.reshape(h.shape[0], -1)], axis=-1)

            # 3) Policy: distribuzione sulle azioni
            dist = self.actor(features)

            # 4) Campioniamo l'azione usando la subkey (diversa a ogni passo)
            action = dist.sample(seed=subkey)

            # 5) Passo del modello di mondo in immaginazione
            next_state, _ = self.rssm.img_step((h, z), action)
            h_next, z_next = next_state

            # 6) Ricostruiamo le feature per reward e value
            feat_next = jnp.concatenate([h_next, z_next.reshape(h_next.shape[0], -1)], axis=-1)
            feat_next = jax.lax.stop_gradient(feat_next)  # Stop gradient qui!

            reward = self.heads(feat_next)[1]  # reward_pred
            value = self.critic(feat_next)

            # 7) Nuovo carry: stato + key aggiornata
            new_carry = ((h_next, z_next), key)

            # 8) Output della scan (sequenze)
            outputs = (reward, value, action, feat_next)
            return new_carry, outputs

        # Inizializziamo la scan con ((start_h, start_z), key)
        init_carry = ((start_h, start_z), key)

        # Secondo argomento (inputs) è None, quindi usiamo una sequenza "vuota" di lunghezza horizon
        (_, _), (dream_rews, dream_vals, dream_acts, dream_feats) = lax.scan(
            dream_step,
            init_carry,
            xs=None,
            length=horizon,
        )

        return dream_rews, dream_vals, dream_acts, dream_feats

    def policy_step(self, prev_state, lidar, goal, prev_action):
        embed = self.encoder(lidar, goal)
        h_next, post_logits, _ = self.rssm(prev_state, prev_action, embed)
        post_probs = nn.softmax(post_logits)
        features = jnp.concatenate([h_next, post_probs.reshape(h_next.shape[0], -1)], axis=-1)
        dist = self.actor(features)
        action = dist.mode()
        next_state = (h_next, post_probs)
        return action, next_state

# --- LOSS HELPERS (Con KL Safe + Free Bits) ---
def compute_kl_loss(post_logits, prior_logits, free_bits=0.1):
    post_probs = nn.softmax(post_logits)
    prior_probs = nn.softmax(prior_logits)

    post_probs = jnp.clip(post_probs, 1e-7, 1.0)
    prior_probs = jnp.clip(prior_probs, 1e-7, 1.0)

    # KL per dimensione
    kl_per_dim = post_probs * (jnp.log(post_probs) - jnp.log(prior_probs))
    kl_per_dim = jnp.sum(kl_per_dim, axis=-1)

    # Free bits più morbido
    kl = jnp.maximum(kl_per_dim, free_bits)

    return jnp.mean(kl)


def calculate_lambda_returns(rewards, values, discount=0.99, lambda_=0.95):
    next_values = jnp.concatenate([values[1:], values[-1:]], axis=0)
    inputs = (rewards, next_values)
    
    def step_fn(last_lambda_ret, inp):
        r, v_next = inp
        v_target = r + discount * ((1 - lambda_) * v_next + lambda_ * last_lambda_ret)
        return v_target, v_target

    last_val = values[-1]
    _, lambda_returns = lax.scan(step_fn, last_val, inputs, reverse=True)
    return lambda_returns


@jax.jit
def train_world_model(state, batch):
    """
    Aggiorna solo il world model (encoder + RSSM + heads).
    Ritorna:
      - new_state: parametri aggiornati
      - logs: dizionario con le loss
      - h_seq, z_seq: stati latenti da usare per il training actor-critic
    """
    def loss_fn(params):
        (pred_obs,
         pred_rew,
         pred_cont,
         post_logits,
         prior_logits,
         features,
         h_seq,
         z_seq) = state.apply_fn(
            params,
            batch["lidar"],
            batch["goal"],
            batch["actions"],
        )

        # Target osservazioni e reward
        target_obs = jnp.concatenate([batch["lidar"], batch["goal"]], axis=-1)
        target_rew = batch["rewards"]

        # Reconstruction loss su osservazioni
        loss_obs = jnp.mean((pred_obs - target_obs) ** 2)

        # Reconstruction loss reward (scala reward di un fattore 0.1)
        loss_rew = jnp.mean((pred_rew.squeeze(-1) - target_rew * 0.1) ** 2)

        # KL priors vs posteriors (con free bits)
        loss_kl = compute_kl_loss(post_logits, prior_logits)

        # Peso KL stile Dreamer (0.5 è un valore ragionevole)
        total_loss = loss_obs + loss_rew + 0.5 * loss_kl

        logs = {
            "loss_total": total_loss,
            "loss_obs": loss_obs,
            "loss_rew": loss_rew,
            "loss_kl": loss_kl,
        }

        # Non facciamo stop_gradient qui: serve il gradiente per il world model
        return total_loss, (logs, h_seq, z_seq)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, (logs, h_seq, z_seq)), grads = grad_fn(state.params)

    # 🔥 Gradient clipping globale
    grads = optax.clip_by_global_norm(grads, 10.0)

    new_state = state.apply_gradients(grads=grads)
    return new_state, logs, h_seq, z_seq


@jax.jit
def train_actor_critic(state, h_seq, z_seq):
    """
    Allena actor e critic usando gli stati latenti del world model
    (h_seq, z_seq) ma SENZA modificare il world model stesso.
    """

    # h_seq: [batch, time, deterministic_size]
    # z_seq: [batch, time, RSSM_STOCHASTIC_SIZE, RSSM_DISCRETE_CLASSES]

    batch, time, _ = h_seq.shape
    batch_time = batch * time

    # Ricostruiamo gli stati iniziali per l'imagination:
    start_h = h_seq.reshape(batch_time, -1)
    start_z = z_seq.reshape(batch_time, RSSM_STOCHASTIC_SIZE, RSSM_DISCRETE_CLASSES)

    # Stop gradient: non vogliamo che actor/critic cambino il world model
    start_h = jax.lax.stop_gradient(start_h)
    start_z = jax.lax.stop_gradient(start_z)

    def loss_fn(params):
        key = jax.random.PRNGKey(0)  # in futuro: passarlo dall'esterno

        dream_rews, dream_vals, dream_acts, dream_feats = state.apply_fn(
            params,
            start_h,
            start_z,
            key,
            method=Dreamer.imagination,
        )

        # Lambda-returns sui reward immaginati (già scalati di 0.1)
        targets = calculate_lambda_returns(dream_rews * 0.1, dream_vals)
        targets = jax.lax.stop_gradient(targets)

        # Critic: MSE tra valore predetto e target
        loss_critic = jnp.mean((dream_vals - targets) ** 2)

        # Actor: massimizzare il target → minimizzare -target
        loss_actor = -jnp.mean(targets)

        total_loss = loss_actor + loss_critic

        logs = {
            "loss_actor": loss_actor,
            "loss_critic": loss_critic,
            "loss_total_ac": total_loss,
        }
        return total_loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logs), grads = grad_fn(state.params)

    # 🔥 Gradient clipping globale
    grads = optax.clip_by_global_norm(grads, 10.0)

    new_state = state.apply_gradients(grads=grads)
    return new_state, logs

