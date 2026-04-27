import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src/jax_env"))

from jax_eval_multi import _build_ppo_shac
from jax_network import EndToEndActorCritic
from jax_wrappers import make_stacked_env, make_autoreset_env
from jax_env_multi import reset_env, step_env

ckpt = os.path.join(project_root, "src/jax_env/checkpoints/ppo_attn_final.msgpack")
_, load_fn, _, _ = _build_ppo_shac()
try:
    params = load_fn(ckpt)
except Exception as e:
    print(f"Error loading {ckpt}: {e}")
    sys.exit()

net_new = EndToEndActorCritic(action_dim=2)
num_envs = 2000
rollout_steps = 50
rng = jax.random.PRNGKey(42)

reset_stacked, step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)
step_auto = make_autoreset_env(reset_stacked, step_stacked)

def _reset_with_dist(key):
    return reset_stacked(key, max_goal_dist=9.0, scenario_idx=-1, ghost_prob=0.0)

vmap_reset = jax.jit(jax.vmap(_reset_with_dist))
# Fix for max_scenario curriculum param (index 6 after self/key/state/action/goal/idx/ghost) in step_auto calls: (key, state, action, max_goal_dist, scenario_idx, ghost_prob, max_scenario)
vmap_step = jax.jit(jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None, None)))

@jax.jit
def infer_actions(obs):
    mean, _, _ = net_new.apply({"params": params}, obs)
    v_out = jax.nn.sigmoid(mean[:, 0])
    w_out = jax.nn.tanh(mean[:, 1])
    return v_out, w_out

# We need a JIT step to rollout
@jax.jit
def collect_rollout(state, obs, rng_key):
    def _env_step(carry, _):
        curr_state, curr_obs, curr_rng = carry
        curr_rng, action_rng, step_rng = jax.random.split(curr_rng, 3)
        
        # We need environment action
        mean, _, _ = net_new.apply({"params": params}, curr_obs)
        max_v = curr_state.env_state.max_v
        
        v_env = jax.nn.sigmoid(mean[:, 0]) * max_v
        w_env = jax.nn.tanh(mean[:, 1])
        env_actions = jnp.stack([v_env, w_env], axis=-1)
        
        step_keys = jax.random.split(step_rng, num_envs)
        next_obs, next_state, _, _, _ = vmap_step(
            step_keys, curr_state, env_actions, 9.0, -1, 0.0, 10
        )
        return (next_state, next_obs, curr_rng), next_obs
        
    _, rollout_obs = jax.lax.scan(_env_step, (state, obs, rng_key), None, length=rollout_steps)
    return rollout_obs

print("Initializing environments...")
rng, reset_rng = jax.random.split(rng)
reset_keys = jax.random.split(reset_rng, num_envs)
env_obs, env_state = vmap_reset(reset_keys)

print(f"Rolling out {rollout_steps} steps across {num_envs} envs to collect generic states...")
rng, rollout_rng = jax.random.split(rng)
all_obs = collect_rollout(env_state, env_obs, rollout_rng)

# Flatten rollout observations
# Shape from (rollout_steps, num_envs, OBS_SIZE) -> (rollout_steps * num_envs, OBS_SIZE)
all_obs_flat = all_obs.reshape(-1, all_obs.shape[-1])
print(f"Collected {all_obs_flat.shape[0]} state observations.")

# We have ~100k states.
num_states = all_obs_flat.shape[0]

print("Assigning specific Vmax conditions across the state buffer...")
import numpy as np
discrete_vmax_choices = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
max_v_data = np.random.choice(discrete_vmax_choices, size=num_states)
max_v_jax = jnp.array(max_v_data)

# First compute and override v ratio
old_max_v = all_obs_flat[:, 11] * 1.8 + 0.2
old_v = all_obs_flat[:, 9] * jnp.maximum(old_max_v, 1e-3)
new_state_vec_0 = old_v / jnp.maximum(max_v_jax, 1e-3)
all_obs_flat = all_obs_flat.at[:, 9].set(new_state_vec_0)

# Then explicitly inject the new vmax condition into the buffer
all_obs_flat = all_obs_flat.at[:, 11].set((max_v_jax - 0.2) / 1.8)

print("Buffer Formulated!")

print("Running inference on all buffered observations...")
raw_v, raw_w = jax.device_get(infer_actions(all_obs_flat))

# Calculate absolute effective applied linear velocity
eff_v = raw_v * max_v_data
print("Inference done.")

fig, axes = plt.subplots(3, 1, figsize=(10, 16))
sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})

for vmax_val in discrete_vmax_choices:
    mask = max_v_data == vmax_val
    if np.any(mask):
        # Utilizzo di histplot con kde=True, stat="density" per la corretta proporzione, e bins per la risoluzione
        sns.histplot(raw_v[mask], ax=axes[0], label=f"v={vmax_val}", kde=True, stat="density", alpha=0.3, edgecolor="none", bins=30)
        sns.histplot(raw_w[mask], ax=axes[1], label=f"v={vmax_val}", kde=True, stat="density", alpha=0.3, edgecolor="none", bins=30)
        sns.histplot(eff_v[mask], ax=axes[2], label=f"v={vmax_val}", kde=True, stat="density", alpha=0.3, edgecolor="none", bins=30)

axes[0].set_title("PPO Extracted LINEAR Velocity vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[0].set_xlabel("Generated Linear Actuation [sigmoid(raw_v) : 0 to 1]", fontsize=11, labelpad=8)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_xlim(0, 1.0)
axes[0].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

axes[1].set_title("PPO Extracted ANGULAR Velocity vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[1].set_xlabel("Generated Angular Actuation [tanh(raw_w) : -1 to 1]", fontsize=11, labelpad=8)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_xlim(-1.0, 1.0)
axes[1].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

axes[2].set_title("Actually Applied Linear Velocity vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[2].set_xlabel("Applied Linear Velocity [m/s]", fontsize=11, labelpad=8)
axes[2].set_ylabel("Density", fontsize=11)
axes[2].set_xlim(0, 2.0)
axes[2].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

plt.tight_layout()

# Ho modificato il nome del file generato per non sovrascrivere il precedente
plot_path = os.path.abspath(os.path.join(current_dir, "ppo_vmax_dist_buffer_hist.png"))
plt.savefig(plot_path, dpi=300)
print(f"Saved plot to {plot_path}")