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
from jax_wrappers import make_stacked_env
from jax_env_multi import reset_env, step_env

ckpt = os.path.join(project_root, "src/jax_env/checkpoints/ppo_attn_final.msgpack")
_, load_fn, _, _ = _build_ppo_shac()
try:
    params = load_fn(ckpt)
except Exception as e:
    print(f"Error loading {ckpt}: {e}")
    sys.exit()

net_new = EndToEndActorCritic(action_dim=2)
num_envs = 100000
rng = jax.random.PRNGKey(42)

reset_stacked, _ = make_stacked_env(reset_env, step_env, stack_dim=3)
def _reset_with_dist(key):
    return reset_stacked(key, max_goal_dist=9.0, scenario_idx=-1, ghost_prob=0.0)

vmap_reset = jax.jit(jax.vmap(_reset_with_dist))
reset_keys = jax.random.split(rng, num_envs)

print("Creating observations...")
env_obs, _ = vmap_reset(reset_keys)

# Stratify max_v explicitly to discrete values [0.2, 0.5, 1.0, 1.5, 2.0]
print("Assigning specific Vmax conditions...")
discrete_vmax_choices = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
max_v_data = np.random.choice(discrete_vmax_choices, size=num_envs)
max_v_jax = jnp.array(max_v_data)

# Index 11 corresponds to max_v in the flattened stacked observation vector
env_obs = env_obs.at[:, 11].set((max_v_jax - 0.2) / 1.8)
print("Observations formulated.")

@jax.jit
def infer_actions(obs):
    mean, _, _ = net_new.apply({"params": params}, obs)
    v_out = jax.nn.sigmoid(mean[:, 0])
    w_out = jax.nn.tanh(mean[:, 1])
    return v_out, w_out

print("Running inference...")
raw_v, raw_w = jax.device_get(infer_actions(env_obs))

# Calculate absolute effective applied linear velocity
eff_v = raw_v * max_v_data
print("Inference done.")

fig, axes = plt.subplots(3, 1, figsize=(10, 16))
sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})

for vmax_val in discrete_vmax_choices:
    mask = max_v_data == vmax_val
    if np.any(mask):
        sns.kdeplot(raw_v[mask], ax=axes[0], label=f"v={vmax_val}", fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(raw_w[mask], ax=axes[1], label=f"v={vmax_val}", fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(eff_v[mask], ax=axes[2], label=f"v={vmax_val}", fill=True, alpha=0.3, linewidth=2)

axes[0].set_title("PPO Extracted LINEAR Velocity (Network Output) Probability Density vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[0].set_xlabel("Generated Linear Actuation [sigmoid(raw_v) : 0 to 1]", fontsize=11, labelpad=8)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_xlim(0, 1.0)
axes[0].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

axes[1].set_title("PPO Extracted ANGULAR Velocity (Network Output) Probability Density vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[1].set_xlabel("Generated Angular Actuation [tanh(raw_w) : -1 to 1]", fontsize=11, labelpad=8)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_xlim(-1.0, 1.0)
axes[1].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

axes[2].set_title("True EFFECTIVE Applied Linear Velocity Probability Density vs Assigned 'Vmax'", fontsize=13, pad=10)
axes[2].set_xlabel("Applied Linear Velocity [m/s]", fontsize=11, labelpad=8)
axes[2].set_ylabel("Density", fontsize=11)
axes[2].set_xlim(0, 2.0)
axes[2].legend(title="Assigned Robot Vmax", fontsize=10, title_fontsize=11)

plt.tight_layout()

plot_path = os.path.abspath(os.path.join(current_dir, "ppo_vmax_dist.png"))
plt.savefig(plot_path, dpi=300)
print(f"Saved plot to {plot_path}")
