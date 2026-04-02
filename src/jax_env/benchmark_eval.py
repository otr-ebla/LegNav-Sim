"""
benchmark_eval.py — High-Speed Evaluation Dashboard
===================================================
Evaluates RL models across all 7 scenarios and generates a visual dashboard.

OOM FIX: Removed the nested vmap over (N_SCENARIOS × N_SPEEDS) that tried to
allocate ~5.4 GiB for a single compiled graph. Evaluation is now a sequential
Python loop over (scenario, speed) pairs; each iteration dispatches a single
vmap over N_ENVS environments, which is the actual parallelism budget the GPU
can handle. Compile time drops to seconds and VRAM stays under 2 GiB.

TRAINING CURVES: A 9th panel plots episode reward over training steps, loaded
from CSV logs written by jax_ppo.py / SACjax.py / TQCjac.py. If a log file is
missing the panel shows a "no data" notice instead of crashing.
"""

import os
import sys
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # suppress pygame banner
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import jax_env
jax_env.USE_LEGS = True

from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT, MAX_STEPS
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState
from jax_legs import LEG_RADIUS
from jax_network import EndToEndActorCritic
from SACjax import ObsEncoder as _SACEncoder, ActorHead as _SACHead
import flax.linen as nn
from TQCjac import TQCActorNetwork as _TQCActorNetwork

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_SIZE   = 666
ACTION_DIM = 2
N_ENVS     = 4096   # parallelism per (scenario, speed) cell — fits in VRAM

MAX_V_TESTS = [0.2, 0.5, 0.75, 1.0, 1.33, 1.66, 2.0]
N_SCENARIOS = 7
N_SPEEDS    = len(MAX_V_TESTS)

# Training-curve CSV paths written by each trainer
TRAINING_LOG_PATHS = {
    "PPO": "checkpoints/ppo_training_log.csv",
    "SAC": "checkpoints_sac/sac_training_log.csv",
    "TQC": "checkpoints_tqc/tqc_training_log.csv",
}

_ppo_net     = EndToEndActorCritic(action_dim=ACTION_DIM)
_sac_enc     = _SACEncoder()
_sac_head    = _SACHead(action_dim=ACTION_DIM)
_tqc_net = _TQCActorNetwork()  # monolithic — ObsEncoder_0/ inside actor_params

# _rollout_body calls net_apply_fn({"params": params}, obs) for ALL networks.
# Both _sac_apply and _tqc_apply mirror that Flax convention:
#   variables["params"] = {"enc": enc_params, "head": head_params}
def _sac_apply(variables, obs):
    p    = variables["params"]
    feat = _sac_enc.apply({"params": p["enc"]}, obs)
    return _sac_head.apply({"params": p["head"]}, feat)



# ── Action Squashing ──────────────────────────────────────────────────────────
def _squash_ppo(mean, max_v):
    v = jax.nn.sigmoid(mean[..., 0]) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)

def _squash_sac_tqc(mean, max_v):
    t = jnp.tanh(mean)
    v = (t[..., 0] + 1.0) * 0.5 * max_v
    w = t[..., 1]
    return jnp.stack([v, w], axis=-1)

# ── Environment Wrappers ──────────────────────────────────────────────────────
@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    base_obs, base_state = reset_env(key, min_dist, scen_idx)
    pose      = base_obs[0:3]
    state_vec = base_obs[3:12]
    lidar     = base_obs[12:]

    base_state = base_state.replace(max_v=target_max_v)
    new_state_vec = jnp.array([
        0.0, 0.0, (target_max_v - 0.2) / 1.8,
        state_vec[3], state_vec[4],
        state_vec[5], state_vec[6], state_vec[7], state_vec[8],
    ])

    lidar_stack = jnp.tile(lidar[None, :], (3, 1))
    pose_stack  = jnp.tile(pose[None, :],  (3, 1))
    stacked_state = StackedEnvState(
        env_state=base_state, lidar_stack=lidar_stack, pose_stack=pose_stack
    )
    flat_obs = jnp.concatenate([pose_stack.flatten(), new_state_vec, lidar_stack.flatten()])
    return flat_obs, stacked_state


@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)
    new_pose      = base_obs[0:3]
    new_state_vec = base_obs[3:12]
    new_lidar     = base_obs[12:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)
    new_stacked_state = StackedEnvState(
        env_state=new_base_state, lidar_stack=new_lidar_stack, pose_stack=new_pose_stack
    )
    flat_obs = jnp.concatenate([new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()])
    return flat_obs, new_stacked_state, reward, done, info


# ── Core Evaluation Kernel ────────────────────────────────────────────────────
# Each network type gets its OWN JIT-compiled function so that `params` always
# belongs to exactly one architecture.  Applying all three networks to the same
# params (as the previous version did with jnp.where) causes Flax to look for
# kernel shapes that don't exist in the checkpoint → ScopeParamNotFoundError.
#
# All three kernels share the same rollout body; only the forward-pass line
# differs.  Each is compiled once on first call and reused for all 49 cells.

def _rollout_body(net_apply_fn, squash_fn, params, scen_idx, target_max_v, rng_key):
    """Inner rollout used by all three per-network eval functions."""
    reset_keys = jax.random.split(rng_key, N_ENVS)
    obs, state = jax.vmap(dynamic_reset_stacked, in_axes=(0, None, None, None))(
        reset_keys, 3.0, scen_idx, target_max_v
    )

    init_dist = jnp.sqrt(
        (state.env_state.goal_x - state.env_state.x) ** 2 +
        (state.env_state.goal_y - state.env_state.y) ** 2
    )

    carry = (
        state, obs,
        jnp.zeros(N_ENVS), jnp.zeros(N_ENVS),   # v_p, av_p
        jnp.zeros(N_ENVS), jnp.zeros(N_ENVS),   # w_p, aw_p
        jnp.zeros(N_ENVS),                        # path_len
        jnp.full(N_ENVS, 100.0),                  # min_human_dist
        jnp.ones(N_ENVS, dtype=jnp.bool_),        # active
        jnp.zeros(N_ENVS),                        # yield_zone_steps (denominator)
        jnp.zeros(N_ENVS),                        # yield_comply_steps (numerator)
    )

    human_r = LEG_RADIUS if jax_env.USE_LEGS else PEOPLE_RADIUS

    YIELD_DIST = 1.5   # m  — must match jax_env_multi.py YIELD_DIST
    YIELD_FOV  = 1.57  # rad — forward 90° each side

    def _step(carry, step_idx):
        state, obs, v_p, av_p, w_p, aw_p, pl, mhd, active, yz_steps, yc_steps = carry
        k_step = jax.random.fold_in(rng_key, step_idx)

        raw_out = net_apply_fn({"params": params}, obs)
        # PPO returns (mean, logstd, value); SAC/TQC return (mean, log_std)
        mean = raw_out[0]
        action = jax.vmap(squash_fn)(mean, state.env_state.max_v)

        step_keys = jax.random.split(k_step, N_ENVS)
        next_obs, next_state, _, done, info = jax.vmap(step_stacked_headless)(
            step_keys, state, action
        )

        v  = next_state.env_state.v
        w  = next_state.env_state.w
        av = (v - v_p) / DT
        aw = (w - w_p) / DT

        jerk_v = jnp.where(active, jnp.abs((av - av_p) / DT), 0.0)
        jerk_w = jnp.where(active, jnp.abs((aw - aw_p) / DT), 0.0)
        pl     = pl + jnp.where(active, v * DT, 0.0)

        ch  = info["closest_human"] - ROBOT_RADIUS - human_r
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"] & active
        c  = info["collision"]    & active
        pc = info["passive_col"]  & active

        # ── Yielding score accumulators ───────────────────────────────────────
        # next_state carries batched tensors: x/y/theta shape (N_ENVS,),
        # people shape (N_ENVS, NUM_PEOPLE, 11).  All ops must be explicitly
        # batched — no implicit broadcasting across the env dimension.
        ppl      = next_state.env_state.people          # (N_ENVS, NUM_PEOPLE, 11)
        px_all   = ppl[:, :, 0]                         # (N_ENVS, NUM_PEOPLE)
        py_all   = ppl[:, :, 1]
        rx_b     = next_state.env_state.x[:, None]      # (N_ENVS, 1)
        ry_b     = next_state.env_state.y[:, None]
        rth_b    = next_state.env_state.theta[:, None]
        dp_x     = px_all - rx_b                        # (N_ENVS, NUM_PEOPLE)
        dp_y     = py_all - ry_b
        dists_p  = jnp.sqrt(dp_x**2 + dp_y**2 + 1e-8)
        rel_ang  = jnp.arctan2(dp_y, dp_x) - rth_b
        rel_ang  = (rel_ang + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        active_p = ppl[:, :, 10] >= 0.0                # (N_ENVS, NUM_PEOPLE)
        in_yz    = (dists_p < YIELD_DIST) & (jnp.abs(rel_ang) < YIELD_FOV) & active_p
        any_in_yz = jnp.any(in_yz, axis=1)             # (N_ENVS,)

        robot_stopped = v <= 0.1                        # (N_ENVS,)

        new_yz_steps = yz_steps + jnp.where(active & any_in_yz, 1.0, 0.0)
        new_yc_steps = yc_steps + jnp.where(active & any_in_yz & robot_stopped, 1.0, 0.0)

        step_data   = (active, done, g, c, pc, jerk_v, jerk_w, ch, v)
        next_active = active & ~done
        return (next_state, next_obs, v, av, w, aw, pl, mhd, next_active,
                new_yz_steps, new_yc_steps), step_data

    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    _, _, _, _, _, _, final_pl, final_mhd, _, final_yz, final_yc = final_carry
    active_mask, _, goals, cols, pcols, jerks_v, jerks_w, step_dists, step_vs = step_data

    ep_lens = active_mask.sum(axis=0)
    ep_goal = goals.any(axis=0)
    ep_col  = cols.any(axis=0)
    ep_pcol = pcols.any(axis=0)

    act_col  = ep_col  & ~ep_pcol & ~ep_goal
    pass_col = ep_pcol & ~ep_goal
    tmo      = ~ep_goal & ~ep_col & ~ep_pcol

    avg_jerk = (jerks_v.sum(axis=0) + jerks_w.sum(axis=0)) / jnp.maximum(ep_lens, 1)
    spl      = ep_goal * (init_dist / jnp.maximum(final_pl, init_dist))
    time_g   = jnp.where(ep_goal, ep_lens * DT, jnp.nan)

    # Yielding score: fraction of yield-zone steps where robot was stopped.
    # NaN when the robot never encountered anyone in its yield zone.
    yield_score = jnp.where(
        final_yz > 0,
        final_yc / final_yz,
        jnp.nan,
    )

    step_dists = jnp.where(active_mask, step_dists, jnp.nan)  # (T, N)
    step_vs    = jnp.where(active_mask, step_vs,    jnp.nan)  # (T, N)

    return {
        "success":     ep_goal.astype(jnp.float32),
        "act_col":     act_col.astype(jnp.float32),
        "pass_col":    pass_col.astype(jnp.float32),
        "timeout":     tmo.astype(jnp.float32),
        "spl":         spl,
        "jerk":        avg_jerk,
        "min_dist":    final_mhd,
        "time":        time_g,
        "yield_score": yield_score,
        "step_dists":  step_dists,   # (MAX_STEPS, N_ENVS)
        "step_vs":     step_vs,      # (MAX_STEPS, N_ENVS)
    }


# One JIT kernel per network — params stay within their own architecture scope.
@jax.jit
def evaluate_cell_ppo(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_ppo_net.apply, _squash_ppo, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_sac(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_sac_apply, _squash_sac_tqc, params,
                         scen_idx, target_max_v, rng_key)

@jax.jit
def evaluate_cell_tqc(params, scen_idx: int, target_max_v: float, rng_key):
    return _rollout_body(_tqc_net.apply, _squash_sac_tqc, params,
                         scen_idx, target_max_v, rng_key)

_EVAL_FN = {"PPO": evaluate_cell_ppo, "SAC": evaluate_cell_sac, "TQC": evaluate_cell_tqc}


# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_checkpoint_safe(path, p_name, dummy_params):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    
    bundle = flax.serialization.msgpack_restore(raw)
    
    # PPO è stato addestrato e salvato come un singolo blocco unificato
    if p_name == "PPO":
        if "actor_params" in bundle: return bundle["actor_params"]
        if "params" in bundle: return bundle["params"]
        return bundle
        
    # SAC: checkpoint salvato come due pytree separati da SACjax.py.
    # Ricostruiamo un dict con chiavi fisse "enc"/"head" usate da _sac_apply.
    # SAC: saved as actor_enc_params + actor_head_params by SACjax.py
    if p_name == "SAC":
        if "actor_enc_params" in bundle and "actor_head_params" in bundle:
            return {"enc": bundle["actor_enc_params"], "head": bundle["actor_head_params"]}
        raise KeyError(f"SAC checkpoint unexpected keys: {list(bundle.keys())}")

    # TQC: monolithic checkpoint — actor_params has ObsEncoder_0/ + Dense_0/1.
    if p_name == "TQC":
        if "actor_params" in bundle:
            return bundle["actor_params"]
        raise KeyError(f"TQC checkpoint unexpected keys: {list(bundle.keys())}")
            
    return bundle


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]

    # 1. Generiamo i parametri fittizi per scoprire la struttura esatta richiesta da Flax
    rng_init = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    ppo_dummy = _ppo_net.init(rng_init, dummy_obs)["params"]

    # 2. Carichiamo mappando le vecchie chiavi sulle nuove strutture
    raw_policies = {
        "PPO": load_checkpoint_safe("checkpoints/ppo_model_best.msgpack", "PPO", ppo_dummy),
        "SAC": load_checkpoint_safe("checkpoints_sac/sac_best.msgpack", "SAC", None),
        "TQC": load_checkpoint_safe("checkpoints_tqc/tqc_best.msgpack", "TQC", None),
    }
    
    policies = {
        name: jax.device_put(p, gpu)
        for name, p in raw_policies.items()
        if p is not None
    }

    if not policies:
        print("No valid checkpoints found. Please train a model first.")
        return

    rng = jax.random.PRNGKey(42)

    # ── Warm-up: compile only the kernel(s) we actually have checkpoints for ──
    print("Compiling evaluation kernels (one per available policy, ~30 s each)...")
    for p_name, params in policies.items():
        rng, k_warmup = jax.random.split(rng)
        jax.block_until_ready(_EVAL_FN[p_name](params, 0, 1.0, k_warmup))
        print(f"  {p_name} kernel compiled.")
    print()

    # ── Sequential evaluation loop ────────────────────────────────────────────
    results      = []
    scatter_data = {}   # {policy_name: (dists_array, vs_array)}
    scen_names = {
        0: "Random", 1: "Parallel", 2: "Perpend", 3: "Circular",
        4: "Bottleneck", 5: "Intersect", 6: "Groups",
    }

    print("Executing evaluation grid (sequential cells, parallel envs)...")
    for p_name, params in policies.items():
        eval_fn  = _EVAL_FN[p_name]
        t_policy = time.time()
        _sd_list, _sv_list = [], []
        for si in range(N_SCENARIOS):
            for vi, v_max in enumerate(MAX_V_TESTS):
                rng, cell_rng = jax.random.split(rng)
                cell = jax.device_get(jax.block_until_ready(
                    eval_fn(params, si, float(v_max), cell_rng)
                ))
                sd = cell["step_dists"].ravel()
                sv = cell["step_vs"].ravel()
                ok = np.isfinite(sd) & np.isfinite(sv)
                _sd_list.append(sd[ok])
                _sv_list.append(sv[ok])
                for i in range(N_ENVS):
                    results.append({
                        "Policy":        p_name,
                        "Scenario":      si,
                        "Max_V":         v_max,
                        "Success":       cell["success"][i],
                        "Active Col":    cell["act_col"][i],
                        "Passive Col":   cell["pass_col"][i],
                        "Timeout":       cell["timeout"][i],
                        "SPL":           cell["spl"][i],
                        "Jerk":          cell["jerk"][i],
                        "Min Dist":      cell["min_dist"][i],
                        "Time to Goal":  cell["time"][i],
                        "Yield Score":   cell["yield_score"][i],
                    })
        scatter_data[p_name] = (
            np.concatenate(_sd_list),
            np.concatenate(_sv_list),
        )
        print(f"  {p_name} done in {time.time() - t_policy:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv("evaluation_raw_data.csv", index=False)
    print("Saved evaluation_raw_data.csv\n")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    print("Generating dashboard...")
    sns.set_theme(style="whitegrid", palette="muted")

    # Layout: 3 rows × 4 cols = 12 panels
    #   Row 1: Success, Active Col, Passive Col, Timeout  — all vs Max_V
    #   Row 2: SPL, Time to Goal, Safety Margin, Jerk
    #   Row 3: Yield Score vs Max_V, Success by Scenario, Overall Outcomes, Training Curves
    fig = plt.figure(figsize=(36, 21))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard",
                 fontsize=28, weight="bold")

    R, C = 3, 4   # rows, cols

    # ── Row 1: outcome rates vs Max_V ────────────────────────────────────────

    # 1. Success vs Speed
    v_suc_df = df.groupby(["Max_V","Policy"])["Success"].mean().reset_index()
    v_suc_df["Success"] *= 100
    ax1 = plt.subplot(R, C, 1)
    sns.lineplot(data=v_suc_df, x="Max_V", y="Success", hue="Policy",
                 marker="o", linewidth=3, markersize=8, ax=ax1)
    ax1.set_title("Success Rate vs. Max Speed", fontsize=13)
    ax1.set_xticks(MAX_V_TESTS)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Max Linear Speed (m/s)")
    ax1.set_ylabel("Rate (%)")

    # 2. Active Collisions vs Speed
    v_acol_df = df.groupby(["Max_V","Policy"])["Active Col"].mean().reset_index()
    v_acol_df["Active Col"] *= 100
    ax2 = plt.subplot(R, C, 2)
    sns.lineplot(data=v_acol_df, x="Max_V", y="Active Col", hue="Policy",
                 marker="X", linewidth=3, markersize=8, ax=ax2)
    ax2.set_title("Active Collisions vs. Max Speed", fontsize=13)
    ax2.set_xticks(MAX_V_TESTS)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Max Linear Speed (m/s)")
    ax2.set_ylabel("Rate (%)")

    # 3. Passive Collisions vs Speed
    v_pcol_df = df.groupby(["Max_V","Policy"])["Passive Col"].mean().reset_index()
    v_pcol_df["Passive Col"] *= 100
    ax3 = plt.subplot(R, C, 3)
    sns.lineplot(data=v_pcol_df, x="Max_V", y="Passive Col", hue="Policy",
                 marker="o", linewidth=3, markersize=8, ax=ax3)
    ax3.set_title("Passive Collisions vs. Max Speed", fontsize=13)
    ax3.set_xticks(MAX_V_TESTS)
    ax3.set_ylim(0, 100)
    ax3.set_xlabel("Max Linear Speed (m/s)")
    ax3.set_ylabel("Rate (%)")

    # 4. Timeout vs Speed
    v_tmo_df = df.groupby(["Max_V","Policy"])["Timeout"].mean().reset_index()
    v_tmo_df["Timeout"] *= 100
    ax4 = plt.subplot(R, C, 4)
    sns.lineplot(data=v_tmo_df, x="Max_V", y="Timeout", hue="Policy",
                 marker="s", linewidth=3, markersize=8, ax=ax4)
    ax4.set_title("Timeout Rate vs. Max Speed", fontsize=13)
    ax4.set_xticks(MAX_V_TESTS)
    ax4.set_ylim(0, 100)
    ax4.set_xlabel("Max Linear Speed (m/s)")
    ax4.set_ylabel("Rate (%)")

    # ── Row 2: quality metrics ────────────────────────────────────────────────
    suc = df[df["Success"] == 1.0]

    # 5. SPL (successful episodes only)
    ax5 = plt.subplot(R, C, 5)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Success-weighted Path Length (SPL)", fontsize=13)
    ax5.set_ylabel("SPL")

    # 6. Time to Goal (successful episodes only)
    ax6 = plt.subplot(R, C, 6)
    sns.boxplot(data=suc, x="Policy", y="Time to Goal", hue="Policy",
                ax=ax6, showfliers=False)
    ax6.set_title("Time to Reach Goal (seconds)", fontsize=13)
    ax6.set_ylabel("seconds")

    # 7. Safety Margin (Min Human Dist) vs Speed
    ax7 = plt.subplot(R, C, 7)
    sns.lineplot(data=df, x="Max_V", y="Min Dist", hue="Policy",
                 marker="^", linewidth=3, markersize=8, ax=ax7)
    ax7.set_title("Safety Margin vs. Max Speed", fontsize=13)
    ax7.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Collision Threshold")
    ax7.set_xticks(MAX_V_TESTS)
    ax7.set_xlabel("Max Linear Speed (m/s)")
    ax7.set_ylabel("Min Human Distance (m)")
    ax7.legend(fontsize=9)

    # 8. Kinematic Jerk
    ax8 = plt.subplot(R, C, 8)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy", ax=ax8, showfliers=False)
    ax8.set_title("Average Kinematic Jerk (Smoothness)", fontsize=13)
    ax8.set_ylabel("Jerk (m/s³ + rad/s³)")

    # ── Row 3: yielding, scenario breakdown, overall, training ───────────────

    # 9. Yielding Score vs Speed
    # YS = fraction of yield-zone steps where robot was stopped (v ≤ 0.1 m/s).
    # NaN episodes (no yield-zone encounters) are excluded from the mean.
    v_ys_df = df.groupby(["Max_V","Policy"])["Yield Score"].mean().reset_index()
    ax9 = plt.subplot(R, C, 9)
    sns.lineplot(data=v_ys_df, x="Max_V", y="Yield Score", hue="Policy",
                 marker="D", linewidth=3, markersize=8, ax=ax9)
    ax9.set_title("Yielding Score vs. Max Speed", fontsize=13)
    ax9.set_xticks(MAX_V_TESTS)
    ax9.set_ylim(0, 1)
    ax9.set_xlabel("Max Linear Speed (m/s)")
    ax9.set_ylabel("Yield Compliance (0–1)")
    ax9.axhline(1.0, color="green", linestyle=":", alpha=0.4)
    ax9.axhline(0.5, color="gray",  linestyle=":", alpha=0.4)

    # 10. Success by Scenario
    scen_names = {
        0: "Random", 1: "Parallel", 2: "Perpend", 3: "Circular",
        4: "Bottleneck", 5: "Intersect", 6: "Groups",
    }
    scen_df = df.groupby(["Scenario","Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(scen_names)
    ax10 = plt.subplot(R, C, 10)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax10)
    ax10.set_title("Success Rate by Layout Topology", fontsize=13)
    ax10.set_xticklabels(ax10.get_xticklabels(), rotation=30, ha="right")
    ax10.set_ylim(0, 100)
    ax10.set_ylabel("Success Rate (%)")

    # 11. Overall Episode Outcomes (bar chart)
    rate_df   = df.groupby("Policy")[["Success","Active Col","Passive Col","Timeout"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100
    ax11 = plt.subplot(R, C, 11)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax11)
    ax11.set_title("Overall Episode Outcomes (%)", fontsize=13)
    ax11.set_ylim(0, 100)
    ax11.set_ylabel("Rate (%)")

    # 12. Training Curves — episode reward over environment steps
    ax12 = plt.subplot(R, C, 12)
    any_log = False
    POLICY_COLORS = {"PPO": "#4C72B0", "SAC": "#DD8452", "TQC": "#55A868"}
    for p_name, log_path in TRAINING_LOG_PATHS.items():
        if os.path.exists(log_path):
            try:
                log_df = pd.read_csv(log_path)
                x_millions = log_df["step"] / 1e6
                w = max(5, len(log_df) // 30)
                smoothed = log_df["mean_ep_reward"].rolling(window=w, min_periods=1).mean()
                ax12.plot(x_millions, log_df["mean_ep_reward"],
                          alpha=0.18, linewidth=1, color=POLICY_COLORS.get(p_name))
                ax12.plot(x_millions, smoothed,
                          label=p_name, linewidth=2.5, color=POLICY_COLORS.get(p_name))
                any_log = True
            except Exception as e:
                print(f"  Warning: could not read {log_path}: {e}")

    if any_log:
        ax12.set_xlabel("Environment Steps (millions)", fontsize=11)
        ax12.set_ylabel("Mean Episode Reward", fontsize=11)
        ax12.set_title("Episode Reward During Training", fontsize=13)
        ax12.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax12.legend(fontsize=10)
    else:
        ax12.text(0.5, 0.5,
                  "No training logs found.\nRun trainers to generate\n"
                  "checkpoints/*/training_log.csv",
                  ha="center", va="center", transform=ax12.transAxes,
                  fontsize=11, color="gray")
        ax12.set_title("Episode Reward During Training", fontsize=13)
        ax12.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("Saved 'Evaluation_Dashboard.png'")

    _plot_proximity_speed(scatter_data)


def _plot_proximity_speed(scatter_data: dict):
    """
    One panel per policy: raw scatter of every active timestep across all
    episodes, scenarios and speed caps.
    x = surface distance to nearest human (m)  — negative means collision
    y = robot linear speed at that moment (m/s)
    """
    policies = list(scatter_data.keys())
    if not policies:
        return

    POLICY_COLORS = {"PPO": "#4C72B0", "SAC": "#DD8452", "TQC": "#55A868"}
    MAX_PTS = 120_000   # sub-sample per policy so the PNG stays manageable

    fig, axes = plt.subplots(
        1, len(policies),
        figsize=(8 * len(policies), 7),
        sharey=True, squeeze=False
    )
    fig.suptitle(
        "Linear Speed vs. Distance to Nearest Human\n"
        "(all active timesteps · all scenarios · all speed caps)",
        fontsize=15, weight="bold"
    )

    for col, p_name in enumerate(policies):
        ax    = axes[0, col]
        color = POLICY_COLORS.get(p_name, "#888888")
        dists, vs = scatter_data[p_name]

        # Discard anything outside plausible physical range
        ok    = (dists >= -0.3) & (dists <= 4.0) & (vs >= 0.0) & (vs <= 2.05)
        dists = dists[ok]
        vs    = vs[ok]
        n     = len(dists)

        # Sub-sample for rendering
        if n > MAX_PTS:
            rng_sc = np.random.default_rng(0)
            idx    = rng_sc.choice(n, MAX_PTS, replace=False)
            dists, vs = dists[idx], vs[idx]

        ax.scatter(
            dists, vs,
            s=1.5, alpha=0.08, color=color,
            rasterized=True
        )

        # Reference lines
        ax.axvline(0.0, color="red",    lw=1.5, ls="--", alpha=0.8,
                   label="Collision (0 m)")
        ax.axvline(0.5, color="orange", lw=1.2, ls=":",  alpha=0.7,
                   label="Comfort (0.5 m)")

        ax.set_xlim(-0.2, 3.8)
        ax.set_ylim(-0.05, 2.1)
        ax.set_xlabel("Surface distance to nearest human (m)", fontsize=12)
        if col == 0:
            ax.set_ylabel("Linear speed (m/s)", fontsize=12)
        ax.set_title(f"{p_name}  —  {n:,} pts", fontsize=13, weight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("proximity_speed_scatter.png", dpi=180)
    print("Saved 'proximity_speed_scatter.png'")
    plt.close(fig)


if __name__ == "__main__":
    main()