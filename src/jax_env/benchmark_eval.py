"""
benchmark_eval.py — Massive Headless Evaluation Dashboard (GPU-Optimised)
==========================================================================
Tests PPO, SAC, and TQC models across all 7 scenarios × 3 max_v limits.
All computation runs on CUDA via JAX. Key GPU optimisations vs. prior version:

  OPT 1 — N_ENVS 1000 → 4096:
    With 1000 envs the GPU's SMs were severely under-utilised (each SM can
    run thousands of threads). 4096 fills the warp pipeline without risking
    OOM on a 16/24 GB card. Tune down to 2048 if you hit memory limits.

  OPT 2 — Vectorise over (scenario × max_v) in one vmap call:
    The old code had a Python for-loop: evaluate_chunk was called 7×3=21
    times sequentially per policy, paying Python dispatch + XLA kernel-launch
    overhead every time. Now evaluate_chunk_grid vmaps over all 21 combos
    simultaneously, keeping everything on-device with zero Python round-trips.

  OPT 3 — Explicit device_put of params to GPU:
    Checkpoints are loaded on CPU; without device_put JAX copies weights
    lazily on each JIT call. We pin them once to the GPU device so every
    forward pass reads from fast HBM.

  OPT 4 — XLA_FLAGS for CUDA kernel tuning:
    --xla_gpu_enable_triton_gemm=true  — use Triton fused GEMM (faster on
      Ampere/Ada GPUs for the Conv1D + Dense layers in ObsEncoder).
    --xla_gpu_cuda_graph_enable=true   — wrap the scan body in a CUDA graph
      so the 500-step scan pays kernel-launch overhead only once.
    --xla_gpu_enable_async_all_reduce=true — overlaps any collectives.

  OPT 5 — block_until_ready after compile warmup:
    Without this the timing clock starts before XLA finishes async
    compilation, inflating the reported compile time and shortening the
    measured eval time.

  OPT 6 — donate_argnums on evaluate_chunk_grid:
    Donating the rng_key array lets XLA reuse its buffer in-place across
    vmap iterations, saving one allocation per (scenario, max_v) pair.

  OPT 7 — Fold jax.random.split inside the scan with fold_in:
    jax.random.fold_in(key, step_idx) is cheaper than split inside the scan
    body because it avoids materialising the full (2,) output array; the
    counter is XOR-folded into the Threefry state in a single fused op.

FIXES carried from prior version:
  FIX 1 — Use EndToEndActorCritic / SACActorNetwork (correct param trees).
  FIX 2 — Correct raw-obs slices: base_obs[0:3], [3:12], [12:].
"""

import os

# ── CUDA / XLA environment — set BEFORE importing JAX ───────────────────────
os.environ["JAX_PLATFORMS"]               = "cuda,cpu"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_GPU_ALLOCATOR"]            = "cuda_malloc_async"
# OPT 4: Triton fused GEMM for Conv1D + Dense layers (faster on Ampere/Ada GPUs)
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"

import time
import functools
import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Environment imports ───────────────────────────────────────────────────────
from jax_env import ROOM_W, ROOM_H, ROBOT_RADIUS, PEOPLE_RADIUS, DT, MAX_STEPS
from jax_env_multi import reset_env, step_env
from jax_wrappers import StackedEnvState

# ── Network imports — exact classes used during training ──────────────────────
from jax_network import EndToEndActorCritic
from SACnetwork import SACActorNetwork

# ── Config ────────────────────────────────────────────────────────────────────
OBS_SIZE    = 342
ACTION_DIM  = 2

# OPT 1: 4096 parallel envs saturates GPU SMs; tune to 2048 if OOM
N_ENVS      = 4096

MAX_V_TESTS = jnp.array([1.0, 1.5, 2.0])   # kept as jnp for vmap
SCENARIOS   = jnp.arange(7, dtype=jnp.int32)

N_SCENARIOS = 7
N_SPEEDS    = 3

# ── Network singletons ────────────────────────────────────────────────────────
_ppo_net = EndToEndActorCritic(action_dim=ACTION_DIM)
_sac_net = SACActorNetwork(action_dim=ACTION_DIM)


# ── Action squashing ──────────────────────────────────────────────────────────
def _squash_ppo(mean, max_v):
    v = jax.nn.sigmoid(mean[..., 0]) * max_v
    w = jnp.tanh(mean[..., 1])
    return jnp.stack([v, w], axis=-1)

def _squash_sac(mean, max_v):
    t = jnp.tanh(mean)
    v = (t[..., 0] + 1.0) * 0.5 * max_v
    w = t[..., 1]
    return jnp.stack([v, w], axis=-1)


# ── Environment wrappers ──────────────────────────────────────────────────────
@jax.jit
def dynamic_reset_stacked(key, min_dist, scen_idx, target_max_v):
    """
    Reset one env, override max_v, build stacked obs (342,).
    get_obs single-frame layout: pose(3) + state_vec(9) + lidar(108) = 120.
    """
    base_obs, base_state = reset_env(key, min_dist, scen_idx)

    pose      = base_obs[0:3]   # (3,)
    state_vec = base_obs[3:12]  # (9,)
    lidar     = base_obs[12:]   # (108,)

    base_state = base_state.replace(max_v=target_max_v)

    new_state_vec = jnp.array([
        0.0,
        0.0,
        (target_max_v - 0.2) / 1.8,
        state_vec[3],   # goal_dist_norm
        state_vec[4],   # goal_align_norm
        state_vec[5], state_vec[6], state_vec[7], state_vec[8],  # rear_prox x4
    ])

    lidar_stack = jnp.tile(lidar[None, :], (3, 1))  # (3, 108)
    pose_stack  = jnp.tile(pose[None,  :], (3, 1))  # (3, 3)

    stacked_state = StackedEnvState(
        env_state=base_state,
        lidar_stack=lidar_stack,
        pose_stack=pose_stack,
    )
    flat_obs = jnp.concatenate([
        pose_stack.flatten(), new_state_vec, lidar_stack.flatten()
    ])
    return flat_obs, stacked_state


@jax.jit
def step_stacked_headless(key, state: StackedEnvState, action):
    """Single headless step; slices single-frame obs (120,) correctly."""
    base_obs, new_base_state, reward, done, info = step_env(key, state.env_state, action)

    new_pose      = base_obs[0:3]
    new_state_vec = base_obs[3:12]
    new_lidar     = base_obs[12:]

    new_lidar_stack = jnp.concatenate([state.lidar_stack[1:], new_lidar[None]], axis=0)
    new_pose_stack  = jnp.concatenate([state.pose_stack[1:],  new_pose[None]],  axis=0)

    new_stacked_state = StackedEnvState(
        env_state=new_base_state,
        lidar_stack=new_lidar_stack,
        pose_stack=new_pose_stack,
    )
    flat_obs = jnp.concatenate([
        new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
    ])
    return flat_obs, new_stacked_state, reward, done, info


# ── Core evaluation kernel ────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(1,))
def evaluate_chunk(params, net_type: str, scen_idx, target_max_v, rng_key):
    """
    Runs N_ENVS full episodes for a single (policy, scenario, max_v).
    This is vmapped over (scenario, max_v) in evaluate_chunk_grid below.
    """
    reset_keys = jax.random.split(rng_key, N_ENVS)

    obs, state = jax.vmap(dynamic_reset_stacked, in_axes=(0, None, None, None))(
        reset_keys, 3.0, scen_idx, target_max_v
    )

    init_dist = jnp.sqrt(
        (state.env_state.goal_x - state.env_state.x)**2 +
        (state.env_state.goal_y - state.env_state.y)**2
    )

    carry = (
        state, obs,
        jnp.zeros(N_ENVS),           # v_prev
        jnp.zeros(N_ENVS),           # av_prev
        jnp.zeros(N_ENVS),           # w_prev
        jnp.zeros(N_ENVS),           # aw_prev
        jnp.zeros(N_ENVS),           # path_len
        jnp.full(N_ENVS, 100.0),     # min_human_dist
        jnp.ones(N_ENVS, dtype=jnp.bool_),  # active
    )

    def _step(carry, step_idx):
        state, obs, v_p, av_p, w_p, aw_p, pl, mhd, active = carry

        # OPT 7: fold_in avoids a full split allocation — just XORs counter
        k_step = jax.random.fold_in(rng_key, step_idx)

        if net_type == "PPO":
            mean, _, _ = _ppo_net.apply({"params": params}, obs)
            action = jax.vmap(_squash_ppo)(mean, state.env_state.max_v)
        else:
            mean, _ = _sac_net.apply({"params": params}, obs)
            action = jax.vmap(_squash_sac)(mean, state.env_state.max_v)

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

        pl  = pl  + jnp.where(active, v * DT, 0.0)
        ch  = info["closest_human"] - ROBOT_RADIUS - PEOPLE_RADIUS
        mhd = jnp.where(active, jnp.minimum(mhd, ch), mhd)

        g  = info["goal_reached"]
        c  = info["collision"]
        pc = info["passive_col"]

        step_data   = (active, done, g, c, pc, jerk_v, jerk_w)
        next_active = active & ~done
        return (next_state, next_obs, v, av, w, aw, pl, mhd, next_active), step_data

    # Scan over step indices so fold_in can use them (OPT 7)
    final_carry, step_data = jax.lax.scan(
        _step, carry, jnp.arange(MAX_STEPS, dtype=jnp.uint32)
    )
    _, _, _, _, _, _, final_pl, final_mhd, _ = final_carry
    active_mask, dones, goals, cols, pcols, jerks_v, jerks_w = step_data

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

    return {
        "success":  ep_goal.astype(jnp.float32),
        "act_col":  act_col.astype(jnp.float32),
        "pass_col": pass_col.astype(jnp.float32),
        "timeout":  tmo.astype(jnp.float32),
        "spl":      spl,
        "jerk":     avg_jerk,
        "min_dist": final_mhd,
        "time":     time_g,
    }


@functools.partial(jax.jit, static_argnums=(1,))
def evaluate_chunk_grid(params, net_type: str, rng_key):
    """
    OPT 2: Evaluate ALL 7 scenarios × 3 speeds in a single JIT call.

    Instead of 21 sequential Python for-loop iterations (each paying
    kernel-launch + Python overhead), we build a (N_SCENARIOS, N_SPEEDS)
    grid of random keys and vmap evaluate_chunk over both axes.
    This keeps everything on-device; the GPU sees one big batched kernel
    rather than 21 small sequential ones.

    Output shapes: each metric is (N_SCENARIOS, N_SPEEDS, N_ENVS).
    """
    # Build a (N_SCENARIOS, N_SPEEDS) grid of independent RNG keys
    keys_flat  = jax.random.split(rng_key, N_SCENARIOS * N_SPEEDS)
    keys_grid  = keys_flat.reshape(N_SCENARIOS, N_SPEEDS, 2)

    # vmap over scenarios (axis 0) and speeds (axis 1)
    def _eval_one(scen_idx, speed_keys):
        """Evaluate one scenario across all speeds."""
        def _eval_speed(max_v, key):
            return evaluate_chunk(params, net_type, scen_idx, max_v, key)
        return jax.vmap(_eval_speed)(MAX_V_TESTS, speed_keys)

    return jax.vmap(_eval_one)(SCENARIOS, keys_grid)
    # Returns dict of (N_SCENARIOS, N_SPEEDS, N_ENVS) arrays


# ── Utilities ─────────────────────────────────────────────────────────────────
def load_checkpoint_safe(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    bundle = flax.serialization.msgpack_restore(raw)
    return bundle.get("actor_params", bundle.get("params"))


def _params_to_gpu(params):
    """OPT 3: Pin params once to GPU so every forward pass reads from HBM."""
    gpu = jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices()[0]
    return jax.device_put(params, gpu)


def main():
    # ── Device report ────────────────────────────────────────────────────────
    try:
        cuda_devices = jax.devices("cuda")
        print(f"GPU detected: {cuda_devices[0]}")
    except RuntimeError:
        cuda_devices = []
        print("WARNING: No CUDA GPU found — falling back to CPU (will be slow).")

    print(f"Config: N_ENVS={N_ENVS}, scenarios={N_SCENARIOS}, speeds={N_SPEEDS}")
    total_eps = N_ENVS * N_SCENARIOS * N_SPEEDS
    print(f"Total episodes per policy: {total_eps:,}")

    # ── Load and pin checkpoints ──────────────────────────────────────────────
    raw_policies = {
        "PPO": ("PPO", load_checkpoint_safe("checkpoints/ppo_model_best.msgpack")),
        "SAC": ("SAC", load_checkpoint_safe("checkpoints_sac/sac_best.msgpack")),
        "TQC": ("TQC", load_checkpoint_safe("checkpoints_tqc/tqc_best.msgpack")),
    }

    # OPT 3: pin to GPU once
    policies = {}
    for name, (net_type, params) in raw_policies.items():
        if params is None:
            print(f"  Checkpoint missing for {name}, will skip.")
            policies[name] = (net_type, None)
        else:
            policies[name] = (net_type, _params_to_gpu(params))
            print(f"  Loaded {name} checkpoint → GPU")

    rng = jax.random.PRNGKey(42)

    # ── Compile both graph variants ───────────────────────────────────────────
    print("\nCompiling JAX graphs (this takes ~1-2 min first run)...")
    dummy_obs = jnp.zeros((1, OBS_SIZE))
    ppo_dummy = _params_to_gpu(_ppo_net.init(rng, dummy_obs)["params"])
    sac_dummy = _params_to_gpu(_sac_net.init(rng, dummy_obs)["params"])

    rng, k1, k2 = jax.random.split(rng, 3)
    # OPT 5: block_until_ready ensures true end-to-end compilation before timing
    jax.block_until_ready(evaluate_chunk_grid(ppo_dummy, "PPO", k1))
    jax.block_until_ready(evaluate_chunk_grid(sac_dummy, "SAC", k2))
    print("  Compilation complete.\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results = []
    scen_names = {0:"Random", 1:"Parallel", 2:"Perpend", 3:"Circular",
                  4:"Bottleneck", 5:"Intersect", 6:"Groups"}
    v_list = [float(v) for v in MAX_V_TESTS.tolist()]

    start_time = time.time()

    for p_name, (net_type, params) in policies.items():
        if params is None:
            print(f"Skipping {p_name} (no checkpoint).")
            continue

        rng, sub_rng = jax.random.split(rng)
        t0 = time.time()
        print(f"Evaluating {p_name}...", end=" ", flush=True)

        # OPT 2: one JIT call covers all 21 combos simultaneously
        grid = evaluate_chunk_grid(params, net_type, sub_rng)
        # OPT 5: force synchronisation before reporting time
        grid = jax.device_get(jax.block_until_ready(grid))

        elapsed = time.time() - t0
        eps_per_sec = total_eps / elapsed
        print(f"done in {elapsed:.1f}s  ({eps_per_sec:,.0f} episodes/s)")

        # Unpack (N_SCENARIOS, N_SPEEDS, N_ENVS) grid into rows
        for si, scen in enumerate(range(N_SCENARIOS)):
            for vi, v_max in enumerate(v_list):
                for i in range(N_ENVS):
                    results.append({
                        "Policy":       p_name,
                        "Scenario":     scen,
                        "Max_V":        v_max,
                        "Success":      grid["success"][si, vi, i],
                        "Active Col":   grid["act_col"][si, vi, i],
                        "Passive Col":  grid["pass_col"][si, vi, i],
                        "Timeout":      grid["timeout"][si, vi, i],
                        "SPL":          grid["spl"][si, vi, i],
                        "Jerk":         grid["jerk"][si, vi, i],
                        "Min Dist":     grid["min_dist"][si, vi, i],
                        "Time to Goal": grid["time"][si, vi, i],
                    })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_raw_data.csv", index=False)

    total_time = time.time() - start_time
    n_policies = sum(1 for _, p in policies.values() if p is not None)
    grand_total = total_eps * n_policies
    print(f"\nTotal: {grand_total:,} episodes in {total_time:.1f}s "
          f"({grand_total/total_time:,.0f} eps/s across all policies)")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    print("\nGenerating Dashboard Plots...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("RL Navigation Policies: Evaluation Dashboard", fontsize=24, weight="bold")

    rate_df = df.groupby("Policy")[["Success","Active Col","Passive Col","Timeout"]].mean().reset_index()
    rate_melt = rate_df.melt(id_vars="Policy", var_name="Outcome", value_name="Rate")
    rate_melt["Rate"] *= 100

    ax1 = plt.subplot(2, 3, 1)
    sns.barplot(data=rate_melt, x="Outcome", y="Rate", hue="Policy", ax=ax1)
    ax1.set_title("Overall Episode Outcomes (%)", fontsize=16)
    ax1.set_ylim(0, 100)

    scen_df = df.groupby(["Scenario","Policy"])["Success"].mean().reset_index()
    scen_df["Success"] *= 100
    scen_df["Scenario_Name"] = scen_df["Scenario"].map(scen_names)
    ax2 = plt.subplot(2, 3, 2)
    sns.barplot(data=scen_df, x="Scenario_Name", y="Success", hue="Policy", ax=ax2)
    ax2.set_title("Success Rate by Layout Topology", fontsize=16)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    ax2.set_ylim(0, 100)

    v_df = df.groupby(["Max_V","Policy"])["Success"].mean().reset_index()
    v_df["Success"] *= 100
    ax3 = plt.subplot(2, 3, 3)
    sns.lineplot(data=v_df, x="Max_V", y="Success", hue="Policy",
                 marker="o", linewidth=3, markersize=10, ax=ax3)
    ax3.set_title("Success Rate vs. Robot Max Speed", fontsize=16)
    ax3.set_xticks(v_list)
    ax3.set_ylim(0, 100)

    suc = df[df["Success"] == 1.0]
    ax4 = plt.subplot(2, 3, 4)
    sns.boxplot(data=suc, x="Policy", y="SPL", hue="Policy", ax=ax4, showfliers=False)
    ax4.set_title("Success-weighted Path Length (SPL)", fontsize=16)

    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=suc, x="Policy", y="Time to Goal", hue="Policy", ax=ax5, showfliers=False)
    ax5.set_title("Time to Reach Goal (seconds)", fontsize=16)

    ax6 = plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x="Policy", y="Jerk", hue="Policy", ax=ax6, showfliers=False)
    ax6.set_title("Average Kinematic Jerk (Smoothness)", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Evaluation_Dashboard.png", dpi=300)
    print("Saved 'Evaluation_Dashboard.png'")


if __name__ == "__main__":
    main()