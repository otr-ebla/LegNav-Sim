"""
ws_lib.py — shared library for the CALF PPO Legs weight-surgery experiment.

* checkpoint I/O (never writes outside experiments/weight_surgery/)
* GPU-vectorised evaluation harness over scenarios 0-12 with paired seeds
  (identical initial conditions + identical env noise keys for every policy)

Evaluation conventions mirror the repo's own eval scripts:
  - deterministic policy (action mean), ghost_prob = 0 (humans react to robot)
  - scenarios 0-6  : benchmark_eval.py style (max_goal_dist 9 m, max_v swept
                     over MAX_V_TESTS)
  - scenarios 7-12 : test_scenarios_eval.py style (multi-waypoint, fresh
                     500-step budget per waypoint segment, scenario max_v)
"""
import os, json, hashlib, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
_f = "--xla_gpu_experimental_enable_fusion_autotuner=false"
if _f not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _f).strip()

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_comp_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

import legnav.core.jax_env as _jax_env
_jax_env.USE_LEGS = True
_jax_env.SENSOR_NOISE = True
from legnav.core.jax_env import DT, MAX_STEPS, _MAX_GOAL_DIST
from legnav.core.jax_env_multi import reset_env, step_env
from legnav.core.jax_wrappers import make_stacked_env, assemble_stacked_obs
from legnav.core.jax_network import EndToEndActorCritic, scale_actions_batched

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ORIGINAL_CKPT = os.path.join(REPO, "checkpoints", "ppo", "ppo_legs_best.msgpack")
ORIGINAL_SHA = "4b1dec6bc550e432f0c48cf02b69206354b742574f2eac027ac8565dfe8e6568"
CAND_DIR = os.path.join(HERE, "candidates")
RES_DIR = os.path.join(HERE, "results")

OBS_SIZE = 668
NET = EndToEndActorCritic(action_dim=2)

SCEN_NAMES = {0: "Random", 1: "Parallel", 2: "Perpendicular", 3: "Circular",
              4: "Bottleneck", 5: "Intersection", 6: "StaticGroups",
              7: "S-Corridor", 8: "ConvergingCrowds", 9: "SequentialRooms",
              10: "ZigzagCounterflow", 11: "LongParallelCorridor", 12: "U-TurnCorridor"}
SCENS = list(range(13))
MAX_V_TESTS = [0.2, 0.5, 0.75, 1.0, 1.33, 1.66, 2.0]

# waypoint table (index 0 is the scenario's own goal → placeholder)
_WP = np.zeros((13, 3, 2), np.float32)
_NWP = np.ones((13,), np.int32)
_wps = {7: [(10.0, 4.0), (2.0, 8.0), (10.5, 10.5)],
        9: [(7.0, -1.0), (10.0, -2.0)],          # test_scenarios_eval override
        11: [(6.0, 12.0), (6.0, 19.2)],
        12: [(1.5, 1.5), (10.5, 1.5), (10.5, 10.5)]}
for _s, _l in _wps.items():
    _NWP[_s] = len(_l)
    for _i, _p in enumerate(_l):
        _WP[_s, _i] = _p
WP = jnp.asarray(_WP)
NWP = jnp.asarray(_NWP)


# ───────────────────────── checkpoint I/O ─────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def assert_original_untouched():
    s = sha256(ORIGINAL_CKPT)
    assert s == ORIGINAL_SHA, f"ORIGINAL CHECKPOINT CHANGED: {s}"
    return s


def load_bundle(path=ORIGINAL_CKPT):
    with open(path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def load_params(path=ORIGINAL_CKPT):
    b = load_bundle(path)
    return jax.tree_util.tree_map(np.asarray, b.get("params", b))


def copy_params(p):
    return jax.tree_util.tree_map(lambda x: np.array(x, dtype=np.float32, copy=True), p)


def save_candidate(params, name, meta):
    """Write a NEW checkpoint in the original bundle format
    ({"params", "opt_state"} msgpack). opt_state is carried over verbatim from
    the original so jax_ppo.load_checkpoint / every eval loader keeps working."""
    os.makedirs(CAND_DIR, exist_ok=True)
    path = os.path.join(CAND_DIR, f"{name}.msgpack")
    assert os.path.abspath(path) != os.path.abspath(ORIGINAL_CKPT)
    assert os.path.abspath(path).startswith(HERE)
    orig = load_bundle()
    bundle = {"params": jax.tree_util.tree_map(lambda x: np.asarray(x, np.float32), params),
              "opt_state": orig["opt_state"]}
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(bundle))
    meta = dict(meta)
    meta.update(name=name, file=os.path.relpath(path, HERE), sha256=sha256(path),
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                original_sha256_at_save=assert_original_untouched())
    with open(os.path.join(CAND_DIR, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


# ───────────────────────── evaluation harness ─────────────────────────
_reset_stacked, _step_stacked = make_stacked_env(reset_env, step_env, stack_dim=3)


def _reset_one(key, scen, v_over):
    is_test = scen >= 7
    mgd = jnp.where(is_test, 20.0, 9.0)
    mind = jnp.where(is_test, 8.0, 0.8)
    obs, st = _reset_stacked(key, max_goal_dist=mgd, ghost_prob=0.0,
                             scenario_idx=scen, min_goal_dist=mind)
    es = st.env_state
    max_v = jnp.where(is_test, es.max_v, v_over)
    st = st.replace(env_state=es.replace(max_v=max_v))
    obs = obs.at[2].set((max_v - 0.2) / 1.8)      # kin_vec[2] = max_v_norm
    return obs, st


def _step_one(key, st, action, wp_idx, scen):
    obs, st2, rew, done, info = _step_stacked(key, st, action)
    es = st2.env_state
    goal = info["goal_reached"]
    adv = goal & (wp_idx + 1 < NWP[scen])
    nxt = WP[scen, jnp.minimum(wp_idx + 1, 2)]
    ny = jnp.where(nxt[1] == -1.0, es.obs_boxes[0, 5] + 1.0,
                   jnp.where(nxt[1] == -2.0, es.obs_boxes[2, 5] + 1.0, nxt[1]))
    gx = jnp.where(adv, nxt[0], es.goal_x)
    gy = jnp.where(adv, ny, es.goal_y)
    es = es.replace(goal_x=gx, goal_y=gy,
                    time_step=jnp.where(adv, 0, es.time_step))
    # goal-dependent obs entries recomputed for the new waypoint (same formulas as get_obs)
    dx, dy = gx - es.x, gy - es.y
    c, s = jnp.cos(-es.theta), jnp.sin(-es.theta)
    gvec = jnp.array([c * dx - s * dy, s * dx + c * dy]) / _MAX_GOAL_DIST
    gdist = jnp.sqrt(dx ** 2 + dy ** 2)
    galign = (jnp.arctan2(dy, dx) - es.theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    kin = obs[:5]
    kin = jnp.where(adv, kin.at[3].set(gdist / _MAX_GOAL_DIST).at[4].set(galign / jnp.pi), kin)
    goal_stack = jnp.where(adv, st2.goal_stack.at[-1].set(gvec), st2.goal_stack)
    st2 = st2.replace(env_state=es, goal_stack=goal_stack)
    obs = assemble_stacked_obs(kin, goal_stack, st2.pose_stack, st2.lidar_stack)
    col = info["collision"] & ~goal
    a_col = info["active_col"] & col
    p_col = info["passive_col"] & col & ~a_col
    o_col = col & ~a_col & ~p_col
    tmo = info["timeout"] & ~goal & ~col
    succ = goal & ~adv
    ep_done = succ | col | tmo
    out = dict(rew=rew, adv=adv, succ=succ, a_col=a_col, p_col=p_col, o_col=o_col, tmo=tmo,
               ep_done=ep_done, gdist=gdist, ch=info["closest_shoe_surface"])
    return obs, st2, jnp.where(adv, wp_idx + 1, wp_idx), out


CHUNK = 100
YIELD_DIST, YIELD_FOV = 1.5, 0.785


def _init_carry(key, scen, v_over):
    n = scen.shape[0]
    keys = jax.random.split(key, n)
    obs, st = jax.vmap(_reset_one)(keys, scen, v_over)
    es = st.env_state
    d0 = jnp.sqrt((es.goal_x - es.x) ** 2 + (es.goal_y - es.y) ** 2)
    z = jnp.zeros(n)
    acc = dict(ret=z, steps=z, path=z, dstar=d0, sum_dw=z, sum_dv=z, sum_jerk=z, sum_absw=z,
               sum_vfrac=z, flips=z, mhd=jnp.full(n, 100.0), yz=z, yc=z, stop_steps=z,
               succ=z, a_col=z, p_col=z, o_col=z, tmo=z, final_dist=d0, wps=z,
               prev_v=z, prev_w=z, prev_aw=z, init_dist=d0, max_v=es.max_v)
    return (st, obs, jnp.zeros(n, jnp.int32), jnp.ones(n, bool), acc)


def _policy_action(params, obs, max_v):
    mean, _, _ = NET.apply({"params": params}, obs)
    return scale_actions_batched(mean, max_v)


def _make_chunk(act_fn, record=False):
    def chunk(params, carry, key, scen, t0):
        n = scen.shape[0]

        def body(carry, i):
            st, obs, wp, active, a = carry
            act = act_fn(params, obs, st.env_state.max_v)
            keys = jax.random.split(jax.random.fold_in(key, t0 + i), n)
            obs2, st2, wp2, o = jax.vmap(_step_one)(keys, st, act, wp, scen)
            es = st2.env_state
            v, w = es.v, es.w
            af = active.astype(jnp.float32)
            aw = (w - a["prev_w"]) / DT
            ppl = es.people
            dpx = ppl[:, :, 0] - es.x[:, None]
            dpy = ppl[:, :, 1] - es.y[:, None]
            rel = (jnp.arctan2(dpy, dpx) - es.theta[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
            in_yz = jnp.any((jnp.sqrt(dpx ** 2 + dpy ** 2 + 1e-8) < YIELD_DIST)
                            & (jnp.abs(rel) < YIELD_FOV) & (ppl[:, :, 10] >= 0.0), axis=1)
            flip = (w * a["prev_w"] < 0) & (jnp.abs(w) > 0.05) & (jnp.abs(a["prev_w"]) > 0.05)
            na = dict(a)
            na["ret"] = a["ret"] + af * o["rew"]
            na["steps"] = a["steps"] + af
            na["path"] = a["path"] + af * v * DT
            na["dstar"] = a["dstar"] + jnp.where(active & o["adv"], o["gdist"], 0.0)
            na["sum_dw"] = a["sum_dw"] + af * jnp.abs(w - a["prev_w"])
            na["sum_dv"] = a["sum_dv"] + af * jnp.abs(v - a["prev_v"])
            na["sum_jerk"] = a["sum_jerk"] + af * jnp.abs((aw - a["prev_aw"]) / DT)
            na["sum_absw"] = a["sum_absw"] + af * jnp.abs(w)
            na["sum_vfrac"] = a["sum_vfrac"] + af * v / jnp.maximum(es.max_v, 1e-3)
            na["flips"] = a["flips"] + af * flip
            na["stop_steps"] = a["stop_steps"] + af * (v < 0.05)
            na["mhd"] = jnp.where(active, jnp.minimum(a["mhd"], o["ch"]), a["mhd"])
            na["yz"] = a["yz"] + af * in_yz
            na["yc"] = a["yc"] + af * (in_yz & (v <= 0.1))
            for k in ("succ", "a_col", "p_col", "o_col", "tmo"):
                na[k] = a[k] + af * o[k]
            na["wps"] = a["wps"] + af * (o["adv"] | o["succ"])
            na["final_dist"] = jnp.where(active, o["gdist"], a["final_dist"])
            na["prev_v"], na["prev_w"], na["prev_aw"] = v, w, aw
            return (st2, obs2, wp2, active & ~o["ep_done"], na), ((obs, active) if record else None)

        carry, rec = jax.lax.scan(body, carry, jnp.arange(CHUNK))
        return (carry, rec) if record else carry
    return chunk


_init_jit = jax.jit(_init_carry)
_chunk_jit = jax.jit(_make_chunk(_policy_action))
_MAX_TOTAL_STEPS = MAX_STEPS * 3 + CHUNK


def make_batch(n_per_scen, scens=SCENS):
    scen = np.repeat(np.array(scens, np.int32), n_per_scen)
    v = np.tile(np.array(MAX_V_TESTS, np.float32), len(scen) // len(MAX_V_TESTS) + 1)[:len(scen)]
    return jnp.asarray(scen), jnp.asarray(v)


def evaluate(params, seed=0, n_per_scen=140, scens=SCENS, chunk_fn=None, verbose=False):
    """Run one paired batch. Returns dict of per-episode numpy arrays."""
    chunk_fn = chunk_fn or _chunk_jit
    scen, v = make_batch(n_per_scen, scens)
    key = jax.random.PRNGKey(seed)
    k_reset, k_step = jax.random.split(key)
    carry = _init_jit(k_reset, scen, v)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    t = 0
    while t < _MAX_TOTAL_STEPS:
        carry = chunk_fn(params, carry, k_step, scen, jnp.int32(t))
        t += CHUNK
        n_act = int(carry[3].sum())
        if verbose:
            print(f"   t={t} active={n_act}", flush=True)
        if n_act == 0:
            break
    a = {k: np.asarray(x) for k, x in carry[4].items()}
    a["unfinished"] = np.asarray(carry[3]).astype(np.float32)
    a["tmo"] = a["tmo"] + a["unfinished"]
    a["scen"] = np.asarray(scen)
    return a


def evaluate_seeds(params, seeds, n_per_scen=140, **kw):
    outs = [evaluate(params, seed=s, n_per_scen=n_per_scen, **kw) for s in seeds]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def episode_metrics(a):
    """Per-episode derived metrics (arrays)."""
    steps = np.maximum(a["steps"], 1)
    succ = a["succ"]
    m = dict(
        success=succ, collision=a["a_col"] + a["p_col"] + a["o_col"],
        active_col=a["a_col"], passive_col=a["p_col"], static_col=a["o_col"],
        timeout=a["tmo"], reward=a["ret"],
        nav_time=np.where(succ > 0, a["steps"] * DT, np.nan),
        path_len=np.where(succ > 0, a["path"], np.nan),
        final_dist=a["final_dist"],
        spl=succ * a["dstar"] / np.maximum(a["path"], a["dstar"]),
        dw=a["sum_dw"] / steps, dv=a["sum_dv"] / steps, ang_jerk=a["sum_jerk"] / steps,
        abs_w=a["sum_absw"] / steps, v_frac=a["sum_vfrac"] / steps,
        w_flip_rate=a["flips"] / steps, stop_frac=a["stop_steps"] / steps,
        min_human_dist=np.where(a["mhd"] < 99, a["mhd"], np.nan),
        yield_score=np.where(a["yz"] > 0, a["yc"] / np.maximum(a["yz"], 1), np.nan),
    )
    return m


METRIC_ORDER = ["success", "collision", "active_col", "passive_col", "static_col", "timeout",
                "reward", "nav_time", "path_len", "final_dist", "spl", "dw", "dv", "ang_jerk",
                "abs_w", "w_flip_rate", "v_frac", "stop_frac", "min_human_dist", "yield_score"]


def summarize(a, by_scen=False):
    m = episode_metrics(a)
    def agg(mask):
        r = {"n": int(mask.sum())}
        for k in METRIC_ORDER:
            x = m[k][mask]
            r[k] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
        return r
    out = {"ALL": agg(np.ones_like(a["scen"], bool)),
           "TRAIN(0-6)": agg(a["scen"] < 7), "TEST(7-12)": agg(a["scen"] >= 7)}
    if by_scen:
        for s in np.unique(a["scen"]):
            out[f"{s}:{SCEN_NAMES[int(s)]}"] = agg(a["scen"] == s)
    return out


def score(a):
    """Scalar used ONLY for search/selection on dev seeds:
    macro-average over scenarios of the true environment episode return."""
    return float(np.mean([a["ret"][a["scen"] == s].mean() for s in np.unique(a["scen"])]))


def fmt_row(name, r):
    return (f"{name:34s} n={r['n']:5d} SR={100*r['success']:5.1f} COL={100*r['collision']:5.1f} "
            f"(a{100*r['active_col']:4.1f} p{100*r['passive_col']:4.1f} s{100*r['static_col']:4.1f}) "
            f"TO={100*r['timeout']:5.1f} R={r['reward']:7.2f} T={r['nav_time']:5.1f}s L={r['path_len']:5.2f}m "
            f"FD={r['final_dist']:4.2f} SPL={r['spl']:.3f} dw={r['dw']:.3f} jerk={r['ang_jerk']:5.2f} "
            f"|w|={r['abs_w']:.3f} flip={r['w_flip_rate']:.3f}")


# ───────────────────────── batched forward helpers ─────────────────────────
@jax.jit
def _mean_jit(params, obs):
    return NET.apply({"params": params}, obs)[0]


def policy_mean(params, obs, bs=4096):
    """Squashed action mean [v_frac, w] for a big numpy batch of observations."""
    params = jax.tree_util.tree_map(jnp.asarray, params)
    out = []
    for i in range(0, len(obs), bs):
        o = np.asarray(obs[i:i + bs], np.float32)
        pad = bs - len(o)
        if pad:
            o = np.concatenate([o, np.zeros((pad, o.shape[1]), np.float32)])
        out.append(np.asarray(_mean_jit(params, jnp.asarray(o)))[:bs - pad])
    return np.concatenate(out)


@jax.jit
def _feat_jit(params, obs):
    _, st = NET.apply({"params": params}, obs, capture_intermediates=True)
    it = st["intermediates"]
    return dict(d0=it["Dense_0"]["__call__"][0], d1=it["Dense_1"]["__call__"][0],
                raw=it["Dense_2"]["__call__"][0], attn=it["FrameStackAttention_0"]["__call__"][0],
                cnn=it["LidarFrameCNN_0"]["__call__"][2])


def policy_feats(params, obs, bs=4096):
    """Intermediate activations (pre-ReLU d0, d1; raw pre-tanh mean; attention out; newest CNN token)."""
    params = jax.tree_util.tree_map(jnp.asarray, params)
    outs = []
    for i in range(0, len(obs), bs):
        o = np.asarray(obs[i:i + bs], np.float32)
        pad = bs - len(o)
        if pad:
            o = np.concatenate([o, np.zeros((pad, o.shape[1]), np.float32)])
        f = _feat_jit(params, jnp.asarray(o))
        outs.append({k: np.asarray(v)[:bs - pad] for k, v in f.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
