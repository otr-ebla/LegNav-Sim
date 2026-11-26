"""
diag_s18.py — Diagnose a trained policy inside the TRAINING env (same flags as the cluster job).

Runs N parallel envs (vmap) for T steps with the actor (stochastic sampling as in training, or the
deterministic mean as in deployment) and reports, per scenario and overall:
  outcome mix (success / active / passive / static collision / group-cross / timeout),
  per-episode sum of every reward term, and an oscillation metric (sign flips of commanded w).

Usage (set the same env flags as g5k/train_asym_scen18.oar.sh first):
  LEGNAV_REAL_ENV=1 LEGNAV_SOCIAL_FT=1 LEGNAV_FT_PRED=1 LEGNAV_GRP_CROSS_TERMINAL=1 \
  LEGNAV_FT_SCEN18_PROB=0.8 LEGNAV_ROBOT_FOOTPRINT=rect:0.72x0.50 PYTHONPATH=. \
  python -m legnav.evaluation.diag_s18 --ckpt <calf.msgpack> --max-scen 4 --dist 6.0 --mode both
"""
import argparse
import os

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from legnav.core.jax_network import EndToEndActorCritic, scale_actions_batched
from legnav.core.jax_wrappers import make_stacked_env, make_autoreset_env

if os.environ.get("LEGNAV_REAL_ENV", "0") == "1":
    from legnav.core.jax_env_real import reset_env_real as reset_env, step_env_real as step_env
else:
    from legnav.core.jax_env_multi import reset_env, step_env

from legnav.core.obs_layouts import OBS_SIZE   # 668, or 672 with LEGNAV_OBS_CMD=1
TERMS = ("progress", "step", "smooth", "rot", "stop/spin", "heading", "yield+social", "terminal")
COLS = []
SOC = ("rew_grp", "rew_cpa", "rew_foot", "rew_wall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-scen", type=int, default=4)
    ap.add_argument("--dist", type=float, default=6.0)
    ap.add_argument("--ghost", type=float, default=0.2)
    ap.add_argument("--mode", choices=("stoch", "det", "both"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.ckpt, "rb") as f:
        bundle = flax.serialization.msgpack_restore(f.read())
    params = bundle.get("params", bundle)
    net = EndToEndActorCritic(action_dim=2)
    sig = np.exp(np.asarray(net.apply({"params": params}, jnp.zeros((1, OBS_SIZE)))[1][0]))
    print(f"policy sigma (v, w) = {sig}")

    reset_s, step_s = make_stacked_env(reset_env, step_env, stack_dim=3)
    step_auto = make_autoreset_env(reset_s, step_s)
    vstep = jax.vmap(step_auto, in_axes=(0, 0, 0, None, None, None, None))
    vreset = jax.vmap(lambda k: reset_s(k, max_goal_dist=a.dist, scenario_idx=-1,
                                        ghost_prob=a.ghost, max_scenario=a.max_scen))

    def run(stochastic):
        @jax.jit
        def rollout(key):
            k0, key = jax.random.split(key)
            obs, st = vreset(jax.random.split(k0, a.envs))

            def body(c, _):
                obs, st, key = c
                key, ka, ks = jax.random.split(key, 3)
                mean, logstd, _ = net.apply({"params": params}, obs)
                raw = mean + stochastic * jax.random.normal(ka, mean.shape) * jnp.exp(logstd)
                act = scale_actions_batched(raw, st.env_state.max_v)
                es = st.env_state   # no scenario id in the state: the 5x9 m room identifies scenario 18
                scen = jnp.where((jnp.abs(es.room_w - 5.0) < 1e-3) & (jnp.abs(es.room_h - 9.0) < 1e-3), 18, 0)
                nobs, nst, r, d, info = vstep(jax.random.split(ks, a.envs), st, act,
                                              a.dist, jnp.int32(-1), a.ghost, jnp.int32(a.max_scen))
                pp = es.people
                dd = jnp.where(pp[..., 0] > -900, jnp.hypot(pp[..., 0] - es.x[:, None], pp[..., 1] - es.y[:, None]), 1e3)
                ih = jnp.argmin(dd, axis=-1)
                hsp = jnp.hypot(jnp.take_along_axis(pp[..., 2], ih[:, None], 1)[:, 0],
                                jnp.take_along_axis(pp[..., 3], ih[:, None], 1)[:, 0])
                out = dict(hsp=hsp, rv=es.v, r=r, d=d, g=info["goal_reached"], ac=info["active_col"], pc=info["passive_col"],
                           sc=info["static_col"], gx=info["grp_cross"], to=info["timeout"],
                           terms=info["rew_terms"], w=act[:, 1], v=act[:, 0], scen=scen,
                           **{k: info[k] for k in SOC})
                return (nobs, nst, key), out
            _, h = jax.lax.scan(body, (obs, st, key), None, length=a.steps)
            return h
        return jax.device_get(rollout(jax.random.PRNGKey(a.seed)))

    modes = {"stoch": [1.0], "det": [0.0], "both": [1.0, 0.0]}[a.mode]
    for s in modes:
        h = run(s)
        report(h, "STOCHASTIC (training)" if s else "DETERMINISTIC (deploy)")


def report(h, title):
    T, N = h["d"].shape
    r = np.asarray(h["r"]); sd = r.std(); term = r[np.asarray(h["d"])]
    print(f"raw per-step reward std {sd:.2f} (PPO divides by this, then clips to +-10): "
          f"goal {30/sd:+.1f}  active {-50/sd:+.1f}  passive {-15/sd:+.1f}  grp {-60/sd:+.1f} | "
          f"terminals clipped: {100*(np.abs(term)/sd > 10).mean():.0f}%")
    # Per-episode accumulation (skip each env's first, possibly truncated-at-start, nothing: all start at reset)
    rows = []
    acc_terms = np.zeros((N, h["terms"].shape[-1]))
    acc_soc = np.zeros((N, len(SOC)))
    flips = np.zeros(N); length = np.zeros(N)
    prev_w = np.zeros(N)
    scen0 = h["scen"][0].copy()
    cur_scen = scen0
    for t in range(T):
        acc_terms += h["terms"][t]
        acc_soc += np.stack([h[k][t] for k in SOC], -1)
        w = np.array(h["w"][t])
        flips += (np.sign(w) != np.sign(prev_w)) & (np.abs(w) > 0.15) & (np.abs(prev_w) > 0.15)
        prev_w = w
        length += 1
        d = h["d"][t]
        for i in np.nonzero(d)[0]:
            if h["g"][t, i]:      o = "success"
            elif h["gx"][t, i]:   o = "grp_cross"
            elif h["ac"][t, i]:   o = "active_col"
            elif h["pc"][t, i]:   o = "passive_col"
            elif h["sc"][t, i]:   o = "static_col"
            elif h["to"][t, i]:   o = "timeout"
            else:                 o = "other"
            if o == "active_col":
                COLS.append((float(h["hsp"][t, i]), float(h["rv"][t, i])))
            rows.append((int(cur_scen[i]), o, length[i], flips[i] / max(length[i], 1),
                         *acc_terms[i], *acc_soc[i]))
            acc_terms[i] = 0; acc_soc[i] = 0; flips[i] = 0; length[i] = 0; prev_w[i] = 0
        if t + 1 < T:
            cur_scen = np.where(d, h["scen"][t + 1], cur_scen)

    if COLS:
        c = np.array(COLS); COLS.clear()
        print(f"active collisions: nearest human speed <0.1 m/s (standing) = {100*(c[:,0]<0.1).mean():.1f}%  "
              f"| robot v at contact mean {c[:,1].mean():.2f} m/s (>=0.2: {100*(c[:,1]>=0.2).mean():.0f}%)")
    print(f"\n==== {title}: {len(rows)} finished episodes ====")
    if not rows:
        return
    scen = np.array([r[0] for r in rows]); out = np.array([r[1] for r in rows])
    num = np.array([r[2:] for r in rows], dtype=float)
    cols = ["len", "w_flip_rate", *TERMS, *SOC]
    outcomes = ["success", "active_col", "passive_col", "static_col", "grp_cross", "timeout", "other"]
    groups = [("ALL", np.ones(len(rows), bool))]
    if (scen >= 0).any():
        groups += [("scen18" if s == 18 else "curric", scen == s) for s in sorted(set(scen.tolist()))]
    for name, m in groups:
        n = m.sum()
        mix = "  ".join(f"{o}={100 * (out[m] == o).mean():4.1f}%" for o in outcomes if (out[m] == o).any())
        print(f"[{name:>8}] n={n:4d}  {mix}")
    print("\nMean per-episode sums (ALL) by outcome:")
    print(f"{'outcome':>12} " + " ".join(f"{c[:10]:>10}" for c in cols))
    for o in outcomes:
        m = out == o
        if m.any():
            print(f"{o:>12} " + " ".join(f"{x:>10.2f}" for x in num[m].mean(0)))


if __name__ == "__main__":
    main()
