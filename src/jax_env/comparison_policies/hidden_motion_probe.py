"""
hidden_motion_probe.py — Does NavRep's frozen M output (h_t) encode MOTION?

Context
-------
NavRep trains to ~70% success but collides with humans ~30% of the time and
drives conservatively.  The controller's ONLY temporal/motion signal is h_t
(the transformer M's hidden state at the latest frame); z_t is a single static
frame.  In pretraining, M trained nearly flat (MSE 52.5 -> 51.4), so it may be
doing persistence (h_t ~ f(z_t)) rather than encoding dynamics.  If so, the
controller cannot anticipate where humans are going -> late reactions, hits,
and a learned slow-down.

This probe answers two questions with cheap closed-form ridge regression on the
FROZEN encoder + M (no controller needed):

  A) Latent motion.  Can h_t linearly predict the recent latent velocity
     (z_t - z_{t-1})?  That delta depends on the PAST frame, which z_t alone
     does NOT contain — so if h_t predicts it well (and far better than z_t
     does), M genuinely carries temporal information.  If h_t is no better
     than z_t, M is effectively static.

  B) Human motion.  For samples with a human nearby, does ADDING h_t to the
     controller's features [z_t, state_r] improve prediction of the nearest
     human's ego-frame velocity?  The marginal R^2 gain from h_t == how much
     human-motion information the controller actually receives.

Run:
    cd src/jax_env
    python comparison_policies/hidden_motion_probe.py
"""

import os
import sys

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_JAX_ENV_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR     = os.path.dirname(_JAX_ENV_DIR)
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
for _p in (_JAX_ENV_DIR, _SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES",           "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("TF_GPU_ALLOCATOR",               "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization

from jax_wrappers import make_stacked_env
from jax_env_multi import reset_env, step_env
from comparison_policies.navrep_network import (
    LidarEncoder, TransformerM, Z_DIM, H_DIM, NUM_RAYS, STACK_DIM,
)

VM_CKPT = os.path.join(_JAX_ENV_DIR, "checkpoints_navrep", "navrep_vm.msgpack")

NUM_ENVS  = 256
N_STEPS   = 128          # -> 32,768 samples
R_MAX     = 3.0          # only probe human-motion when nearest human within this (m)
RIDGE_LAM = 1.0
VAL_FRAC  = 0.2

_STATE_END = 14          # obs[:14] = pose_stack(9) + state_vec(5) = state_r


def ridge_r2(X, Y, tr, va, lam=RIDGE_LAM):
    """Aggregate (variance-weighted) R^2 of a ridge fit X->Y, eval on val split."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xtr = np.c_[Xs[tr], np.ones(len(tr))]
    Xva = np.c_[Xs[va], np.ones(len(va))]
    ymu = Y[tr].mean(0)
    Yc  = Y - ymu
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Yc[tr])
    pred = Xva @ W
    ss_res = float(((Yc[va] - pred) ** 2).sum())
    ss_tot = float((Yc[va] ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    if not os.path.isfile(VM_CKPT):
        raise FileNotFoundError(f"V+M checkpoint not found: {VM_CKPT}")

    # ── Load frozen encoder + M ────────────────────────────────────────────────
    enc = LidarEncoder(z_dim=Z_DIM)
    Mn  = TransformerM(z_dim=Z_DIM)
    enc_skel = enc.init(jax.random.PRNGKey(1), jnp.zeros((1, NUM_RAYS)))["params"]
    m_skel   = Mn.init(jax.random.PRNGKey(2), jnp.zeros((1, STACK_DIM, Z_DIM)))["params"]
    with open(VM_CKPT, "rb") as f:
        bundle = flax.serialization.from_bytes(
            {"encoder": enc_skel, "M": m_skel}, f.read()
        )
    enc_p = bundle["encoder"]
    m_p   = bundle["M"]               # TransformerM params (as loaded by navrep_extract_features)
    print(f"Loaded frozen encoder + M from {VM_CKPT}\n")

    # ── Collect stacked obs + ground-truth robot pose & people ────────────────
    bound_reset = lambda key, max_goal_dist, scenario_idx, ghost_prob, min_goal_dist=0.8, **kw: \
        reset_env(key, max_goal_dist, scenario_idx=scenario_idx, ghost_prob=ghost_prob,
                  min_goal_dist=min_goal_dist, **kw)
    rs, ss = make_stacked_env(bound_reset, step_env, stack_dim=STACK_DIM)
    # ghost_prob=0.5 -> humans often ignore the robot, so they cross its path (motion).
    _reset_one = lambda key, sc: rs(key, max_goal_dist=9.0, scenario_idx=sc, ghost_prob=0.5)
    vmap_reset = jax.vmap(_reset_one, in_axes=(0, 0))
    vmap_step  = jax.vmap(ss)

    rng = jax.random.PRNGKey(0)
    rng, k_reset = jax.random.split(rng)
    env_keys  = jax.random.split(k_reset, NUM_ENVS)
    scenarios = jax.random.randint(k_reset, (NUM_ENVS,), 0, 7)
    obs, state = vmap_reset(env_keys, scenarios)

    def body(carry, step_key):
        obs_c, state_c = carry
        ka, kv, kw = jax.random.split(step_key, 3)
        v = jax.random.uniform(kv, (NUM_ENVS,), minval=0.0,  maxval=1.2)
        w = jax.random.uniform(kw, (NUM_ENVS,), minval=-1.0, maxval=1.0)
        nobs, nstate, _, _, _ = vmap_step(jax.random.split(ka, NUM_ENVS), state_c,
                                          jnp.stack([v, w], -1))
        es = nstate.env_state
        rec = (nobs,
               jnp.stack([es.x, es.y, es.theta], -1),   # (NUM_ENVS, 3)
               es.people)                               # (NUM_ENVS, P, 8)
        return (nobs, nstate), rec

    print("Collecting rollouts (random actions, ghost_prob=0.5)...")
    step_keys = jax.random.split(rng, N_STEPS)
    (_, _), (obs_h, pose_h, ppl_h) = jax.lax.scan(body, (obs, state), step_keys)
    obs_h  = np.asarray(jax.block_until_ready(obs_h)).reshape(-1, obs_h.shape[-1])
    pose_h = np.asarray(pose_h).reshape(-1, 3)
    ppl_h  = np.asarray(ppl_h).reshape(-1, ppl_h.shape[-2], ppl_h.shape[-1])
    M = obs_h.shape[0]
    print(f"  samples={M:,}  people/sample={ppl_h.shape[1]}\n")

    # ── Frozen features: z_t, h_t, and latent delta (z_t - z_{t-1}) ───────────
    @jax.jit
    def feats(ob):
        ls = ob[:, _STATE_END:].reshape(-1, STACK_DIM, NUM_RAYS)
        zc, _ = enc.apply({"params": enc_p}, ls)          # (B, 3, Z)
        _, hs = Mn.apply({"params": m_p}, zc)             # (B, 3, H)
        zstk = zc.reshape(zc.shape[0], -1)                # (B, 3*Z) raw latent stack
        return zc[:, -1, :], hs[:, -1, :], zc[:, -1, :] - zc[:, -2, :], zstk

    z_t, h_t, dz, zstk = [], [], [], []
    B = 8192
    for i in range(0, M, B):
        a, b, c, d = feats(jnp.asarray(obs_h[i:i + B]))
        z_t.append(np.asarray(a)); h_t.append(np.asarray(b))
        dz.append(np.asarray(c)); zstk.append(np.asarray(d))
    z_t = np.concatenate(z_t); h_t = np.concatenate(h_t)
    dz = np.concatenate(dz); zstk = np.concatenate(zstk)
    state_r = obs_h[:, :_STATE_END]

    perm  = np.random.RandomState(0).permutation(M)
    n_val = int(M * VAL_FRAC)
    va, tr = perm[:n_val], perm[n_val:]

    # ── Probe A: latent motion (predict z_t - z_{t-1}) ─────────────────────────
    print("── Probe A: can features predict recent latent motion (z_t - z_{t-1})? ──")
    r2_z  = ridge_r2(z_t,                    dz, tr, va)
    r2_h  = ridge_r2(h_t,                    dz, tr, va)
    r2_zh = ridge_r2(np.c_[z_t, h_t],        dz, tr, va)
    print(f"  R^2 from z_t      (static frame) : {r2_z:+.3f}")
    print(f"  R^2 from h_t      (M output)     : {r2_h:+.3f}")
    print(f"  R^2 from [z_t,h_t]               : {r2_zh:+.3f}")
    print(f"  -> h_t marginal over z_t         : {r2_zh - r2_z:+.3f}")

    # ── Probe B: nearest-human ego-frame velocity ──────────────────────────────
    rx, ry, rth = pose_h[:, 0], pose_h[:, 1], pose_h[:, 2]
    dxp = ppl_h[:, :, 0] - rx[:, None]
    dyp = ppl_h[:, :, 1] - ry[:, None]
    dist = np.sqrt(dxp ** 2 + dyp ** 2)
    j = dist.argmin(1)
    ar = np.arange(M)
    near = dist[ar, j]
    vx, vy = ppl_h[ar, j, 2], ppl_h[ar, j, 3]
    c, s = np.cos(-rth), np.sin(-rth)
    y_hum = np.stack([c * vx - s * vy, s * vx + c * vy], 1)   # ego-frame velocity

    mask = near < R_MAX
    idx  = np.where(mask)[0]
    print(f"\n── Probe B: predict NEAREST-human ego velocity (within {R_MAX} m) ──")
    print(f"  usable samples: {idx.size:,} / {M:,}")
    if idx.size > 2000:
        sub = np.random.RandomState(1).permutation(idx.size)
        nv  = int(idx.size * VAL_FRAC)
        vb, tb = idx[sub[:nv]], idx[sub[nv:]]
        # remap to local arrays
        Xz  = np.c_[z_t, state_r]
        Xzh = np.c_[z_t, h_t, state_r]
        def _r2(X):
            mu, sd = X[tb].mean(0), X[tb].std(0) + 1e-6
            Xs = (X - mu) / sd
            Xtr = np.c_[Xs[tb], np.ones(len(tb))]; Xva = np.c_[Xs[vb], np.ones(len(vb))]
            ymu = y_hum[tb].mean(0); Yc = y_hum - ymu
            A = Xtr.T @ Xtr + RIDGE_LAM * np.eye(Xtr.shape[1])
            W = np.linalg.solve(A, Xtr.T @ Yc[tb]); pred = Xva @ W
            return 1.0 - float(((Yc[vb] - pred) ** 2).sum()) / (float((Yc[vb] ** 2).sum()) + 1e-12)
        r2_base  = _r2(Xz)
        r2_full  = _r2(Xzh)
        r2_stack = _r2(np.c_[zstk, state_r])      # raw 3-frame latent stack + state
        print(f"  R^2 from [z_t, state_r]              : {r2_base:+.3f}")
        print(f"  R^2 from [z_t, h_t, state_r]         : {r2_full:+.3f}")
        print(f"  R^2 from [z0,z1,z2, state_r] (raw)   : {r2_stack:+.3f}")
        print(f"  -> h_t marginal for human motion     : {r2_full - r2_base:+.3f}")
        print(f"  -> raw-stack marginal over z_t       : {r2_stack - r2_base:+.3f}")
    else:
        print("  too few near-human samples; raise R_MAX or N_STEPS")

    print("\nReading:")
    print("  Probe A: if h_t's R^2 >> z_t's, M encodes temporal motion; if ~equal")
    print("           and low, M is effectively static (persistence).")
    print("  Probe B: small h_t marginal => controller gets little human-motion")
    print("           signal => cannot anticipate humans => the avoidance ceiling.")


if __name__ == "__main__":
    main()
