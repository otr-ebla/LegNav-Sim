"""Behaviour distillation into the CALF weights (DAgger).
teacher(obs) = hand-designed, obs-only behaviour wrapped around the FROZEN original network.
student      = copy of the original weights; a chosen subset of layers is re-fitted so that the
               network itself reproduces the teacher (no wrapper needed at inference)."""
import time, numpy as np, jax, jax.numpy as jnp, optax
import ws_lib as L

SUBSETS = {
    "head":  ["Dense_2"],
    "trunk": ["Dense_0", "Dense_1", "Dense_2"],
    "actor": ["LidarFrameCNN_0", "LayerNorm_0", "FrameStackAttention_0", "Dense_0", "Dense_1", "Dense_2"],
}


def collect(policy_params, act_fn, seed, n_per_scen=112, every=2, max_chunks=6):
    """Roll out `act_fn` and return the visited observations (active steps, subsampled)."""
    key = ("rec", act_fn)
    if key not in collect.cache:
        collect.cache[key] = jax.jit(L._make_chunk(act_fn, record=True))
    ck = collect.cache[key]
    scen, v = L.make_batch(n_per_scen)
    k_reset, k_step = jax.random.split(jax.random.PRNGKey(seed))
    carry = L._init_jit(k_reset, scen, v)
    pp = jax.tree_util.tree_map(jnp.asarray, policy_params)
    out = []
    for c in range(max_chunks):
        carry, (obs, act) = ck(pp, carry, k_step, scen, jnp.int32(c * L.CHUNK))
        obs, act = np.asarray(obs[::every]), np.asarray(act[::every])
        out.append(obs[act])
        if int(carry[3].sum()) == 0:
            break
    return np.concatenate(out)
collect.cache = {}


def label(teacher_mean_fn, p0, obs, bs=8192):
    key = teacher_mean_fn
    if key not in label.cache:
        label.cache[key] = jax.jit(teacher_mean_fn)
    f = label.cache[key]; p0 = jax.tree_util.tree_map(jnp.asarray, p0); out = []
    for i in range(0, len(obs), bs):
        o = obs[i:i + bs]; pad = bs - len(o)
        if pad: o = np.concatenate([o, np.zeros((pad, o.shape[1]), np.float32)])
        out.append(np.asarray(f(p0, jnp.asarray(o)))[:bs - pad])
    return np.concatenate(out)
label.cache = {}


def make_trainer(subset, lr=1e-4, l2_anchor=0.0):
    train_keys = SUBSETS[subset]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    def split(p):
        return {k: v for k, v in p.items() if k in train_keys}, {k: v for k, v in p.items() if k not in train_keys}

    def loss_fn(tr, fr, tr0, obs, tgt, wgt):
        m = L.NET.apply({"params": {**fr, **tr}}, obs)[0]
        e = ((m - tgt) ** 2).sum(-1)
        reg = sum(jnp.sum((a - b) ** 2) for a, b in zip(jax.tree_util.tree_leaves(tr), jax.tree_util.tree_leaves(tr0)))
        return jnp.mean(wgt * e) + l2_anchor * reg

    @jax.jit
    def step(tr, fr, tr0, os_, obs, tgt, wgt):
        l, g = jax.value_and_grad(loss_fn)(tr, fr, tr0, obs, tgt, wgt)
        up, os_ = opt.update(g, os_, tr)
        return optax.apply_updates(tr, up), os_, l
    return split, opt, step


def fit(student, p0, obs, tgt, wgt, subset, steps, lr=1e-4, bs=4096, seed=0, state=None, log=print):
    split, opt, step = fit.cache.setdefault((subset, lr), make_trainer(subset, lr))
    tr, fr = split(jax.tree_util.tree_map(jnp.asarray, student)); tr0, _ = split(jax.tree_util.tree_map(jnp.asarray, p0))
    os_ = state if state is not None else opt.init(tr)
    rng = np.random.default_rng(seed); ls = []
    for i in range(steps):
        idx = rng.integers(0, len(obs), bs)
        tr, os_, l = step(tr, fr, tr0, os_, jnp.asarray(obs[idx]), jnp.asarray(tgt[idx]), jnp.asarray(wgt[idx]))
        ls.append(float(l))
        if (i + 1) % max(1, steps // 4) == 0:
            log(f"      fit step {i+1}/{steps} loss={np.mean(ls[-200:]):.5f}")
    return jax.tree_util.tree_map(np.asarray, {**fr, **tr}), os_
fit.cache = {}


def dagger(p0, teacher_mean_fn, subset, iters=4, steps=1500, lr=1e-4, w_changed=5.0, seed0=5000, log=print,
           n_per_scen=112):
    """DAgger: iteration 0 visits teacher states, later iterations visit the student's own states."""
    import ws_teachers as T
    t_act = T.act_from_mean(teacher_mean_fn)
    base_mean = lambda params, obs: L.NET.apply({"params": params}, obs)[0]
    student = L.copy_params(p0); O, Y, W = [], [], []; state = None
    for it in range(iters):
        t = time.time()
        obs = collect(p0, t_act, seed0 + it, n_per_scen) if it == 0 else collect(student, L._policy_action, seed0 + it, n_per_scen)
        y = label(teacher_mean_fn, p0, obs); y0 = label(base_mean, p0, obs)
        changed = np.abs(y - y0).sum(-1) > 1e-3
        O.append(obs); Y.append(y); W.append(np.where(changed, w_changed, 1.0).astype(np.float32))
        Oc, Yc, Wc = np.concatenate(O), np.concatenate(Y), np.concatenate(W)
        ys = L.policy_mean(student, obs)
        log(f"   [dagger {it}] new={len(obs)} total={len(Oc)} teacher-changes-action on {100*changed.mean():.1f}% of states | "
            f"student err before fit: all={np.abs(ys-y).mean():.4f} on-changed={np.abs(ys-y)[changed].mean():.4f}  ({time.time()-t:.0f}s)")
        student, state = fit(student, p0, Oc, Yc, Wc, subset, steps, lr, seed=it, state=state, log=log)
        ys = L.policy_mean(student, obs)
        log(f"   [dagger {it}] after fit: all={np.abs(ys-y).mean():.4f} on-changed={np.abs(ys-y)[changed].mean():.4f} off-changed={np.abs(ys-y)[~changed].mean():.4f}")
    return student
