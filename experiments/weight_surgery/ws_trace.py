"""Per-step tracing rollouts (small batch) — used for activation analysis and visualisation."""
import numpy as np, jax, jax.numpy as jnp
import ws_lib as L


def _make_trace(act_fn, T):
    def run(params, key, scen, v):
        k_reset, k_step = jax.random.split(key)
        carry = L._init_carry(k_reset, scen, v)
        st0 = carry[0].env_state
        n = scen.shape[0]

        def body(c, i):
            st, obs, wp, active = c
            act = act_fn(params, obs, st.env_state.max_v)
            keys = jax.random.split(jax.random.fold_in(k_step, i), n)
            obs2, st2, wp2, o = jax.vmap(L._step_one)(keys, st, act, wp, scen)
            es = st.env_state
            rec = dict(obs=obs, active=active, x=es.x, y=es.y, th=es.theta, act=act,
                       gx=es.goal_x, gy=es.goal_y, ppl=es.people[:, :, :2], pidx=es.people[:, :, 10],
                       rew=o["rew"], succ=o["succ"], col=o["a_col"] | o["p_col"] | o["o_col"], tmo=o["tmo"])
            return (st2, obs2, wp2, active & ~o["ep_done"]), rec

        _, rec = jax.lax.scan(body, carry[:4], jnp.arange(T))
        static = dict(circ=st0.obs_circles, boxes=st0.obs_boxes, room_w=st0.room_w, room_h=st0.room_h,
                      max_v=st0.max_v)
        return rec, static
    return jax.jit(run)


_trace_cache = {}


def trace(params, seed=0, n_per_scen=4, T=600, scens=L.SCENS, act_fn=None):
    act_fn = act_fn or L._policy_action
    k = (act_fn, T)
    if k not in _trace_cache:
        _trace_cache[k] = _make_trace(act_fn, T)
    scen, v = L.make_batch(n_per_scen, scens)
    rec, static = _trace_cache[k](jax.tree_util.tree_map(jnp.asarray, params),
                                  jax.random.PRNGKey(seed), scen, v)
    rec = {a: np.asarray(b) for a, b in rec.items()}
    static = {a: np.asarray(b) for a, b in static.items()}
    static["scen"] = np.asarray(scen)
    return rec, static
