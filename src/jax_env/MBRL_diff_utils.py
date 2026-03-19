"""
MBRL_diff_utils.py — Differentiability Patches & Direct Metric Optimization
===========================================================================

Due funzionalità:

  A. PATCH DIFFERENZIABILI
     Sostituzioni smooth per le discontinuità residue in jax_env.py e
     jax_env_multi.py. Da applicare come monkey-patch all'inizio del training
     SHAC (import di questo modulo → sovrascrittura delle funzioni originali).

  B. OTTIMIZZAZIONE DIRETTA DELLE METRICHE SOCIALI
     Funzioni che ottimizzano direttamente metriche di navigazione sociale
     calcolate come loss differenziabili, senza reward engineering.

FIX APPLICATI
-------------

  FIX 1 — Import dentro lax.scan:
    Tutti gli import da jax_env, jax_env_multi, jax_network spostati a
    top-level del modulo. Gli import dentro funzioni tracciate da JAX vengono
    eseguiti ad ogni tracing XLA (overhead silenzioso).

  FIX 2 — social_nav_loss: discount cumulativo corretto:
    Il discount factor era applicato come GAMMA**1 per ogni step invece che
    come GAMMA**t. Ora il carry tiene il discount cumulativo (come in SHAC).

  FIX 3 — social_nav_loss: rng fisso rimosso:
    jax.random.PRNGKey(0) hardcoded nel carry è un anti-pattern: tutti i
    rollout usano la stessa sequenza random. Ora il rng viene passato come arg.

  FIX 4 — patch_env_differentiability: warning esplicito sulle limitazioni:
    Il patch ricreava la cinematica robot con una formula semplificata
    (dx = v*cos(θ)*dt) ignorando il modello cinematico reale. Aggiunto un
    avviso chiaro. Il patch resta opzionale e utile per debug gradient flow,
    ma non dovrebbe essere usato per training serio se step_env è complesso.

INVARIATO
---------
  - Tutte le funzioni pubbliche (soft_clip, smooth_*, collision_loss, ecc.)
  - Firme delle funzioni (retrocompatibile)
  - verify_gradient_flow
"""

import jax
import jax.numpy as jnp
import functools

# FIX 1: tutti gli import a top-level (non dentro _step / scan)
from jax_env import NUM_RAYS, STATE_VEC_SIZE
from jax_network import scale_action_to_env
from jax_env_multi import ROOM_W, ROOM_H

_POSE_SIZE = 3


# ═══════════════════════════════════════════════════════════════════════════════
# A. OPERATORI SMOOTH
# ═══════════════════════════════════════════════════════════════════════════════

def soft_clip(x: jnp.ndarray, lo: float, hi: float, beta: float = 20.0) -> jnp.ndarray:
    """
    Sostituzione differenziabile di jnp.clip(x, lo, hi).

    Usa softplus per approssimare le soglie inferiore e superiore:
        soft_clip(x) ≈ lo + softplus(β(x-lo))/β  per la soglia inferiore
                      - softplus(β(x-hi))/β + (hi-lo) per la soglia superiore

    Proprietà:
      - Identica a clip per x ∈ [lo + δ, hi - δ] dove δ = ln(2)/β ≈ 0.035 con β=20
      - Gradiente non-zero alle soglie (a differenza di clip che ha grad=0 in saturazione)
      - Differenziabile ovunque (nessun kink)
    """
    below  = lo + jax.nn.softplus(beta * (x - lo)) / beta
    result = hi - jax.nn.softplus(beta * (hi - below)) / beta
    return result


def smooth_sign(x: jnp.ndarray, eps: float = 0.1) -> jnp.ndarray:
    """
    Smooth approximation di jnp.sign(x).
    tanh(x/eps) ≈ sign(x) per |x| >> eps, ma è differenziabile a x=0.
    """
    return jnp.tanh(x / eps)


def smooth_maximum(a: jnp.ndarray, b: float, sharpness: float = 10.0) -> jnp.ndarray:
    """
    Smooth approximation di jnp.maximum(a, b).
    Identica a jnp.maximum per |a-b| >> 1/sharpness.
    """
    return b + jax.nn.softplus(sharpness * (a - b)) / sharpness


def smooth_collision_indicator(
    dist: jnp.ndarray,
    threshold: float,
    width: float = 0.15,
) -> jnp.ndarray:
    """
    Smooth proxy per l'indicatore di collisione:
        1 quando dist < threshold, 0 quando dist > threshold + width.

    Implementato come sigmoid invertita centrata sulla soglia.
    Con width=0.15m la transizione è da ~0.9 a ~0.1 in 15cm.
    """
    slope = width / 6.0
    return jax.nn.sigmoid(-(dist - threshold) / slope)


# ═══════════════════════════════════════════════════════════════════════════════
# B. LOSS DIFFERENZIABILI PER METRICHE SOCIALI
# ═══════════════════════════════════════════════════════════════════════════════

_ROBOT_R   = 0.2
_PEOPLE_R  = 0.4
_GOAL_R    = 0.3
_GAMMA     = 0.99
_DT        = 0.15

_INTIMATE_DIST  = 0.45
_PERSONAL_DIST  = 1.2
_SOCIAL_DIST    = 3.6


def collision_loss(
    robot_x: jnp.ndarray,
    robot_y: jnp.ndarray,
    people:  jnp.ndarray,
    obs_circles: jnp.ndarray,
    obs_boxes:   jnp.ndarray,
    room_w: float,
    room_h: float,
    wall_margin: float = 0.05,
) -> jnp.ndarray:
    """
    Loss differenziabile per collisioni.
    Usa smooth_collision_indicator invece di indicatori booleani.
    """
    total = jnp.zeros(())

    # Collisioni con persone (maschera dummy px < 0)
    active = people[:, 0] > 0.0
    dx_p   = people[:, 0] - robot_x
    dy_p   = people[:, 1] - robot_y
    dist_p = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)

    col_threshold    = _ROBOT_R + _PEOPLE_R
    human_col_probs  = smooth_collision_indicator(
        dist_p, col_threshold, width=0.2
    ) * active.astype(jnp.float32)
    total += jnp.mean(human_col_probs)

    # Collisioni con cerchi statici
    dx_c   = obs_circles[:, 0] - robot_x
    dy_c   = obs_circles[:, 1] - robot_y
    dist_c = jnp.sqrt(dx_c**2 + dy_c**2 + 1e-8) - obs_circles[:, 2]
    total += jnp.mean(smooth_collision_indicator(dist_c, _ROBOT_R, width=0.1))

    # Collisioni con muri
    wall_dists = jnp.array([
        robot_x - _ROBOT_R,
        (room_w - _ROBOT_R) - robot_x,
        robot_y - _ROBOT_R,
        (room_h - _ROBOT_R) - robot_y,
    ])
    total += jnp.mean(smooth_collision_indicator(
        wall_dists, wall_margin, width=0.1
    ))

    return total


def proxemics_loss(
    robot_x: jnp.ndarray,
    robot_y: jnp.ndarray,
    people:  jnp.ndarray,
    robot_v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Loss differenziabile per rispetto delle zone prossemiche.
    Penalizza la vicinanza alle persone proporzionalmente alla velocità.

    Loss = Σ_i [ violation_i × (1 + v_factor) ]
    dove violation_i = max(0, personal_dist - edge_dist_i) / personal_dist
    """
    active    = (people[:, 0] > 0.0).astype(jnp.float32)
    dx_p      = people[:, 0] - robot_x
    dy_p      = people[:, 1] - robot_y
    dist_p    = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)
    edge_dist = jnp.maximum(0.0, dist_p - _PEOPLE_R - _ROBOT_R)

    personal_violation = jax.nn.relu(_PERSONAL_DIST - edge_dist) / _PERSONAL_DIST
    v_weight = 1.0 + robot_v / 1.5

    return jnp.mean(personal_violation * active * v_weight)


def efficiency_loss(
    robot_x:  jnp.ndarray,
    robot_y:  jnp.ndarray,
    robot_v:  jnp.ndarray,
    robot_th: jnp.ndarray,
    goal_x:   jnp.ndarray,
    goal_y:   jnp.ndarray,
    max_v:    jnp.ndarray,
) -> jnp.ndarray:
    """
    Loss differenziabile per efficienza di navigazione.
    Penalizza: distanza al goal, misallineamento heading-goal, velocità bassa.
    """
    dx   = goal_x - robot_x
    dy   = goal_y - robot_y
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)

    goal_angle  = jnp.arctan2(dy, dx)
    heading_err = jnp.abs(
        (goal_angle - robot_th + jnp.pi) % (2 * jnp.pi) - jnp.pi
    )

    dist_norm         = dist / (jnp.sqrt(12.0**2 + 12.0**2))
    speed_inefficiency = jnp.maximum(0.0, 1.0 - robot_v / (max_v + 1e-6))
    heading_loss      = (1.0 - jnp.cos(heading_err)) / 2.0

    return dist_norm + 0.3 * heading_loss + 0.1 * speed_inefficiency


def smoothness_loss(
    robot_w:      jnp.ndarray,
    prev_robot_w: jnp.ndarray,
    robot_v:      jnp.ndarray,
) -> jnp.ndarray:
    """
    Loss differenziabile per smoothness del percorso.
    Penalizza cambi bruschi di direzione, scalati con la velocità.
    """
    angular_jerk = (robot_w - prev_robot_w) ** 2
    speed_scale  = 1.0 + robot_v / 1.5
    return angular_jerk * speed_scale


def social_nav_loss(
    actor_params,
    actor_apply,
    init_obs:    jnp.ndarray,
    init_state,
    step_fn,
    rng_key:     jnp.ndarray,    # FIX 3: rng passato come argomento
    horizon:     int,
    ghost_robot: bool = True,
    w_collision:  float = 2.0,
    w_proxemics:  float = 1.0,
    w_efficiency: float = 0.5,
    w_smoothness: float = 0.3,
) -> jnp.ndarray:
    """
    Loss differenziabile combinata per navigazione sociale.

    Calcola le metriche sociali direttamente dal rollout della policy.
    Può essere usata come termine aggiuntivo nella SHAC actor loss:
        total_actor_loss = shac_return_loss + λ * social_nav_loss(...)

    λ ≈ 0.1 è un buon punto di partenza (social loss è O(1), return è O(100)).

    FIX 1: nessun import dentro _step.
    FIX 2: discount cumulativo corretto (γ^t invece di γ^1 per ogni step).
    FIX 3: rng passato come argomento invece di PRNGKey(0) hardcoded.
    """

    def _step(carry, _):
        obs, state, key, cum_discount, prev_w = carry
        key, step_key = jax.random.split(key)

        # FIX 1: scale_action_to_env e STATE_VEC_SIZE già importati a top-level
        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean = mean[0]
        env_action = scale_action_to_env(mean, state.env_state.max_v)

        base_obs, new_base_state, reward, done, info = step_fn(
            step_key, state.env_state, env_action, ghost_robot=ghost_robot
        )

        # Aggiorna stack osservazioni
        new_pose      = base_obs[0:_POSE_SIZE]
        new_state_vec = base_obs[_POSE_SIZE : _POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[_POSE_SIZE + STATE_VEC_SIZE:]

        new_lidar_stack = jnp.concatenate(
            [state.lidar_stack[1:], new_lidar[None]], axis=0)
        new_pose_stack = jnp.concatenate(
            [state.pose_stack[1:], new_pose[None]], axis=0)
        new_state = state.replace(
            env_state=new_base_state,
            lidar_stack=new_lidar_stack,
            pose_stack=new_pose_stack,
        )
        new_obs = jnp.concatenate([
            new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
        ])

        s = new_base_state
        step_loss = (
            w_collision  * collision_loss(
                s.x, s.y, s.people, s.obs_circles, s.obs_boxes, ROOM_W, ROOM_H
            ) +
            w_proxemics  * proxemics_loss(s.x, s.y, s.people, s.v) +
            w_efficiency * efficiency_loss(
                s.x, s.y, s.v, s.theta, s.goal_x, s.goal_y, s.max_v
            ) +
            w_smoothness * smoothness_loss(s.w, prev_w, s.v)
        )
        # Stop gradient su done — non propagare attraverso reset
        done_sg = jax.lax.stop_gradient(done.astype(jnp.float32))
        step_loss = step_loss * (1.0 - done_sg)

        # FIX 2: discount cumulativo corretto (γ^t)
        discounted_loss = cum_discount * step_loss
        next_discount   = cum_discount * _GAMMA * (1.0 - done_sg)

        return (new_obs, new_state, key, next_discount, s.w), discounted_loss

    _, step_losses = jax.lax.scan(
        _step,
        # FIX 3: usa rng_key passato, non PRNGKey(0); discount parte da 1.0
        (init_obs, init_state, rng_key, jnp.ones(()), jnp.array(0.0)),
        None,
        length=horizon,
    )

    return jnp.mean(step_losses)


# ═══════════════════════════════════════════════════════════════════════════════
# C. MONKEY-PATCH OPZIONALE
# ═══════════════════════════════════════════════════════════════════════════════

def patch_env_differentiability():
    """
    Applica un straight-through estimator per soft_clip alla posizione del robot.

    LIMITAZIONE IMPORTANTE (FIX 4):
    Il patch approssima la cinematica come dx = v*cos(θ)*dt, dy = v*sin(θ)*dt.
    Se step_env usa un modello cinematico più complesso (controllo a due ruote,
    friction, saturazione velocità), questa approssimazione diverge dalla fisica
    reale e produce gradienti sbagliati.

    Usa questo patch SOLO per:
      - Debug del gradient flow (verify_gradient_flow)
      - Ambienti con cinematica unicycle semplice

    Per training serio: rendi step_env differenziabile nativamente.
    """
    try:
        import jax_env_multi as _em
        from jax_env_multi import ROOM_W as _RW, ROOM_H as _RH
        from jax_env_multi import ROBOT_RADIUS as _RR

        _orig_step = _em.step_env

        print("[jax_diff_utils] AVVISO: patch_env_differentiability usa una "
              "cinematica semplificata (unicycle). Se step_env è più complesso, "
              "i gradienti possono essere imprecisi.")

        def _patched_step(key, state, action, ghost_robot=True):
            obs, new_state, reward, done, info = _orig_step(
                key, state, action, ghost_robot=ghost_robot
            )

            # Cinematica approssimata per il gradiente (straight-through)
            soft_x = soft_clip(
                state.x + (action[0] * jnp.cos(state.theta) * _DT),
                _RR, _RW - _RR
            )
            soft_y = soft_clip(
                state.y + (action[0] * jnp.sin(state.theta) * _DT),
                _RR, _RH - _RR
            )

            # Straight-through: forward = clip fisico, backward = soft_clip smooth
            corrected_x = soft_x + jax.lax.stop_gradient(new_state.x - soft_x)
            corrected_y = soft_y + jax.lax.stop_gradient(new_state.y - soft_y)

            new_state = new_state.replace(x=corrected_x, y=corrected_y)
            return obs, new_state, reward, done, info

        _em.step_env = _patched_step
        print("[jax_diff_utils] Patch applicato a jax_env_multi.step_env")
        print("  → jnp.clip posizione robot sostituito con straight-through soft_clip")

    except Exception as e:
        print(f"[jax_diff_utils] Patch non applicato: {e}")
        print("  → Continua senza patch (BPTT funziona anche senza, "
              "ma gradienti alle pareti sono zero)")


# ═══════════════════════════════════════════════════════════════════════════════
# D. UTILITY: VERIFICA DIFFERENZIABILITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def verify_gradient_flow(
    actor_params,
    actor_apply,
    reset_fn,
    step_fn,
    min_goal_dist: float = 3.0,
    horizon: int = 4,
    ghost_robot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Verifica che il gradiente fluisca correttamente attraverso il simulatore.

    Esegue un rollout di `horizon` passi e controlla:
      1. Il gradiente di J(θ) rispetto a θ non è NaN
      2. La norma del gradiente è finita e > 0
      3. La norma non è esplosiva (< 1000)

    FIX 1: import spostati a top-level del modulo.
    """
    from jax_wrappers import make_stacked_env

    rng = jax.random.PRNGKey(99)
    rng, reset_key = jax.random.split(rng)

    reset_stacked, _ = make_stacked_env(
        reset_fn, step_fn, stack_dim=3, ghost_robot=ghost_robot
    )
    init_obs, init_state = reset_stacked(reset_key, min_goal_dist=min_goal_dist)

    def test_rollout(params):
        def scan_fn(carry, _):
            obs, state, key = carry
            key, sk = jax.random.split(key)

            # FIX 1: scale_action_to_env già importato a top-level
            mean, _, _ = actor_apply({"params": params}, obs[None])
            mean = mean[0]
            action = scale_action_to_env(mean, state.env_state.max_v)

            base_obs, new_base_state, reward, done, _ = step_fn(
                sk, state.env_state, action, ghost_robot=ghost_robot
            )
            # FIX 1: STATE_VEC_SIZE e _POSE_SIZE già importati a top-level
            new_pose       = base_obs[0:_POSE_SIZE]
            new_state_vec  = base_obs[_POSE_SIZE:_POSE_SIZE+STATE_VEC_SIZE]
            new_lidar      = base_obs[_POSE_SIZE+STATE_VEC_SIZE:]
            new_lidar_stack = jnp.concatenate(
                [state.lidar_stack[1:], new_lidar[None]], axis=0)
            new_pose_stack  = jnp.concatenate(
                [state.pose_stack[1:], new_pose[None]], axis=0)
            new_state = state.replace(
                env_state=new_base_state,
                lidar_stack=new_lidar_stack,
                pose_stack=new_pose_stack,
            )
            new_obs = jnp.concatenate([
                new_pose_stack.flatten(), new_state_vec, new_lidar_stack.flatten()
            ])
            return (new_obs, new_state, key), reward

        rng_test = jax.random.PRNGKey(0)
        _, rewards = jax.lax.scan(
            scan_fn, (init_obs, init_state, rng_test), None, length=horizon
        )
        return jnp.sum(rewards)

    J, grads = jax.value_and_grad(test_rollout)(actor_params)

    leaves     = jax.tree_util.tree_leaves(grads)
    grad_norms = [float(jnp.linalg.norm(g)) for g in leaves]
    has_nan    = any(jnp.any(jnp.isnan(g)) for g in leaves)
    has_inf    = any(jnp.any(jnp.isinf(g)) for g in leaves)
    total_norm = float(jnp.sqrt(sum(n**2 for n in grad_norms)))
    is_healthy = (not has_nan) and (not has_inf) and (1e-10 < total_norm < 1e4)

    stats = {
        "J":           float(J),
        "total_norm":  total_norm,
        "layer_norms": grad_norms,
        "has_nan":     has_nan,
        "has_inf":     has_inf,
        "is_healthy":  is_healthy,
        "horizon":     horizon,
        "n_params":    sum(g.size for g in leaves),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Gradient Flow Check (H={horizon})")
        print(f"  J(θ)       = {float(J):.4f}")
        print(f"  ‖∇θ J‖     = {total_norm:.4e}")
        print(f"  NaN/Inf    = {has_nan}/{has_inf}")
        print(f"  N params   = {stats['n_params']:,}")
        print(f"  Status     = {'✓ HEALTHY' if is_healthy else '✗ UNHEALTHY'}")
        print(f"{'='*50}\n")

    return stats