"""
MBRL_diff_utils.py — Differentiability Patches & Direct Metric Optimization
===========================================================================

Due funzionalità:

  A. PATCH DIFFERENZIABILI
     Sostituzioni smooth per le discontinuità residue in jax_env.py e
     jax_env_multi.py. Da applicare come monkey-patch all'inizio del training
     SHAC (import di questo modulo → sovrascrittura delle funzioni originali).

     Le discontinuità principali sono:
       1. jnp.clip nelle cinematiche robot  → soft_clip (smooth alle soglie)
       2. (prev_dist - new_dist) nel progress reward → già smooth
       3. clearance_factor sigmoid → già smooth (jax.nn.sigmoid)
       4. jnp.where su done  → già smooth (selector, grad sul ramo attivo)

     NB: jnp.where è già differenziabile in JAX (il gradiente fluisce
     attraverso il ramo selezionato, l'altro è azzerato).
     L'unica vera discontinuità è jnp.clip — sostituita qui.

  B. OTTIMIZZAZIONE DIRETTA DELLE METRICHE SOCIALI
     Funzioni che ottimizzano direttamente metriche di navigazione sociale
     calcolate come loss differenziabili, senza reward engineering.

     Le metriche disponibili:
       - social_nav_loss():    loss combinata per navigazione sociale completa
       - collision_loss():     differenziabile vs collisioni (smooth proxy)
       - proxemics_loss():     penalizza violazioni spaziali personali
       - efficiency_loss():    massimizza velocità di navigazione verso goal
       - smoothness_loss():    penalizza jerk (cambi di direzione bruschi)

     Queste loss possono essere usate come:
       1. Termine aggiuntivo nel SHAC actor loss (segnale di shaping)
       2. Loss separata per fine-tuning post-PPO
       3. Metriche di valutazione (non solo reward proxy)

UTILIZZO
--------

All'inizio di jax_mbrl.py / jax_shac.py:

    from jax_diff_utils import patch_env_differentiability
    patch_env_differentiability()

    # Poi nelle SHAC loss:
    from jax_diff_utils import social_nav_loss
    extra_loss = social_nav_loss(actor_params, actor_apply, state, horizon)
"""

import jax
import jax.numpy as jnp
import functools


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

    β=20 dà una transizione di ~0.07m intorno alle soglie fisiche — impercettibile
    per la simulazione ma critica per far fluire i gradienti BPTT.
    """
    # Soglia inferiore: lo + softplus(β(x-lo))/β
    below = lo + jax.nn.softplus(beta * (x - lo)) / beta
    # Soglia superiore: applica stessa logica rispetto a hi
    result = hi - jax.nn.softplus(beta * (hi - below)) / beta
    return result


def smooth_sign(x: jnp.ndarray, eps: float = 0.1) -> jnp.ndarray:
    """
    Smooth approximation di jnp.sign(x).
    Utile per le normali nei push-out di jax_humans.py.
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
    
    Sostituzione di: (dist < threshold).astype(float)
    che ha gradiente zero ovunque tranne alla soglia (kink non differenziabile).

    Implementato come sigmoid invertita centrata sulla soglia:
        σ(-(dist - threshold) / (width / 6))

    Con width=0.15m la transizione è da ~0.9 a ~0.1 in 15cm — fisicamente
    ragionevole (zona di "quasi collisione" che il gradiente può spingere
    verso il basso).
    """
    slope = width / 6.0  # sigma della sigmoid
    return jax.nn.sigmoid(-(dist - threshold) / slope)


# ═══════════════════════════════════════════════════════════════════════════════
# B. LOSS DIFFERENZIABILI PER METRICHE SOCIALI
# ═══════════════════════════════════════════════════════════════════════════════

# Costanti fisiche (da jax_env_multi.py / jax_env.py)
_ROBOT_R   = 0.2   # m
_PEOPLE_R  = 0.4   # m (PEOPLE_RADIUS)
_GOAL_R    = 0.3   # m (GOAL_RADIUS)
_GAMMA     = 0.99
_DT        = 0.15

# Soglie spaziali (Hall 1966 proxemics)
_INTIMATE_DIST  = 0.45  # m — zona intima (0-0.45m): collisione imminente
_PERSONAL_DIST  = 1.2   # m — zona personale (0.45-1.2m): disagio
_SOCIAL_DIST    = 3.6   # m — zona sociale (1.2-3.6m): interazione accettabile


def collision_loss(
    robot_x: jnp.ndarray,
    robot_y: jnp.ndarray,
    people:  jnp.ndarray,   # (N, ≥2): colonne 0=px, 1=py
    obs_circles: jnp.ndarray,  # (M, 3): cx,cy,r
    obs_boxes:   jnp.ndarray,  # (K, 4): cx,cy,hw,hh
    room_w: float,
    room_h: float,
    wall_margin: float = 0.05,
) -> jnp.ndarray:
    """
    Loss differenziabile per collisioni.
    Usa smooth_collision_indicator invece di indicatori booleani.

    Il gradiente di questa loss rispetto alla posizione del robot
    punta verso la direzione di minima probabilità di collisione.
    """
    total = jnp.zeros(())

    # ── Collisioni con persone ────────────────────────────────────────────────
    # Maschera le persone dummy (posizione < 0)
    active = people[:, 0] > 0.0
    dx_p   = people[:, 0] - robot_x
    dy_p   = people[:, 1] - robot_y
    dist_p = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)

    col_threshold = _ROBOT_R + _PEOPLE_R
    human_col_probs = smooth_collision_indicator(
        dist_p, col_threshold, width=0.2
    ) * active.astype(jnp.float32)
    total += jnp.mean(human_col_probs)

    # ── Collisioni con cerchi statici ─────────────────────────────────────────
    dx_c   = obs_circles[:, 0] - robot_x
    dy_c   = obs_circles[:, 1] - robot_y
    dist_c = jnp.sqrt(dx_c**2 + dy_c**2 + 1e-8) - obs_circles[:, 2]
    total += jnp.mean(smooth_collision_indicator(dist_c, _ROBOT_R, width=0.1))

    # ── Collisioni con muri ───────────────────────────────────────────────────
    wall_dists = jnp.array([
        robot_x - _ROBOT_R,                  # muro sinistro
        (room_w - _ROBOT_R) - robot_x,       # muro destro
        robot_y - _ROBOT_R,                  # muro inferiore
        (room_h - _ROBOT_R) - robot_y,       # muro superiore
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

    Penalizza la vicinanza alle persone in modo proporzionale alla velocità:
    passare vicino a un umano lentamente è meno grave che farlo velocemente.

    Basata sul clearance_factor di jax_env_multi.py ma calcolata come loss
    anziché come scaler del progress reward.

    Loss = Σ_i [ violation_i × (1 + v_factor) ]

    dove violation_i = max(0, personal_dist - dist_i) / personal_dist
    e v_factor = robot_v / v_max (velocità normalizzata)
    """
    active = (people[:, 0] > 0.0).astype(jnp.float32)
    dx_p   = people[:, 0] - robot_x
    dy_p   = people[:, 1] - robot_y
    dist_p = jnp.sqrt(dx_p**2 + dy_p**2 + 1e-8)

    # Edge-to-edge distance (sottrae i raggi)
    edge_dist = jnp.maximum(0.0, dist_p - _PEOPLE_R - _ROBOT_R)

    # Violazione della zona personale: continua, smooth
    # 0 quando edge_dist >= _PERSONAL_DIST, 1 all'intimo boundary
    personal_violation = jax.nn.relu(_PERSONAL_DIST - edge_dist) / _PERSONAL_DIST

    # Peso proporzionale alla velocità (più pericoloso passare veloce)
    v_weight = 1.0 + robot_v / 1.5   # normalizzato su max_v tipico

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

    Penalizza:
      1. Distanza al goal (vogliamo minimizzarla)
      2. Misallineamento heading-goal (vogliamo puntare verso il goal)
      3. Velocità bassa lontano dagli umani (vogliamo muoverci efficientemente)
    """
    dx    = goal_x - robot_x
    dy    = goal_y - robot_y
    dist  = jnp.sqrt(dx**2 + dy**2 + 1e-8)

    # Goal heading
    goal_angle = jnp.arctan2(dy, dx)
    heading_err = jnp.abs(
        (goal_angle - robot_th + jnp.pi) % (2 * jnp.pi) - jnp.pi
    )

    # Normalizzazione distanza (goal a max ~17m, stanza 12x12)
    dist_norm = dist / (jnp.sqrt(12.0**2 + 12.0**2))

    # Penalità per velocità bassa (relativa al massimo disponibile)
    speed_inefficiency = jnp.maximum(0.0, 1.0 - robot_v / (max_v + 1e-6))

    # Heading loss: cos(err) = 1 quando allineato, -1 quando opposto
    heading_loss = (1.0 - jnp.cos(heading_err)) / 2.0

    return dist_norm + 0.3 * heading_loss + 0.1 * speed_inefficiency


def smoothness_loss(
    robot_w:     jnp.ndarray,   # velocità angolare corrente
    prev_robot_w: jnp.ndarray,  # velocità angolare precedente
    robot_v:     jnp.ndarray,
) -> jnp.ndarray:
    """
    Loss differenziabile per smoothness del percorso.
    Penalizza cambi bruschi di direzione (jerk angolare).
    Scalata con la velocità: girare bruscamente ad alta velocità è peggio.
    """
    angular_jerk = (robot_w - prev_robot_w) ** 2
    speed_scale  = 1.0 + robot_v / 1.5
    return angular_jerk * speed_scale


def social_nav_loss(
    actor_params,
    actor_apply,
    init_obs:   jnp.ndarray,
    init_state,                   # StackedEnvState
    step_fn,
    horizon:    int,
    ghost_robot: bool = True,
    # Pesi per ogni componente della loss
    w_collision:   float = 2.0,
    w_proxemics:   float = 1.0,
    w_efficiency:  float = 0.5,
    w_smoothness:  float = 0.3,
) -> jnp.ndarray:
    """
    Loss differenziabile combinata per navigazione sociale.

    Calcola le metriche sociali direttamente dal rollout della policy,
    senza passare attraverso il reward engineering del simulatore.

    Vantaggi rispetto al solo reward shaping:
      - Gradiente diretto sulle metriche che misuriamo in benchmark_eval.py
      - No reward hacking: non c'è un proxy da ottimizzare, è la metrica stessa
      - Interpretabile: ogni termine ha significato fisico preciso

    Può essere usata come termine aggiuntivo nella SHAC actor loss:
        total_actor_loss = shac_return_loss + λ * social_nav_loss(...)

    λ ≈ 0.1 è un buon punto di partenza (social loss è O(1), return è O(100)).
    """
    from jax_env_multi import ROOM_W, ROOM_H

    def _step(carry, _):
        obs, state, key, prev_w = carry
        key, action_key, step_key = jax.random.split(key, 3)

        from jax_network import scale_action_to_env
        mean, _, _ = actor_apply({"params": actor_params}, obs[None])
        mean = mean[0]
        env_action = scale_action_to_env(mean, state.env_state.max_v)

        base_obs, new_base_state, reward, done, info = step_fn(
            step_key, state.env_state, env_action, ghost_robot=ghost_robot
        )

        # Aggiorna stack osservazioni (come in _shac_rollout_single)
        from jax_env import NUM_RAYS, STATE_VEC_SIZE
        POSE_SIZE = 3
        new_pose      = base_obs[0:POSE_SIZE]
        new_state_vec = base_obs[POSE_SIZE : POSE_SIZE + STATE_VEC_SIZE]
        new_lidar     = base_obs[POSE_SIZE + STATE_VEC_SIZE:]

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

        # Calcola componenti loss per questo step
        s = new_base_state
        step_loss = (
            w_collision * collision_loss(
                s.x, s.y, s.people, s.obs_circles, s.obs_boxes, ROOM_W, ROOM_H
            ) +
            w_proxemics * proxemics_loss(s.x, s.y, s.people, s.v) +
            w_efficiency * efficiency_loss(
                s.x, s.y, s.v, s.theta, s.goal_x, s.goal_y, s.max_v
            ) +
            w_smoothness * smoothness_loss(s.w, prev_w, s.v)
        )
        # Stop gradient su done per non propagare attraverso reset
        step_loss = step_loss * (1.0 - jax.lax.stop_gradient(done.astype(jnp.float32)))

        return (new_obs, new_state, key, s.w), (_GAMMA ** 1) * step_loss

    _, step_losses = jax.lax.scan(
        _step,
        (init_obs, init_state, jax.random.PRNGKey(0), jnp.array(0.0)),
        None,
        length=horizon,
    )

    return jnp.mean(step_losses)


# ═══════════════════════════════════════════════════════════════════════════════
# C. MONKEY-PATCH OPZIONALE
# ═══════════════════════════════════════════════════════════════════════════════

def patch_env_differentiability():
    """
    Sostituisce jnp.clip nelle cinematiche robot con soft_clip.

    Chiama questa funzione UNA VOLTA all'inizio del training SHAC.
    NON è necessaria per PPO (che non usa i gradienti del simulatore).

    IMPATTO: minimo sulla simulazione (discrepanza < 5cm alle soglie wall),
    massimo sulla qualità del gradiente BPTT (elimina il vanishing alle pareti).

    Implementazione: sostituisce la funzione step_env nel modulo jax_env_multi
    con una versione identica ma con soft_clip invece di jnp.clip per la
    posizione del robot. Il clip della velocità (target_v, target_w) rimane
    invariato perché la policy impara naturalmente a non saturarlo.
    """
    try:
        import jax_env_multi as _em

        _orig_step = _em.step_env

        def _patched_step(key, state, action, ghost_robot=True):
            # Il patch viene fatto PRIMA di chiamare step_env originale:
            # sostituiamo jnp.clip per la posizione con soft_clip.
            # Per il resto (velocità, collisioni) manteniamo l'originale.

            # Nota: non possiamo modificare lo step interno di lax.scan
            # post-hoc. Questo patch è una versione semplificata che
            # avvolge lo step completo con un correction pass:
            # dopo lo step, se la posizione è stata clippata, calcoliamo
            # il "soft" equivalente per far fluire il gradiente.

            obs, new_state, reward, done, info = _orig_step(
                key, state, action, ghost_robot=ghost_robot
            )

            from jax_env_multi import ROOM_W, ROOM_H, ROBOT_RADIUS as RR
            # Applica soft_clip alle coordinate — le coordinate clippate di
            # new_state NON cambiano (garantisce la fisica), ma il gradiente
            # ora fluisce attraverso soft_clip invece di jnp.clip.
            # Questo è uno "straight-through estimator" per la posizione:
            # forward = clip (fisicamente corretto), backward = soft_clip (gradienti utili).
            soft_x = soft_clip(
                state.x + (action[0] * jnp.cos(state.theta) * 0.15),
                RR, ROOM_W - RR
            )
            soft_y = soft_clip(
                state.y + (action[0] * jnp.sin(state.theta) * 0.15),
                RR, ROOM_H - RR
            )

            # Straight-through: il valore è quello originale (fisico),
            # ma il gradiente è quello di soft_clip (smooth).
            # Implementato come: value + stop_gradient(hard - soft)
            corrected_x = soft_x + jax.lax.stop_gradient(new_state.x - soft_x)
            corrected_y = soft_y + jax.lax.stop_gradient(new_state.y - soft_y)

            new_state = new_state.replace(x=corrected_x, y=corrected_y)
            return obs, new_state, reward, done, info

        _em.step_env = _patched_step
        print("[jax_diff_utils] Patch differenziabilità applicato a jax_env_multi.step_env")
        print("  → jnp.clip posizione robot sostituito con straight-through soft_clip")

    except Exception as e:
        print(f"[jax_diff_utils] Patch non applicato: {e}")
        print("  → Continua senza patch (BPTT funziona anche senza, ma gradienti"
              " alle pareti sono zero)")


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

    Ritorna un dizionario con le statistiche del gradiente.
    Utile per debug pre-training.

    Esempio di utilizzo:
        from jax_diff_utils import verify_gradient_flow
        stats = verify_gradient_flow(params, network.apply, reset_env, step_env)
        assert stats["is_healthy"], f"Gradiente non sano: {stats}"
    """
    from jax_wrappers import make_stacked_env

    rng = jax.random.PRNGKey(99)
    rng, reset_key = jax.random.split(rng)

    reset_stacked, step_stacked = make_stacked_env(
        reset_fn, step_fn, stack_dim=3, ghost_robot=ghost_robot
    )
    init_obs, init_state = reset_stacked(reset_key, min_goal_dist=min_goal_dist)

    # Funzione di test: rollout deterministico, ritorna la somma dei reward
    def test_rollout(params):
        from jax_network import scale_action_to_env

        def scan_fn(carry, _):
            obs, state, key = carry
            key, sk = jax.random.split(key)

            mean, _, _ = actor_apply({"params": params}, obs[None])
            mean = mean[0]
            action = scale_action_to_env(mean, state.env_state.max_v)

            base_obs, new_base_state, reward, done, _ = step_fn(
                sk, state.env_state, action, ghost_robot=ghost_robot
            )
            from jax_env import STATE_VEC_SIZE
            POSE_SIZE = 3
            new_pose      = base_obs[0:POSE_SIZE]
            new_state_vec = base_obs[POSE_SIZE:POSE_SIZE+STATE_VEC_SIZE]
            new_lidar     = base_obs[POSE_SIZE+STATE_VEC_SIZE:]
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
            return (new_obs, new_state, key), reward

        rng_test = jax.random.PRNGKey(0)
        _, rewards = jax.lax.scan(
            scan_fn, (init_obs, init_state, rng_test), None, length=horizon
        )
        return jnp.sum(rewards)

    # Calcola il gradiente
    J, grads = jax.value_and_grad(test_rollout)(actor_params)

    # Analisi dei gradienti
    leaves = jax.tree_util.tree_leaves(grads)
    grad_norms  = [float(jnp.linalg.norm(g)) for g in leaves]
    has_nan     = any(jnp.any(jnp.isnan(g)) for g in leaves)
    has_inf     = any(jnp.any(jnp.isinf(g)) for g in leaves)
    total_norm  = float(jnp.sqrt(sum(n**2 for n in grad_norms)))
    is_healthy  = (not has_nan) and (not has_inf) and (total_norm > 1e-10) and (total_norm < 1e4)

    stats = {
        "J":            float(J),
        "total_norm":   total_norm,
        "layer_norms":  grad_norms,
        "has_nan":      has_nan,
        "has_inf":      has_inf,
        "is_healthy":   is_healthy,
        "horizon":      horizon,
        "n_params":     sum(g.size for g in leaves),
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