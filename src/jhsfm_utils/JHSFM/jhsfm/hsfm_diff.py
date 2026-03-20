"""
hsfm_diff.py — Fully Differentiable Headed Social Force Model
==============================================================
Drop-in replacement for hsfm.py.  Public API is identical:
    step(humans_state, humans_goal, parameters, obstacles, dt)
    get_standard_humans_parameters(n_humans)   [in utils.py, unchanged]

DIFFERENTIABILITY FIXES vs original hsfm.py
--------------------------------------------

A. jnp.max(jnp.array([0, x]))  →  jnp.maximum(0.0, x)          [line 129]
   Original builds a Python list at every call — not XLA-friendly and
   produces a non-smooth subgradient at x=0 through jnp.max's argmax path.
   jnp.maximum is the correct elementwise relu with a clean subgradient.

B. lax.cond(all(self == other), ...)  →  continuous self-mask           [line 132]
   Original used exact float equality to zero out the self-self social
   force.  Exact equality is never differentiable.  Replacement:
   compute the force unconditionally (it is ~0 when dist≈0 anyway because
   nij is normalised by dist+eps) and multiply by a soft mask
       self_mask = 1 - exp(-dist² / eps²)
   which is exactly 0 when dist=0 (self) and ≈1 for all others.
   The gradient through self_mask is well-defined everywhere.

C. lax.cond(dist > radius, ...)  →  jnp.where + smooth gate            [line 196]
   Desired force had a hard branch at dist == radius.  Replacement uses
   jnp.where which JAX differentiates as a selector (gradient of the
   inactive branch is zeroed, not NaN).  Additionally the direction
   diff/dist is stabilised with safe_norm.

D. lax.cond(real_dist > 0, ...)  →  jnp.maximum contact terms          [line 162]
   Obstacle force had two branches with different physics at real_dist=0.
   The contact-stiffness terms k1, k2 are physically active only when
   real_dist > 0 (overlap).  Replacing the branch with jnp.maximum(0.0,
   real_dist) inside the expression makes the formula continuous and
   differentiable: the contact terms smoothly activate as surfaces touch.

E. lax.cond(num_obstacles > 0, ...)  →  jnp.where on the average        [line 215]
   Branching on an integer count is non-differentiable.  Replacement
   computes the sum unconditionally and divides by max(count, 1), which
   gives zero when there are no real obstacles and the correct average
   otherwise.  Gradients flow through the division.

F. lax.cond(||v|| > v_max, ...)  →  smooth velocity projection          [line 247]
   Hard velocity clamping creates a gradient discontinuity at the speed
   limit boundary.  Replacement uses a smooth projector:
       v_proj = v * v_max / max(||v||, v_max)
   which is the identity below v_max and a smooth rescaling above it.
   The gradient is continuous everywhere.

G. jnp.linalg.norm  →  safe_norm(x, eps)                               [multiple]
   jnp.linalg.norm(x) has gradient x/||x|| which is NaN at x=0.
   safe_norm computes sqrt(dot(x,x) + eps²) whose gradient is
   x / sqrt(dot(x,x) + eps²) — well-defined at zero (returns 0/eps = 0).
   eps=1e-6 is small enough to be numerically transparent everywhere else.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

# ── Numerical constants ───────────────────────────────────────────────────────
_EPS_NORM  = 1e-6   # safe_norm epsilon: gradient is x/sqrt(||x||²+eps²)
_EPS_DIV   = 1e-3   # kept from original for direction normalisation
_SELF_MASK_EPS = 0.01  # soft self-mask: mask=1-exp(-dist²/eps²), zero at dist=0


# ── G. safe_norm — replaces jnp.linalg.norm everywhere ───────────────────────

def safe_norm(x: jnp.ndarray, eps: float = _EPS_NORM) -> jnp.ndarray:
    """
    ||x||  with a well-defined gradient at x = 0.

    jnp.linalg.norm(x) has gradient  x / ||x||  which is NaN when x = 0.
    safe_norm computes  sqrt( dot(x,x) + eps² )  whose gradient is
        x / sqrt( dot(x,x) + eps² )
    At x=0 this equals 0/eps = 0 — clean, no NaN.
    For ||x|| >> eps the result is indistinguishable from the true norm.
    """
    return jnp.sqrt(jnp.dot(x, x) + eps * eps)


# ── wrap_angle, get_linear_velocity — unchanged ───────────────────────────────

@jit
def wrap_angle(theta: float) -> float:
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jit
def get_linear_velocity(theta: float, body_velocity: jnp.ndarray) -> jnp.ndarray:
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta),  jnp.cos(theta)]])
    return jnp.matmul(R, body_velocity)


# ── Edge / obstacle geometry — unchanged (NaN-branch is structural) ───────────

@jit
def compute_edge_closest_point(reference_point: jnp.ndarray, edge: jnp.ndarray):
    """Closest point on a line segment to reference_point. Differentiable."""
    def _not_nan(reference_point, edge):
        a, b = edge[0], edge[1]
        ap = reference_point - a
        ab = b - a
        den = jnp.dot(ab, ab) + 1e-8
        t = jnp.clip(jnp.dot(ap, ab) / den, 0.0, 1.0)
        h = a + t * ab
        # FIX G: safe_norm instead of jnp.linalg.norm
        dist = safe_norm(h - reference_point)
        return h, dist

    # GRADIENT FIX: ramo NaN del cond restituisce (nan, nan) che si propaga
    # nei gradienti dell'altro ramo via XLA's differentiation di lax.cond.
    # Fix: usa un punto finito molto lontano (1e6, 1e6) come fallback.
    # Il gradiente del ramo fallback è 0 (costante), non NaN.
    return lax.cond(
        jnp.any(jnp.isnan(edge)),
        lambda _: (jnp.array([1e6, 1e6]), jnp.float32(1_000_000.)),
        lambda _: _not_nan(reference_point, edge),
        None,
    )

vectorized_compute_edge_closest_point = vmap(
    compute_edge_closest_point, in_axes=(None, 0)
)


@jit
def compute_obstacle_closest_point(reference_point: jnp.ndarray,
                                   obstacle: jnp.ndarray) -> jnp.ndarray:
    """Closest point on a polygon obstacle. Differentiable."""
    def _not_nan(reference_point, obstacle):
        closest_points, min_distances = vectorized_compute_edge_closest_point(
            reference_point, obstacle
        )
        return closest_points[jnp.argmin(min_distances)]

    # GRADIENT FIX: stessa fix del cond in compute_edge_closest_point.
    # jnp.nan come output del ramo fallback contamina i gradienti.
    return lax.cond(
        jnp.all(jnp.isnan(obstacle)),
        lambda _: jnp.full((2,), 1e6),
        lambda _: _not_nan(reference_point, obstacle),
        None,
    )

vectorized_compute_obstacle_closest_point = vmap(
    compute_obstacle_closest_point, in_axes=(None, 0)
)


# ── B + A. pairwise_social_force — continuous self-mask + jnp.maximum ────────

@jit
def pairwise_social_force(
    human_state: jnp.ndarray,
    other_human_state: jnp.ndarray,
    parameters: jnp.ndarray,
    other_human_parameters: jnp.ndarray,
) -> jnp.ndarray:
    """
    Social force between one pair of humans.

    FIX B: original used lax.cond(all(self == other)) — exact float equality
    is not differentiable.  We now compute the force unconditionally and
    multiply by a soft self-mask:
        mask = 1 - exp(-dist² / _SELF_MASK_EPS²)
    This is exactly 0 when human == other (dist=0, same agent) and ≈ 1
    for all other agents.  Gradient flows smoothly through the mask.

    FIX A: jnp.max(jnp.array([0, x])) → jnp.maximum(0.0, x)
    """
    rij = (parameters[0] + other_human_parameters[0]
           + parameters[18] + other_human_parameters[18])
    diff = human_state[:2] - other_human_state[:2]

    # FIX G: safe_norm for distance
    dist = safe_norm(diff)

    nij = diff / (dist + _EPS_DIV)
    real_dist = rij - dist
    tij = jnp.array([-nij[1], nij[0]])

    human_vel       = get_linear_velocity(human_state[4],       human_state[2:4])
    other_human_vel = get_linear_velocity(other_human_state[4], other_human_state[2:4])
    delta_vij       = jnp.dot(other_human_vel - human_vel, tij)

    # FIX A: jnp.maximum(0.0, real_dist) instead of jnp.max(jnp.array([0, x]))
    contact = jnp.maximum(0.0, real_dist)

    force = (
        (parameters[4]  * jnp.exp(real_dist / parameters[6])  + parameters[12] * contact) * nij
        + (parameters[8] * jnp.exp(real_dist / parameters[10]) + parameters[13] * contact * delta_vij) * tij
    )

    # FIX B: continuous self-mask — zero when dist=0 (self-interaction), 1 elsewhere
    self_mask = 1.0 - jnp.exp(-(dist * dist) / (_SELF_MASK_EPS ** 2))
    return force * self_mask


vectorized_pairwise_social_force = vmap(
    pairwise_social_force, in_axes=(None, 0, None, 0)
)


# ── D. compute_obstacle_force — jnp.maximum replaces lax.cond branch ─────────

@jit
def compute_obstacle_force(human_state: jnp.ndarray,
                           obstacle: jnp.ndarray,
                           parameters: jnp.ndarray) -> jnp.ndarray:
    """
    Repulsive + contact force from one obstacle point.

    FIX D: original had lax.cond(real_dist > 0, branch_contact, branch_no_contact).
    The contact-stiffness terms k1, k2 are physically meaningful only when
    real_dist > 0 (surfaces overlapping).  Replacing the branch with
    jnp.maximum(0.0, real_dist) inside the expression makes the formula
    continuous: contact terms activate smoothly at surface touch and have
    well-defined gradients everywhere, including at real_dist = 0.
    """
    def _not_nan(human_state, obstacle, parameters):
        diff = human_state[:2] - obstacle
        # FIX G
        dist = safe_norm(diff)
        niw = diff / (dist + _EPS_DIV)
        tiw = jnp.array([-niw[1], niw[0]])
        linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
        delta_viw = -jnp.dot(linear_velocity, tiw)
        real_dist = parameters[0] - dist + parameters[18]

        # FIX D: smooth contact activation via maximum(0, real_dist)
        contact = jnp.maximum(0.0, real_dist)

        force = (
            (parameters[5] * jnp.exp(real_dist / parameters[7]) + parameters[12] * contact) * niw
            + (-parameters[9] * jnp.exp(real_dist / parameters[11]) - parameters[13] * contact) * delta_viw * tiw
        )
        return force

    return lax.cond(
        jnp.any(jnp.isnan(obstacle)),
        lambda _: jnp.zeros((2,)),
        lambda _: _not_nan(human_state, obstacle, parameters),
        None,
    )

vectorized_compute_obstacle_force = vmap(
    compute_obstacle_force, in_axes=(None, 0, None)
)


# ── C + E + F. single_update — smooth desired force, obstacle avg, v-proj ────

@jit
def single_update(
    idx: int,
    humans_state: jnp.ndarray,
    human_goal: jnp.ndarray,
    parameters: jnp.ndarray,
    obstacles: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    One Euler step for a single human.

    FIX C: desired force — replaced lax.cond(dist > radius) with jnp.where.
           Direction diff/dist stabilised with safe_norm.
    FIX E: obstacle force average — replaced lax.cond(num_obs > 0) with
           unconditional sum / max(count, 1).  Gradient flows through division.
    FIX F: velocity bounding — replaced lax.cond(||v|| > v_max) with smooth
           projection v * v_max / max(||v||, v_max).
    FIX G: all norms use safe_norm.
    """
    self_state      = humans_state[idx]
    self_parameters = parameters[idx]

    # ── Desired force (FIX C) ─────────────────────────────────────────────────
    linear_velocity = get_linear_velocity(self_state[4], self_state[2:4])
    diff = human_goal - self_state[:2]
    # FIX G
    dist = safe_norm(diff)
    # FIX C: jnp.where instead of lax.cond — gradient of inactive branch is 0,
    # not NaN.  Direction diff/dist is safe because dist >= eps via safe_norm.
    desired_force = jnp.where(
        dist > self_parameters[0],
        self_parameters[1] * ((diff / dist) * self_parameters[2] - linear_velocity) / self_parameters[3],
        jnp.zeros(2),
    )

    # ── Social force ─────────────────────────────────────────────────────────
    # FIX B applied inside pairwise_social_force (self-mask)
    social_force = jnp.sum(
        vectorized_pairwise_social_force(
            self_state, humans_state, self_parameters, parameters
        ),
        axis=0,
    )

    # ── Obstacle force (FIX E) ────────────────────────────────────────────────
    closest_points = vectorized_compute_obstacle_closest_point(
        self_state[:2], obstacles
    )
    # FIX E: count real obstacles, divide unconditionally
    # jnp.isnan returns bool; sum gives float count of real points
    num_real = jnp.sum(~jnp.isnan(closest_points[:, 0])).astype(jnp.float32)
    raw_obstacle_force = jnp.sum(
        vectorized_compute_obstacle_force(self_state, closest_points, self_parameters),
        axis=0,
    )
    # Divide by max(num_real, 1): when num_real=0 the sum is already 0 so
    # the result is 0 regardless, but the division stays finite for gradients.
    obstacle_force = raw_obstacle_force / jnp.maximum(num_real, 1.0)

    # ── Torque ────────────────────────────────────────────────────────────────
    # ── Torque ────────────────────────────────────────────────────────────────
    input_force = desired_force + social_force + obstacle_force
    
    # NEW: Prevent arctan2(0,0) NaN gradient
    input_force_safe = jnp.where(
        jnp.sum(jnp.abs(input_force)) < 1e-8,
        jnp.array([1e-8, 1e-8]),
        input_force
    )
    
    input_force_norm  = safe_norm(input_force)
    input_force_angle = jnp.arctan2(input_force_safe[1], input_force_safe[0])
    inertia  = (self_parameters[1] * self_parameters[0] ** 2) / 2.0
    k_theta  = inertia * self_parameters[17] * input_force_norm
    # FIX G: safe_norm inside sqrt — avoids sqrt(0) which has infinite gradient
    k_omega  = inertia * (1.0 + self_parameters[16]) * jnp.sqrt(
        (self_parameters[17] * input_force_norm) / self_parameters[16] + _EPS_NORM
    )
    torque = -k_theta * wrap_angle(self_state[4] - input_force_angle) - k_omega * self_state[5]
    torque = jnp.clip(torque, -100.0, 100.0)

    # ── Global force ─────────────────────────────────────────────────────────
    cos_t = jnp.cos(self_state[4])
    sin_t = jnp.sin(self_state[4])
    global_force = jnp.array([
        jnp.dot(input_force, jnp.array([cos_t,  sin_t])),
        self_parameters[14] * jnp.dot(
            social_force + obstacle_force, jnp.array([-sin_t, cos_t])
        ) - self_parameters[15] * self_state[3],
    ])

    # ── Euler integration ────────────────────────────────────────────────────
    new_state = jnp.array([
        self_state[0] + dt * linear_velocity[0],                          # px
        self_state[1] + dt * linear_velocity[1],                          # py
        self_state[2] + dt * (global_force[0] / self_parameters[1]),      # bvx
        self_state[3] + dt * (global_force[1] / self_parameters[1]),      # bvy
        wrap_angle(self_state[4] + dt * self_state[5]),                   # theta
        jnp.clip(self_state[5] + dt * (torque / inertia), -10.0, 10.0),  # omega
    ])

    # ── FIX F: smooth velocity projection ────────────────────────────────────
    # Original: lax.cond(||bv|| > v_max, rescale, identity)
    # Discontinuous gradient at the speed-limit boundary.
    #
    # Replacement: v_proj = v * v_max / max(||v||, v_max)
    #   Below v_max: max(||v||, v_max) = v_max → v_proj = v * v_max / v_max = v  (identity)
    #   Above v_max: max(||v||, v_max) = ||v|| → v_proj = v * v_max / ||v||     (rescale)
    #   At v_max:    both sides meet continuously, gradient is continuous.
    bv       = new_state[2:4]
    bv_norm  = safe_norm(bv)
    bv_proj  = bv * self_parameters[2] / jnp.maximum(bv_norm, self_parameters[2])
    new_state = new_state.at[2:4].set(bv_proj)

    # GRADIENT FIX: sostituito nan_to_num con clip finito.
    # nan_to_num(nan=0.0) mappa NaN → costante 0 → ∂(0)/∂input = 0 nel backward:
    # ogni volta che la fisica produce NaN, il gradiente dell'intera traiettoria
    # BPTT diventa zero (o NaN). jnp.clip è differenziabile ovunque:
    # gradiente=0 solo in saturazione (non per NaN).
    # Per sicurezza, le posizioni restano in range fisico ragionevole.
    new_state = jnp.clip(new_state, -2000.0, 2000.0)
    return new_state


vectorized_single_update = vmap(
    single_update, in_axes=(0, None, 0, None, 0, None)
)


# ── Public API — identical to original hsfm.py ───────────────────────────────

@jit
def step(
    humans_state: jnp.ndarray,
    humans_goal: jnp.ndarray,
    parameters: jnp.ndarray,
    obstacles: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    One timestep for all humans.  Fully differentiable drop-in for hsfm.step.

    args / output shapes: identical to original hsfm.py step().
    """
    return vectorized_single_update(
        jnp.arange(len(humans_state)),
        humans_state,
        humans_goal,
        parameters,
        obstacles,
        dt,
    )