import jax.numpy as jnp
from jax import jit, vmap, lax, debug, random
from src.config import SimConfig

HUMANS_RADIUS, HUMANS_VELOCITY = SimConfig.HUMANS_RADIUS, SimConfig.HUMANS_VELOCITY

def get_standard_humans_parameters(n_humans:int):
    """
    Returns the standard parameters of the HSFM for the humans in the simulation. 
    Parameters are the same for all humans in the form:
    (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    """
    # [Radius, Mass, V_max, Tau, ...]
    single_params = jnp.array([
        HUMANS_RADIUS, 75., HUMANS_VELOCITY, 0.5, 
        2000., 2000., 0.08, 0.08, 120., 120., 
        0.6, 0.6, 120000., 240000., 1., 500., 
        3., 0.1, 0.
    ])
    return jnp.tile(single_params, (n_humans, 1))

def grid_cell_obstacle_occupancy(static_obstacles:jnp.ndarray, cell_size:float, distance_threshold:int):
    obstacle_points = static_obstacles.reshape(-1, 2)
    min_xy = jnp.floor(jnp.nanmin(obstacle_points, axis=0) / cell_size).astype(int)
    max_xy = jnp.ceil(jnp.nanmax(obstacle_points, axis=0) / cell_size).astype(int)
    grid_shape = (max_xy - min_xy) + 2 * distance_threshold
    grid = jnp.zeros((grid_shape[0], grid_shape[1], len(static_obstacles)), dtype=bool)
    
    for obs_idx in range(len(static_obstacles)):
        if jnp.isnan(static_obstacles[obs_idx,0,0]).any(): continue
        obs_edges = static_obstacles[obs_idx]
        for edge in obs_edges:
            p0, p1 = edge
            if jnp.isnan(p0).any() or jnp.isnan(p1).any(): continue
            edge_vec = p1 - p0
            edge_len = jnp.linalg.norm(edge_vec)
            n_samples = max(2, int(jnp.ceil(edge_len / (cell_size * 0.5))))
            ts = jnp.linspace(0, 1, n_samples)
            interp_points = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
            grid_obs_points = jnp.floor(interp_points / cell_size).astype(int) - min_xy + distance_threshold
            for pt in grid_obs_points:
                x, y = pt
                x_min = max(0, x - distance_threshold); x_max = min(grid_shape[0], x + distance_threshold + 1)
                y_min = max(0, y - distance_threshold); y_max = min(grid_shape[1], y + distance_threshold + 1)
                grid = grid.at[x_min:x_max, y_min:y_max, obs_idx].set(True)
                
    nx, ny = grid_shape
    x_coords = jnp.arange(nx) * cell_size + (min_xy[0] - distance_threshold) * cell_size
    y_coords = jnp.arange(ny) * cell_size + (min_xy[1] - distance_threshold) * cell_size
    grid_cell_coords = jnp.stack(jnp.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
    return grid, grid_cell_coords