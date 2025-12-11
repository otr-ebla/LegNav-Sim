import jax.numpy as jnp
from jax import jit, vmap, lax, debug, random

# TODO: Add generate random humans parameters function
# TODO: Add function to generate animation of simulation

from src.core.env_config import N_HUMANS, HUMANS_VELOCITY, HUMANS_RADIUS

def get_standard_humans_parameters(n_humans:int):
    """
    Returns the standard parameters of the HSFM for the humans in the simulation. Parameters are the same for all humans in the form:
    (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)

    args:
    - n_humans: int - Number of humans in the simulation.

    outputs:
    - parameters (n_humans, 19) - Standard parameters for the humans in the simulation.
    """
    #single_params = jnp.array([0.3, 75., 1., 0.5, 2000., 2000., 0.08, 0.08, 120., 120., 0.6, 0.6, 120000., 240000., 1., 500., 3., 0.1, 0.])
    single_params = jnp.array([HUMANS_RADIUS, 75., HUMANS_VELOCITY, 0.5, 2000., 2000., 0.08, 0.08, 120., 120., 0.6, 0.6, 120000., 240000., 1., 500., 3., 0.1, 0.])
    return jnp.tile(single_params, (n_humans, 1))

def grid_cell_obstacle_occupancy(static_obstacles:jnp.ndarray, cell_size:float, distance_threshold:int):
    """
    Returns a grid cell occupancy map for the static obstacles in the simulation.

    args:
    - static_obstacles: jnp.ndarray of shape (n_obstacles, n_edges, 2, 2) - Static obstacles in the simulation.
    - cell_size: float - Resolution of the grid cells.
    - distance_threshold: int - Distance threshold (in cells) to consider a cell occupied by an obstacle.

    outputs:
    - grid_cell_occupancy: jnp.ndarray of booleans of shape (n+distance_threshold,n+distance_threshold,len(static_obstacles)) 
                           where n is the max number of cells necessary to cover all obstacles 
                           in the x and y direction - Grid cell occupancy map for the static obstacles.
    - grid_cell_coords: jnp.ndarray of shape (n+distance_threshold,n+distance_threshold,2) - Coordinates of the min point of grid cells.
    """
    # Flatten all obstacle points
    obstacle_points = static_obstacles.reshape(-1, 2)
    # Find bounds
    min_xy = jnp.floor(jnp.nanmin(obstacle_points, axis=0) / cell_size).astype(int)
    max_xy = jnp.ceil(jnp.nanmax(obstacle_points, axis=0) / cell_size).astype(int)
    # Grid size (add distance_threshold padding)
    grid_shape = (max_xy - min_xy) + 2 * distance_threshold
    # Initialize grid
    grid = jnp.zeros((grid_shape[0], grid_shape[1], len(static_obstacles)), dtype=bool)
    # Mark occupied cells within distance_threshold for each obstacle
    for obs_idx in range(len(static_obstacles)):
        if jnp.isnan(static_obstacles[obs_idx,0,0]).any():
            continue
        obs_edges = static_obstacles[obs_idx]  # shape: (n_edges, 2, 2)
        for edge in obs_edges:
            p0, p1 = edge  # Each edge is two points (2,)
            if jnp.isnan(p0).any() or jnp.isnan(p1).any():
                continue
            # Interpolate points along the edge
            edge_vec = p1 - p0
            edge_len = jnp.linalg.norm(edge_vec)
            n_samples = max(2, int(jnp.ceil(edge_len / (cell_size * 0.5))))
            ts = jnp.linspace(0, 1, n_samples)
            interp_points = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
            # For each interpolated point, mark grid cells
            grid_obs_points = jnp.floor(interp_points / cell_size).astype(int) - min_xy + distance_threshold
            for pt in grid_obs_points:
                x, y = pt
                x_min = max(0, x - distance_threshold)
                x_max = min(grid_shape[0], x + distance_threshold + 1)
                y_min = max(0, y - distance_threshold)
                y_max = min(grid_shape[1], y + distance_threshold + 1)
                grid = grid.at[x_min:x_max, y_min:y_max, obs_idx].set(True)
    # Compute the min coordinate of each cell in the grid
    # grid shape: (nx, ny, n_obstacles)
    nx, ny = grid_shape
    x_coords = jnp.arange(nx) * cell_size + (min_xy[0] - distance_threshold) * cell_size
    y_coords = jnp.arange(ny) * cell_size + (min_xy[1] - distance_threshold) * cell_size
    # Create a meshgrid of cell min coordinates (nx, ny, 2)
    grid_cell_coords = jnp.stack(jnp.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
    return grid, grid_cell_coords

@jit
def filter_obstacles(
    humans_state:jnp.ndarray, 
    static_obstacles:jnp.ndarray, 
    grid_occupancy:jnp.ndarray, 
    grid_coords:jnp.ndarray, 
    cell_size:float
) -> jnp.ndarray:
    """
    Filters the static obstacles for each human based on their position and the grid occupancy map.

    args:
    - humans_state: jnp.ndarray of shape (n_humans, 6) - Current state of the humans in the simulation.
    - static_obstacles: jnp.ndarray of shape (n_obstacles, n_edges, 2, 2) - Static obstacles in the simulation.
    - grid_occupancy: jnp.ndarray of booleans of shape (n+distance_threshold,n+distance_threshold,len(static_obstacles)) 
                        where n is the max number of cells necessary to cover all obstacles 
                        in the x and y direction - Grid cell occupancy map for the static obstacles.
    - grid_coords: jnp.ndarray of shape (n+distance_threshold,n+distance_threshold,2) - Coordinates of the min point of grid cells.
    - cell_size: float - Resolution of the grid cells.

    outputs:
    - filtered_static_obstacles: jnp.ndarray of shape (n_humans, n_obstacles, n_edges, 2, 2) - Filtered static obstacles for each human.
    """
    # Get human positions
    human_positions = humans_state[:, :2]
    # Get grid cell indices for each human
    grid_origin = grid_coords[0, 0]  # shape (2,)
    human_grid_indices = jnp.floor((human_positions - grid_origin) / cell_size).astype(int)
    @jit
    def get_human_obstacles(human_grid_index:jnp.ndarray, grid_occupancy:jnp.ndarray, static_obstacles:jnp.ndarray):
        # Get the grid cell indices for the human
        x, y = human_grid_index
        # Get the occupied cells around the human
        obstacles_in_occupied_cell = lax.cond(
            (x < 0) | (y < 0) | (x >= grid_occupancy.shape[0]) | (y >= grid_occupancy.shape[1]),
            lambda _: jnp.zeros((len(static_obstacles),), dtype=bool),
            lambda _: grid_occupancy[x, y],
            None
        )
        # Create a mask for the first dimension (obstacle index)
        mask = obstacles_in_occupied_cell[:, None, None, None]
        nan_obstacles = jnp.full_like(static_obstacles, jnp.nan)
        filtered_static_obstacles = jnp.where(mask, static_obstacles, nan_obstacles)
        return filtered_static_obstacles
    filtered_static_obstacles = vmap(get_human_obstacles, in_axes=(0, None, None))(human_grid_indices, grid_occupancy, static_obstacles)
    return filtered_static_obstacles