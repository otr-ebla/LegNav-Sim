from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from jhsfm.hsfm import step
from jhsfm.utils import *

# Hyperparameters
grid_cell_size = 1
grid_distance_threshold = 1
dt = 0.01
end_time = 15
humans_state = np.array([
    [7.,0.,0.,0.,jnp.pi,0.],
    [6.8,0.8,0.,0.,jnp.pi,0.],
    [6.8,-0.8,0.,0.,jnp.pi,0.],
    [6.5,1.5,0.,0.,jnp.pi,0.],
    [6.5,-1.5,0.,0.,jnp.pi,0.],
    [6.2,2.2,0.,0.,jnp.pi,0.],
    [6.2,-2.2,0.,0.,jnp.pi,0.],
    [6.0,3.0,0.,0.,jnp.pi,0.],
    [6.0,-3.0,0.,0.,jnp.pi,0.],
    [5.8,3.8,0.,0.,jnp.pi,0.],
    [5.8,-3.8,0.,0.,jnp.pi,0.],
    [5.5,4.5,0.,0.,jnp.pi,0.],
    [5.5,-4.5,0.,0.,jnp.pi,0.],
    [5.2,5.2,0.,0.,jnp.pi,0.],
    [5.2,-5.2,0.,0.,jnp.pi,0.],
    [5.0,6.0,0.,0.,jnp.pi,0.],
    [5.0,-6.0,0.,0.,jnp.pi,0.],
    [4.8,6.8,0.,0.,jnp.pi,0.],
    [4.8,-6.8,0.,0.,jnp.pi,0.],
    [4.5,7.5,0.,0.,jnp.pi,0.],
    [4.5,-7.5,0.,0.,jnp.pi,0.],
    [4.2,8.2,0.,0.,jnp.pi,0.],
    [4.2,-8.2,0.,0.,jnp.pi,0.],
    [4.0,9.0,0.,0.,jnp.pi,0.],
    [4.0,-9.0,0.,0.,jnp.pi,0.],
    [3.8,9.8,0.,0.,jnp.pi,0.],
    [3.8,-9.8,0.,0.,jnp.pi,0.],
    [3.5,10.5,0.,0.,jnp.pi,0.],
    [3.5,-10.5,0.,0.,jnp.pi,0.],
    [3.2,11.2,0.,0.,jnp.pi,0.],
    [3.2,-11.2,0.,0.,jnp.pi,0.],
    [3.0,12.0,0.,0.,jnp.pi,0.],
    [3.0,-12.0,0.,0.,jnp.pi,0.],
    [2.8,12.8,0.,0.,jnp.pi,0.],
    [2.8,-12.8,0.,0.,jnp.pi,0.],
    [2.5,13.5,0.,0.,jnp.pi,0.],
    [2.5,-13.5,0.,0.,jnp.pi,0.],
    [2.2,14.2,0.,0.,jnp.pi,0.],
    [2.2,-14.2,0.,0.,jnp.pi,0.],
    [2.0,15.0,0.,0.,jnp.pi,0.],
    [2.0,-15.0,0.,0.,jnp.pi,0.],
    [1.8,15.8,0.,0.,jnp.pi,0.],
    [1.8,-15.8,0.,0.,jnp.pi,0.],
])
# Static obstacles - example adding some padding edges as dimensions should be equal for the static_obstacles array but obstacles may have different number of edges and could be dfferentiated for each human (for optimization)
static_obstacles = jnp.array([
    [[[-0.1,0.5],[0.1,0.5]],[[0.1,0.5],[0.1,3]],[[0.1,3],[-0.1,3]],[[-0.1,3],[-0.1,0.5]]],
    [[[-0.1,-0.5],[0.1,-0.5]],[[0.1,-0.5],[0.1,-3]],[[0.1,-3],[-0.1,-3]],[[-0.1,-3],[-0.1,-0.5]]],
    [[[2.0, 2.0], [2.5, 2.0]], [[2.5, 2.0], [2.5, 2.5]], [[2.5, 2.5], [2.0, 2.5]], [[2.0, 2.5], [2.0, 2.0]]],
    [[[3.0, -2.0], [3.5, -2.0]], [[3.5, -2.0], [3.5, -1.5]], [[3.5, -1.5], [3.0, -1.5]], [[3.0, -1.5], [3.0, -2.0]]],
    [[[-4.0, 1.0], [-3.5, 1.0]], [[-3.5, 1.0], [-3.5, 1.5]], [[-3.5, 1.5], [-4.0, 1.5]], [[-4.0, 1.5], [-4.0, 1.0]]],
    [[[0.0, 4.0], [0.5, 4.0]], [[0.5, 4.0], [0.5, 4.5]], [[0.5, 4.5], [0.0, 4.5]], [[0.0, 4.5], [0.0, 4.0]]],
    [[[-2.0, -3.0], [-1.5, -3.0]], [[-1.5, -3.0], [-1.5, -2.5]], [[-1.5, -2.5], [-2.0, -2.5]], [[-2.0, -2.5], [-2.0, -3.0]]],
    [[[5.0, 0.0], [5.5, 0.0]], [[5.5, 0.0], [5.5, 0.5]], [[5.5, 0.5], [5.0, 0.5]], [[5.0, 0.5], [5.0, 0.0]]],
    [[[-6.0, -1.0], [-5.5, -1.0]], [[-5.5, -1.0], [-5.5, -0.5]], [[-5.5, -0.5], [-6.0, -0.5]], [[-6.0, -0.5], [-6.0, -1.0]]],
    [[[1.0, -5.0], [1.5, -5.0]], [[1.5, -5.0], [1.5, -4.5]], [[1.5, -4.5], [1.0, -4.5]], [[1.0, -4.5], [1.0, -5.0]]],
    [[[-3.0, 5.0], [-2.5, 5.0]], [[-2.5, 5.0], [-2.5, 5.5]], [[-2.5, 5.5], [-3.0, 5.5]], [[-3.0, 5.5], [-3.0, 5.0]]],
    [[[4.0, -4.0], [4.5, -4.0]], [[4.5, -4.0], [4.5, -3.5]], [[4.5, -3.5], [4.0, -3.5]], [[4.0, -3.5], [4.0, -4.0]]],
    [[[-5.0, 3.0], [-4.5, 3.0]], [[-4.5, 3.0], [-4.5, 3.5]], [[-4.5, 3.5], [-5.0, 3.5]], [[-5.0, 3.5], [-5.0, 3.0]]],
    [[[0.0, -6.0], [0.5, -6.0]], [[0.5, -6.0], [0.5, -5.5]], [[0.5, -5.5], [0.0, -5.5]], [[0.0, -5.5], [0.0, -6.0]]],
    [[[-1.0, 6.0], [-0.5, 6.0]], [[-0.5, 6.0], [-0.5, 6.5]], [[-0.5, 6.5], [-1.0, 6.5]], [[-1.0, 6.5], [-1.0, 6.0]]],
    [[[2.0, -7.0], [2.5, -7.0]], [[2.5, -7.0], [2.5, -6.5]], [[2.5, -6.5], [2.0, -6.5]], [[2.0, -6.5], [2.0, -7.0]]],
    [[[-4.0, 7.0], [-3.5, 7.0]], [[-3.5, 7.0], [-3.5, 7.5]], [[-3.5, 7.5], [-4., 7.,]], [[-4., 7.,], [-4., 7.,]]],
    [[[3.0, 5.0], [3.5, 5.0]], [[3.5, 5.0], [3.5, 5.5]], [[3.5, 5.5], [3.0, 5.5]], [[3.0, 5.5], [3.0, 5.0]]],
    [[[-6.0, -2.0], [-5.5, -2.0]], [[-5.5, -2.0], [-5.5, -1.5]], [[-5.5, -1.5], [-6., -1.,]], [[-6., -1.,], [-6., -2.,]]],
    [[[1.0, 6.0], [1.5, 6.0]], [[1.5, 6.0], [1.5, 6.5]], [[1.5, 6.5], [1.0, 6.5]], [[1.0, 6.5], [1.0, 6.0]]],
    [[[-2.0, -7.0], [-1.5, -7.0]], [[-1.5, -7.0], [-1.5, -6.5]], [[-1.5, -6.5], [-2., -6.,]], [[-2., -6.,], [-2., -7.,]]],
    [[[-3.0, -8.0], [-2.5, -8.0]], [[-2.5, -8.0], [-2.5, -7.5]], [[-2.5, -7.5], [-3., -7.,]], [[-3., -7.,], [-3., -8.,]]],
    [[[-1.0, -9.0], [-0.5, -9.0]], [[-0.5, -9.0], [-0.5, -8.5]], [[-0.5, -8.5], [-1., -8.,]], [[-1., -8.,], [-1., -9.,]]],
    [[[2.0, 8.0], [2.5, 8.0]], [[2.5, 8.0], [2.5, 8.5]], [[2.5, 8.5], [2., 8.,]], [[2., 8.,], [2., 8.,]]],
    [[[-4., -9.,], [-3., -9.,]], [[-3., -9.,], [-3., -8.,]], [[-3., -8.,], [-4., -8.,]], [[-4., -8.,], [-4., -9.]]],
    [[[3., 8.,], [3.5, 8.,]], [[3.5, 8.,], [3.5, 8.5]], [[3.5, 8.5], [3., 8.5]], [[3., 8.5], [3., 8.]]],
    [[[-6., -10.,], [-5., -10.,]], [[-5., -10.,], [-5., -9.]], [[-5., -9.], [-6., -9.,]], [[-6., -9.,], [-6., -10.]]],
    [[[1., 9.,], [1.5, 9.,]], [[1.5, 9.,], [1.5, 9.5]], [[1.5, 9.5], [1., 9.5]], [[1., 9.5], [1., 9.]]],
    [[[-2., -11.,], [-1.5, -11.,]], [[-1.5, -11.,], [-1.5, -10.]], [[-1.5, -10.], [-2., -10.,]], [[-2., -10.,], [-2., -11.]]],
    [[[0., 10.,], [0.5, 10.,]], [[0.5, 10.,], [0.5, 10.5]], [[0.5, 10.5], [0., 10.5]], [[0., 10.5], [0., 10.]]],
    [[[-3., -12.,], [-2.5, -12.,]], [[-2.5, -12.,], [-2.5, -11.]], [[-2.5, -11.], [-3., -11.,]], [[-3., -11.,], [-3., -12.]]],
    [[[2., 11.,], [2.5, 11.,]], [[2.5, 11.,], [2.5, 11.5]], [[2.5, 11.5], [2., 11.5]], [[2., 11.5], [2., 11.]]],
    [[[-4., -13.,], [-3.5, -13.,]], [[-3.5, -13.,], [-3.5, -12.]], [[-3.5, -12.], [-4., -12.,]], [[-4., -12.,], [-4., -13.]]],
    [[[-7., -14.,], [-6.5, -14.,]], [[-6.5, -14.,], [-6.5, -13.]], [[-6.5, -13.], [-7., -13.,]], [[-7., -13.,], [-7., -14.]]],
    [[[-5., 13.,], [-4.5, 13.,]], [[-4.5, 13.,], [-4.5, 13.5]], [[-4.5, 13.5], [-5., 13.5]], [[-5., 13.5], [-5., 13.]]],
    [[[-1., 14.,], [-0.5, 14.,]], [[-0.5, 14.,], [-0.5, 14.5]], [[-0.5, 14.5], [-1., 14.5]], [[-1., 14.5], [-1., 14.]]],
    [[[-3., 15.,], [-2.5, 15.,]], [[-2.5, 15.,], [-2.5, 15.5]], [[-2.5, 15.5], [-3., 15.5]], [[-3., 15.5], [-3., 15.]]],
    [[[0., -16.,], [0.5, -16.,]], [[0.5, -16.,], [0.5, -15.]], [[0.5, -15.], [0., -15.,]], [[0., -15.,], [0., -16.]]],
    [[[-2., 16.,], [-1.5, 16.,]], [[-1.5, 16.,], [-1.5, 16.5]], [[-1.5, 16.5], [-2., 16.5]], [[-2., 16.5], [-2., 16.]]],
    [[[3., 17.,], [3.5, 17.,]], [[3.5, 17.,], [3.5, 17.5]], [[3.5, 17.5], [3., 17.5]], [[3., 17.5], [3., 17.]]],
    [[[-4., -18.,], [-3.5, -18.,]], [[-3.5, -18.,], [-3.5, -17.]], [[-3.5, -17.], [-4., -17.,]], [[-4., -17.,], [-4., -18.]]],
    [[[1., 18.,], [1.5, 18.,]], [[1.5, 18.,], [1.5, 18.5]], [[1.5, 18.5], [1., 18.5]], [[1., 18.5], [1., 18.]]],
    [[[-10., -19.,], [-9.5, -19.,]], [[-9.5, -19.,], [-9.5, -18.]], [[-9.5, -18.], [-10., -18.,]], [[-10., -18.,], [-10., -19.]]],
    [[[5., 19.,], [5.5, 19.,]], [[5.5, 19.,], [5.5, 19.5]], [[5.5, 19.5], [5., 19.5]], [[5., 19.5], [5., 19.]]],
    [[[-7., -20.,], [-6.5, -20.,]], [[-6.5, -20.,], [-6.5, -19.]], [[-6.5, -19.], [-7., -19.,]], [[-7., -19.,], [-7., -20.]]],
])
grid_occupancy, grid_coords = grid_cell_obstacle_occupancy(static_obstacles, grid_cell_size, grid_distance_threshold)
print(f"Number of humans: {len(humans_state)}")
print(f"Number of static obstacles: {len(static_obstacles)}")
print(f"Number of grid cells: {grid_occupancy.shape[0]} x {grid_occupancy.shape[1]} = {grid_occupancy.shape[0]*grid_occupancy.shape[1]}")
print(f"Grid cell size: {grid_cell_size}")
print(f"Grid distance threshold: {grid_distance_threshold}")

# Initial conditions
humans_goal = np.zeros((len(humans_state), 2))
for i in range(len(humans_state)):
    # Goal: (gx, gy)
    humans_goal[i,0] = -7
    humans_goal[i,1] = 0.
humans_state = jnp.array(humans_state)
initial_humans_state = jnp.copy(humans_state)
humans_parameters = get_standard_humans_parameters(len(humans_state))
humans_goal = jnp.array(humans_goal)

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
dummy_static_obstacles = jnp.stack([static_obstacles for _ in range(len(humans_state))])
_ = step(humans_state, humans_goal, humans_parameters, dummy_static_obstacles, dt)
_ = filter_obstacles(humans_state, static_obstacles, grid_occupancy, grid_coords, grid_cell_size)
print(f"\nAvailable devices: {jax.devices()}\n")
steps = int(end_time/dt)

# Simulation FILTERING OBSTACLES
print(f"Starting simulation FILTERING OBSTACLES... - Simulation time: {steps*dt} seconds")
start_time = time.time()
for i in range(steps):
    filtered_static_obstacles = filter_obstacles(humans_state, static_obstacles, grid_occupancy, grid_coords, grid_cell_size)
    humans_state = step(humans_state, humans_goal, humans_parameters, filtered_static_obstacles, dt)
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)

# Simulation NOT FILTERING OBSTACLES
humans_state = jnp.copy(initial_humans_state)
print(f"\nStarting simulation NOT FILTERING OBSTACLES... - Simulation time: {steps*dt} seconds")
start_time = time.time()
for i in range(steps):
    humans_state = step(humans_state, humans_goal, humans_parameters, dummy_static_obstacles, dt)
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)

# Simulation saving state for plotting
humans_state = jnp.copy(initial_humans_state)
all_states = np.empty((steps+1, len(humans_state), 6), np.float32)
all_states[0] = humans_state
for i in range(steps):
    filtered_static_obstacles = filter_obstacles(humans_state, static_obstacles, grid_occupancy, grid_coords, grid_cell_size)
    humans_state = step(humans_state, humans_goal, humans_parameters, filtered_static_obstacles, dt)
    all_states[i+1] = humans_state
end_time = time.time()
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
COLORS = list(mcolors.TABLEAU_COLORS.values())
print("\nPlotting...")
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y')
# Plot the grid given computed grid_coords
for coord in grid_coords.reshape(-1,2):
    rect = plt.Rectangle((coord[0], coord[1]), grid_cell_size, grid_cell_size, facecolor='none', edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
    ax.add_patch(rect)
for h in range(len(humans_state)): 
    ax.plot(all_states[:,h,0], all_states[:,h,1], color=COLORS[h%len(COLORS)], linewidth=0.5, zorder=0)
    ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=COLORS[h%len(COLORS)], zorder=2)
    for k in range(0,steps+1,int(3/dt)):
        head = plt.Circle((all_states[k,h,0] + np.cos(all_states[k,h,4]) * humans_parameters[h,0], all_states[k,h,1] + np.sin(all_states[k,h,4]) * humans_parameters[h,0]), 0.1, color=COLORS[h%len(COLORS)], zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((all_states[k,h,0],all_states[k,h,1]),humans_parameters[h,0], edgecolor=COLORS[h%len(COLORS)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        num = int(k*dt) if (k*dt).is_integer() else (k*dt)
        ax.text(all_states[k,h,0],all_states[k,h,1], f"{num}", color=COLORS[h%len(COLORS)], va="center", ha="center", size=10, zorder=1, weight='bold')
for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
if not os.path.exists(os.path.join(os.path.dirname(__file__),".images")):
    os.makedirs(os.path.join(os.path.dirname(__file__),".images"))
figure.savefig(os.path.join(os.path.dirname(__file__),".images",f"example6.png"), format='png')
plt.show()