from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import pickle
from jhsfm.hsfm import step
from jhsfm.utils import *

# Hyperparameters
circle_radius = 7
dt = 0.01
end_time = 15

# Initial conditions
with open(os.path.join(os.path.dirname(__file__),"custom_configurations/circular_crossing_5_humans.pkl"), 'rb') as f:
    initial_configuration = pickle.load(f)
humans_state = initial_configuration[0]["full_state"]
n_humans = humans_state.shape[0]
humans_goal = initial_configuration[0]["humans_goal"]
humans_parameters = get_standard_humans_parameters(n_humans)
static_obstacles = initial_configuration[0]["static_obstacles"]
static_obstacles_per_human = jnp.stack([static_obstacles for _ in range(len(humans_state))])

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
_ = step(humans_state, humans_goal, humans_parameters, static_obstacles_per_human, dt)

# Simulation 
steps = int(end_time/dt)
print(f"\nAvailable devices: {jax.devices()}\n")
print(f"Starting simulation... - Simulation time: {steps*dt} seconds\n")
start_time = time.time()
all_states = np.empty((steps+1, n_humans, 6), np.float32)
all_states[0] = humans_state
for i in range(steps):
    humans_state = step(humans_state, humans_goal, humans_parameters, static_obstacles_per_human, dt)
    all_states[i+1] = humans_state
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
COLORS = list(mcolors.TABLEAU_COLORS.values())
print("\nPlotting...")
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
for h in range(n_humans): 
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
plt.show()