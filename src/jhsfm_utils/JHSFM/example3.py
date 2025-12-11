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
n_humans = 15
traffic_length = 14
traffic_height = 3
dt = 0.01
end_time = 5
np.random.seed(0) # For reproducibility

# Initial conditions
humans_state = np.zeros((n_humans, 6))
humans_goal = np.zeros((n_humans, 2))
humans_pos = []
for i in range(n_humans):
    while True:
        a = -(traffic_length/2 ) + .3
        b = traffic_length/2 - .3
        pos = np.array([(b - a) * np.random.random() + a, (np.random.random() - 0.5) * traffic_height], dtype=np.float64)
        collide = False
        for j in range(len(humans_pos)):
            other_human_pos = humans_pos[j]
            if np.linalg.norm(pos - other_human_pos) - .7 < 0: # This is  discomfort distance
                collide = True 
                break
        if not collide:
            humans_pos.append(pos)
            ## State: (px, py, bvx, bvy, theta, omega)
            humans_state[i,0] = pos[0]
            humans_state[i,1] = pos[1]
            humans_state[i,2] = 0
            humans_state[i,3] = 0
            humans_state[i,4] = jnp.pi
            humans_state[i,5] = 0
            # Goal: (gx, gy)
            humans_goal[i,0] = -(traffic_length / 2)-3
            humans_goal[i,1] = pos[1]
            break
humans_state = jnp.array(humans_state)
humans_parameters = get_standard_humans_parameters(n_humans)
humans_goal = jnp.array(humans_goal)
# Obstacles
static_obstacles = jnp.array([[[[-traffic_length/2-3,-traffic_height/2-1],[-traffic_length/2-3,-traffic_height/2-0.5]],[[-traffic_length/2-3,-traffic_height/2-0.5],[traffic_length/2,-traffic_height/2-0.5]],[[traffic_length/2,-traffic_height/2-0.5],[traffic_length/2,-traffic_height/2-1]],[[traffic_length/2,-traffic_height/2-1],[-traffic_length/2-3,-traffic_height/2-1]]],
                              [[[-traffic_length/2-3,traffic_height/2+1],[-traffic_length/2-3,traffic_height/2+0.5]],[[-traffic_length/2-3,traffic_height/2+0.5],[traffic_length/2,traffic_height/2+0.5]],[[traffic_length/2,traffic_height/2+0.5],[traffic_length/2,traffic_height/2+1]],[[traffic_length/2,traffic_height/2+1],[-traffic_length/2-3,traffic_height/2+1]]]])
static_obstacles_per_human = jnp.stack([static_obstacles for _ in range(len(humans_state))])
# Make a human traverse obstacles (all obstacles for him are set to Nan)
# human_that_traverses_obstacle = 0
# static_obstacles_per_human = static_obstacles_per_human.at[human_that_traverses_obstacle,:].set(jnp.nan)

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
ax.set(xlabel='X',ylabel='Y',xlim=[-traffic_length/2-4,traffic_length/2+1],ylim=[-traffic_height-1,traffic_height+1])
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
if not os.path.exists(os.path.join(os.path.dirname(__file__),".images")):
    os.makedirs(os.path.join(os.path.dirname(__file__),".images"))
figure.savefig(os.path.join(os.path.dirname(__file__),".images",f"example3.png"), format='png')
plt.show()