# Indoor RL Nav: 2D Simulation Environment for Mobile Robot LiDAR Navigation

<img src="assets/indoor_gif.gif" alt="Environment Demo" width="70%"/>

## Overview

**Indoor RL Nav** is a lightweight, high-performance 2D simulation environment designed specifically for training Deep Reinforcement Learning (DRL) agents to navigate mobile robots.

The core focus of this repository is to solve the navigation in dynamic environments problem using sparse sensor data. Agents are equipped with simulated 2D LiDAR sensors (ray-casts) and must learn to reach target coordinates while avoiding both static geometry (walls) and moving dynamic obstacles (simulated pedestrians).

Built on the standard **Gymnasium** interface, this environment allows for rapid prototyping and training of algorithms like **PPO (Proximal Policy Optimization)** and **SAC (Soft Actor-Critic)** to develop robust obstacle avoidance policies.

## Key Features

- **2D LiDAR Perception**  
  Agents perceive the world solely through sparse 2D ray-cast data (e.g., 108 rays), mimicking the limitations of real-world laser scanners found on platforms like Turtlebots.

- **Dynamic Obstacle Avoidance**  
  Unlike static mazes, this environment populates the world with moving entities, forcing the agent to learn predictive navigation rather than simple map memorization.

- **Deep RL Ready**  
  Fully compatible with **stable-baselines3**, allowing immediate integration with standard implementations of **PPO**, **SAC**, and **TQC**.

- **Fast Simulation**  
  Optimized for high-throughput training steps on standard CPUs, enabling efficient training of navigation policies without requiring heavy 3D rendering engines.

## Repository Structure

```text
.
├── assets/              # Visuals (gifs, images)
├── checkpoints/         # Trained model weights (.zip, .msgpack)
├── logs/                # Tensorboard logs and training metrics
├── scripts/             # Executable scripts
│   ├── train_ppo.py     # Main training loop for PPO/SAC agents
│   ├── run_ppo.py       # Inference script to test/visualize trained agents
│   └── visualMain.py    # Real-time visualization tool
├── src/                 # Core library code
│   ├── agents/          # Agent implementations and trainers
│   └── envs/            # Gymnasium environment definitions
│       ├── gym_nav_env.py    # Core simulation logic
│       └── bouncing.py       # Dynamics for moving obstacles
└── requirements.txt     # Project dependencies
```

## Getting Started
Create new python environment:
```
python3 -m venv 2drlenv
```

Install dependencies:
```
cd indoor-rl-nav
pip install -r requirements.txt
```

Train a new Agent(PPO/SAC):
```
python -m scripts.train_ppo
```

Visualize a Pre-Trained Agent:
```
python -m scripts.run_ppo

```



