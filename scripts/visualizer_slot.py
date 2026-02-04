# visualizer_ppo.py
from pyexpat import model
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.gym_nav_env import GymNavEnv
from src.config import LidarConfig
from models.slot_attention import LidarSlotAttentionExtractor # Modified class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Model name")
    parser.add_argument("--algo", type=str, default="TQC")
    parser.add_argument("--num_people", type=int, default=5)
    args = parser.parse_args()

    # 1. Setup Single Environment (Visual mode requires 1 env)
    env = GymNavEnv(
        render_mode="human",
        num_rays=LidarConfig.NUM_RAYS,
        num_people=args.num_people,
        num_obstacles=10,
        render_skip=1,
        use_legs=True # Better visuals
    )
    # Wrap in DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # 2. Load Normalization (Critical)
    norm_path = f"./checkpoints/{args.name}_vecnormalize.pkl"
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded VecNormalize")
    except:
        print("⚠️  VecNormalize NOT found. Visualization might be wrong!")

    # 3. Load Model
    model_path = f"./checkpoints/{args.name}"
    ModelClass = {"TQC": TQC, "SAC": SAC, "PPO": PPO}[args.algo]
    
    # Need to pass custom objects to ensure it loads the modified class
    custom_objects = {
        "features_extractor_class": LidarSlotAttentionExtractor
    }
    
    model = ModelClass.load(model_path, custom_objects=custom_objects)
    print("🧠 Model Loaded")

    # 4. PREPARE PLOTTING
    plt.ion()
    # Create a figure with 2 subplots: 
    # Top: Robot View (Env render handles this, but we can augment it or use separate win)
    # Bottom: Attention Heatmap
    
    # Note: env.render() creates its own window. We create a second one for Attention.
    fig_attn, ax_attn = plt.subplots(figsize=(10, 5))
    ax_attn.set_title("Real-Time Slot Attention Map")
    ax_attn.set_xlabel("LiDAR Rays (0 = Right, 54 = Front, 108 = Left)")
    ax_attn.set_ylabel("Slot Index (Object ID)")
    
    # Initial empty heatmap
    # Shape: (Num_Slots, Num_Rays) -> (6, 108)
    # We use random data to init the imshow object
    dummy_data = np.zeros((6, 108))
    im = ax_attn.imshow(dummy_data, aspect='auto', cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_attn, label="Attention Weight")
    
    obs = env.reset()

    while True:
        # 1. Get Action from Policy
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. EXTRACT ATTENTION MAP
        # We need to access the internal policy network -> features_extractor
        # obs is numpy, convert to tensor
        obs_tensor = torch.as_tensor(obs).to(model.device)
        
        # Call our custom method
        # The policy object in SB3 is model.policy
        # The extractor is model.policy.features_extractor
        # Accediamo all'extractor specifico dell'Actor
        attn_matrix = model.policy.actor.features_extractor.get_attention_map(obs_tensor)
        
        # Convert to numpy: (Batch=1, Slots=6, Rays=108)
        attn_np = attn_matrix.cpu().numpy()[0]
        
        # 3. UPDATE PLOT
        # We normalize per ray to make it clearer which slot "wins" each ray?
        # Or just raw softmax weights? Raw is usually better to see confidence.
        im.set_data(attn_np)
        fig_attn.canvas.draw()
        fig_attn.canvas.flush_events()
        
        # 4. Step Environment
        obs, rewards, dones, infos = env.step(action)
        env.render() # Standard render
        
        if dones[0]:
            obs = env.reset()
            # Pause briefly on reset
            time.sleep(0.5)

if __name__ == "__main__":
    main()