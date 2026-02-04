import time
import argparse
import os
import math
import numpy as np
import torch
import types  # Necessario per l'override dei metodi dinamico
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import environment e config
from src.envs.gym_nav_env import GymNavEnv
from src.config import LidarConfig

# --- LOGICA SEQUENZIALE CUSTOM ---

def sequential_stalk_logic(self):
    """
    Logica:
    1. Gli umani partono tutti fermi.
    2. Si attiva solo UN umano alla volta (basato su _active_human_idx).
    3. L'umano attivo va verso il robot.
    4. Quando arriva a STOP_DISTANCE, si congela e ATTIVA il prossimo.
    """
    
    STOP_DISTANCE = 0.8   # Metri: Distanza a cui si congelano
    STALK_SPEED = 0.5     # m/s: Velocità
    
    # Reset stato se è iniziata una nuova run (step 0)
    if self.step_count == 0 or not hasattr(self, '_active_human_idx'):
        self._active_human_idx = 0
        # Opzionale: stampa di debug
        # print(f"🔄 Reset: Starting with Human 0 active")

    # Controlliamo se l'umano attivo ha raggiunto il target
    if self._active_human_idx < len(self.people):
        active_p = self.people[self._active_human_idx]
        dist_to_robot = math.hypot(self.x - active_p["x"], self.y - active_p["y"])
        
        if dist_to_robot <= STOP_DISTANCE:
            # Ha raggiunto il robot -> Congela e passa al prossimo
            self._active_human_idx += 1
            print(f"❄️  Human {self._active_human_idx-1} Frozen! Activating Human {self._active_human_idx}...")

    # Ciclo su tutti gli umani per aggiornare posizioni
    for i, p in enumerate(self.people):
        
        # CASO 1: Umano già passato (CONGELATO)
        if i < self._active_human_idx:
            p["vx"] = 0.0
            p["vy"] = 0.0
            # Mantiene l'angolo verso il robot (faccia cattiva)
            # p["angle"] resta quello che era
            
        # CASO 2: Umano Attivo (INSEGUIMENTO)
        elif i == self._active_human_idx:
            dx = self.x - p["x"]
            dy = self.y - p["y"]
            target_angle = math.atan2(dy, dx)
            
            p["vx"] = math.cos(target_angle) * STALK_SPEED
            p["vy"] = math.sin(target_angle) * STALK_SPEED
            p["angle"] = target_angle
            
        # CASO 3: Umano Futuro (IN ATTESA)
        else:
            p["vx"] = 0.0
            p["vy"] = 0.0
            # Guardano verso il robot in attesa
            dx = self.x - p["x"]
            dy = self.y - p["y"]
            p["angle"] = math.atan2(dy, dx)

        # --- FISICA COMUNE ---
        p["x"] += p["vx"] * self.dt
        p["y"] += p["vy"] * self.dt

        # Muri
        p["x"] = max(self.people_radius, min(self.room_width - self.people_radius, p["x"]))
        p["y"] = max(self.people_radius, min(self.room_height - self.people_radius, p["y"]))
        
        # Animazione gambe (Solo se si muove)
        if self.use_legs:
            v_mag = math.hypot(p["vx"], p["vy"])
            if v_mag > 0.01:
                smooth_v = self._get_smooth_speed(i, v_mag)
                self.humans_leg_phase[i] += smooth_v * self.dt * 4.0
                p["legs"] = self._calculate_leg_positions(
                    p["x"], p["y"], smooth_v, p["angle"], self.humans_leg_phase[i]
                )
            # Se è fermo, le gambe restano nell'ultima posizione calcolata (non resettiamo)

# --- ESECUZIONE MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Test Policy: Sequential Frozen Humans")
    parser.add_argument("--name", type=str, required=True, help="Nome modello")
    parser.add_argument("--algo", type=str, default="TQC", choices=["PPO", "SAC", "TQC"])
    parser.add_argument("--num_people", type=int, default=5, help="Numero totale di ostacoli umani")
    parser.add_argument("--use_legs", action="store_true", help="Abilita rendering gambe")
    parser.add_argument("--render_skip", type=int, default=3, help="Velocità render (più alto = più veloce)")
    parser.add_argument("--room_size", type=float, default=8.0, help="Lato stanza")
    
    args = parser.parse_args()

    print(f"--- 🧪 SCENARIO: SEQUENTIAL FROZEN TRAP ---")
    print(f"Model: {args.name} | Algo: {args.algo}")
    print(f"Room: {args.room_size}m | Humans: {args.num_people} (One by one)")

    # 1. Setup Environment
    def make_env():
        env = GymNavEnv(
            render_mode="human",
            num_rays=LidarConfig.NUM_RAYS,
            num_people=args.num_people,
            num_obstacles=0,      # Zero ostacoli statici classici
            use_legs=args.use_legs,
            render_skip=args.render_skip,
            distraction_prob=0.0,
            lidar_noise_enable=False
        )
        
        # Resize stanza
        env.env.room_width = args.room_size
        env.env.room_height = args.room_size
        env.env.max_possible_dist = math.sqrt(2 * args.room_size**2)

        # --- FIX: RICALCOLA LA SCALA GRAFICA ---
        # Aggiorniamo la scala affinché 8 metri riempiano tutti gli 800 pixel
        env.env.scale = env.env.window_size / max(env.env.room_width, env.env.room_height)
        # ---------------------------------------
        
        # Override logica
        print("🔧 Injecting 'Sequential Stalk' logic...")
        env.env._step_people = types.MethodType(sequential_stalk_logic, env.env)
        
        return env

    env = DummyVecEnv([make_env])

    # 2. VecNormalize
    vec_path = f"./checkpoints/{args.name}_vecnormalize.pkl"
    if not os.path.exists(vec_path):
        vec_path = f"{args.name}_vecnormalize.pkl"

    if os.path.exists(vec_path):
        print(f"📥 Loading Norm Stats: {vec_path}")
        env = VecNormalize.load(vec_path, env)
        env.training = False      
        env.norm_reward = False   
    else:
        print("⚠️  WARNING: VecNormalize missed!")

    # 3. Load Model
    model_path = f"./checkpoints/{args.name}"
    ModelClass = {"PPO": PPO, "SAC": SAC, "TQC": TQC}[args.algo]
    try:
        model = ModelClass.load(model_path, env=env)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Load error: {e}")
        return

    # 4. Loop
    obs = env.reset()
    print("\nStarting... (Ctrl+C to exit)\n")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            if dones[0]:
                info = infos[0]
                reason = info.get("termination_reason", "unknown")
                print(f"🏁 End: {reason}")
                
                if "collision" in reason:
                    time.sleep(0.5)
                
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 Stop.")
        env.close()

if __name__ == "__main__":
    main()