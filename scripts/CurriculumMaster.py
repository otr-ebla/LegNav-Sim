import subprocess
import time
import os
import glob
import pandas as pd
import sys

# --- CONFIGURATION ---
PYTHON_EXEC = sys.executable # Uses the current python interpreter
TRAIN_SCRIPT = "scripts.train_ppo"

# CURRICULUM STAGES
# Constraints: Start with 10 obstacles, Max 10 humans.
STAGES = [
    # Stage 1: Static Environment (Learn to navigate around objects)
    {
        "name": "Stage1_Static_Easy",
        "obstacles": 10,
        "humans": 0,
        "threshold": 0.95,      # Very high precision required before moving on
        "min_steps": 200_000,   # Minimum burn-in steps
        "max_steps": 3_000_000  # Fail-safe max steps
    },
    # Stage 2: Dense Static Environment (Cluttered room)
    {
        "name": "Stage2_Static_Hard",
        "obstacles": 20,
        "humans": 0,
        "threshold": 0.90,
        "min_steps": 300_000,
        "max_steps": 5_000_000
    },
    # Stage 3: Low Dynamic (Introduction to humans)
    {
        "name": "Stage3_Dynamic_Low",
        "obstacles": 15,        # Slightly reduced static obstacles to make space for humans
        "humans": 5,
        "threshold": 0.85,      # Humans are unpredictable, slightly lower threshold allowed
        "min_steps": 500_000,
        "max_steps": 8_000_000
    },
    # Stage 4: High Dynamic (Max Complexity)
    {
        "name": "Stage4_Dynamic_High",
        "obstacles": 15,
        "humans": 10,           # Max Humans
        "threshold": 0.85,      # Final Target
        "min_steps": 1_000_000,
        "max_steps": 20_000_000 # Run for a long time
    }
]

def get_success_rate(log_dir, window=100):
    """Reads Monitor CSVs to calculate Success Rate (Reward > 100)."""
    # Find all monitor files (one per cpu core)
    files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    if not files: return 0.0

    dfs = []
    for f in files:
        try:
            # Skip header lines (metadata)
            df = pd.read_csv(f, skiprows=1)
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs: return 0.0
    
    # Combine data from all environments
    full_df = pd.concat(dfs)
    if len(full_df) < window: return 0.0

    # Analyze last 'window' episodes
    last_data = full_df.tail(window)
    
    # Logic: In your env, Goal Reached gives +200. Failures are usually negative.
    # We assume success if reward ('r') > 100.
    successes = last_data[last_data['r'] > 100].count()['r']
    return successes / len(last_data)

def main():
    prev_model_path = None

    for i, stage in enumerate(STAGES):
        print(f"\n{'#'*60}")
        print(f"🚀 STARTING CURRICULUM STAGE {i+1}/{len(STAGES)}: {stage['name']}")
        print(f"   - Obstacles: {stage['obstacles']}")
        print(f"   - Humans:    {stage['humans']}")
        print(f"   - Target SR: {stage['threshold']:.1%}")
        print(f"{'#'*60}\n")

        # 1. Prepare Command
        log_dir = f"./logs/{stage['name']}"
        os.makedirs(log_dir, exist_ok=True)

        cmd = [
            PYTHON_EXEC, "-m", TRAIN_SCRIPT,
            "--training_name", stage['name'],
            "--num_people", str(stage['humans']),
            "--num_obstacles", str(stage['obstacles']),
            "--steps", str(stage['max_steps']),
            "--algo", "TQC" # Or SAC/PPO
        ]

        # Resume from previous stage if available
        if prev_model_path:
            print(f"🔄 Transfer Learning: Loading weights from {prev_model_path}...")
            cmd.extend(["--load_model", prev_model_path])

        # 2. Launch Training Process
        process = subprocess.Popen(cmd)
        
        start_time = time.time()
        
        # 3. Monitor Loop
        try:
            while True:
                time.sleep(30) # Check every 30 seconds

                if process.poll() is not None:
                    print("❌ Training process crashed or finished unexpectedly.")
                    break

                # Check Success Rate
                sr = get_success_rate(log_dir)
                elapsed_min = (time.time() - start_time) / 60
                
                print(f"   [{stage['name']}] Status: SR={sr:.1%} | Time={elapsed_min:.1f}m")

                # Check Termination Conditions
                # Condition A: Success Rate met AND Minimum Steps respected (to avoid luck)
                # Note: We can't easily check steps from outside, so we use time as proxy for min_steps
                # Assuming approx 200 steps/sec -> 200k steps ~ 15 mins.
                min_time_passed = elapsed_min > (stage['min_steps'] / 10000) # heuristic
                
                if sr >= stage['threshold'] and min_time_passed:
                    print(f"\n✅ STAGE CLEARED! ({sr:.1%} >= {stage['threshold']:.1%})")
                    print("🛑 Stopping current training stage...")
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
        except KeyboardInterrupt:
            print("\n🛑 Curriculum interrupted by user.")
            process.terminate()
            return

        # 4. Prepare Next Stage
        # Locate the saved model. train_ppo.py saves to ./checkpoints/{name}.zip
        expected_model = f"./checkpoints/{stage['name']}.zip"
        
        if os.path.exists(expected_model):
            prev_model_path = expected_model
            print(f"💾 Checkpoint found: {prev_model_path}")
        else:
            print(f"⚠️ Warning: Could not find checkpoint {expected_model}. Next stage might start from scratch.")
            prev_model_path = None

    print("\n🎉🎉 CURRICULUM COMPLETED! 🎉🎉")

if __name__ == "__main__":
    main()