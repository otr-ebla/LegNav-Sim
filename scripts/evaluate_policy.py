import numpy as np
from tqdm import tqdm
from sb3_contrib import TQC
from src.envs.gym_nav_env import GymNavEnv

NUM_EPISODES = 1000
NUM_RAYS = 108 # Deve corrispondere al training

# Disabilitiamo il render_mode per velocizzare il test (non serve vedere 1000 ep)
env = GymNavEnv(render_mode=None, num_rays=NUM_RAYS, num_people=20)
model = TQC.load("./checkpoints/30MTQC", env=env)

# Contenitori per le metriche
successes, collisions, timeouts = 0, 0, 0
spl_scores, angular_jerks, min_dists_humans = [], [], []

print(f"Avvio evaluation su {NUM_EPISODES} episodi...")
for _ in tqdm(range(NUM_EPISODES)):
    obs, info = env.reset()
    
    # Variabili per calcolo SPL (Success Weighted by Path Length)
    start_pos = np.array([env.env.x, env.env.y])
    goal_pos = np.array([env.env.goal_x, env.env.goal_y])

    optimal_dist = np.linalg.norm(goal_pos - start_pos)
    current_path_len = 0.0
    ep_jerks = []
    ep_min_human_dist = float('inf')
    prev_w = 0.0 # Per calcolare il jerk (derivata accelerazione angolare)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_pos = np.array([env.env.x, env.env.y])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Calcolo Jerk Angolare: |w_t - w_{t-1}| / dt
        dt = env.env.dt
        ep_jerks.append(abs(action[1] - prev_w) / dt)
        prev_w = action[1]

        # Aggiorna lunghezza percorso corrente
        current_path_len += np.linalg.norm(np.array([env.env.x, env.env.y]) - prev_pos)

        # Accesso diretto allo stato interno di SimpleEnv per le posizioni umani
        people_pos = np.array([[p['x'], p['y']] for p in env.env.people])
        robot_pos = np.array([env.env.x, env.env.y])
        if len(people_pos) > 0:
            dists = np.linalg.norm(people_pos - robot_pos, axis=1)
            ep_min_human_dist = min(ep_min_human_dist, np.min(dists))

    reason = info.get("termination_reason", "unknown")
    angular_jerks.append(np.mean(ep_jerks))
    min_dists_humans.append(ep_min_human_dist)

    if reason == "goal_reached":
        successes += 1

        # SPL = 1 * (Optimal / max(Actual, Optimal)) solo se successo
        spl_scores.append(optimal_dist / max(current_path_len, optimal_dist))
        # Salviamo tempo e path solo per episodi di successo
        # Nota: usiamo una lista globale definita fuori (aggiungila al blocco 2 se vuoi)
        # path_lengths_success.append(current_path_len)

    elif "collision" in reason:
        collisions += 1
        spl_scores.append(0.0)

    elif reason == "max_steps_reached":
        timeouts += 1
        spl_scores.append(0.0)

env.close()

print("\n--- RISULTATI EVALUATION ---")
print(f"Success Rate: {successes/NUM_EPISODES*100:.2f}%")
print(f"Collision Rate: {collisions/NUM_EPISODES*100:.2f}%")
print(f"Timeout Rate: {timeouts/NUM_EPISODES*100:.2f}%")
print(f"SPL (Success weighted by Path Length): {np.mean(spl_scores):.4f}")
print(f"Avg Angular Jerk: {np.mean(angular_jerks):.4f} rad/s^2")
print(f"Avg Min Dist to Humans: {np.mean(min_dists_humans):.2f} m")
        