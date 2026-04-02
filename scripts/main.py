from envs.simple_env import Simple2DEnv
import random

def main():
    env = Simple2DEnv(max_steps = 1000, 
                      num_rays=216,
                      num_people=20)
    obs = env.reset()
    print("Environment reset. Initial obs:", obs)

    done = False
    step_idx = 0

    while not done:
        w = random.uniform(-5, 5)
        v = random.uniform(0.5, 1)
        action = (v, w)

        obs, reward, done, info = env.step(action)
        

        step_idx += 1

        x, y, theta, lidar = obs
        lidar_str = [f"{d:.2f}" for d in lidar]
        print(
            f"Step {step_idx:02d} | "
            f"x={x:.2f} y={y:.2f} theta={theta:.2f} | "
            f"lidar={lidar_str} | "
            f"info={info}"
        )
        env.render()

    print("Episode finished.")
if __name__ == "__main__":
    main()