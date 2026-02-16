import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces

class VecTemporalStack(VecEnvWrapper):
    """
    Wrapper that stacks BOTH 'lidar' and 'pose' over time.
    """
    def __init__(self, venv, stack_dim):
        orig_lidar_space = venv.observation_space["lidar"]
        orig_pose_space = venv.observation_space["pose"]
        orig_state_space = venv.observation_space["state"]
        
        self.num_rays = orig_lidar_space.shape[0]
        self.stack_dim = stack_dim
        self.n_envs = venv.num_envs
        
        new_observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(stack_dim, self.num_rays), dtype=np.float32),
            "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(stack_dim, 3), dtype=np.float32),
            "state": orig_state_space 
        })
        
        super().__init__(venv, observation_space=new_observation_space)
        
        # Buffers for history
        self.lidar_buffer = np.zeros((self.n_envs, stack_dim, self.num_rays), dtype=np.float32)
        self.pose_buffer = np.zeros((self.n_envs, stack_dim, 3), dtype=np.float32)

    def reset(self):
        obs = self.venv.reset() 
        new_lidar = obs['lidar']
        new_pose = obs['pose']
        
        # Fill buffers with the first frame
        for k in range(self.stack_dim):
            self.lidar_buffer[:, k, :] = new_lidar
            self.pose_buffer[:, k, :] = new_pose
            
        return self._get_dict_obs(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        
        # Roll and update buffers
        self.lidar_buffer = np.roll(self.lidar_buffer, -1, axis=1)
        self.pose_buffer = np.roll(self.pose_buffer, -1, axis=1)
        
        self.lidar_buffer[:, -1, :] = obs['lidar']
        self.pose_buffer[:, -1, :] = obs['pose']
        
        for i, done in enumerate(dones):
            if done:
                term_obs = infos[i].get("terminal_observation")
                if term_obs is not None:
                    # Handle terminal observation for Lidar
                    l_hist = self.lidar_buffer[i].copy()
                    l_hist = np.roll(l_hist, -1, axis=0)
                    l_hist[-1, :] = term_obs["lidar"]
                    
                    # Handle terminal observation for Pose
                    p_hist = self.pose_buffer[i].copy()
                    p_hist = np.roll(p_hist, -1, axis=0)
                    p_hist[-1, :] = term_obs["pose"]
                    
                    infos[i]["terminal_observation"] = {
                        "lidar": l_hist,
                        "pose": p_hist,
                        "state": term_obs["state"]
                    }
                # Reset buffers for the specific environment that died
                self.lidar_buffer[i, :, :] = obs['lidar'][i]
                self.pose_buffer[i, :, :] = obs['pose'][i]

        return self._get_dict_obs(obs), rews, dones, infos

    def _get_dict_obs(self, raw_obs):
        return {
            "lidar": self.lidar_buffer.copy(),
            "pose": self.pose_buffer.copy(),
            "state": raw_obs['state']
        }