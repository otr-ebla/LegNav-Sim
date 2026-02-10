from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
import torch as th
import numpy as np

class AdaptiveTQC(TQC):
    """
    Adaptive Risk-Averse TQC (AR-TQC).
    
    PAPER CONTRIBUTION:
    Dynamically adjusts the number of truncated quantiles based on the 
    immediate risk perceived by the LiDAR sensor.
    
    Why it works:
    - High LiDAR value (near 1.0) = Obstacle Close -> High Risk.
    - We increase 'top_quantiles_to_drop' to remove optimistic estimates.
    - The agent becomes "pessimistic" and prioritizes safety (CVaR).
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        ent_coef_losses, ent_coefs = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            # === [MAJOR CONTRIBUTION: SENSOR-DRIVEN RISK] ===
            # 1. Extract LiDAR data. 
            # In your env, obs is: [scalars(4), lidar_data(...)]
            lidar_data = replay_data.observations[:, 4:]
            
            # 2. Estimate Risk.
            # Your env uses INVERSE LiDAR: 1.0 = Collision (Close), 0.0 = Max Range (Far).
            # So, the MAX value in the array represents the closest obstacle.
            #current_risk_metric = th.max(lidar_data).item() # scalar 0.0 to 1.0
            current_risk_metric = th.quantile(lidar_data, 0.99).item() # More robust to outliers than max, captures "worst-case" scenario without being a single noisy spike

            # 3. Dynamic Truncation Calculation
            # Thresholds
            risk_safe = 0.3    # ~30% sensor range (Safe)
            risk_danger = 0.85 # ~85% sensor range (Danger! Very close)
            
            min_drop = 0   # Standard TQC (Optimistic)
            max_drop = 8   # High safety (Pessimistic - cuts top 32% of distribution if 25 quantiles)
            
            if current_risk_metric < risk_safe:
                current_drop = min_drop
            elif current_risk_metric > risk_danger:
                current_drop = max_drop
            else:
                # Linear Interpolation
                ratio = (current_risk_metric - risk_safe) / (risk_danger - risk_safe)
                current_drop = int(min_drop + ratio * (max_drop - min_drop))
            
            # Apply multiplier for n_critics (TQC internal logic)
            n_dropped_total = current_drop * self.critic.n_critics
            
            # Safety clamp: Ensure we keep at least 5 quantiles to learn
            total_quantiles = self.critic.quantiles_total
            if (total_quantiles - n_dropped_total) < 5:
                n_dropped_total = total_quantiles - 5
            
            target_quantiles_count = total_quantiles - n_dropped_total
            # === [END CONTRIBUTION BLOCK] ===

            # Action by the current actor
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient logic
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                
                # Sort quantiles
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                
                # [MODIFIED] Use dynamic truncation here
                next_quantiles = next_quantiles[:, :target_quantiles_count]

                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                from stable_baselines3.common.utils import polyak_update
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
        # Loggare il drop level è cruciale per i grafici del paper!
        self.logger.record("train/adaptive_drop_q", current_drop) 

    def learn(self, **kwargs):
        return super().learn(**kwargs)