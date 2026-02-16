import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class EndToEndNavExtractor(BaseFeaturesExtractor):
    """
    Implements: [Lidar_t -> 1D CNN] + Pose_t -> LSTM -> Concat with State -> Final Output.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We define features_dim as the output size of this extractor. 
        # SB3's policy network will use this as its input.
        super().__init__(observation_space, features_dim)
        
        # 1. Read dimensions from the observation space
        lidar_shape = observation_space["lidar"].shape  # (Stack, Rays), e.g., (5, 72)
        self.seq_len = lidar_shape[0]
        self.rays_dim = lidar_shape[1]
        
        self.pose_dim = observation_space["pose"].shape[1] # Should be 3: [x, y, theta]
        self.state_dim = observation_space["state"].shape[0] # Should be 8 (v, w, goal, etc.)
        
        # 2. 1D CNN (Spatial Feature Extraction per time step)
        cnn_out_channels = 16
        self.cnn = nn.Sequential(
            # Input: (Batch * Seq_len, 1 channel, Rays)
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=cnn_out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.rays_dim)
            cnn_flat_size = self.cnn(dummy_lidar).shape[1]
            
        # 3. LSTM (Temporal Integration)
        # Input to LSTM is the CNN features PLUS the 3 pose variables at time t
        lstm_input_size = cnn_flat_size + self.pose_dim
        self.lstm_hidden_size = 64
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=1, 
            batch_first=True # Expects (Batch, Seq, Features)
        )
        
        # 4. Final Projection (Combining LSTM output with current global state)
        final_concat_size = self.lstm_hidden_size + self.state_dim
        
        self.linear = nn.Sequential(
            nn.Linear(final_concat_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # inputs shape: lidar=(Batch, 5, 72), pose=(Batch, 5, 3), state=(Batch, 8)
        lidar_seq = observations["lidar"]
        pose_seq = observations["pose"]
        current_state = observations["state"]
        
        batch_size = lidar_seq.size(0)
        
        # --- CNN Spatial Processing ---
        # Fold the sequence dimension into the batch dimension to process all frames at once
        lidar_reshaped = lidar_seq.view(batch_size * self.seq_len, 1, self.rays_dim)
        
        # Pass all steps through CNN simultaneously
        cnn_features = self.cnn(lidar_reshaped) 
        
        # Unfold back to sequential format: (Batch, Seq_len, cnn_flat_size)
        cnn_features_seq = cnn_features.view(batch_size, self.seq_len, -1)
        
        # --- Concatenate CNN output with Pose ---
        # Shape: (Batch, Seq_len, cnn_flat_size + 3)
        lstm_input = torch.cat((cnn_features_seq, pose_seq), dim=2)
        
        # --- LSTM Temporal Processing ---
        # C++ optimized sequential processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Discard intermediate steps, keep only the output of the final step T
        # Shape: (Batch, lstm_hidden_size)
        lstm_final_out = lstm_out[:, -1, :] 
        
        # --- Final Concatenation and Output ---
        # Combine temporal summary with current step state
        final_concat = torch.cat((lstm_final_out, current_state), dim=1)
        
        # Return features to SB3's policy network
        return self.linear(final_concat)