import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class RobustHybridCnnMlp(BaseFeaturesExtractor):
    """
    Robust Hybrid Feature Extractor (CNN + MLP).
    
    Designed for:
    1. High-Resolution LiDAR (1080 rays).
    2. Noisy Environments (Sim2Real gap).
    3. Temporal Stacking (Velocity/Acceleration estimation).
    
    The architecture uses a large initial kernel to act as a noise filter
    and downsamples aggressively to maintain computational efficiency.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        num_rays: int = 1080,
        stack_dim: int = 5,
        hidden_dim: int = 256,
    ):
        # We initialize the superclass with the final features_dim
        super().__init__(observation_space, features_dim=hidden_dim)

        self.num_rays = num_rays
        self.stack_dim = stack_dim
        
        print(f"🧠 [RobustHybridCnnMlp] Init: Rays={num_rays}, Stack={stack_dim}")

        # === 1. LIDAR ENCODER (CNN) ===
        # Designed to process [Batch, Stack, Rays]
        
        self.lidar_spatial_encoder = nn.Sequential(
            # Layer 1: Noise Filtering & Initial Feature Extraction
            # Kernel=25: Covers approx ~8 degrees of FOV. 
            # This is crucial to ignore single-ray dropouts/glitches.
            # Stride=2: Immediately halves the dimension (1080 -> 540) for efficiency.
            nn.Conv1d(in_channels=stack_dim, out_channels=32, kernel_size=25, stride=2, padding=12),
            nn.ReLU(),
            # No MaxPool here to preserve spatial coherence for the next layer
            
            # Layer 2: Shape Recognition
            # Kernel=9: Refines features (corners, legs).
            # Stride=2: Downsamples (540 -> 270)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            
            # Layer 3: Global Feature Abstraction
            # Kernel=5: High-level abstraction.
            # Stride=2: Downsamples (270 -> 135)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # Final Pooling to condense features
            nn.MaxPool1d(kernel_size=2) # 135 -> ~67 spatial points
        )

        # === DYNAMIC OUTPUT CALCULATION ===
        # We perform a dummy pass to calculate the exact flattened output size of the CNN.
        # This avoids manual calculation errors when changing kernel/padding/strides.
        with torch.no_grad():
            # Create a dummy tensor matching the expected input shape
            # Shape: [Batch=1, Channels=Stack, Length=Rays]
            dummy_input = torch.zeros(1, stack_dim, num_rays)
            dummy_output = self.lidar_spatial_encoder(dummy_input)
            
            # Flatten to get the vector size
            self.cnn_output_dim = dummy_output.view(1, -1).shape[1]
            
        print(f"   -> CNN Output Flattened Size: {self.cnn_output_dim}")

        # === 2. SCALAR ENCODER (MLP) ===
        # Processes: [Dist_Goal, Heading, Lin_Vel, Ang_Vel]
        self.scalar_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # === 3. SENSOR FUSION ===
        # Concatenates CNN features + Scalar features
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Extra layer for deep reasoning
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            obs: Tensor of shape (batch_size, 4 + num_rays * stack_dim)
        """
        batch_size = obs.shape[0]

        # 1. Split Observations
        # The environment returns a flat vector. We need to split it.
        # Slicing: [0:4] are scalars, [4:] is the flattened LiDAR stack
        scalars = obs[:, :4]
        lidar_flat = obs[:, 4:]

        # 2. Reshape LiDAR for Conv1d
        # Expected Conv1d Input: (Batch, Channels, Length) -> (Batch, Stack, Rays)
        # We assume the stack was flattened in order: [Ray1_t0, Ray2_t0... Ray1_t1...]
        # But usually Gym stacks are: [Frame_t-2, Frame_t-1, Frame_t]
        # So we view it as (Batch, Stack, Rays)
        
        # Safety check for dimensions (optional but recommended for debugging)
        # expected_lidar_size = self.num_rays * self.stack_dim
        # assert lidar_flat.shape[1] == expected_lidar_size, f"Input mismatch: {lidar_flat.shape[1]} vs {expected_lidar_size}"

        lidar = lidar_flat.view(batch_size, self.stack_dim, self.num_rays)

        # 3. Encode LiDAR
        lidar_encoded = self.lidar_spatial_encoder(lidar)
        lidar_encoded = lidar_encoded.view(batch_size, -1) # Flatten

        # 4. Encode Scalars
        scalars_encoded = self.scalar_encoder(scalars)

        # 5. Fusion
        combined = torch.cat([lidar_encoded, scalars_encoded], dim=1)
        features = self.fusion(combined)

        return features