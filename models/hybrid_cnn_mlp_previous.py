from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class HybridCnnMlp(BaseFeaturesExtractor):
    """
    Feature extractor ibrido CNN + MLP per LIDAR + scalari
    """

    def __init__(
        self,
        observation_space,
        num_rays=108,
        stack_dim=3,
        hidden_dim=128,
    ):
        # features_dim = dimensione dell'output finale
        super().__init__(observation_space, features_dim=hidden_dim)

        self.num_rays = num_rays
        self.stack_dim = stack_dim

        # === LIDAR ENCODER ===
        self.lidar_spatial_encoder = nn.Sequential(
            nn.Conv1d(stack_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        lidar_encoding_dim = 64 * 27  # 108 → 54 → 27

        # === SCALAR ENCODER ===
        self.scalar_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # === FUSION ===
        self.fusion = nn.Sequential(
            nn.Linear(lidar_encoding_dim + 64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, 328)
        return: (batch, features_dim)
        """
        batch_size = obs.shape[0]

        scalars = obs[:, :4]
        lidar_flat = obs[:, 4:]

        lidar = lidar_flat.view(batch_size, self.stack_dim, self.num_rays)

        lidar_encoded = self.lidar_spatial_encoder(lidar)
        lidar_encoded = lidar_encoded.view(batch_size, -1)

        scalars_encoded = self.scalar_encoder(scalars)

        combined = torch.cat([lidar_encoded, scalars_encoded], dim=1)
        features = self.fusion(combined)

        return features

