import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Lidar1DCNN(BaseFeaturesExtractor):
    """
    Custom Feature Extractor che usa:
    - 1D CNN per processare lo stack del Lidar (estrae forme geometriche).
    - Pass-through (o piccola MLP) per gli scalari (distanza, angolo, velocità).
    - Fusione finale delle features.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # --- CONFIGURAZIONE ---
        self.n_scalars = 4  # [dist, angle, v, w]
        n_input_features = observation_space.shape[0]
        self.n_lidar_flat = n_input_features - self.n_scalars
        
        # Recuperiamo stack_dim e num_rays ipotizzando che la parte lidar sia (stack * rays)
        # Esempio: 324 punti lidar totali. Se num_rays=108 -> stack=3.
        # È fondamentale sapere num_rays per fare il reshape corretto.
        # Possiamo dedurlo o passarlo come argomento. Qui lo dedurremo dai config se possibile,
        # ma per robustezza assumiamo che l'utente lo passi o lo cabliamo.
        # Cabliamo i default del tuo progetto:
        self.num_rays = 108 
        self.stack_dim = self.n_lidar_flat // self.num_rays 
        
        # --- RAMO LIDAR (CNN 1D) ---
        # Input: (Batch, Channels=Stack_Dim, Length=Num_Rays)
        self.cnn = nn.Sequential(
            # Layer 1: Riconosce feature semplici (linee, punti)
            nn.Conv1d(in_channels=self.stack_dim, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # Layer 2: Riconosce forme (angoli, curve)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: Feature complesse
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calcolo dimensione output CNN facendo un forward fittizio
        with torch.no_grad():
            # (Batch=1, Channels, Length)
            sample_lidar = torch.zeros(1, self.stack_dim, self.num_rays)
            n_flatten_cnn = self.cnn(sample_lidar).shape[1]
            
        # --- RAMO SCALARI ---
        # Semplice processamento per dare peso agli scalari
        self.scalar_net = nn.Sequential(
            nn.Linear(self.n_scalars, 16),
            nn.ReLU()
        )
        
        # --- TESTA FINALE ---
        # Unisce CNN e Scalari
        self.linear = nn.Sequential(
            nn.Linear(n_flatten_cnn + 16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Separazione Input
        # scalars: [Batch, 0:4]
        # lidar:   [Batch, 4:]
        scalars = observations[:, :self.n_scalars]
        lidar_flat = observations[:, self.n_scalars:]
        
        # 2. Reshape Lidar per CNN 1D
        # Da [Batch, 324] a [Batch, 3 (stack), 108 (rays)]
        # PyTorch Conv1d vuole (Batch, Channel, Length)
        lidar_3d = lidar_flat.view(-1, self.stack_dim, self.num_rays)
        
        # 3. Forward CNN
        cnn_out = self.cnn(lidar_3d)
        
        # 4. Forward Scalari
        scalar_out = self.scalar_net(scalars)
        
        # 5. Concatenazione e Output
        combined = torch.cat((cnn_out, scalar_out), dim=1)
        return self.linear(combined)