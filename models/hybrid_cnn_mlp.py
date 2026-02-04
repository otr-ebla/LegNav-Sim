# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch
# import torch.nn as nn


# class HybridCnnMlp(BaseFeaturesExtractor):
#     """
#     Feature extractor ibrido CNN + MLP per LIDAR + scalari
#     """

#     def __init__(
#         self,
#         observation_space,
#         num_rays=108,
#         stack_dim=3,
#         hidden_dim=128,
#     ):
#         # features_dim = dimensione dell'output finale
#         super().__init__(observation_space, features_dim=hidden_dim)

#         self.num_rays = num_rays
#         self.stack_dim = stack_dim

#         # === LIDAR ENCODER ===
#         self.lidar_spatial_encoder = nn.Sequential(
#             nn.Conv1d(stack_dim, 32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#         )

#         lidar_encoding_dim = 64 * 27  # 108 → 54 → 27

#         # === SCALAR ENCODER ===
#         self.scalar_encoder = nn.Sequential(
#             nn.Linear(4, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#         )

#         # === FUSION ===
#         self.fusion = nn.Sequential(
#             nn.Linear(lidar_encoding_dim + 64, hidden_dim),
#             nn.ReLU(),
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         obs: (batch, 328)
#         return: (batch, features_dim)
#         """
#         batch_size = obs.shape[0]

#         scalars = obs[:, :4]
#         lidar_flat = obs[:, 4:]

#         lidar = lidar_flat.view(batch_size, self.stack_dim, self.num_rays)

#         lidar_encoded = self.lidar_spatial_encoder(lidar)
#         lidar_encoded = lidar_encoded.view(batch_size, -1)

#         scalars_encoded = self.scalar_encoder(scalars)

#         combined = torch.cat([lidar_encoded, scalars_encoded], dim=1)
#         features = self.fusion(combined)

#         return features

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch
# import torch.nn as nn

# class HybridCnnMlp(BaseFeaturesExtractor):
#     """
#     Feature extractor ibrido CNN + MLP per LIDAR ad alta risoluzione + scalari
#     """

#     def __init__(
#         self,
#         observation_space,
#         num_rays=1080,   # Default aggiornato a 1080
#         stack_dim=5,     # Aumentato default a 5 per gestire il rumore temporale
#         hidden_dim=256,  # Aumentato leggermente per gestire più info
#     ):
#         super().__init__(observation_space, features_dim=hidden_dim)

#         self.num_rays = num_rays
#         self.stack_dim = stack_dim

#         # === LIDAR ENCODER ===
#         # Con 1080 raggi e rumore, usiamo kernel più ampi per "vedere" le forme 
#         # ignorando i singoli raggi impazziti.
#         self.lidar_spatial_encoder = nn.Sequential(
#             # Layer 1: Rilevamento feature locali (angoli, linee)
#             nn.Conv1d(stack_dim, 32, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2), # 1080 -> 540
            
#             # Layer 2: Astrazione
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2), # 540 -> 270
            
#             # Layer 3: Feature globali
#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2), # 270 -> 135
#         )

#         # CALCOLO DINAMICO DELLA DIMENSIONE DI USCITA
#         # Facciamo un passaggio "finto" per capire quanto è grande l'uscita della CNN
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, stack_dim, num_rays)
#             dummy_output = self.lidar_spatial_encoder(dummy_input)
#             self.cnn_output_dim = dummy_output.view(1, -1).shape[1]
            
#         print(f"🧠 HybridCnnMlp Init: Rays={num_rays}, Stack={stack_dim} -> CNN Output Flattened: {self.cnn_output_dim}")

#         # === SCALAR ENCODER ===
#         self.scalar_encoder = nn.Sequential(
#             nn.Linear(4, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#         )

#         # === FUSION ===
#         self.fusion = nn.Sequential(
#             nn.Linear(self.cnn_output_dim + 64, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), # Un layer in più per processare la fusione
#             nn.ReLU()
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         batch_size = obs.shape[0]

#         # Separazione input: primi 4 sono scalari (dist, head, v, w)
#         scalars = obs[:, :4]
#         lidar_flat = obs[:, 4:]

#         # Reshape per la CNN: (Batch, Stack, Rays)
#         # Assicuriamoci che l'input corrisponda a quello atteso
#         current_rays = lidar_flat.shape[1] // self.stack_dim
        
#         # Safety check: se config.py dice 1080 ma qui arriva altro, evitiamo crash silenziosi
#         if current_rays != self.num_rays:
#              # Se capita questo, probabilmente stack_dim non è allineato nel VecNormalize
#              # Adattiamo dinamicamente per evitare crash, ma è un warning
#              pass 

#         lidar = lidar_flat.view(batch_size, self.stack_dim, self.num_rays)

#         lidar_encoded = self.lidar_spatial_encoder(lidar)
#         lidar_encoded = lidar_encoded.view(batch_size, -1)

#         scalars_encoded = self.scalar_encoder(scalars)

#         combined = torch.cat([lidar_encoded, scalars_encoded], dim=1)
#         features = self.fusion(combined)

#         return features

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class HybridCnnMlp(BaseFeaturesExtractor):
    """
    Feature extractor UNIFICATO.
    Gestisce automaticamente:
    - Vecchi Modelli (108 raggi) -> Kernel 5 (Legacy)
    - Nuovi Modelli (1080 raggi) -> Kernel 7 (High Res)
    """

    def __init__(
        self,
        observation_space,
        num_rays=108,
        stack_dim=3,
        hidden_dim=256, 
    ):
        # Inizializziamo super con un features_dim temporaneo
        super().__init__(observation_space, features_dim=hidden_dim)

        self.num_rays = num_rays
        self.stack_dim = stack_dim

        print(f"🧠 HybridCnnMlp Init: Rays={num_rays}, Stack={stack_dim}")

        # === SELETTORE AUTOMATICO ARCHITETTURA ===
        
        # CASO A: VECCHI MODELLI (Raggi <= 108)
        # Riconosciamo i vecchi pesi e costruiamo la rete vecchia maniera
        if self.num_rays <= 108:
            print("   -> MODE: Legacy (Compatible with NNeasy2)")
            
            # 1. CNN Esatta dei vecchi training
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
            
            # Calcolo dimensione uscita vecchia: 108 -> 54 -> 27. 27*64 = 1728
            self.cnn_output_dim = 1728 
            
            # 2. Fusion Esatta dei vecchi training (128 hidden dim fisso se necessario)
            # Nota: NNeasy2 usava hidden_dim=128. Se il parametro passato è diverso, forziamo qui?
            # Per ora ci fidiamo di 'hidden_dim' passato da run_ppo.py
            
            self.fusion = nn.Sequential(
                nn.Linear(self.cnn_output_dim + 64, hidden_dim),
                nn.ReLU(),
            )

        # CASO B: NUOVI MODELLI (Raggi > 108)
        else:
            print("   -> MODE: High-Res (Kernel 7, Noise Robust)")
            self.lidar_spatial_encoder = nn.Sequential(
                nn.Conv1d(stack_dim, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2), 
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2), 
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2), 
            )

            # Calcolo dinamico
            with torch.no_grad():
                dummy = torch.zeros(1, stack_dim, num_rays)
                out = self.lidar_spatial_encoder(dummy)
                self.cnn_output_dim = out.view(1, -1).shape[1]
                
            self.fusion = nn.Sequential(
                nn.Linear(self.cnn_output_dim + 64, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        # === SCALAR ENCODER (Comune) ===
        self.scalar_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
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