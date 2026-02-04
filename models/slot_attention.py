# models/slot_attention.py

import torch
import torch.nn as nn
import numpy as np

class SlotAttention(nn.Module):
    # ... (Keep __init__ as is) ...
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        # ... same initialization ...
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None, return_attn=False):
        """
        Modified forward to optionally return attention weights.
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs) # Normalizziamo gli inputs perché è critico
        k = self.to_k(inputs) # Proiettiamo in K, le chiavi rappresentano l'indirizz o caratteristiche identificative delle input feature
        v = self.to_v(inputs) # Proiettiamo in V, i valori rappresentano il contenuto effettivo delle input feature
        # k e v sono statiche, perché non cambiano durante le iterazioni

        # We will store the attention of the FINAL iteration
        last_attn = None

        # ... (codice precedente identico) ...
        
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            # 1. Calcolo Softmax (La "Competizione")
            # Questa dice: "Chi possiede questo raggio?" (Valori 0-1)
            attn = dots.softmax(dim=1) + self.eps
            
            # --- MODIFICA QUI ---
            # Salviamo QUESTA versione per la visualizzazione e il ritorno
            if return_attn:
                last_attn = attn.clone() 
            # --------------------

            # 2. Normalizzazione per Weighted Mean (Serve solo per l'update interno)
            # Questa dice: "Quanto pesa questo raggio nella media dello slot?"
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)

            # Usa attn_norm per calcolare gli updates
            updates = torch.einsum('bij,bjd->bid', attn_norm, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, n_s, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        if return_attn:
            return slots, last_attn # Ritorniamo la versione "pura" (Softmax)
            
        return slots

# Update the Extractor Wrapper to expose this
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# In models/slot_attention.py

class LidarSlotAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_rays=108, num_slots=6, slot_dim=64):
        # ... (codice precedente invariato) ...
        features_dim = 4 + (num_slots * slot_dim)
        super().__init__(observation_space, features_dim)
        
        self.num_rays = num_rays
        self.slot_dim = slot_dim  # Importante salvarlo

        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, slot_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- NOVITÀ: Positional Embedding Apprendibile ---
        # Creiamo un vettore unico per ogni angolo (raggio) che si somma alle feature
        self.pos_embedding = nn.Parameter(torch.randn(1, slot_dim, num_rays))
        nn.init.normal_(self.pos_embedding, std=0.01)

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=3
        )

    def forward(self, observations):
        scalars = observations[:, :4]
        lidar = observations[:, 4:]
        lidar_in = lidar.unsqueeze(1)
        
        # 1. Estrazione Feature (Batch, 64, 108)
        features = self.lidar_encoder(lidar_in)
        
        # 2. --- AGGIUNTA POSITIONAL EMBEDDING ---
        # Sommiamo l'informazione "dove sei?" all'informazione "cosa vedi?"
        features = features + self.pos_embedding

        # 3. Permutazione per Slot Attention (Batch, 108, 64)
        features = features.permute(0, 2, 1)
        
        # 4. Slot Attention
        slots = self.slot_attention(features)
        
        slots_flat = slots.flatten(start_dim=1)
        return torch.cat([scalars, slots_flat], dim=1)

    # Ricordati di aggiornare anche get_attention_map con la stessa logica!
    def get_attention_map(self, observations):
        with torch.no_grad():
            lidar = observations[:, 4:]
            lidar_in = lidar.unsqueeze(1)
            features = self.lidar_encoder(lidar_in)
            
            # --- CRITICO: Anche qui aggiungere pos embedding ---
            features = features + self.pos_embedding
            
            features = features.permute(0, 2, 1)
            _, attn_weights = self.slot_attention(features, return_attn=True)
            return attn_weights