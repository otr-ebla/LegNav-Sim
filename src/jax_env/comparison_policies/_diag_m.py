"""Diagnostic: is M's next-z residual target learnable at all?

Loads the trained encoder, collects a small fresh rollout, encodes to z, and
checks (a) latent scale/smoothness and (b) how much of the residual Δz a simple
linear predictor can explain. If even a linear AR model gets ~0 R², the target
is genuinely unpredictable from z-history alone (e.g. needs the action), and no
transformer will help. If a linear model explains a decent chunk but our M
didn't, the problem is optimization/scale, not the data.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))
sys.path.insert(0, _THIS)

import jax, jax.numpy as jnp, numpy as np
import flax.serialization
from jax_train import NUM_ENVS
from comparison_policies.pretrain_navrep import collect_expert_lidar, _encode_all, VM_CKPT
from comparison_policies.navrep_network import Z_DIM

# load trained encoder
with open(VM_CKPT, "rb") as f:
    blob = f.read()
import comparison_policies.navrep_network as nn_mod
target = {"encoder": None, "M": None}
bundle = flax.serialization.from_bytes(
    {"encoder": {}, "M": {}}, blob
)
enc = bundle["encoder"]

rng = jax.random.PRNGKey(123)
_, seq = collect_expert_lidar(NUM_ENVS * 150, rng)   # (150, 1024, 216)
z = _encode_all(enc, np.asarray(seq))                # (T, N, Z)
T, N, Z = z.shape
print(f"\nz shape {z.shape}")

z = z.transpose(1, 0, 2)                             # (N, T, Z) per-trajectory
dz = z[:, 1:, :] - z[:, :-1, :]                      # (N, T-1, Z)
z_t = z[:, :-1, :]
z_tm1 = np.concatenate([z[:, :1, :], z[:, :-2, :]], axis=1)  # shifted history

# scale / smoothness
print(f"z per-dim std (mean over dims): {z.std(axis=(0,1)).mean():.3f}")
print(f"Δz per-dim std (mean over dims): {dz.std(axis=(0,1)).mean():.3f}")
zf = z.reshape(-1, Z)
corr = np.mean([np.corrcoef(z[:, :-1, d].ravel(), z[:, 1:, d].ravel())[0,1]
                for d in range(Z)])
print(f"mean lag-1 autocorr of z (per dim): {corr:.3f}")

# baseline MSE = sum-over-dim variance of Δz (what 'predict 0 residual' achieves)
base_mse = (dz.reshape(-1, Z) ** 2).sum(axis=1).mean()
print(f"\nzero-prediction residual MSE (M's baseline): {base_mse:.3f}")

def lstsq_r2(X, Y):
    # X:(M,F) Y:(M,Z); fit Y≈X W ; report total MSE and R²
    X = np.concatenate([X, np.ones((X.shape[0], 1), np.float32)], axis=1)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ W
    mse = ((pred - Y) ** 2).sum(axis=1).mean()
    ss_res = ((pred - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return mse, 1 - ss_res / ss_tot

X1 = z_t.reshape(-1, Z); Y = dz.reshape(-1, Z)
mse1, r2_1 = lstsq_r2(X1, Y)
X2 = np.concatenate([z_t, z_tm1], axis=-1).reshape(-1, 2*Z)
mse2, r2_2 = lstsq_r2(X2, Y)
print(f"linear Δz ~ z_t        : MSE {mse1:.3f}  R² {r2_1:.3f}")
print(f"linear Δz ~ [z_t,z_t-1]: MSE {mse2:.3f}  R² {r2_2:.3f}")
