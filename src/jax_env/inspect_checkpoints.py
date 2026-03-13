"""
inspect_checkpoints.py — stampa la struttura completa dei checkpoint.
Esegui da ~/indoor-rl-nav/src/jax_env/:
    python3 inspect_checkpoints.py
"""
import os
import flax.serialization
 
def print_tree(d, indent=0):
    prefix = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}/")
                print_tree(v, indent + 1)
            else:
                shape = getattr(v, 'shape', type(v).__name__)
                dtype = getattr(v, 'dtype', '')
                print(f"{prefix}{k}: {shape} {dtype}")
    else:
        shape = getattr(d, 'shape', type(d).__name__)
        print(f"{prefix}{shape}")
 
checkpoints = {
    "SAC": "checkpoints_sac/sac_best.msgpack",
    "TQC": "checkpoints_tqc/tqc_best.msgpack",
    "PPO": "checkpoints/ppo_model_best.msgpack",
}
 
for name, path in checkpoints.items():
    print(f"\n{'='*60}")
    print(f" {name}: {path}")
    print('='*60)
    if not os.path.exists(path):
        print("  [NOT FOUND]")
        continue
    with open(path, "rb") as f:
        bundle = flax.serialization.msgpack_restore(f.read())
    print(f"Top-level keys: {list(bundle.keys())}")
    for top_key in ["actor_params", "actor_enc_params", "actor_head_params"]:
        if top_key in bundle:
            print(f"\n  [{top_key}] subtree:")
            print_tree(bundle[top_key], indent=2)
 