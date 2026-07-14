"""bfloat16 mixed precision for GPU training.

Flax promotes the operands of every layer to their common dtype, so feeding
bf16 activations into fp32 params silently upcasts back to fp32 — no Tensor
Core matmuls. Real bf16 compute needs the *params* cast at apply time too.
`bf16_apply` does that: params and float inputs go in as bf16, outputs come
back as fp32 so losses, log-probs and GAE stay in full precision. The optimizer
never sees bf16 — it keeps the fp32 master weights, and gradients flow back
through the cast as fp32, so checkpoints are unchanged.

Set LEGNAV_BF16=0 to fall back to full fp32.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

USE_BF16      = os.environ.get("LEGNAV_BF16", "1") == "1"
NET_DTYPE     = jnp.bfloat16 if USE_BF16 else jnp.float32
PRECISION_STR = "bfloat16" if USE_BF16 else "float32"


def _cast(tree, dtype):
    def _leaf(x):
        if isinstance(x, (jax.Array, np.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(_leaf, tree)


def cast_net(tree):
    """Cast every float leaf to the network compute dtype."""
    return _cast(tree, NET_DTYPE)


def cast_f32(tree):
    return _cast(tree, jnp.float32)


def bf16_apply(apply_fn):
    """Wrap a Flax `.apply` so the forward pass runs in bf16 and returns fp32.

    Keyword arguments (rngs, method, mutable, …) are passed through untouched.
    """
    if not USE_BF16:
        return apply_fn

    def wrapped(variables, *args, **kwargs):
        out = apply_fn(cast_net(variables), *_cast(args, NET_DTYPE), **kwargs)
        return cast_f32(out)

    return wrapped
