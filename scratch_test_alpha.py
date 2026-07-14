import jax.numpy as jnp
target_entropy = -2.0
log_pi = jnp.array(-3.0)  # Very random
# If log_pi is -3 and target is -2, it's MORE random than target. We want alpha to decrease.
# SAC paper: J(alpha) = E[-alpha * (log_pi + H)]  where H is the target entropy.
# Wait, SAC paper says J(alpha) = E[ alpha * (-log_pi - H) ]? No, usually H is target for -log_pi.
# Let's check TQC paper.
