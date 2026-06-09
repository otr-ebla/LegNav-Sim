import jax
import jax.numpy as jnp
from src.jax_env.jax_env import reset_env, step_env
from src.jax_env.jax_scenarios import _random_scen

key = jax.random.PRNGKey(42)
print("Testing reset...")
obs, state = reset_env(key, 5.0, -1, 1.0)
print("Testing step...")
obs, state, rew, done, info = step_env(key, state, jnp.array([1.0, 0.0]))
print("Done!")
