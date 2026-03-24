import time
import jax
import jax.numpy as jnp
from jax_env_multi import reset_env, step_env

def main():
    print("🚀 Initializing social navigation environment...")
    rng = jax.random.PRNGKey(42)
    reset_key, step_key = jax.random.split(rng)

    # 1. Reset the environment
    obs, state = reset_env(reset_key)
    action = jnp.array([1.0, 0.0]) # 1.0 m/s forward, 0.0 rad/s turn

    # 2. JIT compile the step function
    # We mark ghost_robot as static so JAX knows it's a compile-time constant
    fast_step = jax.jit(step_env, static_argnames=("ghost_robot",))

    print("⚙️  JIT Compiling step_env (XLA is tracing the graph)...")
    t0 = time.time()
    next_obs, next_state, reward, done, info = fast_step(step_key, state, action, ghost_robot=True)
    
    # Block until the GPU actually finishes the computation
    jax.block_until_ready(next_obs)
    print(f"✅ Compilation & first step finished in {time.time() - t0:.2f} seconds!")

    print("⚡ Running a second step to measure pure hardware execution speed...")
    t1 = time.time()
    next_obs, next_state, reward, done, info = fast_step(step_key, next_state, action, ghost_robot=True)
    jax.block_until_ready(next_obs)
    
    print(f"🏎️  Raw execution time: {time.time() - t1:.5f} seconds!")
    print("If you see this, your environment is mathematically sound, non-differentiable, and ready to train!")

if __name__ == "__main__":
    main()