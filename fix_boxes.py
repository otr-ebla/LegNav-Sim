import re

with open("src/jax_env/jax_scenarios.py", "r") as f:
    content = f.read()

# Replace basic zero initializations
content = content.replace("obs_boxes = jnp.zeros((NUM_OBS_BOX, 4))", "obs_boxes = _zero_boxes(NUM_OBS_BOX)")
content = content.replace("obs_boxes   = jnp.zeros((NUM_OBS_BOX, 4))", "obs_boxes = _zero_boxes(NUM_OBS_BOX)")

# We also need to fix `.at[x].set([cx, cy, hw, hh])`
# Example: .at[0].set([wall_width / 2.0, ROOM_H / 2.0, wall_width / 2.0, ROOM_H / 2.0])
# Replace it with .at[0].set(_aabb_to_quad(...))
# Let's use regex for `.at\[(\d+)\].set\(\[(.*?)\]\)`
content = re.sub(r'\.at\[(\d+)\]\.set\(\[(.*?)\]\)', r'.at[\1].set(_aabb_to_quad(\2))', content)

with open("src/jax_env/jax_scenarios.py", "w") as f:
    f.write(content)
