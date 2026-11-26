"""
obs_layouts.py — single source of truth for observation layouts (current + legacy).

Four flat-observation layouts exist in the repo history / checkpoints:

  v668  current  [kin(5) | goal_stack(2S) | ego_deltas(3S) | lidar(216 S)]       S=3
                 kin = [v/vmax, w, max_v_norm, goal_dist, goal_align/pi]. CALF nets (fused
                 input 203), current VanillaE2E.
  v667  legacy   same as v668 without goal_align (fused input 202). Tried for the revision
                 and reverted: PPO learned clearly slower without the explicit goal angle.
  v662  legacy   [pose(3S: gdx,gdy,theta/pi per frame) | state_vec(5) | lidar]   consecutive frames
                 pre-SE(2) PPO (fused input 206), old vanilla MLP, old NavRep.
  v659  legacy   [goal_stack(2S) | kin(5) | lidar]                               consecutive frames
                 DWA / MPPI planners, TAGD, current NavRep.

Evaluation rolls out ONE environment that stores the full-resolution ring buffers
(kin, goal, pose, lidar; length RAW_LEN) and every policy gets its own view of it
through `adapt(raw, layout, stride)`; so all policies face identical episodes and no
policy ever needs the env to be rebuilt.

`detect_*` read a checkpoint's parameter shapes and return an `ObsSpec`, or raise
`Unsupported` (callers turn that into a labelled "unsupported" table row instead of
crashing). Things shapes cannot tell are encoded by file-name rule (see _name_flags)
and by an explicit override.
"""
import os
from dataclasses import dataclass
import jax.numpy as jnp

from legnav.config import RobotConfig

STACK_DIM = 3
CURRENT_STRIDE = RobotConfig.LIDAR_STACK_STRIDE            # stride the current trainers use
RAW_LEN = (STACK_DIM - 1) * max(CURRENT_STRIDE, 1) + 1     # ring-buffer length held by the eval env
KIN, GOAL, POSE, RAYS = 5, 2, 3, 216
RAW_SIZE = KIN + RAW_LEN * (GOAL + POSE + RAYS)

# v672  real-robot track (LEGNAV_OBS_CMD=1, needs LEGNAV_REAL_ENV=1)
#       [kin(5) | cmd(4) | goal_stack(2S) | ego_deltas(3S) | lidar(216 S)]
#       kin's v, w are the MEASURED velocities; cmd = the two commands still in the transport delay,
#       newest first [v/max_v, w, v_prev/max_v, w_prev] (jax_env_real.cmd_obs). The network reads
#       kin+cmd as one 9-wide kin vector (fused input 207). Contract: legnav/deployment/shelfy/.
CMD_OBS = os.environ.get("LEGNAV_OBS_CMD", "0") == "1"
CMD = 4
OBS_LAYOUT = "v672" if CMD_OBS else "v668"

# index of max_v_norm = (max_v-0.2)/1.8 inside each layout (kin[2])
MAX_V_IDX = {"v667": 2, "v668": 2, "v672": 2, "v662": 11, "v659": 8}
LAYOUT_SIZE = {"v667": 667, "v668": 668, "v672": 672, "v662": 662, "v659": 659}
OBS_SIZE = LAYOUT_SIZE[OBS_LAYOUT]                         # what the current trainers produce


class Unsupported(Exception):
    """Checkpoint cannot be evaluated by this code (reason in the message)."""


@dataclass(frozen=True)
class ObsSpec:
    layout: str = "v668"
    stride: int = CURRENT_STRIDE
    legacy_net: bool = False        # old fused-206 network slicing
    se2: bool = True                # parameter-free SE(2) reprojection on/off
    tanh_inside: bool = True
    note: str = ""

    def describe(self):
        return (f"{self.layout}/stride{self.stride}"
                + ("/legacy-net" if self.legacy_net else "")
                + ("" if self.se2 else "/noSE2") + ("" if self.tanh_inside else "/tanh-outside")
                + (f" [{self.note}]" if self.note else ""))


# ── raw ring buffer → flat observation ────────────────────────────────────────
def pack_raw(kin, goal_buf, pose_buf, lidar_buf):
    """(5), (L,2), (L,3), (L,216) -> flat raw vector (carried through the rollout)."""
    return jnp.concatenate([kin, goal_buf.reshape(-1), pose_buf.reshape(-1), lidar_buf.reshape(-1)])


def unpack_raw(raw):
    kin = raw[..., :KIN]
    o = KIN
    goal = raw[..., o:o + RAW_LEN * GOAL].reshape(*raw.shape[:-1], RAW_LEN, GOAL); o += RAW_LEN * GOAL
    pose = raw[..., o:o + RAW_LEN * POSE].reshape(*raw.shape[:-1], RAW_LEN, POSE);  o += RAW_LEN * POSE
    lidar = raw[..., o:].reshape(*raw.shape[:-1], RAW_LEN, RAYS)
    return kin, goal, pose, lidar


def adapt(raw, layout, stride):
    """Single (un-batched) raw vector -> the flat obs a policy of `layout` expects.
    `stride` picks frames o_t, o_{t-stride}, o_{t-2 stride} from the ring buffer."""
    kin, goal, pose, lidar = unpack_raw(raw)
    sel = RAW_LEN - 1 - stride * jnp.arange(STACK_DIM - 1, -1, -1)       # oldest first
    if layout == "v667":        # current: goal_align (kin[4]) is not exposed
        return jnp.concatenate([kin[:4], goal[sel].reshape(-1),
                                _ego(pose[sel]).reshape(-1), lidar[sel].reshape(-1)])
    if layout == "v668":
        return jnp.concatenate([kin, goal[sel].reshape(-1),
                                _ego(pose[sel]).reshape(-1), lidar[sel].reshape(-1)])
    if layout == "v662":
        pz = jnp.concatenate([goal[sel], pose[sel][:, 2:3] / jnp.pi], axis=-1)   # (gdx, gdy, theta/pi)
        return jnp.concatenate([pz.reshape(-1), kin, lidar[sel].reshape(-1)])
    if layout == "v659":
        return jnp.concatenate([goal[sel].reshape(-1), kin, lidar[sel].reshape(-1)])
    if layout == "v672":
        raise Unsupported("v672 needs the pending-command buffer (real-robot env), not in the raw ring buffer")
    raise ValueError(layout)


def _ego(pose_sel):
    from legnav.core.jax_wrappers import _ego_deltas
    return _ego_deltas(pose_sel)


# ── detection from checkpoint parameters ──────────────────────────────────────
def _first_dense_in(params, key="Dense_0"):
    """Input dim of the first fused Dense: params[...]['Dense_0']['kernel'] at the shallowest level."""
    if isinstance(params, dict):
        if key in params and isinstance(params[key], dict) and "kernel" in params[key]:
            return int(params[key]["kernel"].shape[0])
        for v in params.values():
            r = _first_dense_in(v, key)
            if r is not None: return r
    return None


def _name_flags(ckpt_path):
    n = (ckpt_path or "").lower()
    return dict(se2=("noreproj" not in n), tanh_inside=("tanh_outside" not in n))


def detect_shared_encoder(params, ckpt_path="", stride=None):
    """PPO (EndToEndActorCritic) / SAC / TQC (SharedEncoder): fused-input 202 = current (v667), 203 = v668, 206 = pre-SE(2)."""
    dim = _first_dense_in(params)
    f = _name_flags(ckpt_path)
    if dim == 202:
        return ObsSpec("v667", CURRENT_STRIDE if stride is None else stride, False, f["se2"], f["tanh_inside"])
    if dim == 203:       # legacy: kin carried goal_align
        return ObsSpec("v668", CURRENT_STRIDE if stride is None else stride, False, f["se2"], f["tanh_inside"])
    if dim == 207:       # real-robot track: kin + pending commands
        return ObsSpec("v672", CURRENT_STRIDE if stride is None else stride, False, f["se2"], f["tanh_inside"],
                       note="real-robot obs (pending commands)")
    if dim == 206:
        return ObsSpec("v662", 1, True, False, f["tanh_inside"],
                       note="pre-SE(2) checkpoint, old pose layout")
    raise Unsupported(f"unknown fused input dim {dim} (pre-216-ray or foreign architecture)")


def detect_mlp(params, stride=None):
    """VanillaMLP: first Dense input equals the obs size it was trained on."""
    dim = _first_dense_in(params)
    for lay, size in LAYOUT_SIZE.items():
        if dim == size:
            return ObsSpec(lay, (CURRENT_STRIDE if stride is None else stride) if lay in ("v667", "v668") else 1)
    raise Unsupported(f"MLP input dim {dim} matches no known layout")


# ── current flat obs -> what a given (possibly legacy) checkpoint expects ─────
def to_v668(obs):
    """Current v667 flat obs (..., 667) -> legacy v668 (..., 668): re-insert goal_align/pi as
    kin[4]. It is atan2(g_y, g_x)/pi of the NEWEST ego goal, i.e. exactly the value the old
    simulator exposed, so legacy policies see the observations they were trained on.
    Identity on an obs that is already 668-D (current layout)."""
    if obs.shape[-1] == LAYOUT_SIZE["v668"]:
        return obs
    g = obs[..., 4:10][..., -2:]                       # newest ego goal (g_x, g_y)
    align = (jnp.arctan2(g[..., 1], g[..., 0]) / jnp.pi)[..., None]
    return jnp.concatenate([obs[..., :4], align, obs[..., 4:]], axis=-1)


def fit_obs(obs, params):
    """Adapt a v668/v667 flat obs to the checkpoint whose `params` are given (auto-detected from the
    first fused Dense input: 203 / 668 -> v668, 202 / 667 -> v667). Identity when they already match."""
    want = _first_dense_in(params)
    if want in (207, LAYOUT_SIZE["v672"]) or obs.shape[-1] == LAYOUT_SIZE["v672"]:
        if obs.shape[-1] != LAYOUT_SIZE["v672"] or want not in (207, LAYOUT_SIZE["v672"]):
            raise Unsupported(f"v672 policy/obs mismatch: obs {obs.shape[-1]}, policy fused input {want} "
                              "(set LEGNAV_REAL_ENV=1 LEGNAV_OBS_CMD=1 for v672 policies, unset for others)")
        return obs
    if want in (203, LAYOUT_SIZE["v668"]):
        return to_v668(obs)
    if want in (202, LAYOUT_SIZE["v667"]) and obs.shape[-1] == LAYOUT_SIZE["v668"]:
        return jnp.concatenate([obs[..., :4], obs[..., 5:]], axis=-1)     # drop goal_align
    return obs
