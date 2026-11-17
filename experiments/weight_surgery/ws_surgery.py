"""Weight-space programming primitives for the CALF actor-critic.

Trunk:  fused(203)=[attn(192) | kin(5)=v/vmax, w_prev, vmax_n, gdist_n, galign/pi | goal_stack(6)]
        h0=relu(fused·W0+b0) (256) ; h1=relu(h0·W1+b1) (128) ; raw=h1·K+b ; v=(tanh(raw0)/2+.5)·vmax ; w=tanh(raw1)
"""
import numpy as np, ws_lib as L

KIN0 = 192          # index of kin_vec[0] inside the fused trunk input
G_NAMES = ["v_prev", "w_prev", "vmax_n", "gdist_n", "galign", "g4x", "g4y", "g2x", "g2y", "g0x", "g0y"]
G_BOUND = np.array([1.0, 1.0, 1.0, 1.5, 1.0] + [1.5] * 6, np.float32)   # |g_T[i]| bounds


def head_calibrate(p, gv=1.0, bv=0.0, gw=1.0, bw=0.0):
    p = L.copy_params(p); K, b = p["Dense_2"]["kernel"], p["Dense_2"]["bias"]
    K[:, 0] *= gv; b[0] = b[0] * gv + bv; K[:, 1] *= gw; b[1] = b[1] * gw + bw
    return p


def dead_units(p, obs):
    """Units of Dense_0 / Dense_1 that never fire on the given observation set."""
    F = L.policy_feats(p, obs)
    return np.where((F["d0"] > 0).mean(0) == 0)[0], np.where((F["d1"] > 0).mean(0) == 0)[0]


def implant_skip(p, A, dead0, dead1):
    """Implant an exact linear skip connection   raw += A · g_T   (A: 2×11) by re-purposing
    two dead ReLU units in Dense_0 and two in Dense_1 as always-on 'carrier' channels.

    channel c (c=0→raw_v, c=1→raw_w):
        h0[j_c] = relu( A[c]·g_T + off_c )          off_c > max|A[c]·g_T|  → never clips → linear
        h1[k_c] = relu( h0[j_c] )                   = A[c]·g_T + off_c
        raw[c] += h1[k_c] − off_c
    Everything else that touched those units is zeroed, so on states where they were dead the
    original function is preserved exactly and the only change is the added skip term."""
    p = L.copy_params(p)
    W0, b0 = p["Dense_0"]["kernel"], p["Dense_0"]["bias"]
    W1, b1 = p["Dense_1"]["kernel"], p["Dense_1"]["bias"]
    K, b = p["Dense_2"]["kernel"], p["Dense_2"]["bias"]
    W3 = p["Dense_3"]["kernel"]
    A = np.asarray(A, np.float32)
    assert len(dead0) >= 2 and len(dead1) >= 2
    for c in range(2):
        if not np.any(A[c]):
            continue
        j, k = int(dead0[c]), int(dead1[c])
        off = float(np.abs(A[c]) @ G_BOUND) + 1.0
        W0[:, j] = 0.0; W0[KIN0:KIN0 + 11, j] = A[c]; b0[j] = off
        W1[j, :] = 0.0; W1[:, k] = 0.0; W1[j, k] = 1.0; b1[k] = 0.0
        K[k, :] = 0.0; K[k, c] = 1.0; b[c] -= off
        W3[k, :] = 0.0                      # keep the critic unaffected by the carrier
    return p


def lowpass_w(p, alpha, dead0, dead1):
    """Steering EMA filter in weights:  raw_w ← (1-α)·raw_w + α·w_prev."""
    q = head_calibrate(p, gw=1.0 - alpha)
    A = np.zeros((2, 11), np.float32); A[1, 1] = alpha
    return implant_skip(q, A, dead0, dead1)


def build(p0, dead0, dead1, gv=1.0, bv=0.0, gw=1.0, bw=0.0, alpha=0.0, skip=None):
    """Composite edit:  raw_v ← gv·raw_v + bv + Σ skip['v'][name]·g_T[name]
                        raw_w ← (1-α)·gw·raw_w + bw + α·w_prev + Σ skip['w'][name]·g_T[name]"""
    q = head_calibrate(p0, gv=gv, bv=bv, gw=gw * (1.0 - alpha), bw=bw)
    A = np.zeros((2, 11), np.float32); A[1, 1] += alpha
    for c, ch in enumerate("vw"):
        for n, val in ((skip or {}).get(ch, {})).items():
            A[c, G_NAMES.index(n)] += val
    return implant_skip(q, A, dead0, dead1) if np.any(A) else q
