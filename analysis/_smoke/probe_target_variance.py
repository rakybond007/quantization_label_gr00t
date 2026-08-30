"""Per-expert TARGET variance for dexjoco v1 K=4 abs-mode layout, sampled
from train data. Higher variance → harder to predict → router-supervision
disadvantage. Tests our hypothesis that m4 (sum-pair from first 8) has lower
variance than raw (full 16) and that's why router collapsed to m4."""
import glob
import numpy as np
import pandas as pd

PARQUETS = sorted(glob.glob("/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail/data/chunk-*/episode_*.parquet"))
print(f"found {len(PARQUETS)} episode parquets for hammer_nail")

H = 16
A_POS = 3   # arm_pos dims
A_ROT = 4   # arm_rot is quaternion in dataset (4-dim, NOT rotvec); we'll only use pos for variance
A_HAND = 16
N_SAMPLES_PER_EP = 5   # sample 5 random windows per episode

chunks = []   # collect 16-step action windows from random offsets
for p in PARQUETS:
    df = pd.read_parquet(p)
    actions = np.stack(df["action"].values)   # (T, A=22) — dexjoco abs action format
    T = actions.shape[0]
    if T < H:
        continue
    for _ in range(N_SAMPLES_PER_EP):
        start = np.random.randint(0, T - H + 1)
        chunks.append(actions[start:start + H])

chunks = np.stack(chunks)   # (N, 16, 22)
print(f"collected {chunks.shape} action windows")

def block_last(arr, factor):
    T = arr.shape[-2]
    g = arr[..., :T - T % factor, :].reshape(*arr.shape[:-2], T // factor, factor, arr.shape[-1])
    return g[..., -1, :]

# v1 K=4 abs layout
raw_t = chunks                                # (N, 16, 22)  raw 16-step
m8_t  = block_last(chunks[:, :H], 2)          # (N,  8, 22)  block-last from full 16
m4_t  = block_last(chunks[:, :H//2], 2)       # (N,  4, 22)  block-last from FIRST 8
n8_t  = chunks[:, :H//2]                      # (N,  8, 22)  raw first 8

def summarize(arr, label):
    # per-element variance across N samples → average over (T, A) for a scalar
    var_per_elem = arr.var(axis=0)         # (T, A)
    mean_v = var_per_elem.mean()
    pos_v  = var_per_elem[:, :3].mean()                # arm_pos
    rot_v  = var_per_elem[:, 3:7].mean() if arr.shape[-1] >= 7 else 0.0  # quat
    hnd_v  = var_per_elem[:, 7:].mean()                # hand
    print(f"  {label:<6}  shape={arr.shape}  mean_var={mean_v:.5f}   pos={pos_v:.5f}  rot={rot_v:.5f}  hand={hnd_v:.5f}")
    return mean_v

print(f"\n=== per-element variance across {len(chunks)} samples ===")
v_raw = summarize(raw_t, "raw16")
v_m8  = summarize(m8_t,  "m8")
v_m4  = summarize(m4_t,  "m4")
v_n8  = summarize(n8_t,  "n8")

print(f"\n=== variance ratios (smaller = router favors this expert) ===")
print(f"  m4/raw   = {v_m4/v_raw:.3f}")
print(f"  n8/raw   = {v_n8/v_raw:.3f}")
print(f"  m8/raw   = {v_m8/v_raw:.3f}")
print(f"  m4/m8    = {v_m4/v_m8:.3f}")

# Per-step variance trace (arm_pos only) — show start vs end
print(f"\n=== per-step variance of arm_pos (raw chunk) — start vs end ===")
ps = raw_t[:, :, :3].var(axis=0).mean(axis=-1)   # (16,)
for t in range(H):
    print(f"  step {t:2d}: var(arm_pos)={ps[t]:.5f}")
print("\nVAR_PROBE_DONE")
