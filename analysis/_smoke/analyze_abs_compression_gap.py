"""Quantify the difference between raw16/m8/m4 chunks under abs action mode.

For dexjoco abs: m8 = block_last(raw, 2), m4 = block_last(raw[:8], 2).
We measure:
  (a) per-step jump magnitude (how far the robot must move per command) for
      raw16 vs m8 vs m4. Larger jump = controller must move further per step =
      bigger tracking error / instability.
  (b) consecutive-target gap distribution — how sparse the waypoints are.
"""
import glob, numpy as np, pandas as pd

PARQS = sorted(glob.glob("/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail/data/chunk-000/episode_*.parquet"))[:30]
H = 16

def block_last(arr, factor):
    T = arr.shape[-2]
    g = arr[..., :T - T % factor, :].reshape(*arr.shape[:-2], T // factor, factor, arr.shape[-1])
    return g[..., -1, :]

# Collect raw 16-step chunks from random offsets
chunks = []
for p in PARQS:
    df = pd.read_parquet(p)
    acts = np.stack(df["action"].values)   # (T, 22)
    T = acts.shape[0]
    if T < H + 5: continue
    for _ in range(20):
        start = np.random.randint(0, T - H)
        chunks.append(acts[start:start + H])
chunks = np.stack(chunks)   # (N, 16, 22)
print(f"collected {chunks.shape} chunks from dexjoco hammer_nail (abs actions)")

raw16 = chunks                                  # (N, 16, 22)
m8    = block_last(chunks[:, :H], 2)            # (N, 8, 22)
m4    = block_last(chunks[:, :H // 2], 2)       # (N, 4, 22)  block-last from first 8

# Per-step jump = ||a[t] - a[t-1]|| for arm_pos dims (cols 0..3)
def per_step_jump(arr):
    diffs = np.diff(arr[:, :, :3], axis=1)            # (N, T-1, 3)
    jumps = np.linalg.norm(diffs, axis=-1)            # (N, T-1)
    return jumps.mean(), jumps.std(), np.percentile(jumps, 90)

def per_step_jump_hand(arr):
    diffs = np.diff(arr[:, :, 6:22], axis=1)           # (N, T-1, 16) — hand
    jumps = np.linalg.norm(diffs, axis=-1)
    return jumps.mean(), jumps.std(), np.percentile(jumps, 90)

print()
print("=== per-step waypoint gap (||a_t - a_{t-1}||) ===")
print(f"{'expert':<8} {'#steps':<7} | arm_pos: mean / std / p90 | hand: mean / std / p90")
for name, arr, steps in [("raw16", raw16, 15), ("m8", m8, 7), ("m4", m4, 3)]:
    pm, ps, p90 = per_step_jump(arr)
    hm, hs, h90 = per_step_jump_hand(arr)
    print(f"  {name:<6} {steps:<7} | {pm:.4f}  {ps:.4f}  {p90:.4f}  | {hm:.4f}  {hs:.4f}  {h90:.4f}")

print()
print("=== ratio m8/raw16, m4/raw16 (arm_pos) ===")
raw_m, _, _ = per_step_jump(raw16)
m8_m,  _, _ = per_step_jump(m8)
m4_m,  _, _ = per_step_jump(m4)
print(f"  m8/raw16 = {m8_m/raw_m:.2f}x  (expected ~2x if linearly subsampled)")
print(f"  m4/raw16 = {m4_m/raw_m:.2f}x  (expected ~4x or ~2x of m8)")

# Trajectory total length
print()
print("=== total chunk path length (sum of waypoint jumps, arm_pos only) ===")
for name, arr in [("raw16", raw16), ("m8", m8), ("m4", m4)]:
    diffs = np.diff(arr[:, :, :3], axis=1)
    total = np.linalg.norm(diffs, axis=-1).sum(axis=-1)   # (N,)
    print(f"  {name:<6} total path = {total.mean():.4f} +/- {total.std():.4f}")
print("ABS_GAP_DONE")
