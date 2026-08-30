"""Analyze per-step entropy from per-step calibration smoke + AAC h* distribution.

Loads `all_step_entropies` from prediction.txt — list of [16] vectors per chunk.
Computes:
  1. Mean per-step entropy across chunks → does it increase/decrease with i?
  2. AAC h* (paper's algorithm at full step granularity) per chunk → distribution
"""
import os, json, sys, glob
import numpy as np

DEFAULT = "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/robocasa/_smoke_selective/_perstep_check/CoffeeSetupMug/prediction.txt"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

step_entropies = None
for line in open(path):
    if line.startswith("all_step_entropies:"):
        step_entropies = json.loads(line.split(":", 1)[1].strip())
        break

if step_entropies is None:
    print(f"No 'all_step_entropies' line found in {path}")
    sys.exit(1)

A = np.array(step_entropies)        # (n_chunks, 16)
print(f"Loaded {A.shape[0]} chunks × {A.shape[1]} steps")

# 1. Per-step mean
print("\n=== Per-step entropy E_i (mean across chunks) ===")
print(f"  {'i':>3s}  {'mean':>10s}  {'std':>10s}  {'p25':>10s}  {'p50':>10s}  {'p75':>10s}")
for i in range(A.shape[1]):
    col = A[:, i]
    print(f"  {i:>3d}  {col.mean():>10.4f}  {col.std():>10.4f}  "
          f"{np.quantile(col,0.25):>10.4f}  {np.quantile(col,0.50):>10.4f}  {np.quantile(col,0.75):>10.4f}")

mean_curve = A.mean(axis=0)
print(f"\n  Trend: E_0={mean_curve[0]:.2f}, E_15={mean_curve[-1]:.2f}, "
      f"diff(end-start)={mean_curve[-1]-mean_curve[0]:+.2f}")

# 2. AAC h* per chunk
def aac_h_star(s, xi=1):
    s = np.asarray(s, dtype=np.float64)
    H = len(s)
    avgs = np.cumsum(s) / np.arange(1, H + 1)
    diffs = avgs[1:] - avgs[:-1]
    h_star = int(np.argmax(diffs)) + 1
    return max(h_star, xi)

h_stars = np.array([aac_h_star(row, xi=1) for row in A])
print(f"\n=== AAC h* distribution (per-chunk, xi=1, H=16 step level) ===")
print(f"  unique values & counts:")
vals, counts = np.unique(h_stars, return_counts=True)
for v, c in zip(vals, counts):
    print(f"    h*={v:>2d}  count={c:>4d}  ({100.0*c/len(h_stars):.1f}%)")
print(f"  mean h* = {h_stars.mean():.2f}, std = {h_stars.std():.2f}")
print(f"  → mean #pairs to compress (h*/2 floor) = {(h_stars // 2).mean():.2f}")

# 3. AAC h* on per-PAIR scores (current implementation)
print(f"\n=== AAC h* on per-PAIR scores (current impl, H=8 pair level) ===")
pair_E = np.zeros((A.shape[0], 8))
for k in range(8):
    pair_E[:, k] = A[:, 2*k] + A[:, 2*k+1]
h_stars_pair = np.array([aac_h_star(row, xi=1) for row in pair_E])
vals, counts = np.unique(h_stars_pair, return_counts=True)
for v, c in zip(vals, counts):
    print(f"    h*={v:>2d}  count={c:>4d}  ({100.0*c/len(h_stars_pair):.1f}%)")
print(f"  mean h* = {h_stars_pair.mean():.2f}, std = {h_stars_pair.std():.2f}")
