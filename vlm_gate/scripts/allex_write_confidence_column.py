"""Write our labelling into the dataset's parquet as a per-frame column.

Each labeller publishes a column named after themselves, alongside david's:
float32, one value per frame, in [0,1] where 0 is original speed and 1 is the
maximum, with no missing values. This matches that contract exactly.

Two conversions are needed and both are stated here rather than buried.

Our unit is a 16-step CHUNK; the column is per FRAME. A chunk's value is held
across the frames it covers. Frames past the last labelled chunk — the tail of
an episode, shorter than one chunk — take the last chunk's value rather than a
default, because the alternative is an invented number at exactly the moment the
robot is finishing the task.

Our value is a ratio K on the grid {1, 2, 2.5, 3}; the column is normalised.
`hojin = (K - 1) / (3 - 1)`, so K=1 gives 0 and K=3 gives 1, matching david's
clip((alpha-1)/(alpha_max-1)). The raw K stays in the meta JSON for anyone who
needs the grid the robot actually executes.

A ramp is applied across chunk boundaries for the same reason david ramps across
segment boundaries: the underlying decision is piecewise constant, and stepping
between levels within one control tick is not something the robot can follow.

    python allex_write_confidence_column.py <records.jsonl> <dataset-dir> [column] [--apply]

Without --apply it reports what it would write and touches nothing.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REC, DSD = sys.argv[1], sys.argv[2]
rest = [a for a in sys.argv[3:] if not a.startswith("--")]
COL = rest[0] if rest else "hojin"
APPLY = "--apply" in sys.argv
CHUNK, GRID_MAX = 16, 3.0
RAMP = 9                     # frames, matching the ramp david uses (0.3 s at 30 fps)

by_ep = {}
for line in open(REC):
    try:
        r = json.loads(line)
    except Exception:
        continue
    by_ep.setdefault(r["ep"], {})[r["f"]] = float(r.get("K", 1.0))
print(f"{sum(len(v) for v in by_ep.values())} chunks over {len(by_ep)} episodes")

files = sorted(glob.glob(os.path.join(DSD, "data", "*", "*.parquet")))
if not files:
    raise SystemExit(f"no parquet under {DSD}/data")

tot = miss = 0
vals = []
for p in files:
    ep = int(os.path.basename(p).split("episode_")[1].split(".")[0])
    d = pd.read_parquet(p)
    n = len(d)
    ch = by_ep.get(ep)
    if not ch:
        miss += 1
        continue

    # step function over frames, held across each chunk, last chunk covers the tail
    v = np.full(n, np.nan, dtype=np.float64)
    for f, K in ch.items():
        v[f:min(f + CHUNK, n)] = (K - 1.0) / (GRID_MAX - 1.0)
    last = max(ch)
    v[np.isnan(v)] = (ch[last] - 1.0) / (GRID_MAX - 1.0)

    # linear ramp across level changes, so the column is followable
    if RAMP > 1:
        k = np.ones(RAMP) / RAMP
        v = np.convolve(np.pad(v, (RAMP // 2, RAMP // 2), mode="edge"), k, mode="valid")[:n]
    v = np.clip(v, 0.0, 1.0).astype(np.float32)

    vals.append(v)
    tot += n
    if APPLY:
        d[COL] = v
        d.to_parquet(p, index=False)

allv = np.concatenate(vals) if vals else np.array([0.0])
print(f"{'wrote' if APPLY else 'would write'} column {COL!r} to {len(files) - miss} parquet files")
print(f"  {tot:,} frames   range {allv.min():.4f}–{allv.max():.4f}   mean {allv.mean():.4f}")
print(f"  dtype float32, no NaN: {not np.isnan(allv).any()}")
if miss:
    print(f"  {miss} episodes had no labels and were left untouched")
if not APPLY:
    print("\ndry run — pass --apply to write")
