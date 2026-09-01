"""Measure the deterministic constants for THIS dataset (no VLM involved).

v5's limits came from a slower allex recording; reused unchanged they fire on a
third to a half of all chunks here purely because this data moves faster.  This
script prints the numbers that allex_v2_common hard-codes, with their provenance.

  python allex_v2_calibrate.py [n_episodes_stride]
"""
import sys, os, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import descriptors

# Which recording to calibrate against is the whole point of this script, so it
# is an argument. v5's constants came from a different, slower capture and fired
# on a third of chunks here purely because this data moves faster; the same trap
# is waiting for every new recording.
DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CHUNK = 16

step, ds = [], []
# episode count comes from the recording, not a constant: the replay set has 10.
_n_ep = json.load(open(f"{DS}/meta/info.json"))["total_episodes"]
eps = list(range(0, _n_ep, STRIDE))
for ep in eps:
    d = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    step.append(np.linalg.norm(np.diff(A, axis=0), axis=1))
    ds += [descriptors(A, WR, WL, f, CHUNK) for f in range(0, len(A) - CHUNK, CHUNK)]
step = np.concatenate(step)

wr = np.array([x["wrist_rot"] for x in ds])
accum = np.array([wr[max(0, i - 2):i + 1].sum() for i in range(len(ds))])
held = np.array([x["held"] for x in ds], bool)
gr = np.array([x["gap_rate"] for x in ds])

print(f"episodes {eps}")
print(f"steps {len(step)}, chunks {len(ds)}, held (two palms) {held.mean()*100:.1f}%")
print("single-step |dA| rad      p99/p99.5/p99.9/max  "
      f"{np.percentile(step,[99,99.5,99.9]).round(3)} {step.max():.3f}   -> MERGE_LIMIT_V2")
print("rot accum (3 chunks) deg, held only   p50/p75/p90/p95  "
      f"{np.percentile(accum[held],[50,75,90,95]).round(1)}   -> ROT_ACCUM_LIMIT_V2")
print("gap_rate m/step, held only            p50/p75/p90/p95  "
      f"{np.percentile(gr[held],[50,75,90,95]).round(4)}   -> GAP_RATE_LIMIT_V2")
print("merge demand K2 p50/p90/p99  " + str(np.percentile([x["merge_demand_k2"] for x in ds],[50,90,99]).round(3)))
print("hand_change   p50/p90/p99  " + str(np.percentile([x["hand_change"] for x in ds],[50,90,99]).round(4)))
