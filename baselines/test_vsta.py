"""Check VSTA against real trajectories, not synthetic ones.

Run:  python baselines/test_vsta.py
"""
import glob, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vsta import retime, merge_integer, resample_error

# (name, glob, delta dims, discrete dims) -- offsets taken from each dataset's modality.json
SETS = [
    ("robocasa24", "/s3data/robocasa_mg_gr00t_300/v1/data/chunk-000/episode_0000*.parquet",
     [5, 6, 7, 8, 9, 10, 0, 1, 2, 3], [4, 11]),
    ("rc365", "/s3data/rc365-target16/v1/DeliverStraw/*/lerobot/data/chunk-000/episode_00000*.parquet",
     [5, 6, 7, 8, 9, 10, 0, 1, 2, 3], [4, 11]),
    ("libero", "/ckpt/%s/datasets/libero_gr00t_delta/data/chunk-000/episode_00000*.parquet" % __import__("os").environ["USER"],
     [0, 1, 2, 3, 4, 5], [6]),
]

def load(pat, n=5):
    out = []
    for f in sorted(glob.glob(pat))[:n]:
        t = pq.read_table(f)
        col = "action" if "action" in t.column_names else "actions"
        out.append(np.stack(t.column(col).to_numpy()))
    return out

fail = 0
for name, pat, dd, cd in SETS:
    eps = load(pat)
    if not eps:
        print(f"[{name}] no parquet found at {pat} -- skipped")
        continue
    print(f"\n=== {name}: {len(eps)} episodes, action dim {eps[0].shape[1]} ===")

    # 1. integer merge and general retime must agree -- they are the same operation
    a = eps[0]
    for k in (2, 3, 4):
        m = merge_integer(a, k, dd, cd)
        r = retime(a, float(k), dd, cd)
        n = min(len(m), len(r))
        d = np.abs(m[:n][:, dd] - r[:n][:, dd]).max()
        ok = d < 1e-9
        fail += not ok
        print(f"  K={k}: merge_integer vs retime  max|diff|={d:.2e}  {'OK' if ok else 'MISMATCH'}")

    # 2. endpoint displacement must survive re-timing, at every speed
    for s in (0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
        errs = [resample_error(e, s, dd, cd) for e in eps]
        mx = max(x["max_rel_err"] for x in errs)
        steps = f"{errs[0]['steps_in']}->{errs[0]['steps_out']}"
        ok = mx < 1e-6
        fail += not ok
        print(f"  s={s:<5} {steps:>12}  max endpoint rel err={mx:.2e}  {'OK' if ok else 'LOSS'}")

    # 3. what the gripper does under compression -- the dim the paper handles specially
    g = cd[-1]
    orig_tr = int((np.diff(np.sign(a[:, g])) != 0).sum())
    for k in (2, 4):
        mm = merge_integer(a, k, dd, cd)
        print(f"  gripper transitions: original {orig_tr}, after K={k} merge {int((np.diff(np.sign(mm[:, g])) != 0).sum())}")

print(f"\n{'ALL CHECKS PASSED' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
