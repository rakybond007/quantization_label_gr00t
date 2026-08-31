"""How much can one labelcheck number carry?

`qgate labelcheck` reduces a label set to a single Spearman against measured
K=2 damage, and that number now gates whether a labelling is trained on. It
rejected phase6 and it will accept or reject the next one, so it is worth
knowing what its error bars are and where its blind spots lie.

Three things this prints:

1. A bootstrap interval for each rho. n is 24 tasks for RoboCasa and 40 for
   LIBERO, which is small enough that the interval matters.
2. The tie fraction in the damage axis. Damage is a success-rate difference
   over 50 episodes, so it is quantised to 0.02 and many tasks land on exactly
   zero. Spearman over a heavily tied axis is carried by the few untied points.
3. phase5 against phase6 paired on the same 24 tasks, which is the comparison
   the accept/reject decision actually rests on.
"""
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "vlm_gate")
from qgate import labelcheck  # noqa: E402

DSR = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
       "robocasa_mg_gr00t_300")
DSL = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
       "libero_gr00t_delta")

CASES = {
    "phase5": ("assets/labels/robocasa/v6b_phase5_softA.parquet", "robocasa", DSR,
               "baseline_full_v2_with_action_steps", "baseline_compress_K2"),
    "phase6": ("assets/labels/robocasa/v6b_phase6_softA.parquet", "robocasa", DSR,
               "baseline_full_v2_with_action_steps", "baseline_compress_K2"),
    "libero": ("assets/labels/libero/libero_dense_v1.parquet", "libero", DSL,
               "baseline_raw", "baseline_K2"),
}


def boot(d, c, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d, c = np.asarray(d), np.asarray(c)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d))
        if len(set(d[i])) < 3:
            continue
        out.append(spearmanr(d[i], c[i]).statistic)
    return np.percentile(out, [2.5, 97.5])


got = {}
for name, args in CASES.items():
    r = labelcheck.score(*args)
    got[name] = r["rows"]
    d = [x["delta_k2"] for x in r["rows"]]
    c = [x["confidence"] for x in r["rows"]]
    lo, hi = boot(d, c)
    _, cnt = np.unique(d, return_counts=True)
    print(f"{name:8s} n={len(d):3d}  rho={r['spearman']:+.3f}   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]   "
          f"largest tied damage group {cnt.max() / len(d):.0%}   "
          f"distinct damage values {len(cnt)}")

# The accept/reject decision compares two label sets over the same tasks, so
# resampling tasks jointly is the honest test of that difference.
a = {x["task"]: x for x in got["phase5"]}
b = {x["task"]: x for x in got["phase6"]}
tasks = sorted(set(a) & set(b))
d = np.array([a[t]["delta_k2"] for t in tasks])
c5 = np.array([a[t]["confidence"] for t in tasks])
c6 = np.array([b[t]["confidence"] for t in tasks])

rng = np.random.default_rng(1)
diff = []
for _ in range(4000):
    i = rng.integers(0, len(tasks), len(tasks))
    if len(set(d[i])) < 3:
        continue
    diff.append(spearmanr(d[i], c5[i]).statistic - spearmanr(d[i], c6[i]).statistic)
diff = np.array(diff)
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"\nphase5 - phase6, paired over the same {len(tasks)} tasks: "
      f"{diff.mean():+.3f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  resamples where phase5 scores higher: {(diff > 0).mean():.1%}")
