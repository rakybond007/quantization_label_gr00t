"""Independent recheck of the phase6 rejection.

phase6 was rejected on one number: Spearman between per-task mean confidence and
the measured cost of blanket K=2, +0.019 against phase5's +0.420. That number
answers one question well and two others not at all, so this checks all three
before the rejection is treated as settled.

1. Is it robust to the damage reference? The Spearman is computed against one
   pair of evaluation runs. If phase6 tracks damage measured against a different
   compression baseline, the rejection is an artefact of that choice.

2. Does it discriminate per chunk? labelcheck aggregates to per-task means,
   which is the wrong granularity for a gate that decides per chunk. A label set
   can rank tasks badly and still separate the risky chunks inside each one —
   and that is the job. Ground truth here is the gripper command changing within
   the window, the same event score_probe uses.

3. Is the chunk-level signal merely the task-level signal? Subtracting each
   task's mean removes everything labelcheck measures. What survives is pure
   within-task discrimination.
"""
import json
import sys

import numpy as np
import pandas as pd

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
LAB = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/assets/labels/robocasa"
COL = "p_yes_soft"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CHUNK = 16

info = json.load(open(f"{DS}/meta/info.json"))
_cache = {}


def actions(ep):
    if ep not in _cache:
        ch = ep // info["chunks_size"]
        try:
            _cache[ep] = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            _cache[ep] = None
    return _cache[ep]


def auc_ci(y, s, B=2000, seed=0):
    y = np.asarray(y, float); s = np.asarray(s, float)

    def a(y, s):
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 * n0 else np.nan
    base = a(y, s)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            bs.append(a(y[i], s[i]))
    return base, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


gens = {}
for tag in ("phase5", "phase6"):
    d = pd.read_parquet(f"{LAB}/v6b_{tag}_softA.parquet",
                        columns=["episode_index", "frame_index", "task", COL])
    gens[tag] = d.rename(columns={"episode_index": "ep", "frame_index": "f", COL: "p"})

eps = sorted(set(gens["phase5"].ep.unique()) & set(gens["phase6"].ep.unique()))
rng = np.random.default_rng(0)
eps = sorted(rng.choice(eps, size=min(N_EP, len(eps)), replace=False))
print(f"sampled {len(eps)} episodes\n")

# ground truth: does the gripper command change inside the window
lab = {}
for ep in eps:
    A = actions(ep)
    if A is None:
        continue
    g = np.abs(np.diff(A[:, -1], prepend=A[0, -1]))
    lab[ep] = g

print("2. per-chunk discrimination — does low confidence predict a gripper transition")
print(f"   {'gen':8s} {'n':>7s} {'pos':>6s} {'AUC':>7s}   95% CI")
keep = {}
for tag, d in gens.items():
    d = d[d.ep.isin(lab)]
    y, s, t = [], [], []
    for ep, f, p, task in zip(d.ep, d.f, d.p, d.task):
        g = lab[ep]
        if f >= len(g):
            continue
        y.append(int(g[f:f + CHUNK].max() > 0.5)); s.append(1 - p); t.append(task)
    y = np.array(y); s = np.array(s)
    keep[tag] = (y, s, np.array(t))
    A, lo, hi = auc_ci(y, s)
    print(f"   {tag:8s} {len(y):7d} {int(y.sum()):6d} {A:7.3f}   [{lo:.3f}, {hi:.3f}]")

print("\n3. within task only — each task's mean confidence removed first")
print(f"   {'gen':8s} {'n':>7s} {'AUC':>7s}   95% CI")
for tag, (y, s, t) in keep.items():
    df = pd.DataFrame({"y": y, "s": s, "t": t})
    df["s"] = df.s - df.groupby("t").s.transform("mean")
    A, lo, hi = auc_ci(df.y.values, df.s.values)
    print(f"   {tag:8s} {len(df):7d} {A:7.3f}   [{lo:.3f}, {hi:.3f}]")
