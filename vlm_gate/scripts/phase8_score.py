"""Score the single graded text question against phase6's five checks.

Same chunks, same computed flags, same two targets as phase7_score.py, so the
only thing that differs is the VLM half. No logits are read on the phase8 side:
`pick` is the digit the model wrote.

The level is an ordering of how thinnable the moment is, so it enters as a
compressibility directly -- (pick-1)/(L-1) -- rather than as a risk to be
noisy-OR'd. The computed flags stay exactly where they were.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = "/sjw_alinlab/home/hojin2/quantization_agent_workspace"
P8 = f"{WS}/vlm_gate/output/_gate_distill/phase8_single/single.jsonl"
P7 = f"{WS}/vlm_gate/output/_gate_distill/phase7_pilot"
PARQ = f"{WS}/assets/labels/robocasa/v6b_phase6_softA.parquet"
SOFT = ["c_grip_transition_soft", "c_reversal_soft", "c_precise_hold_soft"]
LEVELS = 4


def auc_ci(y, s, n=2000, seed=0):
    def a(y, s):
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
        p, q = y == 1, y == 0
        if p.sum() == 0 or q.sum() == 0:
            return float("nan")
        return (r[p].mean() - r[q].mean()) / len(s) + 0.5
    rng = np.random.default_rng(seed)
    bs = [a(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return a(y, s), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


rows = [json.loads(l) for l in open(P8)]
d8 = pd.DataFrame([r for r in rows if r["pick"] is not None])
d8 = d8.rename(columns={"ep": "episode_index", "f": "frame_index"})
par = pd.read_parquet(PARQ, columns=["episode_index", "frame_index", "task",
                                     "c_grip_transition"] + SOFT)
d = d8.merge(par, on=["episode_index", "frame_index"])

import sys  # noqa: E402
sys.path.insert(0, f"{WS}/vlm_gate")
from qgate import labelcheck  # noqa: E402
DSR = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
       "robocasa_mg_gr00t_300")
dmg = labelcheck.damage("robocasa", "baseline_full_v2_with_action_steps",
                        "baseline_compress_K2")
cls = labelcheck.task_classes(DSR)

print(f"{len(d):,} chunks joined  (파싱실패 {sum(1 for r in rows if r['pick'] is None)})")
print("등급 분포:", dict(sorted(pd.Series(d['pick']).value_counts().items())))

S = d[SOFT].values.astype(float)
comp = (d["pick"].values.astype(float) - 1) / (LEVELS - 1)   # 0=압축불가, 1=자유
risk = 1 - np.prod(1 - S, axis=1)
raw = (1 - risk) * comp
p = np.argsort(np.argsort(raw)) / (len(raw) - 1)

dd = d.assign(p=p, cls=d.episode_index.map(cls))
m = dd.dropna(subset=["cls"]).groupby("cls").p.mean()
t = sorted(set(m.index) & set(dmg))
rho = spearmanr([dmg[x] for x in t], [m[x] for x in t]).statistic
y = (d.c_grip_transition.values > 0.5).astype(int)
A, lo, hi = auc_ci(y, 1 - p)
print(f"\n{'arm':22s} {'unique':>7s} {'per-task rho':>13s} {'per-chunk AUC':>15s}")
print(f"{'phase8-single-text':22s} {len(np.unique(raw)):7d} {rho:+13.3f}   {A:.3f} [{lo:.3f}, {hi:.3f}]")
print(f"\nper-task rho over {len(t)} tasks; per-chunk positives = {int(y.sum())}")
