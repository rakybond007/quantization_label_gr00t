"""Score the phase7 pilot: does a graded answer beat YES/NO?

Three readings of the same 2,000 chunks, off the same image prefill, with
phase6's questions verbatim — only the answer scale differs:

  binary    P(YES) over the {YES,NO} tokens
  expected  probability-weighted mean over five grade tokens
  pick      the grade the model actually picks, (g-1)/4 — the pure text answer

The computed flags are taken from the existing parquet so the aggregation is
identical across arms and only the VLM half varies. Scored two ways, because
this project has demonstrated that they disagree: per task against measured K=2
damage, and per chunk against whether the gripper transitions in the window.

The per-chunk target is weak on its own — the facts text tells the model about
the gripper — but it is told to all three arms equally, so the comparison
between them holds even where the absolute level does not.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = "/sjw_alinlab/home/hojin2/quantization_agent_workspace"
PILOT = f"{WS}/vlm_gate/output/_gate_distill/phase7_pilot"
PARQ = f"{WS}/assets/labels/robocasa/v6b_phase6_softA.parquet"
SOFT = ["c_grip_transition_soft", "c_reversal_soft", "c_precise_hold_soft"]


def load(name, cols):
    rows = []
    for line in open(f"{PILOT}/{name}.jsonl"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    d = pd.DataFrame(rows)
    return d.rename(columns={"ep": "episode_index", "f": "frame_index"})[
        ["episode_index", "frame_index"] + cols]


def auc_ci(y, s, B=2000, seed=0):
    y, s = np.asarray(y, float), np.asarray(s, float)

    def a(y, s):
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 * n0 else np.nan
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            bs.append(a(y[i], s[i]))
    return a(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


b = load("binary", list("ABCDE"))
g = load("graded", list("ABCDE") + [f"{q}_pick" for q in "ABCDE"])
par = pd.read_parquet(PARQ, columns=["episode_index", "frame_index", "task",
                                     "c_grip_transition"] + SOFT)

arms = {
    "binary": b.merge(par, on=["episode_index", "frame_index"]),
    "graded-expected": g.merge(par, on=["episode_index", "frame_index"]),
    "graded-pick": g.rename(columns={f"{q}_pick": q for q in "ABCDE"}
                            ).merge(par, on=["episode_index", "frame_index"]),
}
print(f"{len(arms['binary']):,} chunks joined\n")

# per-task damage, the labelcheck axis
dmg = {}
import sys  # noqa: E402
sys.path.insert(0, f"{WS}/vlm_gate")
from qgate import labelcheck  # noqa: E402
DSR = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
       "robocasa_mg_gr00t_300")
dmg = labelcheck.damage("robocasa", "baseline_full_v2_with_action_steps",
                        "baseline_compress_K2")
cls = labelcheck.task_classes(DSR)

print(f"{'arm':18s} {'unique p':>9s} {'per-task rho':>13s} {'per-chunk AUC':>15s}")
for name, d in arms.items():
    V = d[list("ABCDE")].values.astype(float)
    S = d[SOFT].values.astype(float)
    risk = 1 - np.prod(1 - np.column_stack([S, V[:, 1:]]), axis=1)
    raw = (1 - risk) * (0.5 + 0.5 * V[:, 0])
    p = np.argsort(np.argsort(raw)) / (len(raw) - 1)

    d = d.assign(p=p, cls=d.episode_index.map(cls))
    m = d.dropna(subset=["cls"]).groupby("cls").p.mean()
    t = sorted(set(m.index) & set(dmg))
    rho = spearmanr([dmg[x] for x in t], [m[x] for x in t]).statistic

    y = (d.c_grip_transition.values > 0.5).astype(int)
    A, lo, hi = auc_ci(y, 1 - p)
    print(f"{name:18s} {len(np.unique(raw)):9d} {rho:+13.3f} "
          f"  {A:.3f} [{lo:.3f}, {hi:.3f}]")

print(f"\nper-task rho is over {len(t)} tasks; per-chunk positives = {int(y.sum())}")
