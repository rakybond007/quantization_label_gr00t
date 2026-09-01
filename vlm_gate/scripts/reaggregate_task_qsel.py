"""Re-aggregate with, per task, the questions that cannot order its chunks left out.

A question earns its place in a task's confidence by telling one chunk from
another there. Some do not: within a single task a question can return nearly
the same value on every chunk, and a constant cannot reorder anything — it only
shifts the whole task down. Measured on phase6, 44 of the 120 (task, question)
pairs are like that, and question E is flat on 19 of the 24 tasks.

Two things this deliberately does NOT do:

  It does not drop a question for answering LOW. Low and flat are different: a
  question that sits at zero and spikes to 0.8 twice per episode is exactly the
  decisive signal this gate is for. Only flatness is disqualifying.

  It does not drop question A. It is the one safe axis, it enters the formula
  multiplicatively rather than through the risk product, and it is the only
  question that discriminates on all 24 tasks.

The threshold is relative to each task's own questions rather than an absolute
number, because absolute thresholds carried between datasets are how this
project has been bitten twice — allex inherited a merge limit from a slower
recording and flagged 32% of chunks spuriously, and RoboCasa's clip threshold
blocked 18.7% on a property of the simulator. A question is dropped when it
varies less than FLOOR of the widest-varying question in that same task, so the
comparison is always within one task's own scale.

    python reaggregate_task_qsel.py [--base BASE] [--floor 0.10] [--out PATH]
"""
import argparse
import json

import numpy as np
import pandas as pd

WS = "/sjw_alinlab/home/hojin2/quantization_agent_workspace"
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
Q = list("ABCDE")
SOFT = ["c_grip_transition_soft", "c_reversal_soft",
        "c_precise_hold_soft", "c_infeasible_merge_soft"]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="v6b_phase6_softA")
ap.add_argument("--floor", type=float, default=0.10,
                help="drop a question varying less than this share of the task's widest")
ap.add_argument("--out", default=None)
a = ap.parse_args()
OUT = a.out or f"{WS}/assets/labels/robocasa/{a.base}_qsel.parquet"

d = pd.read_parquet(f"{WS}/assets/labels/robocasa/{a.base}.parquet")
cls = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    e = json.loads(line)
    c = [t for t in e.get("tasks", []) if isinstance(t, str) and " " not in t and t != "Valid"]
    if c:
        cls[e["episode_index"]] = c[0]
d["t"] = d.episode_index.map(cls)
have = d.t.notna().values

V = d[[f"q_{q}" for q in Q]].values.astype(float)
S = d[[c for c in SOFT if c in d.columns]].values.astype(float)

sd = d[have].groupby("t")[[f"q_{q}" for q in Q]].std()
keep = np.ones_like(V)                     # 1 = question counts for this chunk
dropped = {}
for t, row in sd.iterrows():
    widest = row.max()
    sel = (d.t == t).values
    out = [q for j, q in enumerate(Q)
           if q != "A" and row[f"q_{q}"] < a.floor * widest]
    for q in out:
        keep[sel, Q.index(q)] = 0
    if out:
        dropped[t] = out

# identical aggregation, with the flat questions simply not entering the product
risk = 1 - np.prod(1 - S, axis=1) * np.prod(1 - V[:, 1:] * keep[:, 1:], axis=1)
raw = (1 - risk) * (0.5 + 0.5 * V[:, 0])
rank = np.argsort(np.argsort(raw)) / (len(raw) - 1)

before = d.p_yes.values.copy()
d["p_raw_soft"] = d["p_raw"] = raw
d["p_yes_soft"] = d["p_yes"] = rank
d["quantize"] = (rank >= 0.5).astype(int)
d.drop(columns=["t"]).to_parquet(OUT, index=False)

from scipy.stats import spearmanr  # noqa: E402
n_pairs = sum(len(v) for v in dropped.values())
print(f"floor = {a.floor:.2f} of each task's widest-varying question")
print(f"dropped {n_pairs} (task, question) pairs over {len(sd)} tasks")
for q in Q:
    n = sum(1 for v in dropped.values() if q in v)
    if n:
        print(f"   {q} dropped on {n:2d} tasks")
print(f"\n{len(d):,} rows -> {OUT}")
print(f"  rank correlation with {a.base}      : {spearmanr(before, rank).statistic:+.4f}")
print(f"  decisions that flip at tau=0.5      : {np.mean((before >= .5) != (rank >= .5)):.2%}")
