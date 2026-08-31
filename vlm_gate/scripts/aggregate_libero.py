"""LIBERO label aggregation — same structure as phase6, different embodiment.

The five axes are one safe and four risks, as in RoboCasa phase6:
  A gripper in empty space        -> safe
  B setting onto a small flat target it must sit squarely on
  C moving into or out of an enclosed space
  D taking up or releasing an object's weight
  E reaching in among objects close enough to knock over

Two things differ from RoboCasa and neither is cosmetic.

`reversal` is not in the computed set. LIBERO trajectories are smooth —
adjacent-step cosine median 0.998, cos < 0 on 0.02% of steps — so a reversal
detector fires on nothing and a flag that never fires carries no information.
A continuous `turn`, cut at this data's own 5th percentile, replaced it.

The computed flags arrive continuous already, so there is no binary variant to
recompute. The saturation that made 29.51% of RoboCasa's binary chunks collapse
to exactly zero cannot happen here; the tie count is reported below so that
claim is checked rather than assumed.

A separate script rather than an argument to the RoboCasa one, for the reason
the phase6 script gives: a generation has to stay regenerable by the script
that made it, or comparisons stop being comparisons.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
DS = os.environ.get(
    "DS", "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta")
TAG = os.environ.get("TAG", "libero_dense")
OUTP = os.path.expanduser(os.environ.get(
    "OUTP", "~/quantization_agent_workspace/assets/labels/libero/libero_dense_v1.parquet"))

COMPUTED = ("grip_transition", "turn", "precise_hold", "infeasible_merge")
RISK_Q = "BCDE"
SAFE_Q = "A"

instr = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", [])
         if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

rows = {}
for p in sorted(glob.glob(f"{BASE}/output/_gate_distill/{TAG}_s16_*.jsonl")):
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if all(k in r for k in ("A", "B", "C", "D", "E", *COMPUTED)):
            rows[(r["ep"], r["f"])] = r
print(f"수집 {len(rows)}프레임")
if not rows:
    raise SystemExit(f"{TAG}_s16_*.jsonl 에서 읽은 게 없다")

keys = sorted(rows)
CR = np.array([[rows[k][c] for c in COMPUTED] for k in keys], dtype=float)
V = np.array([[rows[k][q] for q in "ABCDE"] for k in keys], dtype=float)

risk = 1 - np.prod(
    1 - np.column_stack([CR] + [V[:, "ABCDE".index(q)] for q in RISK_Q]), axis=1)
safe = 0.5 + 0.5 * V[:, "ABCDE".index(SAFE_Q)]
raw = (1 - risk) * safe
rank = (np.argsort(np.argsort(raw)) / (len(raw) - 1)).astype(np.float64)

df = pd.DataFrame({
    "episode_index": [k[0] for k in keys], "frame_index": [k[1] for k in keys],
    "task": [instr.get(k[0], "") for k in keys], "p_yes": rank, "p_raw": raw,
    "quantize": (rank >= 0.5).astype(int)})
for i, c in enumerate(COMPUTED):
    df[f"c_{c}"] = CR[:, i]
for i, q in enumerate("ABCDE"):
    df[f"q_{q}"] = V[:, i]

os.makedirs(os.path.dirname(OUTP), exist_ok=True)
df.to_parquet(OUTP, index=False)

# The tie fraction is the direct indicator of noisy-OR saturation: binary flags
# inflate it, and every tied chunk has thrown its VLM answer away.
ties = int((df.p_raw == df.p_raw.min()).sum())
print(f"저장 {OUTP}  {df.shape}")
print(f"p_raw 최저값 동점 {ties} ({ties / len(df):.2%})")
print("계산플래그", {c: round(df[f'c_{c}'].mean(), 4) for c in COMPUTED})
print("VLM문항", {q: round(df[f'q_{q}'].mean(), 3) for q in "ABCDE"})
