"""Re-aggregate a phase6 label set with one computed flag left out.

`infeasible_merge` fires when merging two steps would push a command past the
simulator's action clip. Releasing that clip changes nothing: naive K=2 scores
0.5983 with the clip in place and 0.5950 with it scaled x3, at 214.0 and 213.6
steps. The clipping is not what the merge costs, so a flag that fires on 18.7%
of chunks for it is blocking on a property of the harness.

Nothing is recomputed here. The soft flag columns and the VLM answers are
already in the parquet, so dropping one term is arithmetic on what is stored —
which also means the variant is exactly the original with one factor removed,
and nothing else moves.

    python reaggregate_drop_flag.py [flag-to-drop] [src] [out]
"""
import sys

import numpy as np
import pandas as pd

WS = "/sjw_alinlab/home/hojin2/quantization_agent_workspace"
DROP = sys.argv[1] if len(sys.argv) > 1 else "infeasible_merge"
SRC = sys.argv[2] if len(sys.argv) > 2 else f"{WS}/assets/labels/robocasa/v6b_phase6_softA.parquet"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{WS}/assets/labels/robocasa/v6b_phase6_softA_nomerge.parquet"

ALL = ["grip_transition", "reversal", "precise_hold", "infeasible_merge"]
keep = [f for f in ALL if f != DROP]
if len(keep) == len(ALL):
    raise SystemExit(f"{DROP!r} is not one of {ALL}")

d = pd.read_parquet(SRC)
S = d[[f"c_{f}_soft" for f in keep]].values
V = d[["q_A", "q_B", "q_C", "q_D", "q_E"]].values

# Identical to the shipped aggregation, one factor short.
risk = 1 - np.prod(1 - np.column_stack([S, V[:, 1:]]), axis=1)
safe = 0.5 + 0.5 * V[:, 0]
raw = (1 - risk) * safe
rank = np.argsort(np.argsort(raw)) / (len(raw) - 1)

before = d.p_yes.values.copy()
d["p_raw_soft"] = d["p_raw"] = raw
d["p_yes_soft"] = d["p_yes"] = rank
d["quantize"] = (rank >= 0.5).astype(int)
d.to_parquet(OUT, index=False)

dropped = d[f"c_{DROP}_soft"].values
from scipy.stats import spearmanr  # noqa: E402
print(f"dropped c_{DROP}_soft   mean {dropped.mean():.4f}   "
      f"fires >0.5 on {np.mean(dropped > 0.5):.2%} of chunks")
print(f"kept: {', '.join(keep)}")
print(f"\n{len(d):,} rows -> {OUT}")
print(f"  rank correlation with the original labelling : {spearmanr(before, rank).statistic:+.3f}")
print(f"  chunks whose compress/block decision flips   : {np.mean((before >= .5) != (rank >= .5)):.2%}")
print(f"  ties at the bottom of raw                    : {np.mean(raw == raw.min()):.4%}")
