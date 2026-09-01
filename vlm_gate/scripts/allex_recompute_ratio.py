"""Recompute p, K_max and K_pre for an allex labelling with this recording's own constants.

The labelling client writes every VLM answer (s1_A..s1_D, s2_A..s2_D) and every
computed descriptor alongside the ratio it derived. So when a constant turns out
to have been carried in from a different recording, the fix costs no VLM calls
at all — the two prompt stages are rerun as arithmetic over what is already on
disk.

That has now happened twice. v5's merge limit came from a slower capture and
flagged 32% of v1 as infeasible for being faster than a demonstration that was
not theirs. v1's limit then went to v3 unchanged, and v3 is faster again: its
p99.9 single-step move is 0.478 rad against v1's 0.385, its held-rotation p90 is
78.4 deg against 55.0, its held gap rate p95 is 0.0097 against 0.0065.

Set the constants for the recording, then run:

    ALLEX_MERGE_LIMIT=0.478 ALLEX_ROT_LIMIT=78.4 ALLEX_GAP_LIMIT=0.0097 \\
      python allex_recompute_ratio.py <label-dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import (MERGE_LIMIT_V2, ROT_ACCUM_LIMIT_V2, GAP_RATE_LIMIT_V2,  # noqa: E402
                             ceiling_from_stage2, final_ratio, stage1_confidence)

D = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v3")

print(f"constants in effect: merge {MERGE_LIMIT_V2}  rot {ROT_ACCUM_LIMIT_V2}  "
      f"gap {GAP_RATE_LIMIT_V2}")

files = sorted(glob.glob(f"{D}/labels_*.jsonl"))
if not files:
    raise SystemExit(f"no labels_*.jsonl under {D}")

changed = n = 0
moved = {}
for f in files:
    rows = []
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        # the two stages, rerun over the answers already stored
        p = stage1_confidence((r["s1_A"], r["s1_B"], r["s1_C"], r["s1_D"]), r)
        k_max = ceiling_from_stage2(r["task"], r["s2_A"], r["s2_B"], r["s2_C"], r["s2_D"])
        k_pre = final_ratio(p, k_max)
        if abs(float(r.get("K_pre", -1)) - k_pre) > 1e-9:
            changed += 1
            moved[(r.get("K_pre"), k_pre)] = moved.get((r.get("K_pre"), k_pre), 0) + 1
        r["p"], r["K_max"], r["K_pre"] = float(p), float(k_max), float(k_pre)
        rows.append(r)
    with open(f, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n{n} chunks, {changed} ratios changed ({changed / max(n, 1):.1%})")
if moved:
    print("largest moves:")
    for (a, b), c in sorted(moved.items(), key=lambda x: -x[1])[:8]:
        print(f"   {a} -> {b}   {c}")
print("\nnow rerun allex_v2_aggregate.py to apply the hard blocks and write records.jsonl")
