"""Could the YES/NO text answer replace the logit-derived confidence?

The judge writes both: `ans` is what the model actually said ("NO,NO,NO,NO,NO"),
and A-E are the continuous P(YES) values read from the {YES,NO} logits at the
same slots. Every labelling run has stored both all along, so the question of
whether the text alone would do can be answered from the labels rather than by
running anything.

Three things decide it:

1. How many distinct states the text can express. Five binary answers give at
   most 32, against 266,693 chunks — so the text alone forces enormous ties,
   and the pipeline rank-normalizes p_yes and cuts it at tau. A tie is an
   arbitrary order.
2. Whether the text moves at all. A question answered NO everywhere carries no
   information in text form however informative its logit is.
3. Whether the continuous value is doing work *inside* one text answer. If
   chunks that all say NO still spread over a wide P(YES), the text is
   discarding a real signal rather than summarizing it.
"""
import collections
import glob
import json

import numpy as np

SHARDS = ("/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/output/"
          "_gate_distill/libero_dense_s16_*.jsonl")
Q = "ABCDE"

rows = []
for p in sorted(glob.glob(SHARDS)):
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "ans" in r and all(q in r for q in Q):
            rows.append(r)
print(f"{len(rows):,} chunks with both the text answer and the logit values\n")

# 1 — how many states the text can express
combos = collections.Counter(r["ans"] for r in rows)
print("1. distinct text answers")
print(f"   {len(combos)} of the 32 a five-question form allows")
for a, n in combos.most_common(5):
    print(f"     {a:24s} {n:8,}  {100*n/len(rows):5.1f}%")
top = combos.most_common(1)[0]
print(f"   largest single state holds {100*top[1]/len(rows):.1f}% of all chunks\n")

# 2 — does the text move per question
print("2. per question: YES rate in text, and the logit value it hides")
print(f"   {'q':2s} {'text YES':>9s} {'P(YES) | text NO':>18s} {'P(YES) | text YES':>19s}")
for i, q in enumerate(Q):
    v = np.array([r[q] for r in rows], dtype=float)
    said_yes = np.array([r["ans"].split(",")[i].strip().upper().startswith("Y")
                         if len(r["ans"].split(",")) > i else False for r in rows])
    lo = v[~said_yes]
    hi = v[said_yes]
    print(f"   {q:2s} {said_yes.mean():8.2%} "
          f"{lo.mean():9.3f} (sd {lo.std():.3f})"
          f"{hi.mean():10.3f} (sd {hi.std():.3f})" if len(hi)
          else f"   {q:2s} {said_yes.mean():8.2%} {lo.mean():9.3f} (sd {lo.std():.3f})"
               f"        never said YES")

# 3 — spread that survives inside one text answer
print("\n3. spread inside the single most common text state")
sel = [r for r in rows if r["ans"] == top[0]]
print(f"   {len(sel):,} chunks all answered {top[0]}")
for i, q in enumerate(Q):
    v = np.array([r[Q[i]] for r in sel], dtype=float)
    print(f"     {Q[i]}  P(YES) min {v.min():.3f}  p50 {np.median(v):.3f}  "
          f"max {v.max():.3f}  sd {v.std():.3f}")
