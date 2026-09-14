"""D 가 왜 안 갈리나. 태스크별로 D 등급을 본다."""
import json, sys, os
import numpy as np
sys.path.insert(0, "vlm_gate/scripts")
from libero_v2_verify import _ep_to_task, RISK, STABLE
E = _ep_to_task()
rows = [json.loads(l) for l in open(
    "vlm_gate/output/_gate_distill/libero_v2_pools_s1_0.jsonl")]
by = {}
for r in rows:
    t = E.get(r["ep"])
    if t: by.setdefault(t, []).append(r)
print("%-12s %5s | %5s %5s %5s %5s %5s  | 풀" % ("task", "n", "A", "B", "C", "D", "E"))
for t in list(RISK) + list(STABLE):
    rs = by.get(t)
    if not rs: continue
    m = [np.mean([r[k] for r in rs]) for k in "ABCDE"]
    print("%-12s %5d | %5.2f %5.2f %5.2f %5.2f %5.2f  | %s"
          % (t, len(rs), *m, "위험" if t in RISK else "안정"))
