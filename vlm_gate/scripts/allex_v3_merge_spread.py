"""샤드를 합치고, 칸별로 확신을 띠에 편다.

라벨링은 샤드로 나눠 도는데 띠에 펴는 것은 한 칸의 청크가 다 모여야 한다 --
분위수를 그 칸 안에서 매기기 때문이다. 그래서 샤드는 확신까지만 쓰고 나오고,
펴는 것은 여기서 한 번에 한다.
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v3_checks import TASK_RANGE, DEFAULT_RANGE, band_place, snap  # noqa: E402

OUT = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v5tempo_v3"))
rows, seen = [], set()
for p in sorted(glob.glob(f"{OUT}/records_s*_*.jsonl")):
    n = 0
    for l in open(p):
        r = json.loads(l)
        k = (r["ep"], r["f"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
        n += 1
    print(f"  {os.path.basename(p)}  {n}")
print(f"합계 {len(rows)} 청크")

byc = collections.defaultdict(list)
for i, r in enumerate(rows):
    byc[r.get("cell")].append(i)
for cell, idx in sorted(byc.items(), key=lambda kv: str(kv[0])):
    lo, hi = TASK_RANGE.get(cell, DEFAULT_RANGE)
    z = band_place([rows[i]["conf"] for i in idx], lo, hi)
    for i, zz in zip(idx, z):
        rows[i]["K_spread"] = round(float(zz), 3)
        rows[i]["K"] = snap(float(zz))
    c = collections.Counter(rows[i]["K"] for i in idx)
    print(f"  {str(cell):<15} n={len(idx):6d}  [{lo:g},{hi:g}]   " +
          "  ".join(f"{k:g}x {100*v/len(idx):.0f}%" for k, v in sorted(c.items())))

rows.sort(key=lambda r: (r["ep"], r["f"]))
dst = f"{OUT}/records.jsonl"
with open(dst, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"-> {dst}")
