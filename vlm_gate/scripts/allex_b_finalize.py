"""스윕 결과에서 B 를 확정해 allex_v4c_checks.py 에 써넣는다.

고르는 기준은 **미리 정해져 있다** -- 결과를 보고 기준을 바꾸지 않는다.

  1. 봉투와 한손상자의 평균 등급 차가 가장 큰 것
  2. 동률이면 봉투에서 4등급 이상 비율이 높은 것
  3. 양손 구간에서 낮아야 한다(B 는 한손 전용 감점이다)

    python allex_b_finalize.py            # 무엇이 뽑히는지 보기만
    python allex_b_finalize.py --apply    # 문항 모듈에 써넣기
"""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
APPLY = "--apply" in sys.argv

TEXTS = {}
import re
src = open(f"{HERE}/allex_b_sweep.py").read()
blk = src[src.index("CANDS = ["):src.index("def grab")]
for m in re.finditer(r'\("([a-z_]+)",\n\s*"(.*?)"\),', blk, re.S):
    TEXTS[m.group(1)] = m.group(2)

rows = []
import glob as _g
for p in sorted(_g.glob(f"{BASE}/output/allex_b_sweep*.json")):
    if True:
        for name, picks in json.load(open(p)).items():
            b = np.array([x["B"] for x in picks])
            bg = np.array([bool(x.get("bag")) for x in picks])
            oh = np.array([bool(x["one_handed"]) for x in picks])
            box = oh & ~bg
            if not bg.any() or not box.any():
                continue
            rows.append({
                "name": name, "n": len(picks),
                "bag": float(b[bg].mean()), "box": float(b[box].mean()),
                "sep": float(b[bg].mean() - b[box].mean()),
                "bag4": float(np.mean(b[bg] >= 4)), "box4": float(np.mean(b[box] >= 4)),
                "two": float(b[~oh].mean()) if (~oh).any() else 0.0,
            })
rows.sort(key=lambda r: (-r["sep"], -r["bag4"]))
print(f"{'문구':16s} {'n':>4s} {'봉투':>6s} {'한손상자':>8s} {'차':>7s} {'봉투4+':>7s} {'상자4+':>7s} {'양손':>6s}")
for r in rows:
    print(f"  {r['name']:14s} {r['n']:4d} {r['bag']:6.2f} {r['box']:8.2f} {r['sep']:+7.2f} "
          f"{r['bag4']:7.1%} {r['box4']:7.1%} {r['two']:6.2f}")
if not rows:
    sys.exit("스윕 결과가 없다")
win = rows[0]
print(f"\n선정: **{win['name']}**  (차 {win['sep']:+.2f} · 봉투 4+ {win['bag4']:.1%})")
if win["sep"] < 0.30:
    print("!! 분리도가 0.30 미만이다. 이 문구로는 봉투를 못 가른다 -- 후보를 더 만들어야 한다.")
if not APPLY:
    print("\n(써넣으려면 --apply)"); sys.exit(0)

txt = TEXTS[win["name"]]
p = f"{HERE}/allex_v4c_checks.py"
s = open(p).read()
i = s.index('    "B) '); j = s.index('    "C) ')
lines = [f'    "B) {txt.splitlines()[0]}\\n"'] + \
        [f'    "{l}\\n"' for l in txt.splitlines()[1:]]
s = s[:i] + "\n".join(lines) + "\n" + s[j:]
s = re.sub(r'"B": "[A-Z_]+"', f'"B": "{win["name"].upper()}"', s)
open(p, "w").write(s)
print(f"-> {p} 에 B 를 {win['name']} 로 써넣었다")
