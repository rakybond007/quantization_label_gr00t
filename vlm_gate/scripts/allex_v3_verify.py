"""검증. 임계값은 PROMPT_LOOP 가 얼린다.

구조가 바뀌면서 무엇을 재야 하는지도 바뀌었다. 문항은 더 이상 "이게 어느
태스크인가" 를 맞히지 않는다 -- 그건 주석이 답한다. 문항이 하는 일은 단위
행동을 알아보고 그 태스크의 [하한, 상한] 안에서 어디에 설지를 정하는 것이다.

그래서 상한 위반은 판정이 아니라 버그 점검이 된다. K 는 구조적으로 그 범위를
못 벗어나므로 0 이 아니면 코드가 틀린 것이다. 대신 태스크 안에서 확신이
퍼지는지를 새로 잰다 -- 같은 태스크의 모든 청크가 같은 값이면 상한을 통째로
쓰는 것과 같고, 그러면 게이트가 하는 일이 없다.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v3_checks import ACTIVE, SIGN, TASK_RANGE, snap  # noqa: E402

PARSE_MIN = 99.0        # 1
CONTRAST_MIN = 1.3      # 3
CORR_MAX = 0.7          # 4
SPREAD_MIN = 0.15       # 6  태스크 안 확신의 표준편차

OUT = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3loop"))
rs = [json.loads(l) for l in open(f"{OUT}/records.jsonl")]
Q = ACTIVE
NAME = {"TURN": "뒤집는 중", "HEFT": "두 손이 필요한 것",
        "SHOVE": "밀어 보냄", "FIRM": "단단한 것 운반", "FREE": "빈손 통과"}
# 각 문항이 어느 층에서 높아야 하는가. E 는 못 박은 문항이라 순위에서 뺀다.
OWN = {"TURN": ["turn+box", "turn+bag"], "HEFT": ["turn+box", "move+box"],
       "SHOVE": ["move+box", "move+bag"], "FIRM": ["move+box"]}
gates = {}

by = collections.defaultdict(list)
for r in rs:
    by[r.get("cell")].append(r)
print(f"청크 {len(rs)}개   층 " + "  ".join(f"{k} {len(v)}" for k, v in sorted(by.items())))

full = sum(1 for r in rs if all(r.get(q) is not None for q in Q))
gates["1 형식"] = (100 * full / len(rs), PARSE_MIN, 100 * full / len(rs) >= PARSE_MIN)
print(f"\n[1] 형식 준수  {gates['1 형식'][0]:.1f}%   {'통과' if gates['1 형식'][2] else '미달'}")

print("[2] 진단")
for q in Q:
    c = collections.Counter(r[q] for r in rs if r.get(q) is not None)
    top = 100 * c.most_common(1)[0][1] / max(1, sum(c.values()))
    print(f"      {q} {NAME[q]:<12} 최빈 {top:5.1f}%  쓴 등급 {sorted(c)}"
          f"{'   죽은 칸 있음' if len(c) < 4 else ''}")

print(f"[3] 층 간 대비   자기 층 - 나머지,  합격선 {CONTRAST_MIN}")
for q in Q:
    if q not in OWN:      # FREE 는 못 박은 문항, 순위에서 뺀다
        own = [r[q] for v in by.values() for r in v if r.get(q) is not None]
        print(f"      {q} {NAME[q]:<12} 못 박은 문항, 순위에서 뺌 (평균 {np.mean(own):.2f})")
        continue
    own = [r[q] for c in OWN[q] for r in by.get(c, []) if r.get(q) is not None]
    oth = [r[q] for c, v in by.items() if c not in OWN[q] for r in v if r.get(q) is not None]
    if not own or not oth:
        gates[f"3 {q}"] = (float("nan"), CONTRAST_MIN, False)
        print(f"      {q} 층이 비어 판정 불가")
        continue
    d = float(np.mean(own) - np.mean(oth))
    gates[f"3 {q}"] = (d, CONTRAST_MIN, d >= CONTRAST_MIN)
    print(f"      {q} {NAME[q]:<12} {'+'.join(OWN[q]):<18} {np.mean(own):.2f}  나머지 "
          f"{np.mean(oth):.2f}   차 {d:+.2f}  {'통과' if d >= CONTRAST_MIN else '미달'}")

print(f"[4] 문항 상관   합격선 {CORR_MAX} 이하")
M = np.array([[r[q] for q in Q] for r in rs if all(r.get(q) is not None for q in Q)], float)
mx = 0.0
for i in range(len(Q)):
    for j in range(i + 1, len(Q)):
        if M[:, i].std() < 1e-9 or M[:, j].std() < 1e-9:
            print(f"      {Q[i]}-{Q[j]}  상수라 계산 불가")
            mx = 1.0
            continue
        c = float(np.corrcoef(M[:, i], M[:, j])[0, 1])
        mx = max(mx, abs(c))
        if abs(c) > 0.4:
            print(f"      {Q[i]}-{Q[j]}  {c:+.3f}  {'겹침' if abs(c) > CORR_MAX else ''}")
gates["4 상관"] = (mx, CORR_MAX, mx <= CORR_MAX)
print(f"      최대 {mx:+.3f}  {'통과' if mx <= CORR_MAX else '미달'}")

print("[5] 범위 이탈   구조가 막는다. 0 이 아니면 버그")
bad = [r for r in rs if r.get("cell") in TASK_RANGE
       and not (TASK_RANGE[r["cell"]][0] - 1e-9 <= r["K"] <= TASK_RANGE[r["cell"]][1] + 1e-9)]
gates["5 이탈"] = (len(bad), 0, not bad)
print(f"      {len(bad)}개  {'통과' if not bad else '버그'}")

print(f"[6] 태스크 안 확신 분산   표준편차 {SPREAD_MIN} 이상")
worst = 1e9
for cell in sorted(by):
    v = [r["conf"] for r in by[cell] if "conf" in r]
    if not v:
        continue
    sd = float(np.std(v))
    worst = min(worst, sd)
    ks = [r["K"] for r in by[cell]]
    lo, hi = TASK_RANGE.get(cell, (None, None))
    sn = collections.Counter(snap(k) for k in ks)
    print(f"      {cell:<9} 확신 평균 {np.mean(v):.2f} 표준편차 {sd:.2f}   "
          f"K {np.mean(ks):.2f} [{lo:g},{hi:g}]   " +
          "  ".join(f"{k:g}x {100*n/len(ks):.0f}%" for k, n in sorted(sn.items())))
gates["6 분산"] = (worst, SPREAD_MIN, worst >= SPREAD_MIN)

print("\n=== 판정 ===")
for k, (v, lim, ok) in gates.items():
    print(f"  {k:<10} {v:8.3f}  기준 {lim:6.2f}   {'통과' if ok else '미달'}")
print(f"  전체: {'통과' if all(o for _, _, o in gates.values()) else '미달'}")
json.dump({k: {"값": v, "기준": lim, "통과": ok} for k, (v, lim, ok) in gates.items()},
          open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
