"""PROMPT_METHOD 의 검증을 층화 표본 위에서 잰다. 임계값은 PROMPT_LOOP 가 얼린다.

바뀐 것 둘.

최빈답 비율은 **판정에서 뺐다.** 그 숫자는 문항이 죽은 것과 현상이 원래 드문
것을 못 가른다. 에피소드 0~3 에서 Rotate PolyBag 은 청크의 5.4% 였으니, 제대로
작동하는 문항이라도 95% 가 한 등급에 몰리는 게 맞다. 층화가 이걸 푼다 -- 각
층이 25% 면 작동하는 문항은 반드시 갈린다. 그래서 최빈답과 등급 사용 폭은
진단으로만 남기고, 판정은 층 간 대비가 한다.

대비의 합격선 1.3 은 라벨이 실제로 움직이는 최소치다. 등급 차 d 는 가중치
d/4 로 들어가고 표의 폭 1.5 에 곱해지므로 K 를 d*1.5/4 만큼 움직인다. 후보
배속의 간격이 0.5 이므로 한 칸이라도 옮기려면 d >= 0.5*4/1.5 = 1.33 이다.
그보다 작은 차이는 유의하든 말든 최종 라벨을 바꾸지 못한다.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v3_checks import CEILING, snap  # noqa: E402

PARSE_MIN = 99.0        # 검증 1
CONTRAST_MIN = 1.3      # 검증 3
CORR_MAX = 0.7          # 검증 4

OUT = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3loop"))
rs = [json.loads(l) for l in open(f"{OUT}/records.jsonl")]
Q = "ABCD"
# 각 문항이 뽑혀 나온 국면 = 그 문항이 걸려야 할 층.
OWN = {"A": "move+bag", "B": "turn+box", "C": "turn+bag", "D": "move+box"}
OWNQ = {v: k for k, v in OWN.items()}
NAME = {"A": "무른 것을 들어 놓음", "B": "단단한 것을 두 손 사이에서",
        "C": "무른 것을 판 위에서", "D": "단단한 것이 빈 곳에"}
# 상한이지 목표가 아니다. 넘으면 틀린 것이고, 밑돌면 안전하되 속도를 못
# 챙긴 것뿐이다. 그래서 판정은 위반율이 하고, 여유는 진단으로만 본다 --
# 그러지 않으면 모든 청크에 1.0 을 찍는 라벨러가 만점을 받는다.
CEIL_OF = {"move+box": 3.0, "move+bag": 2.0, "turn+box": 1.5, "turn+bag": 2.5}
VIOLATE_MAX = 5.0        # 그 층의 문항이 걸린 청크 중 상한을 넘긴 비율, %
gates = {}

by = collections.defaultdict(list)
for r in rs:
    by[r.get("cell")].append(r)
print(f"청크 {len(rs)}개   층 " + "  ".join(f"{k} {len(v)}" for k, v in sorted(by.items())))

# 1 -------------------------------------------------------------- 형식 준수 (판정)
full = sum(1 for r in rs if all(r.get(q) is not None for q in Q))
gates["1 형식"] = (100 * full / len(rs), PARSE_MIN, 100 * full / len(rs) >= PARSE_MIN)
print(f"\n[1] 형식 준수  {gates['1 형식'][0]:.1f}%  (합격선 {PARSE_MIN})  "
      f"{'통과' if gates['1 형식'][2] else '미달'}")

# 2 ------------------------------------------------------------------ 진단만
print("[2] 진단 (판정에 안 씀)")
for q in Q:
    c = collections.Counter(r[q] for r in rs if r.get(q) is not None)
    top = 100 * c.most_common(1)[0][1] / max(1, sum(c.values()))
    print(f"      {q} 최빈 {top:5.1f}%   쓴 등급 {sorted(c)}"
          f"{'   등급표에 죽은 칸 있음' if len(c) < 4 else ''}")

# 3 ------------------------------------------------------------ 층 간 대비 (판정)
print(f"[3] 층 간 대비   자기 층 - 나머지 층,  합격선 {CONTRAST_MIN}")
for q in Q:
    own = [r[q] for r in by.get(OWN[q], []) if r.get(q) is not None]
    oth = [r[q] for cell, v in by.items() if cell != OWN[q] for r in v if r.get(q) is not None]
    if not own or not oth:
        print(f"      {q} {NAME[q]:<20} 층이 비어 판정 불가")
        gates[f"3 {q}"] = (float("nan"), CONTRAST_MIN, False)
        continue
    d = float(np.mean(own) - np.mean(oth))
    gates[f"3 {q}"] = (d, CONTRAST_MIN, d >= CONTRAST_MIN)
    print(f"      {q} {NAME[q]:<20} {OWN[q]:<9} {np.mean(own):.2f}  나머지 {np.mean(oth):.2f}"
          f"   차 {d:+.2f}  {'통과' if d >= CONTRAST_MIN else '미달'}")

# 4 -------------------------------------------------------------- 문항 상관 (판정)
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

# 5 ------------------------------------------------------------ 상한 위반 (판정)
# 상한은 그 서브태스크를 통째로 재생했을 때의 값이라 사실상 그 구간 청크들의
# 최솟값이다. 청크 단위로는 Rotate Box 안의 접근 구간이 3.0 이어도 맞을 수
# 있고 -- 변속 라벨링을 하는 이유가 그것이다 -- 아무 동작도 없는 대기 청크는
# 기준값 2.5 를 받기로 되어 있다. 그래서 위반은 **그 층의 문항이 실제로 걸린
# 청크** 에서만 따진다.
print(f"[5] 상한 위반   그 층의 문항이 걸린 청크에서, 합격선 {VIOLATE_MAX:.0f}% 이하")
worst = 0.0
for cell in sorted(CEIL_OF):
    v = by.get(cell, [])
    own = [r for r in v if (r.get(OWNQ[cell]) or 1) >= 4]      # 그 국면이라고 답한 것
    if not own:
        print(f"      {cell:<9} 그 국면으로 답한 청크 없음")
        worst = 100.0
        continue
    bad = [r for r in own if r["K"] > CEIL_OF[cell] + 1e-9]
    pct = 100 * len(bad) / len(own)
    worst = max(worst, pct)
    ks = [r["K"] for r in own]
    print(f"      {cell:<9} 국면 {len(own):3d}청크  평균 K {np.mean(ks):.2f}  "
          f"상한 {CEIL_OF[cell]:g}  위반 {pct:4.1f}%  "
          f"여유 {100*np.mean(ks)/CEIL_OF[cell]:3.0f}%  "
          f"{'통과' if pct <= VIOLATE_MAX else '미달'}")
idle = [r for v in by.values() for r in v if not any((r.get(q) or 1) > 1 for q in Q)]
if idle:
    ki = collections.Counter(snap(r["K"]) for r in idle)
    print(f"      대기 {len(idle)}청크  " + "  ".join(f"{k:g}x {n}" for k, n in sorted(ki.items()))
          + "   (기준값 2.5)")
gates["5 위반"] = (worst, VIOLATE_MAX, worst <= VIOLATE_MAX)

print("\n=== 판정 ===")
for k, (v, lim, ok) in gates.items():
    print(f"  {k:<10} {v:8.3f}  기준 {lim:6.2f}   {'통과' if ok else '미달'}")
allok = all(ok for _, _, ok in gates.values())
print(f"  전체: {'통과' if allok else '미달'}")
json.dump({k: {"값": v, "기준": lim, "통과": ok} for k, (v, lim, ok) in gates.items()},
          open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
