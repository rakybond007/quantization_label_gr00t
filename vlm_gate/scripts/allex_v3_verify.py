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
from allex_v3_checks import (ACTIVE, DEFAULT_RANGE, NGRADE, SIGN,  # noqa: E402
                             TASK_RANGE,
                             band_place, snap)

STRATUM = "cell"   # 층은 주석이 주는 여섯 칸이다 (Rotate/Bring/Pass x Box/PolyBag).

PARSE_MIN = 99.0        # 1
# 합격선은 구조에서 나온다. K = 하한 + 확신 x (상한-하한) 이고 확신에 /2 가
# 들어가므로, 문항 하나가 등급 d 만큼 갈리면 K 는 d/(2*(등급수-1)) x 띠폭 만큼
# 움직인다. 가장 넓은 띠가 1.0 이므로 한 칸(0.5)을 움직이려면 d = 등급수-1,
# 즉 눈금 전체다. 라벨을 바꿀 수 있는 최소치는 그 절반이다.
# 1.3 은 5단 등급표와 옛 구조(상한 폭 1.5, /2 없음)에서 나온 값이라 폐기.
CONTRAST_MIN = (NGRADE - 1) / 2.0      # 3
CORR_MAX = 0.7          # 4
# 한 칸 안에서 절반 넘는 장면이 같은 답을 받으면, 문항이 장면이 아니라 칸을
# 서술하고 있는 것이다. 띠 폭과 무관하게 성립하는 선이라 이렇게 잡는다.
TIE_MAX = 0.50          # 6  칸 안 최빈 답 조합 비율의 상한
# [7] 은 합격선에서 뺐다. v2 는 정답지가 아니라 다른 프롬프트로 뽑은 같은
# 모델의 출력이다. 거기 맞추라고 게이트를 걸면 v3 의 목표가 v2 의 재현이 되고,
# 그러면 v3 를 만들 이유가 없다. 더 근본적으로, 게이트가 정답지를 요구하면
# 이 절차는 정답지 없는 새 태스크로 못 옮겨 간다. 참고로만 찍는다.

OUT = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3loop"))
rs = [json.loads(l) for l in open(f"{OUT}/records.jsonl")]
Q = ACTIVE
NAME = {"CLAMP": "두 손 사이에 붙듦", "LOOSE": "쥔 자리만으로 붙듦",
        "SHOVE": "밀어 보냄", "FLIP": "쉽게 잡히는 것 뒤집음", "FREE": "그저 옮기는 중",
        "IDLE": "아무것도 안 닿음"}
# 각 문항이 어느 층에서 높아야 하는가. E 는 못 박은 문항이라 순위에서 뺀다.
# 각 문항이 높아야 할 서브태스크. 위험 풀은 Rotate Box 와 Bring PolyBag 인데
# 후자는 주석에 없으므로 Bring Object 안에 섞여 있다.
OWN = {"CLAMP": ["Rotate Box"], "LOOSE": ["Bring PolyBag"],
       "SHOVE": ["Pass Box", "Pass PolyBag"], "FLIP": ["Rotate PolyBag"]}
gates = {}

by = collections.defaultdict(list)
for r in rs:
    by[r.get(STRATUM)].append(r)
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

# [3] 층 간 대비는 판정에서 뺐다. 상한·하한이 주석의 칸에서 나오고 확신은
# lo + conf*(hi-lo) 로 그 칸 안에서만 쓰이므로, 칸마다 더해지는 상수는 최종
# 배속에 닿지 않는다. 이 지표는 바로 그 상수를 재고 있어서, 통과시키려고
# 문항을 흔들면 칸 안 신호가 깨진다 -- LOOSE 를 좁혔을 때 실제로 그랬다.
# 후보를 고를 때 칸을 가르는지 보는 것은 생성 단계 규칙으로 남긴다.
info = {}
print(f"[3] 층 간 대비   자기 층 - 나머지  (참고, 판정 아님)")
for q in Q:
    if q not in OWN:      # FREE 는 못 박은 문항, 순위에서 뺀다
        own = [r[q] for v in by.values() for r in v if r.get(q) is not None]
        print(f"      {q} {NAME[q]:<12} 못 박은 문항, 순위에서 뺌 (평균 {np.mean(own):.2f})")
        continue
    own = [r[q] for c in OWN[q] for r in by.get(c, []) if r.get(q) is not None]
    oth = [r[q] for c, v in by.items() if c not in OWN[q] for r in v if r.get(q) is not None]
    if not own or not oth:
        info[f"3 {q}"] = float("nan")
        print(f"      {q} 층이 비어 판정 불가")
        continue
    d = float(np.mean(own) - np.mean(oth))
    info[f"3 {q}"] = d
    print(f"      {q} {NAME[q]:<12} {'+'.join(OWN[q]):<18} {np.mean(own):.2f}  나머지 "
          f"{np.mean(oth):.2f}   차 {d:+.2f}")

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
# 문항 겹침도 판정에서 뺐다. 겹치는 것을 뺄 때마다 결과가 나빠졌다 -- 다섯을
# 한 프롬프트에서 같이 답하므로 겹치는 문항이 서로를 붙들어 준다. 한 문항에
# 한 축의 전권을 주는 것이 오히려 위험하다. 0.70 이라는 선도 내가 지어낸
# 값이었고, 겹침이 해로운지 이로운지는 상관계수가 아니라 결과가 말한다.
info["4 상관"] = mx
print(f"      최대 {mx:+.3f}  {'통과' if mx <= CORR_MAX else '미달'}")

# [5] 범위 이탈은 뺐다. 상한·하한은 후처리이고 구조가 보장한다 -- 문항 답변의
# 판정이 아니다. 같은 장면이 같은 배속을 받는 것도 결함이 아니다.

# 칸 안에서 답이 갈리는지 보던 검사는 뺐다. 어떤 동작은 처음부터 끝까지 정말
# 비슷할 수 있고, 그러면 모든 청크가 같은 답을 내는 것이 맞다. 답이 같다는
# 사실만으로는 문항이 못 알아보는 것인지 장면이 실제로 같은 것인지 못 가른다.
# 그 선(50%)도 측정이 아니라 내가 지어낸 값이었고, 0.019 차이를 맞추려고
# 문항을 열 번 넘게 갈아 끼우게 만들었다.
#
# 칸별 배속 분포는 그대로 찍는다 -- 판정이 아니라 눈으로 보기 위해서다.
print("칸별 배속 (판정 아님, 참고)")
for cell in sorted(by):
    v = by[cell]
    lo, hi = TASK_RANGE.get(cell, (None, None))
    ks = list(band_place([r["conf"] for r in v], lo, hi))
    sn = collections.Counter(snap(k) for k in ks)
    print(f"      {cell:<14} n={len(v):4d}  [{lo:g},{hi:g}]   " +
          "  ".join(f"{k:g}x {100*n/len(ks):.0f}%" for k, n in sorted(sn.items())))

# v2 가 매긴 배속과 얼마나 같은가 ------------------------------------------
# 이게 실질적인 확인이다. v2 는 사람이 보고 쓸 만하다고 판단한 라벨이고,
# 상한/하한도 같은 표를 쓰므로 배속끼리 바로 견줄 수 있다. 지어낸 선이 없다 --
# 몇 %가 같은지, 다른 것은 얼마나 벌어졌는지 그대로 찍는다.
V2 = os.environ.get("ALLEX_V2", "")
if V2 and os.path.exists(V2):
    g = {}
    for line in open(V2):
        r = json.loads(line)
        k = r.get("K_snap", r.get("K"))
        if k is not None:
            g[(r["ep"], r["f"])] = snap(float(k))
    mine = {}
    for cell, v in by.items():
        lo, hi = TASK_RANGE.get(cell, DEFAULT_RANGE)
        for r, k in zip(v, band_place([r["conf"] for r in v], lo, hi)):
            mine[(r["ep"], r["f"])] = snap(float(k))
    pair = [(g[k], mine[k]) for k in mine if k in g]
    if pair:
        same = sum(1 for a, b in pair if abs(a - b) < 1e-9)
        d = [b - a for a, b in pair]
        print(f"\nv2 가 매긴 배속과 견주기   겹치는 청크 {len(pair)}개")
        print(f"      같은 배속 {100*same/len(pair):.1f}%   "
              f"한 칸(0.5) 차이 {100*sum(1 for x in d if abs(abs(x)-0.5)<1e-9)/len(pair):.1f}%   "
              f"그 이상 {100*sum(1 for x in d if abs(x)>0.5+1e-9)/len(pair):.1f}%")
        print(f"      평균 배속  v2 {np.mean([a for a,_ in pair]):.3f}  "
              f"지금 {np.mean([b for _,b in pair]):.3f}   "
              f"치우침 {np.mean(d):+.3f}")
        for cell in sorted(by):
            q = [(g[(r["ep"], r["f"])], mine[(r["ep"], r["f"])]) for r in by[cell]
                 if (r["ep"], r["f"]) in g]
            if not q:
                continue
            s_ = sum(1 for a, b in q if abs(a - b) < 1e-9)
            print(f"      {cell:<14} n={len(q):4d}  같음 {100*s_/len(q):5.1f}%   "
                  f"v2 {np.mean([a for a,_ in q]):.2f} -> 지금 {np.mean([b for _,b in q]):.2f}")
    else:
        print("\nv2 와 겹치는 청크가 없다")
else:
    print("\nv2 라벨 경로(ALLEX_V2)가 없어 견주지 못했다")

print("\n=== 판정 ===")
for k, v in info.items():
    print(f"  {k:<10} {v:8.3f}  참고")
for k, (v, lim, ok) in gates.items():
    print(f"  {k:<10} {v:8.3f}  기준 {lim:6.2f}   {'통과' if ok else '미달'}")
print(f"  전체: {'통과' if all(o for _, _, o in gates.values()) else '미달'}")
json.dump({k: {"값": float(v), "기준": float(lim), "통과": bool(ok)}
           for k, (v, lim, ok) in gates.items()},
          open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
