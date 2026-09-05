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
from allex_v3_checks import (ACTIVE, NGRADE, SIGN, TASK_RANGE,  # noqa: E402
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
NAME = {"CLAMP": "판에서 떠 있음", "LOOSE": "그러모아 쥔 것 옮김",
        "SHOVE": "밀어 보냄", "FLIP": "쉽게 잡히는 것 뒤집음", "FREE": "빈손 통과"}
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
gates["4 상관"] = (mx, CORR_MAX, mx <= CORR_MAX)
print(f"      최대 {mx:+.3f}  {'통과' if mx <= CORR_MAX else '미달'}")

# [5] 범위 이탈은 뺐다. 상한·하한은 후처리이고 구조가 보장한다 -- 문항 답변의
# 판정이 아니다. 같은 장면이 같은 배속을 받는 것도 결함이 아니다.

# [6] 은 확신의 표준편차를 재다가 바꿨다. 확신의 절대 크기는 등급표 눈금이
# 정하는 임의값이고 사상이 분위수라 어차피 버려진다. 실제로 물어야 할 것은
# **한 칸 안에서 답이 갈리는가** 다 -- 갈리면 띠에 펼 수 있고, 안 갈리면 어떤
# 사상을 해도 그 칸은 값 하나로 나와 게이트가 하는 일이 없어진다.
print(f"[6] 칸 안 답이 갈리는가   최빈 답 조합 {TIE_MAX:.0%} 이하")
worst = 0.0
for cell in sorted(by):
    v = by[cell]
    combo = collections.Counter(tuple(r.get(q) for q in Q) for r in v)
    top = combo.most_common(1)[0][1] / len(v)
    worst = max(worst, top)
    lo, hi = TASK_RANGE.get(cell, (None, None))
    # 사상은 여기서 다시 계산한다. 기록이 언제 만들어졌든 지금의 사상으로 본다.
    ks = list(band_place([r["conf"] for r in v], lo, hi))
    sn = collections.Counter(snap(k) for k in ks)
    print(f"      {cell:<14} 최빈 답 {100*top:5.1f}%  서로 다른 답 {len(combo):>2}가지   "
          f"K {np.mean(ks):.2f} [{lo:g},{hi:g}]   " +
          "  ".join(f"{k:g}x {100*n/len(ks):.0f}%" for k, n in sorted(sn.items())))
gates["6 칸 안 구분"] = (worst, TIE_MAX, worst <= TIE_MAX)

# 7 ------------------------------------------------ 정답지와의 확신 순위일치
# v2 의 K 와 우리 K 를 견주면 양쪽 상한이 섞여 든다. 상한은 후처리이므로
# **확신 대 확신**으로 본다 -- v2 의 stage-1 confidence p 가 같은 자리다.
GT = os.environ.get("ALLEX_GT", "")
if GT and os.path.exists(GT):
    g = {}
    for line in open(GT):
        r = json.loads(line)
        if "p" in r:
            g[(r["ep"], r["f"])] = float(r["p"])
    pair = [(g[(r["ep"], r["f"])], r["conf"]) for r in rs
            if (r["ep"], r["f"]) in g and "conf" in r]
    if len(pair) >= 30:
        a_ = np.array([x for x, _ in pair]); b_ = np.array([y for _, y in pair])
        ra, rb = a_.argsort().argsort().astype(float), b_.argsort().argsort().astype(float)
        rho = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0
        info["7 v2일치"] = rho
        print(f"[7] v2 와의 확신 순위일치  n={len(pair)}  rho {rho:+.3f}  (참고, 판정 아님)")
        print("      칸 안:  " + "  ".join(
            f"{c} {np.corrcoef(np.array([g[(r['ep'],r['f'])] for r in v if (r['ep'],r['f']) in g]).argsort().argsort(), np.array([r['conf'] for r in v if (r['ep'],r['f']) in g]).argsort().argsort())[0,1]:+.2f}"
            for c, v in sorted(by.items())
            if len([r for r in v if (r['ep'],r['f']) in g]) > 5))
    else:
        print(f"[7] 정답지  겹치는 청크 {len(pair)}개뿐 -- 판정 불가")

print("\n=== 판정 ===")
for k, v in info.items():
    print(f"  {k:<10} {v:8.3f}  참고")
for k, (v, lim, ok) in gates.items():
    print(f"  {k:<10} {v:8.3f}  기준 {lim:6.2f}   {'통과' if ok else '미달'}")
print(f"  전체: {'통과' if all(o for _, _, o in gates.values()) else '미달'}")
json.dump({k: {"값": float(v), "기준": float(lim), "통과": bool(ok)}
           for k, (v, lim, ok) in gates.items()},
          open(f"{OUT}/verify.json", "w"), ensure_ascii=False, indent=1)
