"""**상한표가 자기 자신을 재현하는가.** 같은 조건으로 다시 재면 같은 값이 나오나.

    python vlm_gate/scripts/ceiling_stability.py robocasa

새 평가를 안 돌린다. `LADDERS.json` 에 태스크·칸마다 성공 개수와 시행 수가
있으므로, 그 비율로 50번을 다시 뽑아 상한 판정을 다시 돌리기만 하면 된다.

## 왜 이걸 재나

우리가 하는 모든 것이 이 표 위에 서 있다. 배속 라벨의 띠도, 문항의 위험/안정
풀도, 방금 한 "현오차가 상한을 맞히는가" 도 전부 이 표를 정답으로 놓는다.

**그런데 태스크마다 50에피다.** 성공률 0.5 언저리에서 1표준오차가 7%p 다.
그 위에서 상한을 네 칸(1.0·1.5·2.0·2.5)으로 가른다. 판정폭(TOL 0.10,
REL_TOL 0.25)이 표준오차와 비슷한 크기라, 한 태스크가 어느 칸에 떨어지는지가
운에 달릴 수 있다.

**과녁이 흔들리면 아무도 못 맞힌다.** 상한을 맞히는 시험에서 무엇이 떨어졌을
때, 그것이 못 맞힌 것인지 맞힐 수 없는 것인지부터 갈라야 한다.

## 무엇을 못 재나

`derive_ceilings` 의 **짝지은 스텝 검정은 흉내내지 않는다** -- 에피소드별 스텝
수가 필요한데 사다리 요약에는 없다. 그 검정은 상한을 낮추기만 하므로, 여기서
나오는 안정도는 **실제보다 낙관적인 상한선**이다.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = (1.0, 1.5, 2.0, 2.5)
TOL, REL_TOL, FLOOR = 0.10, 0.25, 0.35


def snap(r):
    return min(GRID, key=lambda g: abs(g - r))


def ceiling_of(rate, rungs):
    """`derive_ceilings` 의 성공률 부분을 그대로 옮긴 것."""
    b0 = rate[rungs[0]]
    if b0 < FLOOR:
        return 1.0
    ceil = 1.0
    for r in rungs[1:]:
        s = rate[r]
        if s < b0 - TOL:
            break
        if b0 > 0 and (b0 - s) / b0 > REL_TOL:
            break
        ceil = snap(r)
    return ceil


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "robocasa"
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    L = json.load(open(os.path.join(HERE, "..", "analysis", "eval_results",
                                    "LADDERS.json")))["ladders"][f"{bench}/uniform"]
    real = json.load(open(os.path.join(HERE, "..", "analysis",
                                       f"{bench}_task_ceilings.json")))

    cnt = {}
    for r in L["rungs"]:
        for t, v in r["tasks"].items():
            cnt.setdefault(t, {})[float(r["ratio"])] = (
                int(round(v["success"] * v["n"])), int(v["n"]))
    rungs = sorted({k for t in cnt for k in cnt[t]})
    tasks = sorted(cnt)
    print(f"태스크 {len(tasks)} · 칸 {rungs} · 에피 {cnt[tasks[0]][1.0][1]} · "
          f"재추출 {draws}회\n")

    rng = np.random.default_rng(0)
    hits, dist = {}, {}
    for t in tasks:
        if len(cnt[t]) < len(rungs):
            continue
        got = []
        for _ in range(draws):
            rate = {r: rng.binomial(cnt[t][r][1], cnt[t][r][0] / cnt[t][r][1])
                    / cnt[t][r][1] for r in rungs}
            got.append(ceiling_of(rate, rungs))
        c = Counter(got)
        obs = float(real[t][1] if isinstance(real[t], list) else real[t])
        hits[t] = c[obs] / draws
        dist[t] = c

    print(f"{'태스크':24s} {'기록된 상한':>10s} {'같은 값이 나오는 비율':>20s}   "
          f"{'재추출 분포':>28s}")
    for t in sorted(hits, key=lambda t: hits[t]):
        obs = float(real[t][1] if isinstance(real[t], list) else real[t])
        d = " ".join(f"{g}:{dist[t][g]/draws:.0%}" for g in GRID if dist[t][g])
        print(f"{t:24s} {obs:10.1f} {hits[t]:20.0%}   {d:>28s}")

    v = np.array(list(hits.values()))
    print(f"\n재현율 평균 {v.mean():.1%} · 중앙 {np.median(v):.1%} · "
          f"{(v < 0.5).sum()}/{len(v)} 태스크는 절반도 재현 안 됨")

    # 최빈 상한만 찍는 바닥선이 이 흔들림 아래서 얼마나 나오나 -- 상한 맞히기
    # 시험의 천장을 알려면 필요하다.
    mode = Counter(float(real[t][1] if isinstance(real[t], list) else real[t])
                   for t in hits).most_common(1)[0]
    print(f"\n최빈 상한 {mode[0]} 을 {mode[1]}/{len(hits)} 태스크가 갖는다 "
          f"= 바닥선 {mode[1]/len(hits):.1%}")
    print(f"**어떤 방법이든 상한 맞히기의 천장이 재현율 평균 {v.mean():.1%} 다.**")
    print("그 표를 정답으로 놓고 채점하는 이상, 그 위로는 못 올라간다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
