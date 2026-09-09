"""부호와 등급 환산을 바꿔 가며 **같은 등급에서** 신뢰도·배속을 다시 낸다.

    python vlm_gate/scripts/sign_variants.py robocasa <labels_ratio.parquet>

VLM 을 다시 안 부른다. 등급은 그대로 두고 그 뒤 계산만 바꾼다. 몇 분이다.

## 판 셋

    v0  지금 그대로            A 감점 0.667 · C 가점 0.267 · 환산 (g-1)/4
    v1  부호만 뒤집음          A 가점 · C 감점 · 환산 그대로
    v2  부호 + 환산            v1 에 더해 3등급을 위험에서 뺀다

**셋을 다 내는 이유는 두 고침의 몫을 갈라 보기 위해서다.** 한꺼번에 바꾸면
분포가 달라진 것이 부호 때문인지 환산 때문인지 모른다.

## v1 -- 부호

사다리 실측이 지금 부호와 반대다(2.0배속 손상, 24태스크 x 50에피).

    A 계열(누름·돌림) 8태스크   -6.7%   ← 오히려 오른다   그런데 감점, 가중치 최대
    C 계열(경첩·레일) 6태스크  +14.7%                     그런데 가점

phase6 에서 이미 갈라 놓은 것이다 -- "버튼·노브(+0.030)와 문·서랍(-0.117)을
함께 물어서 둘 다 같은 답을 받았다"(PROMPT_METHOD.md). 부호가 반대라서 갈랐고,
가른 뒤 서로 바꿔 달렸다.

가중치는 **덮는 태스크 수를 그대로 두고 부호만 옮긴 뒤 각 변을 다시 1로**
맞춘다. 세는 방식을 같이 바꾸면 무엇이 분포를 옮겼는지 또 안 갈린다.

    감점  B 2 · C 4   ->  B 1/3 · C 2/3
    가점  A 4 · D 5 · E 6  ->  4/15 · 5/15 · 6/15

## v2 -- 환산

등급표는 **거리 눈금**이다(5 지금 · 4 한 동작 남음 · 3 향해 가는 중이고 아직
멀다 · 2 있지만 딴 일 · 1 없음). 그런데 `(g-1)/4` 로 선형 환산해 세기처럼
더한다. 그러면 3등급이 위험의 절반을 낸다.

"손이 버튼으로 가고 있고 아직 멀다" 는 **빈 공간 이동이고 가장 압축이 잘 되는
순간이다.** 그것이 최대 가중치 문항에서 0.333 의 위험을 낸다. 청크는 약 1초라
그 안에 드는 것은 4·5 뿐이다.

    v0/v1   1:0.00  2:0.25  3:0.50  4:0.75  5:1.00
    v2      1:0.00  2:0.00  3:0.00  4:0.50  5:1.00

**가점 문항도 같은 환산을 쓴다.** 한쪽만 바꾸면 두 변의 크기가 달라져 0.5 가
더 이상 두 변이 같아지는 자리가 아니게 된다.
"""
import json
import sys
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SLOTS = "ABCDE"

VARIANTS = {
    "v0 지금": dict(
        sign={"A": -1, "B": -1, "C": +1, "D": +1, "E": +1},
        weight={"A": 4/6, "B": 2/6, "C": 4/15, "D": 5/15, "E": 6/15},
        map=np.array([0.00, 0.25, 0.50, 0.75, 1.00])),
    "v1 부호": dict(
        sign={"A": +1, "B": -1, "C": -1, "D": +1, "E": +1},
        weight={"B": 2/6, "C": 4/6, "A": 4/15, "D": 5/15, "E": 6/15},
        map=np.array([0.00, 0.25, 0.50, 0.75, 1.00])),
    "v2 부호+환산": dict(
        sign={"A": +1, "B": -1, "C": -1, "D": +1, "E": +1},
        weight={"B": 2/6, "C": 4/6, "A": 4/15, "D": 5/15, "E": 6/15},
        map=np.array([0.00, 0.00, 0.00, 0.50, 1.00])),
}


def conf_of(df, v):
    """등급 -> 신뢰도. `ratio_label.confidence` 와 같은 식이되 환산표만 갈아 끼운다.

    등급 분포가 있으면(eg_*) 기댓값을 쓴다. 기댓값은 1~5 사이의 실수라 환산표를
    **선형 보간**해서 읽는다 -- 정수만 받으면 기댓값을 쓰는 뜻이 없어진다.
    """
    m = v["map"]
    risk = np.zeros(len(df))
    safe = np.zeros(len(df))
    for k in SLOTS:
        col = f"eg_{k}" if f"eg_{k}" in df.columns else k
        g = df[col].to_numpy(float)
        val = np.interp(g, np.arange(1, 6), m)
        (risk if v["sign"][k] < 0 else safe)[:] += v["weight"][k] * val
    return np.clip((1.0 + safe - risk) / 2.0, 0.0, 1.0)


def hist(x, lo=0.0, hi=1.0, bins=20, width=44):
    c, edges = np.histogram(x, bins=bins, range=(lo, hi))
    top = max(1, c.max())
    for i in range(bins):
        bar = "#" * int(round(width * c[i] / top))
        print(f"  {edges[i]:5.2f}-{edges[i+1]:5.2f} {c[i]:8,d} {bar}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    bench, lab = sys.argv[1:3]
    from ratio_label import assign_levels, band_of, CHECKS   # 같은 배치 규칙을 쓴다

    ceil = json.load(open(os.path.join(HERE, "..", "analysis",
                                       f"{bench}_task_ceilings.json")))
    need = ["task", "fixed"] + list(SLOTS) + [f"eg_{k}" for k in SLOTS]
    df = pd.read_parquet(lab, columns=None)
    df = df[[c for c in need if c in df.columns]]
    fixed = (df["fixed"].to_numpy() == 1) if "fixed" in df.columns \
        else np.zeros(len(df), bool)
    print(f"행 {len(df):,} · 태스크 {df.task.nunique()} · "
          f"접촉 고정 {fixed.mean():.2%}\n")

    got = {}
    for name, v in VARIANTS.items():
        c = conf_of(df, v)
        # 배속은 지금 쓰는 규칙(levels) 그대로 -- 규칙을 같이 바꾸면 안 갈린다.
        ratio = np.ones(len(df))
        for tk, idx in df.groupby("task").indices.items():
            lo, hi = band_of(ceil, tk)
            free = idx[~fixed[idx]]
            if len(free):
                ratio[free] = assign_levels(c[free], lo, hi)
        got[name] = (c, ratio)

    print("== 신뢰도 ==")
    print(f"{'판':14s} {'최소':>6s} {'25%':>6s} {'중앙':>6s} {'75%':>6s} "
          f"{'최대':>6s} {'폭':>6s} {'표준편차':>8s}")
    for n, (c, _) in got.items():
        q = np.percentile(c, [0, 25, 50, 75, 100])
        print(f"{n:14s} " + " ".join(f"{x:6.3f}" for x in q)
              + f" {q[4]-q[0]:6.3f} {c.std():8.3f}")
    for n, (c, _) in got.items():
        print(f"\n  [{n}]")
        hist(c)

    print("\n== 배속 (levels 규칙 그대로) ==")
    grid = [1.0, 1.5, 2.0, 2.5]
    print(f"{'판':14s} " + " ".join(f"{g:>7}" for g in grid) + f" {'평균':>7s}")
    for n, (_, r) in got.items():
        print(f"{n:14s} " + " ".join(f"{(r==g).mean():7.1%}" for g in grid)
              + f" {r.mean():7.3f}")

    print("\n== 판 사이에서 배속이 바뀐 행 ==")
    ns = list(got)
    for i in range(len(ns)-1):
        a, b = got[ns[i]][1], got[ns[i+1]][1]
        print(f"  {ns[i]} -> {ns[i+1]}   {(a!=b).mean():6.1%} 이동, "
              f"평균 {b.mean()-a.mean():+.3f}")
    a, b = got[ns[0]][1], got[ns[-1]][1]
    print(f"  {ns[0]} -> {ns[-1]}   {(a!=b).mean():6.1%} 이동, "
          f"평균 {b.mean()-a.mean():+.3f}")

    print("\n== 문항별 평균 등급 (참고) ==")
    for k in SLOTS:
        col = f"eg_{k}" if f"eg_{k}" in df.columns else k
        g = df[col].to_numpy(float)
        ig = df[k].to_numpy(int)
        d = np.bincount(ig, minlength=6)[1:6] / len(ig)
        print(f"  {k}  기댓값 평균 {g.mean():.2f}   정수 분포 "
              + " ".join(f"{v:4.0%}" for v in d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
