"""배속 문항 채점. 실측 최대 안전 배율에 대해 문항 판을 견준다.

**판정의 축은 태스크 평균을 이기는가다.** 에피 단위 목표의 태스크 안 표준편차가
0.95 로 태스크 간 0.75 보다 크다. 태스크 이름만 아는 예측기(룩업표)는 이미
있으므로, 문항이 값을 하려면 그 위를 넘어야 한다.

    python ratio_score.py <라벨.jsonl> [...]
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LEVELS = np.array([1.0, 2.0, 2.67, 4.0])


def expected(gp):
    return [sum((i + 1) * float(p) for i, p in enumerate(row)) for row in gp]


def load(path):
    meta = json.load(open(path.replace(".jsonl", "_meta.json")))
    C = importlib.import_module(meta["checks"])
    rs = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rs).drop_duplicates(["task", "ep"], keep="last")
    Q = sorted(C.SIGN)
    eg = [expected(g) if isinstance(g, list) else None for g in d.get("gp", [None] * len(d))]
    for i, q in enumerate(Q):
        d["e" + q] = [(r[i] if r else float(d[q].iloc[j])) for j, r in enumerate(eg)]
    if len(Q) == 1:                      # 직접 분류 판
        s = d["e" + Q[0]]
    else:
        g = {q: (d["e" + q] - 1.0) / (C.NGRADE - 1) for q in Q}
        risk = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] < 0)
        safe = sum(C.WEIGHT[q] * g[q] for q in Q if C.SIGN[q] > 0)
        s = (1.0 + safe - risk) / 2.0
    d["score"] = s
    return d, C, meta


def fit_map(score, y):
    """점수 -> 배율. 분위를 목표 분포에 맞춘다(단조, 파라미터 없음)."""
    q = np.argsort(np.argsort(score)) / max(len(score) - 1, 1)
    cut = np.cumsum([np.mean(y == l) for l in LEVELS])[:-1]
    return LEVELS[np.searchsorted(cut, q)]


def report(path):
    d, C, meta = load(path)
    y = d.max_ratio.to_numpy()
    s = d.score.to_numpy()
    tk = d.task.to_numpy()
    print(f"\n{'='*70}\n{os.path.basename(path)}  ({len(d)}에피 · 문항 {len(C.SIGN)})")
    ngp = int(d.get("gp", pd.Series([None] * len(d))).notna().sum())
    print(f"  등급 분포 {ngp}/{len(d)} = {ngp/max(len(d),1):.0%}")
    for q in sorted(C.SIGN):
        v = d[q].to_numpy()
        print(f"    {q} {C.NAME[q]:14s} 평균 {d['e'+q].mean():.2f} · "
              f"최고등급 {(v==C.NGRADE).mean():5.1%} · 최저 {(v==1).mean():5.1%}")

    pred = fit_map(s, y)
    tmean = pd.Series(y).groupby(tk).transform("mean").to_numpy()
    gmean = np.full_like(y, y.mean())
    print(f"\n  {'예측기':22s} {'MAE':>6s} {'RMSE':>6s} {'순위상관':>9s}")
    for nm, p in (("전역 평균", gmean), ("태스크 평균 (룩업표)", tmean), ("문항", pred)):
        rho, pv = spearmanr(p, y)
        print(f"  {nm:22s} {np.abs(p-y).mean():6.3f} {np.sqrt(((p-y)**2).mean()):6.3f} "
              f"{rho:+.3f}(p{pv:.3f})")

    # 태스크 평균을 지운 뒤에도 남는가 -- 이게 룩업표를 이기는지의 판정
    ry = y - tmean
    rs_ = s - pd.Series(s).groupby(tk).transform("mean").to_numpy()
    rho, pv = spearmanr(rs_, ry)
    print(f"\n  태스크 평균 제거 후 점수 대 목표: 순위상관 {rho:+.3f} (p={pv:.4f})")
    print(f"    -> 양수·유의하면 문항이 태스크 이름 너머의 것을 읽는다")
    print(f"  전체 점수 대 목표: {spearmanr(s,y)[0]:+.3f}")
    print(f"  태스크 평균 점수 대 태스크 평균 목표: "
          f"{spearmanr(pd.Series(s).groupby(tk).mean(), pd.Series(y).groupby(tk).mean())[0]:+.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
