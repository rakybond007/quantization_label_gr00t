"""라벨이 **폐루프 압축 강건성**과 맞는가. 이것이 최종 판정이다.

태스크별로 압축 없이 돌린 성공률과 K배 압축으로 돌린 성공률이 있으면, 그 비(유지율)
가 그 태스크가 압축을 얼마나 견디는지다. 라벨의 태스크 평균이 그것과 **양의** 상관
이어야 라벨이 쓸 수 있다.

기하 손상(selfagg_damage_check.py)과는 다른 질문이다. 기하는 "무엇을 버리는가",
이것은 "버려도 되는가" 다. 24 태스크 실측에서 둘은 일치하지 않았다 -- 계산 특징은
기하를 잘 맞추지만(태스크 안 상관 -0.60) 유지율은 못 맞춘다(-0.23, p=0.28).

    python label_vs_retention.py <라벨.jsonl> [더 많은 라벨...]

라벨 파일은 task 와 (Z 또는 conf) 를 담은 jsonl 이면 된다. 여러 개 주면 합친다
(같은 라벨을 장면 반씩 나눠 돌린 경우).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_RUN = "baseline_full_v2_with_action_steps"
COMP_RUNS = {2: "baseline_compress_K2", 3: "baseline_compress_K3"}


def per_task(run):
    out = {}
    for p in glob.glob(f"output/robocasa/{run}/*/prediction.txt"):
        t = os.path.basename(os.path.dirname(p))
        s = [m.group(1) == "True" for m in
             (re.search(r"is_success:\s*\[\s*(True|False)\s*\]", l) for l in open(p))
             if m]
        if len(s) >= 20:
            out[t] = (float(np.mean(s)), len(s))
    return out


def rank_partial(x, y, c):
    """c 의 순위를 뺀 뒤의 x-y 순위상관."""
    R = (lambda v: pd.Series(v).rank().to_numpy())
    x, y, c = R(x), R(y), R(c)
    rx = x - np.poly1d(np.polyfit(c, x, 1))(c)
    ry = y - np.poly1d(np.polyfit(c, y, 1))(c)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []
    for p in sys.argv[1:]:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    d = pd.DataFrame(rows).drop_duplicates(subset=["ep", "f"], keep="last")
    cols = [c for c in ("Z", "conf") if c in d and d[c].notna().any()]
    print(f"라벨 {len(d)} 장면 · 태스크 {d.task.nunique()} · 열 {cols}")

    base = per_task(BASE_RUN)
    for K, run in COMP_RUNS.items():
        comp = per_task(run)
        T = [t for t in base if t in comp and t in set(d.task)]
        g = pd.DataFrame({"task": T,
                          "base": [base[t][0] for t in T],
                          "comp": [comp[t][0] for t in T]})
        g = g[g.base > 0.05]
        g["keep"] = g.comp / g.base
        m = d.groupby("task")[cols].mean()
        g = g.set_index("task").join(m, how="inner")
        print(f"\n=== K={K} · 태스크 {len(g)} · 유지율 평균 {g.keep.mean():.3f} ===")
        print(f"  {'라벨':<6}{'vs 유지율':>12}{'p':>8}{'기준통제후':>12}  판정")
        for c in cols:
            rho, p = spearmanr(g[c], g.keep)
            pr = rank_partial(g[c], g.keep, g.base)
            verdict = ("쓸 수 있다" if rho > 0.3 and p < .05 else
                       "뒤집어야 한다" if rho < -0.3 and p < .05 else "무관")
            print(f"  {c:<6}{rho:>+12.3f}{p:>8.3f}{pr:>+12.3f}  {verdict}")
        if K == 2:
            print("\n  태스크별 (유지율 낮은 쪽부터)")
            for t, r in g.sort_values("keep").iterrows():
                lab = "  ".join(f"{c}={r[c]:.2f}" for c in cols)
                print(f"    {r.keep:>5.2f}  기준 {r.base:.2f} -> {r.comp:.2f}   {lab}"
                      f"   {t}")


if __name__ == "__main__":
    main()
