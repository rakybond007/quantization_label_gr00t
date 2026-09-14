"""청크마다 배율별 요구량을 계산하고, 국면별 한계를 실측 사다리에 맞춘다.

**왜 국면 구성으로 역산하지 않는가.** 태스크마다 국면 구성이 거의 같아서
(접근 ~50% · 운반 ~35%, 파지·해제는 태스크 간 표준편차 0.04) 설계행렬 조건수가
1158 이다. 태스크 사다리에서 국면별 배율을 직접 뽑을 수 없다.

대신 **청크마다 다른 것**을 쓴다. 배율 r 로 건너뛰면 컨트롤러가 한 스텝에
요구받는 변위가 청크마다 다르고, 그건 계획된 액션에서 바로 계산된다. 국면은
그 요구량을 얼마나 허용할지(한계)만 정한다 -- 파지는 빡빡하게, 접근은 느슨하게.

    K̂(청크) = max { r : demand_r(청크) <= limit[국면(청크)] }

한계는 **태스크 평균 K̂ 가 실측 최대 안전 배율을 재현하도록** 적합한다.
관측 24개(태스크)에 파라미터는 국면 수만큼이다.
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
import phase_scorecard as ps                                       # noqa: E402

RATIOS = [1.0, 1.5, 2.0, 2.5, 2.67, 4.0]
RET_MIN = float(os.environ.get("RET_MIN", "0.85"))
NCHUNK = int(os.environ.get("NCHUNK", "100"))


def demand_curve(a, f, n=16):
    """배율 r 로 합쳤을 때 한 스텝이 요구하는 최대 변위. r 마다 하나.

    델타 액션이므로 건너뛴 스텝의 변위가 합쳐져 한 번에 요구된다. 누적 변위
    곡선에서 길이 r 창의 최대 변화량으로 재면 정수가 아닌 배율도 같은 식으로
    다룬다(r=2.5 는 사다리에 실제로 있는 칸이다).
    """
    w = a[f:f + n, 5:8]
    C = np.vstack([np.zeros(3), np.cumsum(w, axis=0)])       # (n+1, 3)
    t = np.arange(len(C), dtype=float)
    out = []
    for r in RATIOS:
        if r <= 1.0:
            out.append(float(np.linalg.norm(w, axis=1).max()) if len(w) else 0.0)
            continue
        s = np.arange(0, len(C) - 1 - r, 1.0)
        if len(s) == 0:
            out.append(0.0); continue
        A = np.stack([np.interp(s, t, C[:, k]) for k in range(3)], 1)
        B = np.stack([np.interp(s + r, t, C[:, k]) for k in range(3)], 1)
        out.append(float(np.linalg.norm(B - A, axis=1).max()))
    return out


def main():
    L = json.load(open(f"{BASE}/analysis/eval_results/LADDERS.json"))
    rungs = L["ladders"]["robocasa/uniform"]["rungs"]
    assert [r["ratio"] for r in rungs] == RATIOS, "사다리 칸이 바뀌었다"

    target = {}
    for t in rungs[0]["tasks"]:
        base = rungs[0]["tasks"][t]["success"]
        if base <= 0.05:
            continue
        mx = 1.0
        for i, r in enumerate(RATIOS):
            if rungs[i]["tasks"][t]["success"] / base >= RET_MIN:
                mx = r
            else:
                break
        target[t] = mx
    print(f"목표: 태스크 {len(target)}개 · 최대 안전 배율 평균 {np.mean(list(target.values())):.2f}")

    d = pd.read_parquet(f"{BASE}/_tmp/hf_upload/labels/robocasa_v7.parquet",
                        columns=["episode_index", "frame_index", "task"])
    rows = []
    for t, g in d[d.task.isin(target)].groupby("task"):
        s = g.sample(n=min(NCHUNK, len(g)), random_state=11)
        for r_ in s.itertuples():
            ep, f = int(r_.episode_index), int(r_.frame_index)
            try:
                a = ps.get(ep)
                ph = ps.ph6(ep, f)
            except Exception:
                continue
            rows.append({"task": t, "phase": ph, **{f"d{i}": v for i, v in
                                                    enumerate(demand_curve(a, f))}})
        print(f"  {t[:30]:30s} {len(rows)}", flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(f"{BASE}/analysis/robocasa_ladder/chunk_demand.parquet", index=False)
    print(f"\n청크 {len(D):,} · 국면 {sorted(D.phase.unique())}")

    PH = sorted(D.phase.unique())
    D["pi"] = D.phase.map({p: i for i, p in enumerate(PH)})
    Dm = D[[f"d{i}" for i in range(len(RATIOS))]].to_numpy()
    pi = D.pi.to_numpy()
    tasks = sorted(target)
    ti = D.task.map({t: i for i, t in enumerate(tasks)}).to_numpy()
    y = np.array([target[t] for t in tasks])
    R = np.array(RATIOS)

    def khat(limits):
        lim = limits[pi][:, None]                    # 청크마다 자기 국면의 한계
        ok = Dm <= lim                               # (청크, 배율)
        ok[:, 0] = True                              # 1.0 배는 언제나 가능
        idx = (ok * np.arange(len(R))).max(1)
        return R[idx]

    def loss(z):
        k = khat(np.exp(z))
        pred = np.array([k[ti == i].mean() for i in range(len(tasks))])
        return float(np.mean((pred - y) ** 2))

    z0 = np.log(np.full(len(PH), np.quantile(Dm, 0.5)))
    best = minimize(loss, z0, method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    lim = np.exp(best.x)
    k = khat(lim)
    pred = np.array([k[ti == i].mean() for i in range(len(tasks))])
    rho, p = spearmanr(pred, y)

    print(f"\n국면별 한계 (작을수록 빡빡함)")
    for p_, v in sorted(zip(PH, lim), key=lambda x: x[1]):
        n = int((D.phase == p_).sum())
        print(f"  {p_:10s} {v:.4f}   (청크 {n})")
    print(f"\n적합 후 RMSE {np.sqrt(best.fun):.3f} 배 · 태스크 평균 K̂ 대 실측 "
          f"순위상관 {rho:+.3f} (p={p:.4f})")
    print(f"청크별 K̂ 분포: " + " · ".join(
        f"{r}x:{(k == r).mean():.1%}" for r in RATIOS))
    print(f"태스크 안 K̂ 표준편차 평균 {np.mean([k[ti==i].std() for i in range(len(tasks))]):.3f}"
          "  (0 이면 태스크 상수라 게이트가 하는 일이 없다)")
    print(f"\n{'태스크':26s} {'실측':>6s} {'예측':>6s}")
    for i, t in enumerate(tasks):
        print(f"  {t[:26]:26s} {y[i]:6.2f} {pred[i]:6.2f}")


if __name__ == "__main__":
    main()
