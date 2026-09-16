"""국면별 conf 분포에서 역치(tau)를 찾는다.

요구사항 (사용자 지시, 2026-09-16):
  long  내려놓기를 60~70% 는 압축 안 하도록 -- 즉 conf < tau 인 비율이 0.60~0.70
  long  운반은 반대로 대체로 압축 -- conf >= tau 가 높아야 한다
  pnp   상관 없음. 다만 물러남이 낮아진 것이 문제인지 본다

conf 는 그 판의 부호·가중으로 계산한다. 가중은 아직 확정이 아니므로 **잠정값**임을
표에 적는다.

  python humandata_tau.py <labels.jsonl>
"""
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
# **판본을 환경변수로 고른다.** 기본은 v1 재현용(libero 가중을 런타임에 패치)이고,
# v2 부터는 판본이 부호·가중을 직접 들고 있어 패치가 필요 없다.
#   CHECKS=humandata_v2_checks python humandata_tau.py output/humandata_v2/labels.jsonl
_CK_NAME = os.environ.get("CHECKS", "libero_v3c_checks")
CK = importlib.import_module(_CK_NAME)
_OWNS = hasattr(CK, "NAME") and "F" not in CK.SIGN and _CK_NAME != "libero_v3c_checks"

ORD = ["1집으러", "2파지", "3운반", "4내려놓으러", "5물러남"]


def main():
    path = sys.argv[1]
    R = [r for r in map(json.loads, open(path)) if r.get("A")]
    if _OWNS:
        # 판본이 부호·가중을 직접 들고 있다 -- 손대지 않는다.
        Q = tuple(sorted(CK.SIGN))
        S, W = dict(CK.SIGN), dict(CK.WEIGHT)
        has_F = False
    else:
        has_F = any("F" in r for r in R)
        Q = tuple("ABCDEF") if has_F else tuple("ABCDE")
        S = dict(CK.SIGN)
        W = dict(CK.WEIGHT)
    if has_F:
        S["F"] = -1
        W["F"] = W["B"]
        risk = [q for q in S if S[q] < 0]
        t = sum(W[q] for q in risk)
        for q in risk:
            W[q] /= t
    _g = CK.NGRADE - 1
    conf = lambda r: 0.5 * (1 + sum(S[q] * W[q] * ((r[q] - 1) / _g) for q in Q))

    # 국면은 라벨에 실려 있으면 그것을 쓰고(전량 스크립트), 없으면 액션에서 다시 낸다
    # (표본 스크립트는 phase 를 안 적는다). 문턱은 에피소드별 최대값으로 정규화한다.
    import glob

    import pandas as pd
    PM = {}
    for ds in ("pnp_task", "long_horizon_task"):
        root = f"/sjw_alinlab2/home/taekwan/Data/human_data/{ds}/lerobot"
        for f in sorted(glob.glob(root + "/data/*/*.parquet")):
            ep = int(f.split("episode_")[1][:6])
            a = np.stack(pd.read_parquet(f)["action"].values)
            g = a[:, 7]
            thr = max(0.15, 0.5 * float(g.max()))
            gz = g > thr
            tr = np.flatnonzero(np.diff(gz.astype(int)))
            cl = [x for x in tr if gz[x + 1]]
            op = [x for x in tr if not gz[x + 1]]
            cyc = []
            for c in cl:
                nx = [o for o in op if o > c]
                if nx and (not cyc or c > cyc[-1][1]) and nx[0] - c >= 24:
                    cyc.append((c, nx[0]))
            for c0, o0 in cyc:
                PM[(ds, ep, max(0, c0 - 12))] = "1집으러"
                PM[(ds, ep, c0)] = "2파지"
                PM[(ds, ep, (c0 + o0) // 2)] = "3운반"
                PM[(ds, ep, max(c0 + 1, o0 - 12))] = "4내려놓으러"
                PM[(ds, ep, o0)] = "5물러남"

    def ph(r):
        return r.get("phase") or PM.get((r["ds"], r["ep"], r["f"]))
    R = [r for r in R if ph(r)]
    print(f"청크 {len(R)} · 문항 {''.join(Q)} · 가중(잠정) "
          f"{ {k: round(v, 3) for k, v in W.items()} }\n")

    for d in ("pnp_task", "long_horizon_task"):
        v = [r for r in R if r["ds"] == d]
        if not v:
            continue
        print(f"── {d}  (n={len(v)})")
        print(f"{'국면':<12}{'n':>4}{'평균':>8}{'p10':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p90':>7}")
        for p in ORD:
            c = np.array([conf(r) for r in v if ph(r) == p])
            if not len(c):
                continue
            print(f"{p:<12}{len(c):>4}{c.mean():>8.3f}"
                  + "".join(f"{np.percentile(c, q):>7.3f}" for q in (10, 25, 50, 75, 90)))
        print()

    # long 의 역치 탐색
    L = [r for r in R if r["ds"] == "long_horizon_task"]
    place = np.array([conf(r) for r in L if ph(r) == "4내려놓으러"])
    carry = np.array([conf(r) for r in L if ph(r) == "3운반"])
    if len(place) and len(carry):
        print("long · 역치 후보 (conf < tau 면 압축 안 함)")
        print(f"{'tau':>7}{'내려놓기 차단율':>16}{'운반 차단율':>13}{'차이':>8}")
        best = None
        for tau in np.arange(0.30, 0.85, 0.01):
            a = float((place < tau).mean())
            b = float((carry < tau).mean())
            if 0.60 <= a <= 0.70:
                mark = "  <- 요구 구간"
                if best is None or a - b > best[0]:
                    best = (a - b, tau, a, b)
            else:
                mark = ""
            if abs(tau * 100 % 5) < 1e-6:
                print(f"{tau:>7.2f}{a:>16.0%}{b:>13.0%}{a - b:>8.0%}{mark}")
        if best:
            _, tau, a, b = best
            print(f"\n  **요구를 만족하는 tau 있음: {tau:.2f}** "
                  f"(내려놓기 {a:.0%} 차단 · 운반 {b:.0%} 차단 · 차 {a - b:.0%})")
        else:
            lo, hi = (place < 0.5).mean(), (carry < 0.5).mean()
            print(f"\n  **요구 구간(60~70%)을 만족하는 tau 가 없다.**")
            print(f"     내려놓기 차단율 범위 {min((place<t).mean() for t in np.arange(.3,.85,.01)):.0%}"
                  f" ~ {max((place<t).mean() for t in np.arange(.3,.85,.01)):.0%}")


if __name__ == "__main__":
    main()
