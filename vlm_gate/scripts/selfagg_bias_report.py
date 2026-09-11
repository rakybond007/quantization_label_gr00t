"""집계를 모델에 맡긴 판이 쓸 만한가 -- 쏠림·태스크의 몫·우리 산수와의 관계.

판정 기준을 먼저 적어 둔다.

1. **쏠림.** 한 등급이 장면의 대부분을 먹으면 그 라벨로는 아무것도 못 가른다.
   5단계가 고르면 0.2씩이고, 최다 등급이 0.6 을 넘으면 사실상 상수다.
2. **태스크가 정하는 몫.** 우리 5문항 산수는 0.221 -- conf 분산의 78% 가
   태스크만 보고 정해진다. 모델에 맡기는 값이 이보다 태스크 의존이 **작아야**
   맡길 이유가 있다. 크면 그냥 태스크 상수를 배우는 것이다.
3. **이진 균형.** stride 4 로 라벨하고 학습 때 전 프레임으로 펼 것이므로,
   소수 클래스가 15% 아래로 내려가면 복원 재현율이 91% 밑으로 떨어진다.
4. **우리 산수와의 관계.** 완전히 같으면 맡길 이유가 없고, 무관하면 둘 중
   하나가 신호가 아니다.

    python selfagg_bias_report.py z_pct.jsonl [z_abs.jsonl]
"""
import json
import sys

import numpy as np
import pandas as pd


def load(p):
    rs = []
    for line in open(p):
        try:
            rs.append(json.loads(line))
        except Exception:
            pass
    d = pd.DataFrame(rs).drop_duplicates(subset=["ep", "f"], keep="last")
    return d


def var_share(d, col):
    """태스크 평균이 설명하는 분산의 몫. 1 이면 전부 태스크가 정한다."""
    y = d[col].to_numpy(float)
    if y.std() == 0:
        return 1.0
    m = d.groupby("task")[col].transform("mean").to_numpy(float)
    return float(np.var(m) / np.var(y))


def report(d, name):
    n = len(d)
    ok = d.Z.notna().sum()
    print(f"\n{'='*66}\n{name}  장면 {n}  형식실패 {n-ok} ({(n-ok)/max(n,1):.1%})")
    d = d[d.Z.notna()].copy()
    d["Z"] = d.Z.astype(int)
    vc = d.Z.value_counts().sort_index()
    print("\n[1] 등급 분포")
    for z in range(1, 6):
        c = int(vc.get(z, 0))
        print(f"  Z={z} {c:5d} {c/len(d):6.1%} {'#'*int(60*c/len(d))}")
    top = vc.max() / len(d)
    print(f"  최다 등급 몫 {top:.1%}  -> {'쏠림 심함' if top > .6 else '쓸 만함'}"
          f" · 쓰인 등급 {len(vc)}/5")

    print("\n[2] 태스크가 정하는 몫")
    vs = var_share(d, "Z")
    print(f"  모델 집계 Z      {vs:.3f}  (태스크가 {vs:.0%})")
    if "conf" in d:
        vc2 = var_share(d, "conf")
        print(f"  우리 5문항 conf  {vc2:.3f}  (태스크가 {vc2:.0%})")
        print(f"  -> 모델에 맡기면 태스크 의존이 "
              f"{'줄어든다' if vs < vc2 else '늘어난다'} ({vs-vc2:+.3f})")

    print("\n[3] 이진으로 잘랐을 때 균형")
    for th in (3, 4, 5):
        p = (d.Z >= th).mean()
        mino = min(p, 1 - p)
        print(f"  Z>={th}: 압축허용 {p:6.1%} · 소수클래스 {mino:6.1%}"
              f" -> {'stride4 복원 OK' if mino >= .15 else '소수클래스가 얇다'}")

    if "conf" in d:
        print("\n[4] 우리 산수와의 관계")
        r = np.corrcoef(d.Z, d.conf)[0, 1]
        print(f"  상관 {r:+.3f}")
        print("  Z별 conf 평균: " + "  ".join(
            f"Z{z}={g.conf.mean():.3f}(n{len(g)})" for z, g in d.groupby("Z")))

    print("\n[5] 태스크별 Z 평균 (낮을수록 압축 불가 판정)")
    t = d.groupby("task").Z.agg(["mean", "std", "size"]).sort_values("mean")
    for k, r in t.iterrows():
        print(f"  {r['mean']:.2f} ±{0 if pd.isna(r['std']) else r['std']:.2f} "
              f"n{int(r['size']):3d}  {k[:52]}")
    return d


def main():
    ds = {}
    for p in sys.argv[1:]:
        nm = "분위(pct)" if "pct" in p else "절대(abs)"
        ds[nm] = report(load(p), nm)
    if len(ds) == 2:
        a, b = ds.values()
        j = list(ds.values())[0].merge(list(ds.values())[1], on=["ep", "f"],
                                      suffixes=("_1", "_2"))
        print(f"\n{'='*66}\n[6] 속도 표현만 바꿨을 때 (같은 {len(j)} 장면)")
        print(f"  같은 등급 {100*(j.Z_1 == j.Z_2).mean():.1f}%"
              f" · 평균 {j.Z_1.mean():.2f} -> {j.Z_2.mean():.2f}"
              f" · 상관 {np.corrcoef(j.Z_1, j.Z_2)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
