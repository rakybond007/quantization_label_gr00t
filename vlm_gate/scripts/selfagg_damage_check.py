"""라벨이 **실제 압축 손상**을 맞추는가. 모델 없이, 액션 숫자만으로 잰다.

로보카사는 EE delta 를 더해서 병합한다(SUM). 그래서 K=2 병합은 끝점을 정확히
보존하고, 잃는 것은 **중간 경로**다 -- 두 스텝을 하나로 합치면 그 사이의 점을
지나지 않고 직선으로 간다. 그 벗어난 거리가 이 순간 압축이 물리적으로 무엇을
버리는지다.

    dev = 두 스텝 쌍마다 |중간점 - 두 끝점을 잇는 직선| 의 창 안 최대/평균

날 거리(dev_raw)는 큰 동작에서 자동으로 커진다 -- 많이 움직이면 중간점도 멀다.
그래서 창 안 이동거리로 나눈 값(dev_rel)도 같이 낸다. 거친 큰 동작은
dev_raw 가 커도 dev_rel 은 작다. 어느 쪽이 라벨과 맞는지가 곧 라벨이 무엇을
재고 있었는지다.

    python selfagg_damage_check.py <scenes_or_z.jsonl> [...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)
N = 16


def damage(a, f, k=2):
    """K=k 병합이 버리는 중간 경로. -> (최대 날거리, 최대 상대거리, 이동거리)"""
    d = a[f:f + N, 5:8]
    if len(d) < k + 1:
        return np.nan, np.nan, np.nan
    p = np.vstack([np.zeros(3), np.cumsum(d, axis=0)])      # 창 안 상대 위치
    trav = float(np.linalg.norm(d, axis=1).sum())
    raws = []
    for s in range(0, len(d) - k + 1, k):
        A, B = p[s], p[s + k]                                # 병합 후 지나는 두 점
        seg = B - A
        L = np.linalg.norm(seg)
        for t in range(s + 1, s + k):                        # 버려지는 중간점
            v = p[t] - A
            if L < 1e-9:
                raws.append(float(np.linalg.norm(v)))
            else:
                raws.append(float(np.linalg.norm(v - seg * (v @ seg) / L**2)))
    if not raws:
        return np.nan, np.nan, trav
    mx = float(np.max(raws))
    return mx, (mx / trav if trav > 1e-9 else np.nan), trav


def rows(p):
    out = []
    if p.endswith(".json"):
        out = json.load(open(p))
    else:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    d = pd.DataFrame(out)
    return d.drop_duplicates(subset=["ep", "f"], keep="last")


def main():
    d = rows(sys.argv[1])
    for p in sys.argv[2:]:
        e = rows(p)[["ep", "f"] + [c for c in rows(p).columns if c == "Z"]]
        d = d.merge(e, on=["ep", "f"], how="left")
    acts = {}
    rs = []
    for r in d.itertuples():
        ep = int(r.ep)
        if ep not in acts:
            if len(acts) > 32:
                acts.clear()
            acts[ep] = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ep // CH:03d}/"
                f"episode_{ep:06d}.parquet")["action"].values)
        mx2, rel2, trav = damage(acts[ep], int(r.f), 2)
        mx3, rel3, _ = damage(acts[ep], int(r.f), 3)
        rs.append({"dev2": mx2, "rel2": rel2, "dev3": mx3, "rel3": rel3,
                   "trav": trav})
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(rs)], axis=1).dropna(
        subset=["dev2"])
    print(f"장면 {len(d)}\n")
    print("[손상 분포] K=2 병합이 버리는 중간 경로")
    for c, nm in (("dev2", "날거리 K2"), ("rel2", "상대 K2"),
                  ("dev3", "날거리 K3"), ("trav", "창내 이동거리")):
        q = d[c].quantile([.1, .5, .9]).values
        print(f"  {nm:<12} 10% {q[0]:.4f} · 50% {q[1]:.4f} · 90% {q[2]:.4f}")

    labs = [c for c in ("Z", "conf") if c in d and d[c].notna().any()]
    print(f"\n[라벨이 손상을 맞추는가]  (음의 상관이면 옳다 -- 손상 크면 압축 불가)")
    print(f"  {'라벨':<6}" + "".join(f"{k:>12}" for k in
                                     ("dev2", "rel2", "dev3", "rel3", "trav")))
    for L in labs:
        x = d[d[L].notna()]
        row = f"  {L:<6}"
        for c in ("dev2", "rel2", "dev3", "rel3", "trav"):
            row += f"{np.corrcoef(x[L].astype(float), x[c])[0,1]:>+12.3f}"
        print(row)

    print("\n[태스크 안에서만]  태스크 상수를 지우고 봐도 남는가")
    print(f"  {'라벨':<6}" + "".join(f"{k:>12}" for k in ("dev2", "rel2", "dev3")))
    for L in labs:
        x = d[d[L].notna()].copy()
        row = f"  {L:<6}"
        for c in ("dev2", "rel2", "dev3"):
            a = x[L].astype(float) - x.groupby("task")[L].transform("mean").astype(float)
            b = x[c] - x.groupby("task")[c].transform("mean")
            row += f"{np.corrcoef(a, b)[0,1]:>+12.3f}" if a.std() > 0 else f"{'n/a':>12}"
        print(row)

    print("\n[라벨 등급별 평균 손상]")
    for L in labs:
        x = d[d[L].notna()].copy()
        if L == "conf":
            x["_b"] = pd.qcut(x.conf, 5, labels=False, duplicates="drop")
        else:
            x["_b"] = x[L].astype(int)
        g = x.groupby("_b").agg(n=("dev2", "size"), dev2=("dev2", "mean"),
                                rel2=("rel2", "mean"), trav=("trav", "mean"))
        print(f"  {L}:")
        for k, r in g.iterrows():
            print(f"    {k}  n{int(r['n']):4d}  날거리 {r['dev2']:.4f}"
                  f"  상대 {r['rel2']:.4f}  이동 {r['trav']:.3f}")


if __name__ == "__main__":
    main()
