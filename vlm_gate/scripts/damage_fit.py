"""**질러간 양이 손상을 설명하는가.** 있는 두 파일을 붙여서 잰다.

    python vlm_gate/scripts/damage_fit.py robocasa <chord_error.parquet>

새 평가도 재라벨링도 GPU 도 안 쓴다. `chord_error.parquet`(태스크·시점·K 마다
현오차)과 `LADDERS.json`(태스크·K 마다 실측 성공률)이 이미 있다.

    24태스크 x 사다리 칸 = "이만큼 질러갔더니 성공률이 이만큼 떨어졌다" 쌍

**역치를 지어내지 않는다.** 오차가 얼마일 때 성공률이 어떻게 되는지는 우리가
모르는 값이고, 계단이라고 가정할 근거도 없다. 그래서 가정하지 않고 잰다.

## 셋을 따로 본다

한 눈으로만 보면 속는다.

**A. 전체 쌍** -- 현오차가 크면 손상이 큰가. 쉬운 시험이다. K 가 커지면 둘 다
   커지므로 K 효과만으로도 상관이 나온다. **통과해도 별 뜻이 없다.**

**B. 같은 K 안에서 태스크끼리** -- 같은 배속에서, 더 많이 질러가는 태스크가 더
   많이 깨지는가. **이게 본 시험이다.** K 효과가 빠져 있어서, 통과하면 현오차가
   태스크 차이를 실제로 잡은 것이다.

**C. 상한을 맞히는가** -- 규칙 `배속 = 통계 <= θ 를 만족하는 가장 큰 K` 로 각
   태스크의 상한을 예측해 실측과 맞춰 본다. **θ 는 그 태스크를 빼고 정한다**
   (하나 빼고 맞추기). 안 그러면 시험지로 공부한 뒤 그 시험을 보는 것이다.

   그리고 **바닥선을 같이 낸다** -- 아무것도 안 보고 제일 흔한 상한을 찍으면
   몇 개를 맞히나. 그걸 못 넘으면 현오차는 아무 일도 안 한 것이다.

## 통계를 많이 시험하지 않는다

n=24 다. 통계를 쉰 개 시험해 제일 좋은 것을 고르면 그건 잡음을 고른 것이다.
그래서 **미리 정한 다섯**만 본다. 그것도 다섯 중 최고는 낙관적이라는 것을
같이 적는다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 미리 정한 다섯. 늘리지 않는다.
STATS = {
    "평균":      lambda v: float(v.mean()),
    "중앙":      lambda v: float(np.median(v)),
    "p90":       lambda v: float(np.percentile(v, 90)),
    "최대":      lambda v: float(v.max()),
    "큰것비율":  lambda v: float((v > 0.15).mean()),   # 0.15 는 전체 p90 언저리
}


def kendall(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = d = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = (x[i] - x[j]) * (y[i] - y[j])
            c += s > 0
            d += s < 0
    return (c - d) / max(1, c + d), c + d


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    bench, chord = sys.argv[1:3]

    L = json.load(open(os.path.join(HERE, "..", "analysis", "eval_results",
                                    "LADDERS.json")))["ladders"][f"{bench}/uniform"]
    succ, nrun = {}, {}
    for r in L["rungs"]:
        for t, v in r["tasks"].items():
            succ.setdefault(t, {})[float(r["ratio"])] = v["success"]
            nrun.setdefault(t, {})[float(r["ratio"])] = v.get("n", 50)
    ceil = json.load(open(os.path.join(HERE, "..", "analysis",
                                       f"{bench}_task_ceilings.json")))
    hi = {t: float(v[1] if isinstance(v, list) else v) for t, v in ceil.items()}

    df = pd.read_parquet(chord)
    KS = sorted(df.K.unique())
    print(f"현오차 {len(df):,}행 · 태스크 {df.task.nunique()} · K {KS}")

    # (태스크, K) 마다 통계와 실측 손상
    rec = []
    for (t, K), g in df.groupby(["task", "K"]):
        if t not in succ or K not in succ[t] or 1.0 not in succ[t]:
            continue
        base = succ[t][1.0]
        if base <= 0:
            continue                      # 기준선이 0 이면 상대 손상이 뜻이 없다
        v = g["pos"].to_numpy()
        row = {"task": t, "K": K,
               "damage": (base - succ[t][K]) / base, "base": base,
               "n": nrun[t].get(K, 50)}
        row.update({k: f(v) for k, f in STATS.items()})
        rec.append(row)
    R = pd.DataFrame(rec)
    miss = sorted(set(df.task.unique()) - set(R.task.unique()))
    print(f"쌍 {len(R)}개 · 태스크 {R.task.nunique()}"
          + (f"   [!] 사다리에 없어 빠진 태스크 {miss}" if miss else "") + "\n")

    print("=" * 62)
    print("A. 전체 쌍 -- 쉬운 시험. 통과해도 별 뜻 없다 (K 효과가 들어 있다)")
    for k in STATS:
        tau, n = kendall(R[k], R.damage)
        print(f"   {k:9s} 타우 {tau:+.3f}  (쌍 {n})")

    print("\n" + "=" * 62)
    print("B. 같은 K 안에서 태스크끼리 -- **본 시험**. K 효과가 빠져 있다")
    for k in STATS:
        out, tot_c, tot_d = [], 0, 0
        for K in KS:
            sub = R[R.K == K]
            if len(sub) < 8:
                continue
            tau, _ = kendall(sub[k], sub.damage)
            out.append(f"K{K}:{tau:+.2f}")
            # 칸끼리 합쳐 하나로도 본다 -- 칸마다 n=24 라 하나씩은 약하다
            x, y = sub[k].to_numpy(), sub.damage.to_numpy()
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    s = (x[i]-x[j]) * (y[i]-y[j])
                    tot_c += s > 0
                    tot_d += s < 0
        tau_all = (tot_c - tot_d) / max(1, tot_c + tot_d)
        se = (2 / (tot_c + tot_d)) ** 0.5 if tot_c + tot_d else 0
        print(f"   {k:9s} 합쳐서 타우 {tau_all:+.3f} ({abs(tau_all)/se:.1f}SE)"
              f"   칸별 " + " ".join(out))

    print("\n" + "=" * 62)
    print("C. 상한을 맞히는가 -- 하나 빼고 맞추기")
    tasks = sorted(R.task.unique())
    rungs = sorted(set(KS) | {1.0})
    # 바닥선: 아무것도 안 보고 제일 흔한 상한
    from collections import Counter
    for k in STATS:
        exact = within = 0
        for t in tasks:
            other = [x for x in tasks if x != t]
            # 그 태스크를 뺀 나머지로 θ 를 고른다
            best, best_hit = None, -1
            for th in np.percentile(R[k], np.arange(5, 100, 5)):
                hit = 0
                for o in other:
                    s = R[R.task == o].set_index("K")[k]
                    ok = [K for K in KS if K in s.index and s[K] <= th]
                    pred = max(ok) if ok else 1.0
                    hit += abs(pred - hi.get(o, 0)) < 1e-6
                if hit > best_hit:
                    best_hit, best = hit, th
            s = R[R.task == t].set_index("K")[k]
            ok = [K for K in KS if K in s.index and s[K] <= best]
            pred = max(ok) if ok else 1.0
            exact += abs(pred - hi.get(t, 0)) < 1e-6
            i1 = rungs.index(min(rungs, key=lambda r: abs(r - pred)))
            i2 = rungs.index(min(rungs, key=lambda r: abs(r - hi.get(t, 1.0))))
            within += abs(i1 - i2) <= 1
        print(f"   {k:9s} 정확 {exact:2d}/{len(tasks)} = {exact/len(tasks):5.1%}"
              f"   한 칸 안 {within:2d}/{len(tasks)} = {within/len(tasks):5.1%}")
    mode = Counter(hi.get(t, 1.0) for t in tasks).most_common(1)[0]
    base_exact = mode[1] / len(tasks)
    bw = sum(abs(rungs.index(min(rungs, key=lambda r: abs(r-mode[0])))
                 - rungs.index(min(rungs, key=lambda r: abs(r-hi.get(t,1.0))))) <= 1
             for t in tasks) / len(tasks)
    print(f"   {'바닥선':9s} 정확 {mode[1]:2d}/{len(tasks)} = {base_exact:5.1%}"
          f"   한 칸 안 {bw*len(tasks):.0f}/{len(tasks)} = {bw:5.1%}"
          f"   (제일 흔한 상한 {mode[0]} 만 찍기)")
    print("\n   **바닥선을 못 넘으면 현오차는 아무 일도 안 한 것이다.**")
    print("   통계 다섯 중 최고를 고른 값이라 낙관적이다 -- 차이가 작으면 잡음이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
