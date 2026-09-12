"""라벨이 **태스크 상수로 갈음되는가.** 분산 몫이 아니라 정확도로 답한다.

분산 몫(태스크 평균이 설명하는 몫)은 해석이 어렵다. 여기서 내는 것은
"모든 프레임의 라벨을 그 태스크의 최빈값 하나로 바꿨을 때 몇 %가 그대로인가" 다.
그 값이 높으면 프레임별로 라벨링할 이유가 없고 24행 표로 끝난다.

문항을 국면 기준으로 써 두어도 이 값이 높게 나올 수 있다 -- 판정기가 받는 그림이
t 한 순간의 세 뷰뿐이면 국면을 볼 근거가 이미지에 없어서, 모델이 태스크 정체성으로
떨어진다. PAIR=1 (t 와 t+16 을 함께) 판과 견주려고 만든 자리다.

    python label_task_collapse.py <라벨.jsonl> [...]

같은 라벨을 여러 파일로 나눠 돌린 경우 여러 개를 주면 합친다. 파일 이름이 라벨
이름으로 쓰이므로, 다른 판을 함께 보려면 따로 호출한다.
"""
import json
import sys

import numpy as np
import pandas as pd

BIN_TH = 4          # Z>=4 를 "압축 허용" 으로 본다


def load(paths):
    rs = []
    for p in paths:
        for line in open(p):
            try:
                rs.append(json.loads(line))
            except Exception:
                pass
    d = pd.DataFrame(rs).drop_duplicates(subset=["ep", "f"], keep="last")
    return d[d.Z.notna()].copy()


def main():
    d = load(sys.argv[1:])
    d["Z"] = d.Z.astype(int)
    n_t = d.task.nunique()
    print(f"장면 {len(d)} · 태스크 {n_t} · 장면/태스크 "
          f"{d.groupby('task').size().min()}~{d.groupby('task').size().max()}\n")

    mode = d.groupby("task").Z.agg(lambda s: s.mode().iloc[0])
    acc5 = float((d.Z.to_numpy() == d.task.map(mode).to_numpy()).mean())
    b = (d.Z >= BIN_TH).astype(int)
    maj = b.groupby(d.task).agg(lambda s: int(s.mean() >= 0.5))
    accb = float((b.to_numpy() == d.task.map(maj).to_numpy()).mean())
    y = d.Z.astype(float)
    share = float(np.var(d.groupby("task").Z.transform("mean").astype(float)) / np.var(y))
    zero5 = int((d.groupby("task").Z.std() == 0).sum())
    zerob = int((b.groupby(d.task).nunique() == 1).sum())

    print("태스크 최빈값 하나로 전부 갈음했을 때")
    print(f"  5등급 그대로        {acc5:6.1%}")
    print(f"  Z>={BIN_TH} 이진        {accb:6.1%}"
          f"   <- 프레임별 라벨링의 값어치는 이 값이 낮을 때 생긴다")
    print(f"  태스크가 정하는 분산 몫 {share:6.3f}")
    print(f"  전 장면 같은 등급인 태스크  {zero5}/{n_t}")
    print(f"  전 장면 같은 이진 답인 태스크 {zerob}/{n_t}")

    print(f"\n태스크별 이진(Z>={BIN_TH}) 쏠림 -- 100% 면 그 태스크는 상수다")
    g = pd.DataFrame({"task": d.task.to_numpy(), "b": b.to_numpy()}
                     ).groupby("task").b.agg(["mean", "size"])
    g["skew"] = g["mean"].apply(lambda p: max(p, 1 - p))
    for t, r in g.sort_values("skew", ascending=False).iterrows():
        side = "압축" if r["mean"] >= .5 else "불가"
        print(f"  {r['skew']:>6.0%} {side:<4} {'#' * int(30 * r['skew']):<30} {t}")


if __name__ == "__main__":
    main()
