"""배속 게이트 판정. 같은 배속에서 균등보다 성공률이 높은가.

**스텝으로 보간하면 안 된다.** 균등 사다리의 성공-스텝 곡선은 스텝에 대해
단조가 아니다 -- robocasa OpenDoubleDoor 에서 2.67배가 393스텝인데 4.0배가
464스텝이다. 고압축에서 어느 에피가 살아남았느냐에 따라 성공 에피 평균 스텝이
뒤집힌다. 스텝 기준 보간은 그래서 1라운드에 +0.62 라는 없는 이득을 냈다.

**배속으로 보간한다.** 클라이언트가 `realised_ratio` 로 실현 압축비를 적으므로
이름값이 아니라 그 값을 쓴다.

    python ratio_gate_verdict.py <게이트출력디렉터리> [균등.parquet]
"""
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p):
    o = []
    for l in open(p):
        m = re.match(r"episode (\d+) is_success: \[\s*(True|False)\]"
                     r"\s*action_steps:\s*(\d+)", l)
        if m:
            o.append((int(m.group(1)), m.group(2) == "True", int(m.group(3))))
    return pd.DataFrame(o, columns=["ep", "succ", "steps"])


def footer(p):
    o = {}
    for l in open(p):
        if l.startswith("episode"):
            continue
        m = re.match(r"(\w+):\s*([\d.]+)", l)
        if m:
            try:
                o[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return o


def main():
    gate = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/output/robocasa/ratio_gate_r2"
    uni = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/analysis/ratio_prompt/uniform_ep0_19_all.parquet"
    u = pd.read_parquet(uni)
    won = tie = lost = 0
    print(f"{'태스크':20s} {'실현배속':>7s} {'게이트성공':>8s} {'균등기대':>8s} "
          f"{'차':>6s}  {'무압축대비 스텝':>14s}")
    for d in sorted(os.listdir(gate)):
        p = f"{gate}/{d}/prediction.txt"
        if not os.path.exists(p):
            continue
        g = read(p)
        if len(g) < 5:
            print(f"{d:20s} 에피 {len(g)}개뿐 -- 건너뜀"); continue
        f = footer(p)
        rr = f.get("realised_ratio")
        if rr is None:
            print(f"{d:20s} realised_ratio 없음 -- 건너뜀"); continue
        uu = u[u.task == d]
        if not len(uu):
            print(f"{d:20s} 균등 자료 없음"); continue
        agg = uu.groupby("ratio").apply(
            lambda x: pd.Series({"succ": x.succ.mean(),
                                 "steps": x.steps[x.succ].mean() if x.succ.any() else np.nan}),
            include_groups=False).sort_index()
        exp = float(np.interp(rr, agg.index.to_numpy(), agg.succ.to_numpy()))
        gs = g.succ.mean()
        gt = g.steps[g.succ].mean() if g.succ.any() else np.nan
        b = agg.loc[1.0] if 1.0 in agg.index else None
        cut = f"{100*(1-gt/b.steps):.0f}%" if b is not None else "-"
        diff = gs - exp
        mark = "이김" if diff > 0.02 else ("짐" if diff < -0.02 else "동률")
        won += diff > 0.02; tie += abs(diff) <= 0.02; lost += diff < -0.02
        print(f"{d:20s} {rr:7.2f} {gs:8.2f} {exp:8.2f} {diff:+6.2f}  "
              f"{cut:>6s} 단축, 성공 {b.succ:.2f}->{gs:.2f}" if b is not None else "")
    print(f"\n이김 {won} · 동률 {tie} · 짐 {lost}")
    print("주의: 태스크당 20에피다. 0.05 차이는 1에피다.")


if __name__ == "__main__":
    main()
