"""**얼마나 질러가면 성공률이 얼마나 떨어지나.** 관계가 있냐 없냐가 아니라 모양.

    python vlm_gate/scripts/damage_shape.py robocasa <chord_error.parquet>

`damage_fit.py` 는 상관계수만 냈다. 그것은 관계가 있냐 없냐만 말하고 **얼마나
질러가면 얼마나 떨어지는지는 안 알려준다.** 판단하려면 모양을 봐야 한다.

세로가 질러간 양, 가로가 배속, 칸이 그때 성공률이 얼마나 떨어졌나다.

**세로로 읽는 것이 본론이다.** 같은 배속에서 아래로 내려갈수록(더 많이
질러갈수록) 손상이 커져야 이 값으로 배속을 정할 수 있다. 가로로 커지는 것은
당연한 것이라 아무 말도 안 해 준다 -- 배속을 올리면 나빠진다는 뜻일 뿐이다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    bench, chord = sys.argv[1:3]
    L = json.load(open(os.path.join(HERE, "..", "analysis", "eval_results",
                                    "LADDERS.json")))["ladders"][f"{bench}/uniform"]
    succ = {}
    for r in L["rungs"]:
        for t, v in r["tasks"].items():
            succ.setdefault(t, {})[float(r["ratio"])] = v["success"]

    df = pd.read_parquet(chord)
    KS = sorted(df.K.unique())

    rec = []
    for (t, K), g in df.groupby(["task", "K"]):
        if t not in succ or K not in succ[t] or succ[t].get(1.0, 0) <= 0:
            continue
        b = succ[t][1.0]
        rec.append({"task": t, "K": K, "cut": float(np.median(g["pos"])),
                    "drop": (b - succ[t][K]) / b, "base": b})
    R = pd.DataFrame(rec)

    # 구간은 전체 분위로 나눈다. 값을 지어낸 자리에서 자르지 않는다.
    qs = np.percentile(R.cut, [0, 20, 40, 60, 80, 100])
    qs[-1] += 1e-9
    print(f"쌍 {len(R)}개 · 태스크 {R.task.nunique()} · 배속 {KS}")
    print(f"질러간 양 구간(전체 5분위) {' '.join(f'{q:.3f}' for q in qs)}\n")

    print("        질러간 양      " + "".join(f"{f'{K}배속':>12s}" for K in KS))
    print("        (중앙값)       " + "".join(f"{'손상   (n)':>12s}" for K in KS))
    print("  " + "-" * (16 + 12 * len(KS)))
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        line = f"  {lo:.3f}~{hi:.3f}  "
        for K in KS:
            s = R[(R.K == K) & (R.cut >= lo) & (R.cut < hi)]
            line += (f"{s.drop.mean():+8.1%}({len(s):2d})" if len(s)
                     else f"{'-':>12s}")
        print(line)

    print("\n  ** 세로로 읽는다.** 같은 배속 안에서 아래로 내려갈수록 손상이")
    print("  커져야 이 값으로 배속을 정할 수 있다. 가로로 커지는 것은 그냥")
    print("  '배속 올리면 나빠진다' 라 아무 말도 안 해 준다.\n")

    # 세로 기울기를 숫자로도. 태스크당 50에피라 손상 하나의 표준오차가 크다.
    print("  배속별 세로 기울기 (제일 많이 질러간 5분위 - 제일 적게 질러간 5분위)")
    for K in KS:
        s = R[R.K == K]
        if len(s) < 10:
            continue
        top = s[s.cut >= np.percentile(s.cut, 80)].drop
        bot = s[s.cut <= np.percentile(s.cut, 20)].drop
        # 태스크당 50에피 -> 성공률 표준오차 약 0.07, 상대손상은 기준선으로 나눔
        se = float(np.sqrt(top.var() / max(1, len(top))
                           + bot.var() / max(1, len(bot))))
        d = float(top.mean() - bot.mean())
        print(f"    {K}배속   {d:+7.1%}   ({d/se:.1f}SE, 위 {len(top)}개 "
              f"아래 {len(bot)}개)")
    print("\n  0 에 가까우면 그 배속에서는 질러간 양이 손상을 안 가른다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
