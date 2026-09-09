"""문항이 **시점을 가르는가, 태스크를 부르는가** — 저장된 등급만으로 판정한다.

    python vlm_gate/scripts/question_audit.py robocasa <labels_ratio.parquet>

VLM 을 다시 안 부른다. 등급이 이미 parquet 에 있고, 여기서 묻는 둘은 등급만
있으면 답이 나온다. 몇 분이면 끝난다.

## 무엇을 재나

**1. 태스크 안에서 등급이 움직이는가 (`within_frac`)**

    within_frac = 태스크 안 분산 / 전체 분산

시점마다 배속을 정하려고 만든 문항인데 한 태스크 안에서 등급이 상수면, 그
문항은 "지금 무엇을 하는가" 가 아니라 "이게 어느 태스크인가" 를 답한 것이다.
그러면 상한표와 같은 사실을 두 번 세는 것이고(CLAUDE.md 4번), 신뢰도에 새로
들어오는 정보가 없다. **0.2 아래면 그 문항은 태스크 이름이다.**

**2. 태스크 평균 등급이 실측 상한과 같은 방향인가 (`tau`)**

감점 문항은 등급이 높을수록 상한이 낮아야 하고(음의 상관), 가점 문항은 그
반대다. 부호가 어긋나면 그 문항은 **거꾸로 달려 있다** -- 가중치가 클수록
신뢰도를 더 크게 망친다.

태스크 24개짜리 상관이라 약하다. 그래서 이건 **판정이 아니라 경보**다.
어긋난 문항을 시점 단위로 다시 보라는 뜻이지, 이것만으로 부호를 뒤집지 않는다.

**그리고 타우는 한 갈래에 먹힌다.** robocasa 는 PnP 계열 상한이 낮고(중앙 1.5)
Turn 계열이 높다(중앙 2.5). 그래서 PnP 에서 뜨는 문항은 무엇이든 타우가 음수로,
Turn 에서 뜨는 문항은 무엇이든 양수로 나온다 -- 문항이 맞고 틀리고와 무관하다.
D("넓은 상판에 내려놓기")와 E("빈 공간을 가로질러 나른다")가 음수인 것은 둘 다
PnP 에서 뜨기 때문이지 거꾸로 달려서가 아닐 수 있다.

**타우를 부호 근거로 쓸 수 있는 것은 그 문항이 태스크 단위일 때뿐이다.** 그때는
문항이 곧 태스크 갈래라서 둘이 같은 것을 말한다. 시점 단위 문항이면 타우는
"이 문항이 어느 갈래에서 뜨나" 를 잰 것이고, 그 갈래의 상한은 이미 상한표에
있다. 그러므로 **분산 몫이 낮은 문항의 타우만 부호 근거가 된다.**

## 못 재는 것

**시점 단위 정답이 없다.** 사다리는 태스크 단위이고, 같은 태스크 안에서 어느
순간이 위험한지는 아직 아무도 안 쟀다. 그래서 여기서 나오는 것은 "이 문항이
쓸모 있는가" 가 아니라 **"이 문항이 쓸모 있을 수 있는가"** 다. 1번을 통과 못
하면 시점 실험을 할 필요도 없고, 통과해도 시점 실험은 따로 해야 한다.
"""
import json
import sys

import numpy as np
import pandas as pd

SLOTS = "ABCDE"


def kendall(x, y):
    """켄달 타우. 태스크 24개라 피어슨보다 이쪽이 눈금 가정을 덜 한다."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = (x[i] - x[j]), (y[i] - y[j])
            if a * b > 0:
                c += 1
            elif a * b < 0:
                d += 1
    return (c - d) / max(1, c + d)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    bench, lab = sys.argv[1:3]
    # `ratio_label.CHECKS` 와 같은 표를 쓴다 -- 부호·가중치의 사본을 여기 또
    # 두면 한쪽만 고쳐 놓고 두 값을 비교하게 된다.
    from ratio_label import CHECKS
    mod = __import__(CHECKS[bench], fromlist=["SIGN", "WEIGHT"])
    SIGN, WEIGHT = mod.SIGN, mod.WEIGHT

    # 있는 열만 읽는다. phase9 라벨에는 등급 분포(eg_*)가 없어서, 무조건
    # 요구하면 폴백에 닿기 전에 죽는다.
    import pyarrow.parquet as _pq
    have = set(_pq.ParquetFile(lab).schema_arrow.names)
    cols = [c for c in ["task"] + list(SLOTS) + [f"eg_{k}" for k in SLOTS]
            if c in have]
    df = pd.read_parquet(lab, columns=cols)
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ceil = json.load(open(os.path.join(here, "..", "analysis",
                                       f"{bench}_task_ceilings.json")))
    hi = {t: (v[1] if isinstance(v, list) else v) for t, v in ceil.items()}

    print(f"행 {len(df):,} · 태스크 {df.task.nunique()}\n")
    print(f"{'문항':4s} {'부호':4s} {'가중':>5s} {'태스크안 분산몫':>14s} "
          f"{'상한과의 타우':>12s}  {'등급 분포 1..5':>22s}  판정")
    out = {}
    for k in SLOTS:
        col = f"eg_{k}" if f"eg_{k}" in df.columns else k
        g = df[col].to_numpy(float)
        # 태스크 안 분산 몫
        tm = df.groupby("task")[col].transform("mean").to_numpy(float)
        within = float(np.var(g - tm))
        total = float(np.var(g)) or 1e-12
        wf = within / total
        # 태스크 평균 등급 대 상한
        tt = df.groupby("task")[col].mean()
        common = [t for t in tt.index if t in hi]
        tau = kendall([tt[t] for t in common], [hi[t] for t in common])
        # 정수 등급 분포
        ig = df[k].to_numpy(int)
        dist = np.bincount(ig, minlength=6)[1:6] / len(ig)

        want = -1 if SIGN[k] < 0 else 1        # 감점이면 상한과 음의 상관이어야
        flag = []
        if wf < 0.20:
            flag.append("태스크 이름")
        if tau * want < 0 and abs(tau) > 0.15:
            # 태스크 단위 문항일 때만 부호 근거가 된다. 시점 단위 문항의 타우는
            # 그 문항이 어느 태스크 갈래에서 뜨는지를 잰 것이라 부호와 무관하다.
            flag.append("부호 거꾸로" if wf < 0.20 else "타우 어긋남(갈래 탓일 수 있음)")
        if (dist[1] + dist[3]) < 0.05:
            flag.append("2·4등급 죽음")
        out[k] = {"within_frac": round(wf, 3), "tau_vs_ceiling": round(tau, 3),
                  "dist": [round(float(v), 3) for v in dist], "flags": flag}
        print(f"{k:4s} {'감점' if SIGN[k]<0 else '가점':4s} {WEIGHT[k]:5.3f} "
              f"{wf:14.3f} {tau:12.3f}  "
              + " ".join(f"{v:4.0%}" for v in dist)
              + "  " + (" · ".join(flag) if flag else "ok"))

    print()
    print("태스크안 분산몫 0.20 아래 = 그 문항은 시점이 아니라 태스크를 부른다.")
    print("타우 부호가 기대와 어긋나면 거꾸로 달린 것이다 -- 시점 단위로 다시 본다.")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
