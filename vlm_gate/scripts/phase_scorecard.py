"""국면균등 장면에서 문항 판을 채점한다. **프레임별 판단력이 기준이다.**

태스크당 유지율 24개에 맞추는 목표는 라벨을 태스크 상수로 몰고 간다(상관 +0.552 일 때
태스크 몫 0.984). 그래서 이 채점표의 주 지표는 국면 쪽이고, 태스크 상관은 참고로 둔다.

    python phase_scorecard.py <ph_*.jsonl> [...]

각 파일은 gp_conf_check.py 산출물이고 장면 집합은 scenes_phase.json (국면별 균등)이다.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robocasa_descriptors import descriptors           # noqa: E402
from judge_ab import phase_of, vert_of                 # noqa: E402

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)
FINE = ("접근", "파지", "해제", "전이(둘다)", "운반", "놓기")
# 손으로 분류한 "좁은 자리" -- B 가 그것을 보는지 재는 기준
TIGHT = {"PnPCabToCounter", "PnPMicrowaveToCounter", "PnPStoveToCounter",
         "PnPSinkToCounter", "CoffeeServeMug", "OpenDrawer", "OpenSingleDoor",
         "OpenDoubleDoor", "CoffeeSetupMug"}
_A = {}


def get(ep):
    if ep not in _A:
        if len(_A) > 40:
            _A.clear()
        _A[ep] = np.stack(pd.read_parquet(
            f"{DS}/data/chunk-{ep // CH:03d}/"
            f"episode_{ep:06d}.parquet")["action"].values)
    return _A[ep]


def ph6(ep, f):
    a = get(ep)
    x = descriptors(a, f)
    x["vert"] = vert_of(a, f)
    p = phase_of(x)
    if p == "전이":
        p = ("파지" if (x["grip_close"] and not x["grip_open"]) else
             "해제" if (x["grip_open"] and not x["grip_close"]) else "전이(둘다)")
    return p


def per_task(run):
    o = {}
    for p in glob.glob(f"output/robocasa/{run}/*/prediction.txt"):
        t = os.path.basename(os.path.dirname(p))
        s = [m.group(1) == "True" for m in
             (re.search(r"is_success:\s*\[\s*(True|False)\s*\]", l) for l in open(p)) if m]
        if len(s) >= 20:
            o[t] = float(np.mean(s))
    return o


def main():
    base = per_task("baseline_full_v2_with_action_steps")
    keep = {}
    for k in (2, 3):
        c = per_task(f"baseline_compress_K{k}")
        keep[k] = {t: c[t] / base[t] for t in base if t in c and base[t] > 0.05}

    for path in sys.argv[1:]:
        nm = os.path.basename(path).replace("ph_", "").replace(".jsonl", "")
        d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates(["ep", "f"])
        d = d[d.conf_exp.notna()].reset_index(drop=True)
        d["c6"] = [ph6(int(r.ep), int(r.f)) for r in d.itertuples()]
        eg = pd.DataFrame(d.eg.tolist(), columns=list("ABCDE"))
        d = pd.concat([d, eg], axis=1)
        y = d.conf_exp.astype(float)
        tm = d.groupby("task").conf_exp.transform("mean").astype(float)
        res = y - tm
        pm = res.groupby(d.c6).transform("mean")
        gr = [res[d.c6 == p].to_numpy() for p in FINE if (d.c6 == p).sum() >= 5]
        H, pv = kruskal(*gr)
        m = d.groupby("task").conf_exp.mean()
        rr = []
        for k in (2, 3):
            T = [t for t in m.index if t in keep[k]]
            rr.append(spearmanr([m[t] for t in T], [keep[k][t] for t in T]))
        print(f"\n{'='*72}\n[{nm}]  장면 {len(d)} · 태스크 {d.task.nunique()}")
        print(f"  태스크몫 {np.var(tm)/np.var(y):.3f} · 국면몫 {np.var(pm)/np.var(res):.3f}"
              f" · 남는몫 {np.var(res-pm)/np.var(y):.3f} · p(국면) {pv:.5f}")
        print(f"  K2 {rr[0][0]:+.3f}(p{rr[0][1]:.3f}) · K3 {rr[1][0]:+.3f}(p{rr[1][1]:.3f})"
              f" · conf 평균 {y.mean():.3f}")
        print("  국면별 conf: " + "  ".join(
            f"{p}={y[d.c6==p].mean():.3f}(n{(d.c6==p).sum()})"
            for p in FINE if (d.c6 == p).sum() > 0))
        gp = y[d.c6 == "파지"]
        ga = y[d.c6 == "접근"]
        print(f"  파지-접근 {gp.mean()-ga.mean():+.4f}  (파지 n{len(gp)} 접근 n{len(ga)})")

        # 문항별: 상위6 대 하위6 유지율 태스크에서의 등급 차
        tm2 = d.groupby("task")[list("ABCDE")].mean()
        tm2["keep"] = [keep[2].get(t, np.nan) for t in tm2.index]
        tm2 = tm2.dropna(subset=["keep"]).sort_values("keep")
        lo, hi = tm2.head(6), tm2.tail(6)
        print("  문항 상위6-하위6 등급차 (가점은 +, 감점은 - 여야 옳다):")
        for q, sgn in (("A", -1), ("B", -1), ("C", 1), ("D", 1), ("E", 1)):
            dv = hi[q].mean() - lo[q].mean()
            ok = "옳음" if (dv > 0) == (sgn > 0) and abs(dv) > 0.15 else \
                 ("반대" if (dv > 0) != (sgn > 0) and abs(dv) > 0.15 else "무신호")
            print(f"    {q}) {'감점' if sgn<0 else '가점'} {dv:+.2f}  {ok}")
        # B 가 "좁은 자리" 를 보는가
        tb = d.groupby("task").B.mean()
        ref = pd.Series({t: (1 if t in TIGHT else 0) for t in tb.index})
        r, p = spearmanr(tb, ref[tb.index])
        print(f"  B) vs 손분류 좁은자리  {r:+.3f} (p={p:.4f})"
              f" · 좁은 {tb[ref[tb.index]==1].mean():.2f} 대 아닌 {tb[ref[tb.index]==0].mean():.2f}")


if __name__ == "__main__":
    main()
