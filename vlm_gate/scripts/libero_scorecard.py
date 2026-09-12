"""LIBERO 문항 판 채점표. robocasa phase_scorecard.py 의 LIBERO 판.

주 지표는 국면 쪽이다. 태스크당 유지율은 40개뿐이라 거기에 맞추면 라벨이 태스크
상수로 수렴한다(robocasa 에서 겪었다 -- 상관을 올리는 동안 태스크몫이 0.98 이 됐다).
태스크 상관은 **부호 확인용**으로만 본다.

유지율은 `analysis/libero_retention/README.md` 의 정본을 쓴다 -- clip 을 푼 런이고,
런 이름의 K 가 배속이 아니므로 실제 배속으로 읽은 것이다.

    python libero_scorecard.py <라벨.jsonl> [...]
"""
import csv
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
BASE = os.path.dirname(HERE)
SU = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PHASES = ("접근", "파지", "해제", "운반", "들어올림", "내려놓기")
RUNS = {
    "1.53x": ("dir", "/sjw_alinlab2/home/taekwan/Data/libnaive_clipoff_s7_K2_allk2"),
    "1.76x": ("csv", f"{BASE}/analysis/libero_retention/K2p5_clipoff.csv"),
    "1.80x": ("dir", "/sjw_alinlab2/home/taekwan/Isaac-GR00T/output/libero/"
                     "libnaive_clipoff_s7_K4_allk2"),
}
BASE_RUN = "/sjw_alinlab2/home/taekwan/Data/libero_all_Fine"


def from_dir(root):
    o = {}
    for s in SU:
        for p in sorted(glob.glob(f"{root}/{s}/*_results.txt")):
            t = f"{s}/{os.path.basename(p).split('_')[0]}"
            ok = [m.group(2) == "True" for m in
                  (re.match(r"\s*(\d+)\s+(True|False)\s+(\d+)", l) for l in open(p)) if m]
            if ok:
                o[t] = float(np.mean(ok))
    return o


def from_csv(p):
    return {f"{r['suite']}/{r['task_id']}": int(r["successes"]) / int(r["episodes"])
            for r in csv.DictReader(open(p))}


def keeps():
    b = from_dir(BASE_RUN)
    out = {}
    for k, (kind, p) in RUNS.items():
        d = from_dir(p) if kind == "dir" else from_csv(p)
        out[k] = {t: d[t] / b[t] for t in d if t in b and b[t] > 0.05}
    return out


def main():
    K = keeps()
    for path in sys.argv[1:]:
        nm = os.path.basename(path).replace(".jsonl", "")
        d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates(["ep", "f"])
        d = d[d.conf_exp.notna()].reset_index(drop=True)
        eg = pd.DataFrame([r if r else [np.nan] * 5 for r in d.eg],
                          columns=list("ABCDE"))
        pk = pd.DataFrame(d.picks.tolist(), columns=[q + "i" for q in "ABCDE"])
        d = pd.concat([d, eg, pk], axis=1)
        y = d.conf_exp.astype(float)
        tm = d.groupby("cell").conf_exp.transform("mean").astype(float)
        res = y - tm
        pm = res.groupby(d.phase).transform("mean")
        gr = [res[d.phase == p].to_numpy() for p in PHASES if (d.phase == p).sum() >= 5]
        H, pv = kruskal(*gr) if len(gr) >= 2 else (np.nan, np.nan)
        print(f"\n{'='*74}\n[{nm}]  장면 {len(d)} · 셀 {d.cell.nunique()}"
              f" · gp {d.eg.notna().mean():.0%}")
        print(f"  태스크몫 {np.var(tm)/np.var(y):.3f} · 국면몫 {np.var(pm)/np.var(res):.3f}"
              f" · 남는몫 {np.var(res-pm)/np.var(y):.3f} · p(국면) {pv:.5f}"
              f" · conf 평균 {y.mean():.3f}")
        m = d.groupby("cell").conf_exp.mean()
        cells = []
        for k, kk in K.items():
            T = [t for t in m.index if t in kk]
            r, p = spearmanr([m[t] for t in T], [kk[t] for t in T])
            cells.append(f"{k} {r:+.3f}(p{p:.3f})")
        print("  유지율 상관: " + " · ".join(cells) + "   (양수여야 옳다)")
        print("  국면별 conf: " + "  ".join(
            f"{p}={y[d.phase==p].mean():.3f}(n{(d.phase==p).sum()})"
            for p in PHASES if (d.phase == p).sum()))
        t = d.groupby("phase")[list("ABCDE")].mean()
        t = t.loc[[p for p in PHASES if p in t.index]]
        print(f"\n  {'국면':<8}{'n':>5}" + "".join(f"{q:>7}" for q in "ABCDE"))
        for p in t.index:
            print(f"  {p:<8}{(d.phase==p).sum():>5}"
                  + "".join(f"{t.loc[p,q]:>7.2f}" for q in "ABCDE"))
        print("  최고국면: " + "  ".join(f"{q}={t[q].idxmax()}" for q in "ABCDE"))
        print("  4등급이상: " + "  ".join(
            f"{q}={(d[q+'i']>=4).mean():.0%}" for q in "ABCDE"))
        print("  국면간 표준편차: " + "  ".join(f"{q}={t[q].std():.3f}" for q in "ABCDE"))


if __name__ == "__main__":
    main()
