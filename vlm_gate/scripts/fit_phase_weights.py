"""문항 가중을 **국면을 얼마나 가르는지**에 비례해 정한다. 반으로 나눠 검증한다.

가중을 짐작으로 두면 실제로 아무것도 안 가르는 문항이 큰 무게를 쥔다 -- B 는 국면 간
표준편차가 0.12 인데 감점의 40% 를 쥐고 있었다. 각 문항이 국면 사이에서 실제로 흔들리는
폭(국면 평균들의 표준편차)에 비례시키면 그 문제가 사라진다.

이 규칙은 **목표(태스크 유지율)를 보지 않는다** -- 라벨 자체에서 계산된다. 다만 같은
장면에서 정하고 채점하면 순환이므로, 에피소드를 반으로 갈라 한쪽에서 가중을 정하고
다른 쪽에서 채점한다.

    python fit_phase_weights.py <ph_*.jsonl>
"""
import json
import os
import re
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ratio_label import confidence                       # noqa: E402

SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}
BASE_W = {"A": .60, "B": .40, "C": .50, "D": .15, "E": .35}


def per_task(run):
    o = {}
    for p in glob.glob(f"output/robocasa/{run}/*/prediction.txt"):
        t = os.path.basename(os.path.dirname(p))
        s = [m.group(1) == "True" for m in
             (re.search(r"is_success:\s*\[\s*(True|False)\s*\]", l) for l in open(p)) if m]
        if len(s) >= 20:
            o[t] = float(np.mean(s))
    return o


def weights_from(d):
    """국면 평균들의 표준편차에 비례. 감점 풀과 가점 풀에서 각각 1 로 정규화."""
    eg = pd.DataFrame(d.eg.tolist(), columns=list("ABCDE"))
    eg["ph"] = d.ph.to_numpy()
    sd = eg.groupby("ph")[list("ABCDE")].mean().std()
    out = {}
    for pool in ("AB", "CDE"):
        s = sum(float(sd[q]) for q in pool)
        out.update({q: float(sd[q]) / s for q in pool})
    return out, {q: round(float(sd[q]), 3) for q in "ABCDE"}


def score(d, W, keep):
    cs = []
    for r in d.itertuples():
        rec = {q: int(v) for q, v in zip("ABCDE", r.picks)}
        rec["gp"] = r.gp
        cs.append(confidence(rec, SIGN, W, 5))
    y = pd.Series(cs)
    tm = y.groupby(d.task.to_numpy()).transform("mean")
    res = y - tm
    pm = res.groupby(d.ph.to_numpy()).transform("mean")
    m = y.groupby(d.task.to_numpy()).mean()
    rr = []
    for k in (2, 3):
        T = [t for t in m.index if t in keep[k]]
        rr.append(spearmanr([m[t] for t in T], [keep[k][t] for t in T]))
    return dict(phase=np.var(pm) / np.var(res), rest=np.var(res - pm) / np.var(y),
                task=np.var(tm) / np.var(y),
                gap=y[d.ph == "파지"].mean() - y[d.ph == "접근"].mean(),
                k2=rr[0][0], p2=rr[0][1], k3=rr[1][0], p3=rr[1][1], mean=y.mean())


def show(tag, s):
    print(f"  {tag:<22} 국면몫 {s['phase']:.3f} · 남는몫 {s['rest']:.3f} · 태스크몫 {s['task']:.3f}"
          f" · 파지-접근 {s['gap']:+.4f} · K2 {s['k2']:+.3f}(p{s['p2']:.3f})"
          f" · K3 {s['k3']:+.3f}(p{s['p3']:.3f})")


def main():
    base = per_task("baseline_full_v2_with_action_steps")
    keep = {}
    for k in (2, 3):
        c = per_task(f"baseline_compress_K{k}")
        keep[k] = {t: c[t] / base[t] for t in base if t in c and base[t] > 0.05}
    sc = {(s["ep"], s["f"]): s["phase"]
          for s in json.load(open("_tmp/selfagg_bias/scenes_phase.json"))}
    d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1])]).drop_duplicates(["ep", "f"])
    d = d[d.conf_exp.notna()].reset_index(drop=True)
    d["ph"] = [sc.get((int(r.ep), int(r.f))) for r in d.itertuples()]
    print(f"{os.path.basename(sys.argv[1])} · 장면 {len(d)} · 에피 {d.ep.nunique()}")

    W, sd = weights_from(d)
    print(f"\n국면 간 표준편차 {sd}")
    print("가중  " + "  ".join(f"{q}={W[q]:.3f}" for q in "ABCDE"))
    print("\n[전체 762 로 정하고 전체로 채점 -- 순환이므로 참고]")
    show("기존 가중", score(d, BASE_W, keep))
    show("국면 비례 가중", score(d, W, keep))

    # 에피소드를 반으로 갈라: 한쪽에서 가중, 다른 쪽에서 채점
    eps = np.array(sorted(d.ep.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    half = set(eps[:len(eps) // 2])
    a = d[d.ep.isin(half)].reset_index(drop=True)
    b = d[~d.ep.isin(half)].reset_index(drop=True)
    print(f"\n[분할 검증] 가중은 에피 {a.ep.nunique()}개({len(a)}장면)에서, 채점은 "
          f"에피 {b.ep.nunique()}개({len(b)}장면)에서")
    Wa, _ = weights_from(a)
    print("  정한 가중  " + "  ".join(f"{q}={Wa[q]:.3f}" for q in "ABCDE"))
    show("기존 가중(홀드아웃)", score(b, BASE_W, keep))
    show("국면 비례(홀드아웃)", score(b, Wa, keep))


if __name__ == "__main__":
    main()
