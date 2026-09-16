"""robocasa v7 등급 위에서 부호·가중을 실측 사다리에 재적합한다. 등급은 안 건드린다.

  python scripts/robocasa_refit_weights.py            # 적합 + 검증
  python scripts/robocasa_refit_weights.py --emit OUT # conf 를 다시 계산해 내보낸다

검증은 태스크 하나 빼기(LOO)다. 가중 5개로 20점을 외우는 것을 잡아낸다. 적합에 쓰지
않은 정답(천장 표 · K3 유지율)에도 대 본다 -- 거기서도 양수면 외삽이 되는 것이다.

`analysis/robocasa_ladder/GROUND_TRUTH_TRAPS.md` 의 규칙을 따른다: 천장 lo 는 상수라
쓰지 않고, 1x 성공률 0.35 미만 태스크는 뺀다.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")
Q = "ABCDE"
NGRADE = 5
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
LABELS = f"{BASE}/output/_gate_distill/phase9_full/labels_merged.jsonl"


def task_of():
    m = {}
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(l)
        c = [t for t in d["tasks"]
             if isinstance(t, str) and len(t.split()) == 1 and t != "Valid"]
        if c:
            m[d["episode_index"]] = c[0]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", help="conf 를 다시 계산해 이 파일로 쓴다")
    a = ap.parse_args()
    ep2t = task_of()
    nq = pd.read_csv(f"{BASE}/analysis/robocasa_ladder/nq_noclip.csv").set_index("task")
    lad = (pd.read_parquet(f"{BASE}/analysis/robocasa_ladder/episode_target.parquet")
           .groupby("task").max_ratio.mean().to_dict())
    ceil = json.load(open(f"{BASE}/analysis/robocasa_task_ceilings.json"))

    acc = {}
    for l in open(LABELS):
        r = json.loads(l)
        if not all(isinstance(r.get(q), int) for q in Q):
            continue
        t = ep2t.get(r["ep"])
        if t is None or t not in nq.index or t not in lad:
            continue
        s = acc.setdefault(t, [np.zeros(5), 0])
        s[0] += [r[q] for q in Q]
        s[1] += 1
    ts = [t for t in sorted(acc) if nq.allfine_succ[t] / 100 >= 0.35]
    g = (np.array([acc[t][0] / acc[t][1] for t in ts]) - 1) / (NGRADE - 1)
    y = np.array([lad[t] for t in ts])
    b = np.linalg.lstsq(np.c_[g, np.ones(len(ts))], y, rcond=None)[0][:5]
    sign = {q: (1 if v > 0 else -1) for q, v in zip(Q, b)}
    mag = {q: abs(v) for q, v in zip(Q, b)}
    w = {}
    for s in (+1, -1):                       # 각 변의 합이 1 (기존 규약)
        side = [q for q in Q if sign[q] == s]
        tot = sum(mag[q] for q in side)
        for q in side:
            w[q] = mag[q] / tot

    loo = np.array([g[i] @ np.linalg.lstsq(np.c_[g[j], np.ones(len(j))], y[j],
                                           rcond=None)[0][:5]
                    for i in range(len(ts))
                    for j in [np.array([k for k in range(len(ts)) if k != i])]])
    score = g @ np.array([sign[q] * w[q] for q in Q])
    hi = np.array([ceil[t][1] for t in ts if t in ceil])
    k3 = np.array([nq.k3_succ[t] / nq.allfine_succ[t] for t in ts])
    print(f"태스크 {len(ts)} · 청크 {sum(acc[t][1] for t in ts):,}")
    print(f"  사다리   전체적합 {spearmanr(score, y).statistic:+.3f} "
          f"· LOO {spearmanr(loo, y).statistic:+.3f}")
    print(f"  천장 hi  {spearmanr(score[:len(hi)], hi).statistic:+.3f}   (적합에 안 쓴 정답)")
    print(f"  K3 유지  {spearmanr(score, k3).statistic:+.3f}   (적합에 안 쓴 정답)")
    print(f"  SIGN   {sign}")
    print(f"  WEIGHT {({k: round(v, 3) for k, v in w.items()})}")

    out = f"{BASE}/analysis/robocasa_ladder/refit_weights.json"
    json.dump({"sign": sign, "weight": w, "tasks": ts,
               "rho_full": float(spearmanr(score, y).statistic),
               "rho_loo": float(spearmanr(loo, y).statistic)},
              open(out, "w"), indent=1)
    print(f"  -> {out}")

    if a.emit:
        n = 0
        with open(a.emit, "w") as fh:
            for l in open(LABELS):
                r = json.loads(l)
                if not all(isinstance(r.get(q), int) for q in Q):
                    continue
                r["conf"] = 0.5 * (1 + sum(sign[q] * w[q] * ((r[q] - 1) / (NGRADE - 1))
                                           for q in Q))
                fh.write(json.dumps(r) + "\n")
                n += 1
        print(f"  conf 재계산 {n:,} -> {a.emit}")


if __name__ == "__main__":
    main()
