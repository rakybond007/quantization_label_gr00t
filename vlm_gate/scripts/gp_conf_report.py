"""정수 등급 conf 와 등급 기댓값 conf 가 얼마나 다른가, 그리고 결론이 바뀌는가.

gp_conf_check.py 의 산출물을 받는다. 두 conf 가 사실상 같으면(상관 ~1, 순위 동일)
지금까지의 측정은 그대로 유효하고, 다르면 폐루프 상관을 새 conf 로 다시 읽어야 한다.

    python gp_conf_report.py <gp_conf.jsonl>
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
    d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1])])
    has = d.gp.notna()
    print(f"장면 {len(d)} · gp 있는 행 {has.sum()} ({has.mean():.1%})")
    if not has.any():
        raise SystemExit("gp 가 하나도 없다 -- 단일 경로가 여전히 분포를 안 낸다")
    d = d[has].copy()
    di = (d.conf_exp - d.conf_int).abs()
    print(f"\n[두 conf 의 차이]")
    print(f"  상관 {np.corrcoef(d.conf_int, d.conf_exp)[0,1]:+.4f}"
          f" · 순위상관 {spearmanr(d.conf_int, d.conf_exp)[0]:+.4f}")
    print(f"  |차이| 평균 {di.mean():.4f} · 중앙 {di.median():.4f} · 최대 {di.max():.4f}")
    print(f"  정수 conf 평균 {d.conf_int.mean():.4f} -> 기댓값 conf {d.conf_exp.mean():.4f}")
    print(f"  0.05 넘게 벌어진 행 {100*(di > .05).mean():.1f}%"
          f" · 0.10 넘게 {100*(di > .10).mean():.1f}%")
    # 등급 하나가 얼마나 확신 있었나
    eg = np.concatenate([np.asarray(r) for r in d.eg if r is not None])
    pk = np.concatenate([np.asarray(r, dtype=float) for r in d.picks])
    print(f"\n[등급 자체] 뽑힌 정수 대 기댓값: |차이| 평균 {np.abs(eg-pk).mean():.3f}"
          f" · 최대 {np.abs(eg-pk).max():.3f}")

    base, k2 = per_task("baseline_full_v2_with_action_steps"), per_task("baseline_compress_K2")
    keep = {t: k2[t] / base[t] for t in base if t in k2 and base[t] > 0.05}
    m = d.groupby("task")[["conf_int", "conf_exp"]].mean()
    T = [t for t in m.index if t in keep]
    print(f"\n[폐루프 K2 유지율과의 상관]  태스크 {len(T)}개"
          f" (장면/태스크 {d.groupby('task').size().min()}~{d.groupby('task').size().max()})")
    for c in ("conf_int", "conf_exp"):
        rho, p = spearmanr([m[c][t] for t in T], [keep[t] for t in T])
        print(f"  {c:<9} {rho:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
