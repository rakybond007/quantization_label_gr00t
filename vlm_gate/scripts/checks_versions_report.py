"""문항 판(v1/v2/v3)을 **폐루프 압축 유지율** 기준으로 나란히 본다.

이것이 채택 판정이다. 태스크 평균 conf 가 유지율과 **양의** 상관이어야 쓸 수 있다.
v1 은 -0.705 (p<0.001) 로 거꾸로였다.

    python checks_versions_report.py <라벨1.jsonl> [라벨2.jsonl ...]

각 파일은 gp_conf_check.py 산출물(task, conf_int, conf_exp, picks, gp, eg)이다.
파일 이름에서 판 이름을 딴다.
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
    base = per_task("baseline_full_v2_with_action_steps")
    keeps = {}
    for k in (2, 3):
        c = per_task(f"baseline_compress_K{k}")
        keeps[k] = {t: c[t] / base[t] for t in base if t in c and base[t] > 0.05}

    print(f"{'판':<12}{'장면':>6}{'태스크':>7}{'K2 상관':>10}{'p':>7}"
          f"{'K3 상관':>10}{'p':>7}{'태스크몫':>9}{'conf평균':>9}")
    keep_rows = {}
    for p in sys.argv[1:]:
        nm = re.sub(r"^gp_?|\.jsonl$", "", os.path.basename(p)) or "v1"
        d = pd.DataFrame([json.loads(l) for l in open(p)])
        if "conf_exp" not in d:
            print(f"{nm}: conf_exp 가 없다 -- 건너뜀"); continue
        keep_rows[nm] = d
        m = d.groupby("task").conf_exp.mean()
        y = d.conf_exp.astype(float)
        share = np.var(d.groupby("task").conf_exp.transform("mean").astype(float)) / np.var(y)
        cells = []
        for k in (2, 3):
            T = [t for t in m.index if t in keeps[k]]
            rho, pv = spearmanr([m[t] for t in T], [keeps[k][t] for t in T])
            cells += [rho, pv]
        print(f"{nm:<12}{len(d):>6}{d.task.nunique():>7}{cells[0]:>+10.3f}{cells[1]:>7.3f}"
              f"{cells[2]:>+10.3f}{cells[3]:>7.3f}{share:>9.3f}{y.mean():>9.3f}")

    # **VLM 이 의도한 것을 보는가.** 기제가 맞는지와 별개의 질문이다 -- A 는
    # "닫아야 하는가", B 는 "그 자리가 좁은가" 를 묻는데, 모델이 그림에서 그것을
    # 실제로 가려내는지는 모른다. 손으로 분류한 단계와 문항별 등급을 견준다.
    PUSH = {"CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave", "TurnOnStove",
            "TurnOffStove", "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
            "CloseDrawer", "CloseSingleDoor", "CloseDoubleDoor"}
    TIGHT = {"PnPCabToCounter", "PnPMicrowaveToCounter", "PnPStoveToCounter",
             "PnPSinkToCounter", "CoffeeServeMug", "OpenDrawer", "OpenSingleDoor",
             "OpenDoubleDoor", "CoffeeSetupMug"}
    for nm, d in keep_rows.items():
        if "picks" not in d:
            continue
        pk = pd.DataFrame(d.picks.tolist(), columns=list("ABCDE"))
        pk["task"] = d.task.to_numpy()
        m = pk.groupby("task").mean(numeric_only=True)
        grasp = pd.Series({t: 0 if t in PUSH else 1 for t in m.index})
        tight = pd.Series({t: 1 if t in TIGHT else 0 for t in m.index})
        print(f"\n[{nm}] 모델이 의도한 것을 보는가 (태스크 평균 등급 대 손분류)")
        for q, ref, want in (("A", grasp, "파지 있음"), ("B", tight, "좁은 자리")):
            rho, pv = spearmanr(m[q], ref[m.index])
            print(f"  {q}) vs {want:<8} Spearman {rho:+.3f} (p={pv:.4f})"
                  f"  · {want} 평균 {m[q][ref[m.index]==1].mean():.2f}"
                  f" 대 아닌 쪽 {m[q][ref[m.index]==0].mean():.2f}")

    # 채택 판정과, 태스크별로 어디가 달라졌는가
    if len(keep_rows) >= 2:
        nms = list(keep_rows)
        print(f"\n[태스크별 conf 평균]  유지율 낮은 쪽부터")
        ms = {n: keep_rows[n].groupby("task").conf_exp.mean() for n in nms}
        T = sorted(set.intersection(*[set(m.index) for m in ms.values()]) & set(keeps[2]),
                   key=lambda t: keeps[2][t])
        head = "".join(f"{n:>10}" for n in nms)
        print(f"  {'K2유지':>7}{head}   태스크")
        for t in T:
            print(f"  {keeps[2][t]:>7.2f}" + "".join(f"{ms[n][t]:>10.3f}" for n in nms)
                  + f"   {t}")


if __name__ == "__main__":
    main()
