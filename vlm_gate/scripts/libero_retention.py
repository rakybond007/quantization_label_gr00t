"""LIBERO 압축 유지율 표. **clip 을 푼 런만 쓴다.**

clip 을 켠 채 잰 옛 런(output/libero/baseline_*)은 압축 비용을 크게 부풀린다 --
같은 K=2 에서 유지율이 0.587 로 나오는데, clip 을 풀면 0.990 이다. 하네스 제약을
데이터의 성질로 읽게 되므로 쓰지 않는다.

    python libero_retention.py            표 출력
    python libero_retention.py --json OUT  analysis/ 에 저장

에피소드 줄에서 다시 센다 -- 파일 머리의 `Total success rate` 요약을 쓰지 않는다.
robocasa 에서 요약 헤더로 성공률을 계산했다가 K4 가 K3 보다 좋다는 불가능한 결과를
얻은 적이 있다.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
# 배속 -> 결과 위치. results.txt 판과 csv 판이 섞여 있다.
RUNS = {
    1.0: ("dir", "/sjw_alinlab2/home/taekwan/Data/libero_all_Fine"),
    2.0: ("dir", "/sjw_alinlab2/home/taekwan/Data/libnaive_clipoff_s7_K2_allk2"),
    2.5: ("csv", f"{BASE}/analysis/libero_retention/K2p5_clipoff.csv"),
    4.0: ("dir", "/sjw_alinlab2/home/taekwan/Isaac-GR00T/output/libero/"
                 "libnaive_clipoff_s7_K4_allk2"),
}


def from_dir(root):
    o = {}
    for s in SUITES:
        for p in sorted(glob.glob(f"{root}/{s}/*_results.txt")):
            t = f"{s}/{os.path.basename(p).split('_')[0]}"
            ok = [m.group(2) == "True" for m in
                  (re.match(r"\s*(\d+)\s+(True|False)\s+(\d+)", l) for l in open(p)) if m]
            if ok:
                o[t] = (float(np.mean(ok)), len(ok))
    return o


def from_csv(path):
    o = {}
    for r in csv.DictReader(open(path)):
        n = int(r["episodes"])
        o[f"{r['suite']}/{r['task_id']}"] = (int(r["successes"]) / n, n)
    return o


def load_all():
    out = {}
    for k, (kind, p) in RUNS.items():
        if kind == "dir" and not os.path.isdir(p):
            print(f"  [!] K={k} 경로 없음: {p}", file=sys.stderr)
            continue
        if kind == "csv" and not os.path.exists(p):
            print(f"  [!] K={k} 파일 없음: {p}", file=sys.stderr)
            continue
        out[k] = from_dir(p) if kind == "dir" else from_csv(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    S = load_all()
    b = S[1.0]
    ks = [k for k in sorted(S) if k != 1.0]

    print(f"{'배속':>6}{'태스크':>7}{'평균성공':>9}{'유지율':>9}   suite 별 유지율")
    print(f"{'':>6}{'':>7}{'':>9}{'':>9}   " + "".join(f"{s.replace('libero_',''):>10}" for s in SUITES))
    print(f"{1.0:>6.1f}{len(b):>7}{np.mean([v[0] for v in b.values()]):>9.3f}{1.0:>9.3f}")
    for k in ks:
        d = S[k]
        T = [t for t in d if t in b and b[t][0] > 0.05]
        keep = np.mean([d[t][0] / b[t][0] for t in T])
        line = f"{k:>6.1f}{len(d):>7}{np.mean([v[0] for v in d.values()]):>9.3f}{keep:>9.3f}   "
        for s in SUITES:
            ts = [t for t in T if t.startswith(s)]
            line += f"{np.mean([d[t][0]/b[t][0] for t in ts]):>10.3f}"
        print(line)

    print("\n[태스크별 유지율]  K=2.5 낮은 쪽부터")
    d25 = S.get(2.5, {})
    rows = sorted(((d25[t][0] / b[t][0], t) for t in d25 if t in b and b[t][0] > 0.05))
    hdr = "".join(f"{('K'+str(k)):>8}" for k in ks)
    print(f"  {'기준':>6}{hdr}   태스크")
    for _, t in rows:
        cells = "".join(f"{(S[k][t][0]/b[t][0]) if t in S[k] else float('nan'):>8.2f}" for k in ks)
        print(f"  {b[t][0]:>6.2f}{cells}   {t}")

    if a.json:
        out = {t: {"base": b[t][0], **{f"K{k}": (S[k][t][0] / b[t][0])
                                       for k in ks if t in S[k]}} for t in b}
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
