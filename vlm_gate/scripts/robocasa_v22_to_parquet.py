"""robocasa v22 라벨 jsonl -> 배포용 parquet.

v7 변환기(`phase9_v7_to_parquet.py`)는 등급 확률 `gp` 로 기댓값 등급 eA..eE 를
같이 실었다. **v22 는 Gemini 판정이라 확률이 없다** -- 정수 등급뿐이다
(`gemini-3.8-flash` 는 logprob 을 주지 않는다). 그래서 정수 등급 A..F 와 conf 만 싣고,
가중을 바꿔 conf 를 다시 낼 수 있도록 **등급을 그대로 남긴다**.

**접촉 경계 플래그는 싣지 않는다** (사용자 지시). 그래서 기존 배포 파일에 의존할
이유가 없어졌고, 태스크·지시문도 데이터셋 meta 에서 직접 읽는다.

라벨 디렉터리를 여러 개 받는다 -- 타일이 없어 빠졌던 207 에피를 영상에서 직접 라벨한
`output/robocasa_v22_fill` 을 같이 넣는다.

  CHECKS=phase9_checks_v22 python robocasa_v22_to_parquet.py <출력.parquet> <라벨디렉터리>...
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
CK = importlib.import_module(os.environ.get("CHECKS", "phase9_checks_v22"))
Q = sorted(CK.SIGN)


def conf_of(r):
    g = {q: (r[q] - 1) / (CK.NGRADE - 1) for q in Q}
    s = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] > 0)
    k = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] < 0)
    return (1 + s - k) / 2


def main():
    out, srcs = sys.argv[1], sys.argv[2:]
    rows, bad, seen = [], 0, set()
    for src in srcs:
        n0 = len(rows)
        for l in open(f"{src}/labels.jsonl"):
            r = json.loads(l)
            if not all(q in r for q in Q):
                bad += 1
                continue
            k = (r["ep"], r["f"])
            if k in seen:          # 두 디렉터리가 겹치면 앞의 것을 남긴다
                continue
            seen.add(k)
            rows.append(r)
        print(f"  {src}: +{len(rows)-n0:,}")
    print(f"라벨 {len(rows):,} (문항 누락으로 버린 행 {bad})")

    d = pd.DataFrame({
        "episode_index": np.array([r["ep"] for r in rows], dtype=np.int32),
        "frame_index": np.array([r["f"] for r in rows], dtype=np.int32),
        **{q: np.array([r[q] for r in rows], dtype=np.int8) for q in Q},
        "conf": np.array([conf_of(r) for r in rows], dtype=np.float32),
        "speed_mean": np.array([r.get("speed_mean", np.nan) for r in rows],
                               dtype=np.float32),
    })

    # 태스크·지시문은 데이터셋 meta 에서. robocasa 는 에피당 하나다.
    task, instr = {}, {}
    for ln in open(f"{DS}/meta/episodes.jsonl"):
        e = json.loads(ln)
        cands = [t for t in e.get("tasks", [])
                 if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
        short = [t for t in e.get("tasks", []) if isinstance(t, str) and t not in cands
                 and t != "Valid"]
        if cands:
            instr[e["episode_index"]] = cands[0]
        if short:
            task[e["episode_index"]] = short[0]
    d["task"] = d["episode_index"].map(task)
    d["instruction"] = d["episode_index"].map(instr)
    miss = int(d["instruction"].isna().sum())
    print(f"meta 에서 태스크·지시문 병합 · 지시문 없는 행 {miss}")

    d = d.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    d.to_parquet(out, index=False)
    print(f"-> {out}  {len(d):,}행 · {os.path.getsize(out)/1e6:.1f} MB")
    print("열:", list(d.columns))
    print(f"conf 평균 {d['conf'].mean():.4f} · 값 종류 {d['conf'].nunique()}")
    print("태스크", d["task"].nunique(), "· 에피", d["episode_index"].nunique())


if __name__ == "__main__":
    main()
