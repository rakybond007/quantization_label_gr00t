"""human_data v3 라벨 jsonl -> 배포용 parquet.

두 데이터셋(pnp_task · long_horizon_task)이 한 파일에 있고 `ds` 열로 가른다.
에피소드 번호가 데이터셋마다 따로 매겨지므로 **`ds` 없이 (ep, f) 만으로 키가 되지
않는다.**

정수 등급 A~D 를 그대로 싣는다. Gemini 판정이라 등급 확률이 없다(logprob 미제공).
등급이 있으면 어떤 가중으로도 conf 를 다시 낼 수 있다.

  CHECKS=humandata_v3_checks python humandata_v3_to_parquet.py <라벨디렉터리> <출력.parquet>
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CK = importlib.import_module(os.environ.get("CHECKS", "humandata_v3_checks"))
Q = sorted(CK.SIGN)
ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"


def conf_of(r):
    g = {q: (r[q] - 1) / (CK.NGRADE - 1) for q in Q}
    s = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] > 0)
    k = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] < 0)
    return (1 + s - k) / 2


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows, bad = [], 0
    for l in open(f"{src}/labels.jsonl"):
        r = json.loads(l)
        if not all(q in r and r[q] for q in Q):
            bad += 1
            continue
        rows.append(r)
    print(f"라벨 {len(rows):,} (문항 누락 {bad})")

    instr = {}
    for ds in ("pnp_task", "long_horizon_task"):
        p = f"{ROOT}/{ds}/lerobot/meta/episodes.jsonl"
        for ln in open(p):
            d = json.loads(ln)
            t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 2]
            if t:
                instr[(ds, d["episode_index"])] = t[0]

    d = pd.DataFrame({
        "ds": [r["ds"] for r in rows],
        "episode_index": np.array([r["ep"] for r in rows], dtype=np.int32),
        "frame_index": np.array([r["f"] for r in rows], dtype=np.int32),
        **{q: np.array([r[q] for r in rows], dtype=np.int8) for q in Q},
        "conf_vlm": np.array([conf_of(r) for r in rows], dtype=np.float32),
        "phase": [r.get("phase") for r in rows],
        "merge_k2": np.array([r.get("merge_k2", np.nan) for r in rows], dtype=np.float32),
        "grip_closed_frac": np.array([r.get("grip_closed_frac", np.nan) for r in rows],
                                     dtype=np.float32),
    })
    d["instruction"] = [instr.get((a, b)) for a, b in zip(d.ds, d.episode_index)]
    d = d.sort_values(["ds", "episode_index", "frame_index"]).reset_index(drop=True)
    d.to_parquet(out, index=False)
    print(f"-> {out}  {len(d):,}행")
    for ds, g in d.groupby("ds"):
        print(f"  {ds:<20} {len(g):>5}행 · 에피 {g.episode_index.nunique():>3} · "
              f"conf 평균 {g.conf_vlm.mean():.3f}")
    print(f"지시문 없는 행 {int(d.instruction.isna().sum())}")


if __name__ == "__main__":
    main()
