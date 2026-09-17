"""libero v3d 라벨 jsonl -> ATQ 베이킹용 parquet.

`GR00T-action-quantization/scripts/materialize_conf_v7.py` 가 기대하는 스키마다:
`episode_index` · `frame_index` · 게이트 열. 에피마다 프레임 0 에서 시작해야 하고,
스크립트가 `np.interp` 로 전 프레임에 펴며 마지막 라벨 뒤는 `valid=0` 으로 둔다.

**게이트는 `conf_vlm` 하나다.** 접촉 열을 쓰지 않는다(사용자 지시) -- v3c 가 쓰던
`conf_both`·`contact_valid` 는 만들지 않는다. 따라서 베이킹 때 `--valid-col` 도 비운다.

  CHECKS=libero_v3d_checks python libero_v3d_to_parquet.py <라벨디렉터리> <출력.parquet>
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CK = importlib.import_module(os.environ.get("CHECKS", "libero_v3d_checks"))
Q = sorted(CK.SIGN)


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

    d = pd.DataFrame({
        "episode_index": np.array([r["ep"] for r in rows], dtype=np.int32),
        "frame_index": np.array([r["f"] for r in rows], dtype=np.int32),
        **{q: np.array([r[q] for r in rows], dtype=np.int8) for q in Q},
        "conf_vlm": np.array([conf_of(r) for r in rows], dtype=np.float32),
        "phase": [r.get("phase") for r in rows],
    }).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # 베이킹이 요구하는 것: 에피마다 프레임 0 이 있어야 한다.
    g = d.groupby("episode_index").frame_index.min()
    bad0 = g[g != 0]
    if len(bad0):
        raise SystemExit(f"프레임 0 라벨이 없는 에피 {len(bad0)}: {list(bad0.index[:5])}")
    d.to_parquet(out, index=False)
    print(f"-> {out}  {len(d):,}행 · 에피 {d.episode_index.nunique():,}")
    print(f"conf_vlm 평균 {d.conf_vlm.mean():.4f} · 값 종류 {d.conf_vlm.nunique()}")
    for t in (0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"   tau {t:.2f} -> 압축 비율 {(d.conf_vlm >= t).mean():.1%}")


if __name__ == "__main__":
    main()
