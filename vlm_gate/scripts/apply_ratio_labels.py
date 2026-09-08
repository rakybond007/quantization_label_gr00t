"""배속 라벨을 LeRobot 데이터셋 parquet 에 붙인다. **어느 기계에서나 돌아간다.**

이 저장소를 클론하고 데이터셋만 있으면 된다. 우리 서버 경로도, 우리 캐시도,
VLM 도 필요 없다 -- 라벨은 이미 만들어져 `assets/labels/` 에 들어 있다.

    python vlm_gate/scripts/apply_ratio_labels.py \\
        --labels assets/labels/libero/libero_v2_ratio.parquet \\
        --dataset /어디에나/lerobot/libero_gr00t_delta \\
        --out     /어디에나/libero_with_ratio

무엇이 붙나
------------
에피소드·프레임마다 열 넷이 붙는다.

    ratio        이 시점에 쓸 배속. 1.0 · 1.5 · 2.0 · 2.5 중 하나다.
    ratio_conf   그 배속을 정한 신뢰도 0~1 (순서 값이지 확률이 아니다)
    ratio_fixed  접촉이라 1.0 으로 고정된 시점이면 1
    ratio_valid  라벨이 있는 시점이면 1. 없으면 0 이고 ratio 는 1.0 이다.

`ratio_valid` 가 있는 이유는, 라벨이 없는 시점을 "1배속이 정답" 으로 학습시키면
안 되기 때문이다. 손실에서 빼라는 표시다.

왜 이 형태인가
--------------
학습을 여기서 할지 다른 서버에서 할지 정해지지 않아서, **라벨과 적용을 데이터
파이프라인에서 떼어 놨다.** 받는 쪽은 parquet 하나와 이 스크립트만 있으면 되고,
우리 쪽 라벨링을 다시 돌릴 필요가 없다.

배속을 이산값으로 적는 이유는 `RATIOS_ARE_NOT_K.md` 에 있다 -- 디코더가 이산이라
연속값을 적으면 결국 반올림해서 쓰게 되고, 그러면 결정이 반올림에서 일어난다.

    --dry-run 을 주면 쓰지 않고 붙는 비율만 보여 준다. 먼저 그것부터 해 볼 것.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRID = (1.0, 1.5, 2.0, 2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True,
                    help="assets/labels/<bench>/*_ratio.parquet")
    ap.add_argument("--dataset", required=True,
                    help="LeRobot 데이터셋 루트 (meta/ 와 data/ 가 있는 곳)")
    ap.add_argument("--out", default="",
                    help="쓸 곳. 비우면 --in-place 를 줘야 한다")
    ap.add_argument("--in-place", action="store_true",
                    help="데이터셋을 그 자리에서 고친다. 사본이 없으면 위험하다")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lab = pd.read_parquet(a.labels)
    need = {"episode_index", "frame_index", "ratio"}
    miss = need - set(lab.columns)
    if miss:
        raise SystemExit(f"라벨에 열이 없다: {sorted(miss)}")
    bad = sorted(set(np.round(lab.ratio.unique(), 6)) - set(GRID))
    if bad:
        raise SystemExit(f"눈금 밖의 배속이 있다: {bad} (허용 {GRID})")
    print(f"라벨 {len(lab):,}행 · 에피소드 {lab.episode_index.nunique():,}")
    print("  배속 분포  " + "  ".join(
        f"{g}x:{(lab.ratio == g).mean():5.1%}" for g in GRID))

    ds = Path(a.dataset)
    files = sorted(ds.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"데이터셋 parquet 을 못 찾았다: {ds}/data/**/*.parquet")
    print(f"데이터셋 파일 {len(files):,}  <- {ds}")

    if not a.dry_run:
        if a.in_place:
            out = ds
        elif a.out:
            out = Path(a.out)
            if out.exists():
                raise SystemExit(f"{out} 가 이미 있다")
            # meta 는 통째로 복사하고 data 만 다시 쓴다
            shutil.copytree(ds, out, ignore=shutil.ignore_patterns("data", "videos"))
            print(f"메타 복사 -> {out}")
        else:
            raise SystemExit("--out 또는 --in-place 가 필요하다")

    key = lab.set_index(["episode_index", "frame_index"])
    ncov = ntot = 0
    for i, f in enumerate(files):
        d = pd.read_parquet(f)
        if "episode_index" not in d.columns or "frame_index" not in d.columns:
            raise SystemExit(f"{f} 에 episode_index/frame_index 가 없다")
        idx = pd.MultiIndex.from_arrays([d.episode_index, d.frame_index])
        j = key.reindex(idx)
        d["ratio"] = j.ratio.fillna(1.0).to_numpy(dtype=np.float32)
        d["ratio_conf"] = (j["conf"].to_numpy(dtype=np.float32)
                           if "conf" in j.columns else np.float32(0.5))
        d["ratio_fixed"] = (j["fixed"].fillna(0).to_numpy(dtype=np.int8)
                            if "fixed" in j.columns else np.int8(0))
        d["ratio_valid"] = (~j.ratio.isna()).to_numpy(dtype=np.int8)
        ncov += int(d.ratio_valid.sum())
        ntot += len(d)
        if not a.dry_run:
            g = (out / f.relative_to(ds)) if not a.in_place else f
            g.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(g, index=False)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}  덮인 프레임 {ncov:,}/{ntot:,}", flush=True)

    print(f"\n프레임 {ntot:,} 중 라벨 있는 것 {ncov:,} = {ncov / max(1, ntot):.1%}")
    if a.dry_run:
        print("(--dry-run 이라 아무것도 쓰지 않았다)")
        return
    meta = {"labels": os.path.basename(a.labels), "grid": list(GRID),
            "frames_total": int(ntot), "frames_labelled": int(ncov)}
    (Path(out) / "meta" / "ratio_labels.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
