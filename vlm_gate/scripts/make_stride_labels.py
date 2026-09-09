"""배속 라벨에서 stride 간격만 남긴 판을 만든다. 프레임 캐시를 굽는 목록이다.

캐시는 `--labels` 가 주는 (에피소드, 프레임) 만 굽는다. 지금 캐시는 stride 4 로
구워져 있어서, 라벨이 204만이어도 학습에 쓸 수 있는 것은 258,809 뿐이다.

    python vlm_gate/scripts/make_stride_labels.py --stride 2

  stride 1  2,044,657 행 · 캐시 약 288 GB
  stride 2  약 102만    · 약 144 GB
  stride 4  258,809     · 37 GB (지금)

**프레임 번호가 stride 로 나눠떨어지는 것만 남긴다.** 에피소드마다 몇 번째냐로
세면 길이가 다른 에피소드에서 서로 다른 시점이 뽑혀 기존 캐시와 안 겹친다.
"""
import argparse
import os

import pandas as pd

W = os.path.expanduser("~/quantization_agent_workspace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=f"{W}/vlm_gate/output/_gate_distill/"
                                    "robocasa_contact_ratio.parquet")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default="")
    a = p.parse_args()
    out = a.out or f"{W}/assets/labels/robocasa/ratio_contact_stride{a.stride}.parquet"

    d = pd.read_parquet(a.src)
    sub = d[d.frame_index % a.stride == 0].reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.to_parquet(out, index=False)
    print(f"{len(d):,} -> {len(sub):,} ({len(sub)/len(d):.1%})  stride {a.stride}")
    print("  배속 " + "  ".join(f"{v}x:{(sub.ratio == v).mean():.1%}"
                               for v in sorted(sub.ratio.unique())))
    print(f"  캐시 예상 {len(sub) * 37 / 258809:.0f} GB")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
