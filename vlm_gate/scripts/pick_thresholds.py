"""온라인 게이트의 역치 둘을 **있는 데이터에서** 정한다. 지어내지 않는다.

    python vlm_gate/scripts/pick_thresholds.py \
        <labels_ratio.parquet> <chord_error.parquet>

    1단계  --judge-threshold   신뢰도가 이 값 이상이면 압축한다
    2단계  --rate-by-chord     현오차가 이 값 이하인 가장 큰 배속을 고른다

## 어떻게 정하나

역치를 값으로 고르지 않고 **비율로 고른다.** "신뢰도 0.5" 는 지어낸 선이지만
"위쪽 절반" 은 분포가 정해 주는 자리다. 그리고 게이트 평가는 **같은 평균 배속의
균일 대조군**과 견주므로, 역치는 그 평균 배속을 맞추도록 정하는 것이 맞다.

    1단계  압축 비율 p 를 정하면 -> 신뢰도 분포의 (1-p) 분위가 역치
    2단계  목표 평균 배속을 정하면 -> 현오차 분포에서 그 평균을 내는 theta

## 대조군

나온 평균 배속으로 균일 실행을 같이 돌린다. **그것이 바가 되는 것이지 사다리가
바가 되는 것이 아니다** -- 사다리는 태스크 단위이고 우리가 묻는 것은 시점
단위다. 평균이 같은데 게이트가 더 나으면 시점을 가른 것이고, 아니면 아니다.
"""
import sys

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    lab, chord = sys.argv[1:3]

    print("=" * 66)
    print("1단계 -- 압축할지 말지 (신뢰도 역치)")
    d = pd.read_parquet(lab, columns=["conf", "fixed"])
    c = d["conf"].to_numpy(float)
    print(f"  라벨 {len(c):,}행 · 신뢰도 최소 {c.min():.3f} 중앙 "
          f"{np.median(c):.3f} 최대 {c.max():.3f}")
    print(f"  {'압축 비율':>10s} {'역치':>8s}")
    for p in (0.75, 0.65, 0.50, 0.35, 0.25):
        print(f"  {p:10.0%} {np.percentile(c, 100 * (1 - p)):8.3f}")
    print("\n  ** 평가 때 신뢰도는 VLM 이 그 자리에서 냅니다. 라벨과 같은 문항·")
    print("  같은 식이라 분포가 비슷할 것이나 같지는 않습니다 -- 스모크에서")
    print("  실제 분포를 보고 한 번 다시 맞추십시오.")

    print("\n" + "=" * 66)
    print("2단계 -- 압축한다면 몇 배속 (현오차 역치)")
    e = pd.read_parquet(chord, columns=["task", "f", "K", "pos"])
    grid = [1.5, 2.0, 2.5]
    have = [k for k in grid if k in set(e.K.unique())]
    if len(have) < len(grid):
        print(f"  [!] chord 파일에 {sorted(set(e.K.unique()))} 만 있다. "
              f"{grid} 이 다 필요하다")
        return 1
    # 시점마다 배속별 현오차를 나란히 놓는다
    w = e.pivot_table(index=["task", "f"], columns="K", values="pos")
    w = w.dropna(subset=grid)
    print(f"  시점 {len(w):,}개")
    print(f"  {'θ':>8s} {'평균 배속':>10s}   " + "  ".join(f"{g}배" for g in grid))
    rows = []
    for th in np.percentile(w[grid].to_numpy().ravel(), np.arange(5, 100, 5)):
        r = np.full(len(w), grid[0], dtype=float)
        for g in grid:
            r = np.where(w[g].to_numpy() <= th, g, r)
        share = [f"{(r == g).mean():5.0%}" for g in grid]
        rows.append((th, r.mean()))
        print(f"  {th:8.4f} {r.mean():10.3f}   " + "  ".join(share))

    print("\n  ** 목표 평균 배속을 정하고 그 줄의 θ 를 씁니다. 그리고 **같은")
    print("  평균 배속의 균일 실행**을 대조군으로 같이 돌립니다.")
    print("  사다리가 바가 아닙니다 -- 사다리는 태스크 단위이고 우리가 묻는")
    print("  것은 시점 단위입니다. 평균이 같은데 게이트가 나으면 시점을 가른")
    print("  것이고, 아니면 아닙니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
