"""eps 를 움직일 때 실현 압축비가 얼마나 따라 움직이는지 오프라인으로 잰다.

confidence 를 eps 에 태우는 구조는, eps 가 배속을 실제로 움직여야 성립한다.
곡선이 평평하면 손잡이가 아니라 장식이므로 구조를 접어야 한다. 상한은 여기서
일부러 크게 열어 둔다 -- 상한이 걸리면 eps 의 효과가 아니라 상한의 효과를
재는 셈이 되어 곡선이 위쪽에서 거짓으로 눕는다.

폐루프가 아니라 시연 데이터라, 여기서 나온 배속은 실제 eval 보다 높게 나온다
(정책 액션이 시연보다 덜 매끄럽다). 절대값이 아니라 기울기를 보기 위한 것이다.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interp_compress import segment, _cap_ratio


def chunks_from_parquet(files, action_cols, horizon, stride, max_chunks):
    out = []
    for f in files:
        df = pd.read_parquet(f, columns=list(action_cols) + ["episode_index"])
        for _, g in df.groupby("episode_index", sort=False):
            a = np.stack([np.stack(g[c].to_numpy()) if g[c].dtype == object
                          else g[c].to_numpy()[:, None] for c in action_cols], axis=1)
            a = a.reshape(len(g), -1).astype(float)
            for s in range(0, len(a) - horizon + 1, stride):
                out.append(a[s:s + horizon])
                if len(out) >= max_chunks:
                    return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="LeRobot-style dataset root")
    p.add_argument("--action-cols", default="", help="comma separated; empty = every action.* column")
    p.add_argument("--gripper-dim", type=int, default=-1,
                   help="index into the flattened action vector that latches (gripper). "
                        "-1 = none.")
    p.add_argument("--drop-const", action="store_true",
                   help="drop dimensions that never move (unused base/torso channels), "
                        "which would otherwise pull every interpolation error toward zero")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--max-chunks", type=int, default=3000)
    p.add_argument("--max-files", type=int, default=40)
    p.add_argument("--eps", default="0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.20")
    p.add_argument("--ratio-max", type=float, default=16.0, help="deliberately loose")
    p.add_argument("--kmax", type=int, default=16)
    p.add_argument("--space", default="path")
    a = p.parse_args()

    files = sorted(glob.glob(str(Path(a.data) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {a.data}/data")
    cols = [c for c in a.action_cols.split(",") if c]
    if not cols:
        cols = [c for c in pd.read_parquet(files[0]).columns if c.startswith("action")]
    print(f"[i] {len(files)} parquet files, action cols: {cols}")

    chunks = chunks_from_parquet(files[:a.max_files], cols, a.horizon, a.stride, a.max_chunks)
    print(f"[i] {len(chunks)} chunks of {a.horizon}")
    if not chunks:
        raise SystemExit("no chunks")

    gi = a.gripper_dim if a.gripper_dim >= 0 else None
    if a.drop_const:
        stacked = np.concatenate(chunks, axis=0)
        keep = [i for i in range(stacked.shape[1]) if stacked[:, i].std() > 1e-9]
        if gi is not None and gi not in keep:
            keep.append(gi); keep.sort()
        print(f"[i] keeping dims {keep} of {stacked.shape[1]}")
        gi = keep.index(gi) if gi is not None else None
        chunks = [c[:, keep] for c in chunks]
    print(f"{'eps':>7} {'중앙':>6} {'평균':>6} {'p10':>6} {'p90':>6} {'>2.0':>6} {'>3.0':>6}")
    for e in [float(x) for x in a.eps.split(",")]:
        rs = []
        for c in chunks:
            cont = c if gi is None else np.delete(c, gi, axis=1)
            disc = np.zeros((len(c), 0)) if gi is None else c[:, [gi]]
            sp = _cap_ratio(segment(cont, disc, e, a.kmax, a.space), len(c), a.ratio_max)
            rs.append(len(c) / len(sp))
        rs = np.array(rs)
        print(f"{e:>7.3f} {np.median(rs):>6.2f} {rs.mean():>6.2f} "
              f"{np.percentile(rs,10):>6.2f} {np.percentile(rs,90):>6.2f} "
              f"{100*(rs>2.0).mean():>5.1f}% {100*(rs>3.0).mean():>5.1f}%")


if __name__ == "__main__":
    main()
