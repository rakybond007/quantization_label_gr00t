"""Pre-decode labeled frames into a disk cache for full-scale gate training.

Reads the merged labels parquet (episode_index, frame_index), decodes the 3
camera views once, resizes to --res, and writes a uint8 memmap shard
(N, 9, res, res) plus an index parquet mapping rows -> (episode, frame).
The cache is judge-agnostic: cosmos/gemma label the same sampled frames, so
one cache serves both training runs.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

VIEW_KEYS = ["observation.images.left_view", "observation.images.right_view",
             "observation.images.wrist_view"]


def read_frames(mp4_path, idxs):
    import av
    out = {}
    want = sorted(set(idxs))
    with av.open(mp4_path) as c:
        i = 0
        for frame in c.decode(video=0):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
            i += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--labels", required=True, help="merged labels parquet (any judge)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    import cv2
    ds = args.dataset_path
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    cs = info.get("chunks_size", 1000)

    lab = pd.read_parquet(args.labels, columns=["episode_index", "frame_index"])
    eps = sorted(lab["episode_index"].unique())
    eps = [e for e in eps if e % args.num_shards == args.shard]
    lab = lab[lab["episode_index"].isin(eps)].sort_values(["episode_index", "frame_index"])
    n = len(lab)
    os.makedirs(args.out_dir, exist_ok=True)
    mm_path = os.path.join(args.out_dir, f"frames_shard{args.shard}.u8")
    ix_path = os.path.join(args.out_dir, f"index_shard{args.shard}.parquet")
    if os.path.exists(ix_path):
        done = pd.read_parquet(ix_path)
        if len(done) == n:
            print(f"[cache] shard {args.shard} already complete ({n} rows), skip")
            return
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8,
                                   shape=(n, 9, args.res, args.res))
    row = 0
    rows_ix = []
    for ei, g in lab.groupby("episode_index"):
        idxs = g["frame_index"].tolist()
        chunk = ei // cs
        views = {}
        for vk in VIEW_KEYS:
            mp4 = os.path.join(ds, info["video_path"].format(
                episode_chunk=chunk, video_key=vk, episode_index=ei))
            views[vk] = read_frames(mp4, idxs)
        for fi in idxs:
            planes = []
            for vk in VIEW_KEYS:
                im = views[vk].get(fi)
                if im is None:
                    im = np.zeros((args.res, args.res, 3), np.uint8)
                else:
                    im = cv2.resize(im, (args.res, args.res), interpolation=cv2.INTER_AREA)
                planes.append(im.transpose(2, 0, 1))
            mm[row] = np.concatenate(planes, axis=0)
            rows_ix.append({"row": row, "episode_index": int(ei), "frame_index": int(fi)})
            row += 1
        if (len(rows_ix) % 5000) < len(idxs):
            print(f"[cache] shard {args.shard}: {row}/{n} rows", flush=True)
    mm.flush()
    pd.DataFrame(rows_ix).to_parquet(ix_path, index=False)
    print(f"[cache] shard {args.shard} done: {row} rows -> {mm_path}")


if __name__ == "__main__":
    main()
