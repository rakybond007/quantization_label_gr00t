"""Frame cache for the real DROID pnp dataset (2 camera views).

Same memmap format as build_gate_frame_cache.py — (N, 9, res, res) uint8 —
so train_gate_module.py's CachedGateFrames reads it unchanged. The real rig
only has exterior + wrist views, so the third 3-channel plane is zeros.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

VIEW_KEYS = ["observation.images.exterior_image_1_left",
             "observation.images.wrist_image_left"]


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
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--res", type=int, default=128)
    args = p.parse_args()

    import cv2
    ds = args.dataset_path
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    cs = info.get("chunks_size", 1000)

    lab = pd.read_parquet(args.labels, columns=["episode_index", "frame_index"])
    lab = lab.sort_values(["episode_index", "frame_index"])
    n = len(lab)
    os.makedirs(args.out_dir, exist_ok=True)
    mm_path = os.path.join(args.out_dir, "frames_shard0.u8")
    ix_path = os.path.join(args.out_dir, "index_shard0.parquet")
    if os.path.exists(ix_path) and len(pd.read_parquet(ix_path)) == n:
        print(f"[cache] already complete ({n} rows), skip")
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
                    im = cv2.resize(im, (args.res, args.res),
                                    interpolation=cv2.INTER_AREA)
                planes.append(im.transpose(2, 0, 1))
            planes.append(np.zeros((3, args.res, args.res), np.uint8))
            mm[row] = np.concatenate(planes, axis=0)
            rows_ix.append({"row": row, "episode_index": int(ei),
                            "frame_index": int(fi)})
            row += 1
        print(f"[cache] ep {ei}: {row}/{n} rows", flush=True)
    mm.flush()
    pd.DataFrame(rows_ix).to_parquet(ix_path, index=False)
    print(f"[cache] done: {row} rows -> {mm_path}")


if __name__ == "__main__":
    main()
