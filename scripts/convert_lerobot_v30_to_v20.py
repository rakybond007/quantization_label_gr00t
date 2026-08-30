"""Convert a LeRobot v3.0 dataset (aggregated data/video files) to the v2.0
per-episode layout that the GR00T LeRobotSingleDataset loader expects.

v3.0:  data/chunk-XXX/file-YYY.parquet (many episodes), videos/{key}/chunk/file.mp4,
       meta/episodes/chunk/file.parquet, meta/info.json (codebase_version v3.0).
v2.0:  data/chunk-XXX/episode_NNNNNN.parquet (one episode), videos/chunk/{key}/episode.mp4,
       meta/episodes.jsonl, meta/info.json (v2.0).

modality.json + stats.json are copied through unchanged (GR00T reads stats.json;
our loader guard already skips scalar 'count' entries). Usage:
    python convert_lerobot_v30_to_v20.py --src <v3.0 dir> --dst <v2.0 dir>
"""
import argparse, glob, json, os, shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import imageio
import imageio.v3 as iio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--chunks_size", type=int, default=1000)  # v2.0 episode-chunk size
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)

    info = json.load(open(f"{src}/meta/info.json"))
    fps = int(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    eps = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(f"{src}/meta/episodes/chunk-*/file-*.parquet"))],
                    ignore_index=True)
    # cache source data parquet files
    data_cache = {}
    def get_data(ci, fi):
        key = (ci, fi)
        if key not in data_cache:
            data_cache[key] = pd.read_parquet(
                f"{src}/data/chunk-{ci:03d}/file-{fi:03d}.parquet")
        return data_cache[key]

    ep_lines = []
    ep_len = {}
    for _, e in eps.iterrows():
        ei = int(e["episode_index"])
        ech = ei // args.chunks_size
        # ---- data: slice the aggregated parquet for this episode ----
        df = get_data(int(e["data/chunk_index"]), int(e["data/file_index"]))
        sub = df[df["episode_index"] == ei].reset_index(drop=True)
        # GR00T resolves language via an int annotation column indexing tasks.jsonl.
        # DexJoCo only carries task_index, so mirror it under the expected key.
        sub["annotation.human.action.task_description"] = sub["task_index"]
        ddir = f"{dst}/data/chunk-{ech:03d}"; os.makedirs(ddir, exist_ok=True)
        sub.to_parquet(f"{ddir}/episode_{ei:06d}.parquet")
        n = len(sub); ep_len[ei] = n
        tasks = list(e["tasks"]) if hasattr(e["tasks"], "__len__") else [e["tasks"]]
        ep_lines.append({"episode_index": ei, "tasks": tasks, "length": n})

    # ---- videos: one streaming pass per aggregated AV1 file, split per episode,
    #      transcode to h264 (GR00T's decord backend cannot read AV1). ----
    for vk in video_keys:
        groups = defaultdict(list)  # (vci,vfi) -> [(ei, start_frame, length)]
        for _, e in eps.iterrows():
            ei = int(e["episode_index"])
            groups[(int(e[f"videos/{vk}/chunk_index"]), int(e[f"videos/{vk}/file_index"]))].append(
                (ei, int(round(float(e[f"videos/{vk}/from_timestamp"]) * fps)), ep_len[ei]))
        for (vci, vfi), lst in groups.items():
            lst.sort(key=lambda x: x[1])  # by start frame
            srcvid = f"{src}/videos/{vk}/chunk-{vci:03d}/file-{vfi:03d}.mp4"
            it = iio.imiter(srcvid, plugin="pyav")
            cur = 0  # absolute frame index in the source video
            for (ei, start, length) in lst:
                while cur < start:
                    next(it); cur += 1
                frames = []
                for _ in range(length):
                    frames.append(next(it)); cur += 1
                ech = ei // args.chunks_size
                vdir = f"{dst}/videos/chunk-{ech:03d}/{vk}"; os.makedirs(vdir, exist_ok=True)
                iio.imwrite(f"{vdir}/episode_{ei:06d}.mp4", np.stack(frames),
                            plugin="pyav", codec="libx264", fps=fps,
                            out_pixel_format="yuv420p")

    # ---- meta/episodes.jsonl ----
    with open(f"{dst}/meta/episodes.jsonl", "w") as f:
        for line in ep_lines:
            f.write(json.dumps(line) + "\n")
    # ---- meta/tasks.jsonl (from tasks.parquet) ----
    tk = pd.read_parquet(f"{src}/meta/tasks.parquet")
    # tasks.parquet: index=task string, column task_index (or vice versa)
    with open(f"{dst}/meta/tasks.jsonl", "w") as f:
        if "task_index" in tk.columns:
            it = ((row["task_index"], idx) for idx, row in tk.iterrows())
            for ti, task in it:
                f.write(json.dumps({"task_index": int(ti), "task": str(task)}) + "\n")
        else:
            for ti, task in tk["task"].items():
                f.write(json.dumps({"task_index": int(ti), "task": str(task)}) + "\n")
    # ---- meta/info.json (v2.0) ----
    info2 = dict(info)
    info2["codebase_version"] = "v2.0"
    info2["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info2["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info2["chunks_size"] = args.chunks_size
    info2["total_chunks"] = (len(ep_lines) + args.chunks_size - 1) // args.chunks_size
    json.dump(info2, open(f"{dst}/meta/info.json", "w"), indent=4)
    # ---- copy stats.json + modality.json ----
    for fn in ("stats.json", "modality.json"):
        if os.path.exists(f"{src}/meta/{fn}"):
            shutil.copy(f"{src}/meta/{fn}", f"{dst}/meta/{fn}")
    print(f"DONE: {len(ep_lines)} episodes -> {dst}")


if __name__ == "__main__":
    main()
