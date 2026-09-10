"""Write meta/ for the retimed dataset and check it against what was actually built.

meta/stats.json is deliberately omitted: summed deltas are about twice the magnitude of
the originals, so the source statistics would normalise the new actions onto the wrong
scale with nothing raising an error. LeRobotSingleDataset recomputes and writes stats
when the file is absent (gr00t/data/dataset.py:324-337), which is the behaviour we want.

Everything else is derived from the parquets that exist on disk, not from the source
meta, so a shard that failed shows up as a missing episode rather than a length that
silently disagrees with its video.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    dst_meta = os.path.join(args.dst, "meta")
    os.makedirs(dst_meta, exist_ok=True)

    src_eps = {}
    with open(os.path.join(args.src, "meta", "episodes.jsonl")) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                src_eps[int(r["episode_index"])] = r

    # select cameras by key, not dtype: LIBERO declares them "image" even though the
    # frames live in videos/*.mp4, which made this count zero of them
    vkeys = [k for k in info["features"] if k.startswith("observation.images.")]
    assert vkeys, f"no camera keys in {args.src}/meta/info.json"
    rows, total_frames, bad = [], 0, []
    for f in sorted(glob.glob(os.path.join(args.dst, "data", "chunk-*", "*.parquet"))):
        tid = int(os.path.basename(f).split("_")[-1].split(".")[0])
        n = pq.read_metadata(f).num_rows
        chunk = tid // info["chunks_size"]
        for vk in vkeys:
            for cand in (vk, vk.replace("observation.images.", "")):
                p = os.path.join(args.dst, info["video_path"].format(
                    episode_chunk=chunk, episode_index=tid, video_key=cand))
                if os.path.exists(p):
                    break
            else:
                bad.append((tid, "missing video", vk))
        r = dict(src_eps.get(tid, {"episode_index": tid, "tasks": []}))
        r["episode_index"], r["length"] = tid, int(n)
        rows.append(r)
        total_frames += n

    rows.sort(key=lambda r: r["episode_index"])
    with open(os.path.join(dst_meta, "episodes.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    new_info = dict(info)
    new_info["total_episodes"] = len(rows)
    new_info["total_frames"] = int(total_frames)
    new_info["total_videos"] = len(rows) * len(vkeys)
    new_info["splits"] = {"train": f"0:{len(rows)}"}
    json.dump(new_info, open(os.path.join(dst_meta, "info.json"), "w"), indent=4)

    for name in ("tasks.jsonl", "modality.json"):
        src_f = os.path.join(args.src, "meta", name)
        if os.path.exists(src_f):
            # copyfile, not copy: copy() also chmods, which the S3 mount refuses
            shutil.copyfile(src_f, os.path.join(dst_meta, name))
    assert os.path.exists(os.path.join(dst_meta, "modality.json")), \
        f"modality.json not written -- GR00T cannot open the dataset without it"
    for name in ("stats.json", "episodes_stats.jsonl", "stats_gr00t.json"):
        p = os.path.join(dst_meta, name)
        if os.path.exists(p):
            os.remove(p)          # force recomputation on first load

    src_frames = info["total_frames"]
    print(f"episodes {len(rows)} / source {info['total_episodes']}")
    print(f"frames   {total_frames} / source {src_frames}  -> ratio {src_frames/max(total_frames,1):.3f}")
    print(f"videos   {len(rows)*len(vkeys)} expected, missing: {len(bad)}")
    if bad:
        print("  first problems:", bad[:5])
    print(f"meta written to {dst_meta} (stats.json intentionally absent)")
    if len(rows) != info["total_episodes"]:
        print(f"  WARNING: {info['total_episodes']-len(rows)} episodes missing -- a shard "
              f"probably failed; rerun it before training")


if __name__ == "__main__":
    main()
