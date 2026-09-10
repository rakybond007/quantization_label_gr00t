"""Write the accelerated LeRobot dataset: parquet + re-encoded videos + meta.

Per episode, the frames the skip rule keeps become the new episode. Actions are summed
within each group (delta action space -- see retime_dataset.py), state is indexed at the
kept frames (absolute, so indexing is right), and each camera is re-encoded from exactly
those frames so observations stay aligned with actions.

meta/stats.json is deliberately NOT copied. Summed deltas are roughly twice the
magnitude of the originals, so the source statistics would normalise the new actions
onto the wrong scale -- silently, since nothing checks it. LeRobotSingleDataset
recomputes and writes stats.json when it is absent, which is what we want.

Also records, per episode, the fraction of merged commands that would leave the
controller's +-1 range. That is the direct measure of how much temporal redundancy the
demonstrations actually had, which is what decides whether a data-side method like this
has anything to work with at a given recording frequency.
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np

PRECISION, CASUAL = 0, 1
CLIP = 1.0


def skip_indices(labels, low_v=2, high_v=4):
    T = len(labels); idx, i = [], -1
    while i < T:
        if i >= 0 and labels[i] == CASUAL:
            if i + high_v < T and np.all(labels[i:i + high_v] == CASUAL):
                i += high_v; idx.append(i)
            else:
                nz = np.flatnonzero(labels[i + 1:] == PRECISION)
                if nz.size:
                    i = i + 1 + int(nz[0]); idx.append(i)
                else:
                    break
        else:
            if i + low_v < T:
                i += low_v; idx.append(i)
            else:
                break
    return np.array([j for j in idx if 0 <= j < T], dtype=int)


def group_bounds(keep):
    """(start, end) original-step span for each kept frame."""
    out, s = [], 0
    for e in keep:
        out.append((s, int(e)))
        s = int(e) + 1
    return out


def retime_actions(act, keep, ddims, cdims):
    out = np.zeros((len(keep), act.shape[1]), dtype=np.float64)
    for k, (s, e) in enumerate(group_bounds(keep)):
        grp = act[s:e + 1] if e >= s else act[e:e + 1]
        out[k, ddims] = grp[:, ddims].sum(axis=0)
        if len(cdims):
            out[k, cdims] = grp[-1, cdims]
    return out


def reencode(src_path, dst_path, keep, fps):
    import av, decord
    vr = decord.VideoReader(src_path)
    frames = vr.get_batch(np.clip(keep, 0, len(vr) - 1)).asnumpy()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    c = av.open(dst_path, "w")
    st = c.add_stream("libx264", rate=int(round(fps)))
    st.height, st.width = frames.shape[1], frames.shape[2]
    st.pix_fmt = "yuv420p"
    for f in frames:
        for pk in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            c.mux(pk)
    for pk in st.encode():
        c.mux(pk)
    c.close()
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--delta-dims", required=True)
    ap.add_argument("--discrete-dims", default="")
    ap.add_argument("--low-v", type=int, default=2)
    ap.add_argument("--high-v", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    a, b = (int(v) for v in args.delta_dims.split(":"))
    ddims = np.arange(a, b)
    cdims = np.array([int(v) for v in args.discrete_dims.split(",") if v != ""], dtype=int)

    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    fps, chunks = info["fps"], info["chunks_size"]
    # LIBERO declares its cameras as dtype "image" even though the frames are mp4 under
    # videos/ -- selecting on dtype=="video" silently produced a dataset with no videos at
    # all, and the length assert below passed vacuously on the empty list. Select on the
    # key name and require at least one.
    vkeys = [k for k in info["features"] if k.startswith("observation.images.")]
    assert vkeys, f"no camera keys in {args.src}/meta/info.json"
    lab = np.load(args.labels)
    tids = sorted(int(k) for k in lab.files)[args.shard::args.nshards]
    os.makedirs(args.dst, exist_ok=True)

    stats = []
    for tid in tids:
        chunk = tid // chunks
        src_pq = os.path.join(args.src, info["data_path"].format(episode_chunk=chunk, episode_index=tid))
        df = pd.read_parquet(src_pq)
        acol = "action" if "action" in df.columns else "actions"
        act = np.stack(df[acol].to_numpy()).astype(np.float64)
        L = lab[str(tid)]
        n = min(len(L), len(act))
        keep = skip_indices(L[:n], args.low_v, args.high_v)
        if keep.size < 2:
            continue

        # how much redundancy was there? fraction of merged commands that would clip
        merged = np.array([act[s:e + 1][:, ddims].sum(0) for s, e in group_bounds(keep)])
        clip_frac = float((np.abs(merged) > CLIP).mean())

        new = df.iloc[keep].reset_index(drop=True).copy()
        new[acol] = list(retime_actions(act[:n], keep, ddims, cdims).astype(np.float32))
        if "timestamp" in new:
            new["timestamp"] = (np.arange(len(keep)) / fps).astype(np.float32)
        if "frame_index" in new:
            new["frame_index"] = np.arange(len(keep), dtype=np.int64)
        # LIBERO stores its camera frames inline as struct<bytes, path> alongside the mp4s.
        # GR00T reads frames from videos/ and never touches those columns, but
        # calculate_dataset_statistics walks every column and dies on the structs with
        # "float() argument must be ... not 'dict'". The source dataset never hits this
        # because it ships a stats.json; ours deliberately omits one so the summed-delta
        # actions get fresh statistics. Dropping the unused columns fixes the crash and
        # removes the bulk of the file at the same time.
        drop = [c for c in new.columns if c.startswith("observation.images.")]
        if drop:
            new = new.drop(columns=drop)
        dst_pq = os.path.join(args.dst, info["data_path"].format(episode_chunk=chunk, episode_index=tid))
        os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(new, preserve_index=False), dst_pq)

        nfr = []
        for vk in vkeys:
            sub = vk.replace("observation.images.", "")
            sv = os.path.join(args.src, info["video_path"].format(
                episode_chunk=chunk, episode_index=tid, video_key=vk))
            if not os.path.exists(sv):
                sv = os.path.join(args.src, info["video_path"].format(
                    episode_chunk=chunk, episode_index=tid, video_key=sub))
            dv = os.path.join(args.dst, info["video_path"].format(
                episode_chunk=chunk, episode_index=tid, video_key=os.path.basename(os.path.dirname(sv))))
            nfr.append(reencode(sv, dv, keep, fps))

        assert len(nfr) == len(vkeys) and all(x == len(keep) for x in nfr), (
            f"ep {tid}: expected {len(vkeys)} videos of {len(keep)} frames, got {nfr}")
        stats.append({"episode_index": tid, "T": int(n), "kept": int(keep.size),
                      "ratio": float(n / keep.size), "clip_frac": clip_frac,
                      "casual_frac": float((L[:n] == CASUAL).mean())})
        if len(stats) % 50 == 0:
            print(f"  {len(stats)}/{len(tids)} episodes", flush=True)

    out = os.path.join(args.dst, f"_build_shard{args.shard}.json")
    json.dump({"episodes": stats}, open(out, "w"), indent=2)
    r = np.array([s["ratio"] for s in stats]); c = np.array([s["clip_frac"] for s in stats])
    print(f"shard {args.shard}: {len(stats)} episodes, "
          f"{sum(s['T'] for s in stats)} -> {sum(s['kept'] for s in stats)} frames, "
          f"ratio {r.mean():.3f}, merged-command clip fraction {c.mean():.4f}")


if __name__ == "__main__":
    main()
