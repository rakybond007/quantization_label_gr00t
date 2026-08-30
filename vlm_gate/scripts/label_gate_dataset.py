"""Label a LeRobot-format dataset with the VLM gate's quantizability signal.

For every sampled frame (every N steps, matching the eval subchunk cadence) the
judge server is queried with the episode's camera views + task instruction +
the BEST evolved guidance, and P(YES) is stored. The resulting parquet is the
distillation training set for a lightweight gate module (so eval no longer
needs the heavy VLM).

Output follows the cluster conventions:
  labels/ckpt artifacts -> /rlwrld-unified-checkpoints/hojin2/checkpoints/<run>/
  logs                  -> vlm_gate/output/_gate_distill/
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate  # noqa: E402

VIEW_KEYS = ["observation.images.left_view", "observation.images.right_view",
             "observation.images.wrist_view"]


def read_frames(mp4_path, idxs):
    """Return {frame_idx: HxWx3 uint8} for the requested indices."""
    try:
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
    except ImportError:
        import imageio
        r = imageio.get_reader(mp4_path)
        out = {}
        want = set(idxs)
        for i, frame in enumerate(r):
            if i in want:
                out[i] = np.asarray(frame)
                if len(out) == len(want):
                    break
        r.close()
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--guidance", default="", help="raw text or @file")
    p.add_argument("--every", type=int, default=8, help="label every N frames (subchunk cadence)")
    p.add_argument("--episodes", type=int, default=3, help="number of episodes (from the start) to label")
    p.add_argument("--episode-offset", type=int, default=0)
    p.add_argument("--episode-list", default="", help="comma-separated explicit episode indices (overrides --episodes)")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--out", required=True, help="output parquet path")
    args = p.parse_args()

    ds = args.dataset_path
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    chunk_size = info.get("chunks_size", 1000)
    # episodes.jsonl is the authoritative instruction source (the per-frame
    # annotation index columns are unreliable in this dataset — ep1 points at
    # the "Valid" row). Pick the description-looking entry: a lowercase,
    # multi-word sentence (excludes CamelCase task names and "Valid").
    ep_instr = {}
    for ln in open(os.path.join(ds, "meta", "episodes.jsonl")):
        d = json.loads(ln)
        cands = [t for t in d.get("tasks", [])
                 if " " in t and t[:1].islower() and t != "Valid"]
        ep_instr[d["episode_index"]] = cands[0] if cands else (d.get("tasks") or [""])[0]

    gate = VLMGate(args.judge_url)
    g = args.guidance
    guidance = open(g[1:]).read() if g.startswith("@") and os.path.exists(g[1:]) else g

    rows = []
    ep_iter = ([int(x) for x in args.episode_list.split(",") if x != ""] if args.episode_list
               else range(args.episode_offset, args.episode_offset + args.episodes))
    for ei in ep_iter:
        chunk = ei // chunk_size
        pq = os.path.join(ds, info["data_path"].format(episode_chunk=chunk, episode_index=ei))
        df = pd.read_parquet(pq, columns=["episode_index"])
        T = len(df)  # frame index == row order in this dataset (no frame_index column)
        instr = ep_instr.get(ei, "")
        idxs = list(range(0, T, args.every))
        views = {}
        for vk in VIEW_KEYS:
            mp4 = os.path.join(ds, info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ei))
            views[vk] = read_frames(mp4, idxs)
        n_yes = 0
        for fi in idxs:
            imgs = [views[vk].get(fi) for vk in VIEW_KEYS]
            if any(im is None for im in imgs):
                continue
            res = gate.judge(imgs, instr, guidance)
            conf = float(res.get("confidence", 0.0))
            q = int(conf >= args.tau)
            n_yes += q
            rows.append({"episode_index": ei, "frame_index": fi, "task": instr,
                         "p_yes": conf, "quantize": q})
        print(f"[label] ep {ei}: {len(idxs)} frames, quantize {n_yes}/{len(idxs)} "
              f"({100*n_yes/max(len(idxs),1):.0f}%), task={instr[:60]!r}", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[label] saved {len(out)} rows -> {args.out}")
    print(f"[label] overall quantize rate: {out['quantize'].mean():.3f}, "
          f"p_yes mean/median: {out['p_yes'].mean():.3f}/{out['p_yes'].median():.3f}")


if __name__ == "__main__":
    main()
