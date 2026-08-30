"""Label a real-robot LeRobot v2.0 teleop dataset (e.g. MoSS DROID pnp) with
the VLM quantizability gate.

Generic over LeRobot v2 layout: reads meta/info.json for video keys and
episode counts, meta/tasks.jsonl + per-episode parquet task_index for the
instruction, samples every Nth frame from each episode's videos, and queries
the running judge server (same /judge HTTP interface as sim labeling).

Output parquet columns match the sim labelers: episode_index, frame_index,
task, p_yes, quantize (+ n_views for audit).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate  # noqa: E402


def read_frames(mp4, idxs):
    from decord import VideoReader
    vr = VideoReader(mp4)
    idxs = [min(i, len(vr) - 1) for i in idxs]
    return {i: f for i, f in zip(idxs, vr.get_batch(idxs).asnumpy())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--guidance", required=True, help="guidance text or @file")
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--episodes", default="", help="comma list; empty = all")
    p.add_argument("--out", required=True)
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    g = args.guidance
    guidance = open(g[1:]).read().strip() if g.startswith("@") else g

    info = json.load(open(os.path.join(args.dataset, "meta", "info.json")))
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    # wrist view last, matching sim convention (wrist is the contact-detail view)
    vkeys = sorted(vkeys, key=lambda k: ("wrist" in k, k))
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(os.path.join(args.dataset, "meta", "tasks.jsonl"))}
    n_eps = info["total_episodes"]
    eps = ([int(e) for e in args.episodes.split(",") if e != ""]
           if args.episodes else list(range(n_eps)))

    gate = VLMGate(args.judge_url, timeout=120.0)
    rows = []
    for ei, ep in enumerate(eps):
        chunk = ep // info["chunks_size"]
        df = pd.read_parquet(os.path.join(
            args.dataset, info["data_path"].format(episode_chunk=chunk, episode_index=ep)))
        instr = tasks[int(df["task_index"].iloc[0])] if "task_index" in df else tasks[0]
        n = len(df)
        idxs = list(range(0, n, args.every))
        views = {}
        for vk in vkeys:
            mp4 = os.path.join(args.dataset, info["video_path"].format(
                episode_chunk=chunk, episode_index=ep, video_key=vk))
            views[vk] = read_frames(mp4, idxs)
        for fi in idxs:
            imgs = [views[vk][min(fi, max(views[vk]))] for vk in vkeys]
            res = gate.judge(imgs, instr, guidance)
            if "error" in res:
                print(f"[label-real] ep {ep} f{fi} judge error: {res['error'][:100]}", flush=True)
                continue
            p_yes = float(res["confidence"])  # server confidence IS P(YES)
            rows.append(dict(episode_index=ep, frame_index=fi, task=instr,
                             p_yes=p_yes, quantize=bool(p_yes >= args.tau),
                             n_views=len(imgs)))
        if ei % 10 == 0:
            d = pd.DataFrame(rows)
            print(f"[label-real] ep {ep} ({ei+1}/{len(eps)}) rows={len(d)} "
                  f"qrate={d.quantize.mean():.2f}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out)
    print(f"[label-real] saved {len(out)} rows -> {args.out} "
          f"(qrate={out.quantize.mean():.2f}, p_yes mean={out.p_yes.mean():.3f})")


if __name__ == "__main__":
    main()
