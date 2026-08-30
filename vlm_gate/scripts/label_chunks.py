"""One labeller for every benchmark. Benchmarks differ by configuration, not code.

Until now each embodiment had its own labelling script — RoboCasa's tile-based
one, plus a copy for LIBERO and another for dexjoco — and they drifted: three
places to fix a resume bug, three places where a view count could be wrong.
What actually differs between them is small and declarative, so it lives in
BENCHMARKS below: where the data is, which camera views to send and in what
order, which deterministic descriptor module to use, which guidance and
question files to read, and how to key an episode.

Frames are decoded from video on demand rather than read from pre-baked tiles.
At the stride this now runs (every frame) RoboCasa alone would need 2.07M tile
files, which this filesystem cannot hold; decoding an episode in order is also
cheaper than reading a million small files back.

    python label_chunks.py <benchmark> <port> <shard> <n_shards>
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
sys.path.insert(0, f"{BASE}/scripts")
from vlm_gate import VLMGate                                   # noqa: E402

EVOLVER = f"{BASE}/analysis/_evolver"

BENCHMARKS = {
    "robocasa": {
        "dataset": "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/"
                   "kimtaey/robocasa_mg_gr00t_300",
        "descriptors": "robocasa_descriptors.py",
        "guidance": f"{EVOLVER}/_varkA/robocasa_guidance_phase_v5.txt",
        "questions": None,          # ASK6 lives in cosmos_1call_v6 for this one
        "views": ("left", "right", "wrist"),
        "tail": 4,                  # last frames cannot start a full chunk
    },
    "libero": {
        "dataset": "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/"
                   "kimtaey/libero_gr00t_delta",
        "descriptors": "libero_descriptors.py",
        "guidance": f"{EVOLVER}/_libero/libero_guidance_v1.txt",
        "questions": f"{EVOLVER}/_libero/libero_questions_v1.txt",
        "views": ("front", "wrist"),
        "tail": 4,
    },
    "dexjoco": {
        "dataset": os.path.expanduser(
            "~/quantization_agent_workspace/assets/datasets/dexjoco_lerobot/"
            "dexjoco_lerobot_datasets"),
        "descriptors": "dexjoco_descriptors.py",
        "guidance": f"{EVOLVER}/_dexjoco/dexjoco_guidance_v1.txt",
        "questions": f"{EVOLVER}/_dexjoco/dexjoco_questions_v2.txt",
        "views": ("wrist", "ego|front"),
        "tail": 16,
        "per_task": True,           # six task datasets, each with its own episode numbering
        "ep_stride": 1000,          # global ep = ep_stride * task_id + local
    },
}


def load_module(filename):
    p = f"{BASE}/scripts/{filename}"
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pick_views(info, wanted):
    """Video keys in the order the guidance assumes, matched by substring.

    Two things this has to tolerate. LIBERO declares its camera features with
    dtype "image" rather than "video" even though video_path points at mp4, so
    the dtype alone cannot identify them. And dexjoco names its second view
    differently per task — `ego_right` on click_mouse, `front` on water_plant —
    so a slot may list alternatives separated by "|", first match winning.
    """
    keys = [k for k, v in info["features"].items()
            if v.get("dtype") in ("video", "image") and k.startswith("observation.images")]
    out = []
    for slot in wanted:
        hit = None
        for alt in slot.split("|"):
            cands = [k for k in keys if alt in k and k not in out]
            # "right" must not swallow "right_wrist" when both are asked for
            if alt != "wrist":
                cands = [k for k in cands if "wrist" not in k] or cands
            if cands:
                hit = cands[0]
                break
        if hit is None:
            raise RuntimeError(f"view slot {slot!r} matched nothing; dataset has {keys}")
        out.append(hit)
    return out


def read_questions(cfg):
    if cfg["questions"]:
        return open(cfg["questions"]).read().strip()
    import cosmos_1call_v6 as ref
    return ref.ASK6


def count_slots(ask):
    """Slot letters are read off the question text, so evolving the set needs no code change."""
    import re
    return "".join(sorted({m.group(1) for m in re.finditer(r"^([A-Z])\)", ask, re.M)}))


def episode_sources(cfg):
    """(episode_key, dataset_root, local_episode_index) for every episode."""
    if not cfg.get("per_task"):
        yield from ((e, cfg["dataset"], e) for e in _episodes(cfg["dataset"]))
        return
    for tid, task in enumerate(sorted(os.path.basename(p) for p in
                                      glob.glob(f"{cfg['dataset']}/*") if os.path.isdir(p))):
        root = f"{cfg['dataset']}/{task}"
        for e in _episodes(root):
            yield cfg["ep_stride"] * tid + e, root, e


def _episodes(root):
    with open(f"{root}/meta/episodes.jsonl") as f:
        return [json.loads(l)["episode_index"] for l in f if l.strip()]


def main():
    bench, port, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cfg = BENCHMARKS[bench]
    stride = int(os.environ.get("LABEL_STRIDE", "1"))
    tag = os.environ.get("TAG", f"{bench}_dense")
    out_path = f"{BASE}/output/_gate_distill/{tag}_s{nsh}_{shard}.jsonl"

    desc = load_module(cfg["descriptors"])
    guidance = open(cfg["guidance"]).read().strip()
    ask = read_questions(cfg)
    slots = count_slots(ask)
    print(f"[label] {bench} shard{shard}/{nsh} stride={stride} "
          f"질문 {len(slots)}개({slots}) -> {os.path.basename(out_path)}", flush=True)

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["ep"], r["f"]))
            except Exception:
                pass
    print(f"[label] 재개 {len(done)}개", flush=True)

    import pandas as pd
    from decord import VideoReader
    gate = VLMGate(f"http://127.0.0.1:{port}", timeout=180)
    out = open(out_path, "a")
    n = 0
    for ep_key, root, ep_local in episode_sources(cfg):
        if ep_key % nsh != shard:
            continue
        info = json.load(open(f"{root}/meta/info.json"))
        cs = info.get("chunks_size", 1000)
        ch = ep_local // cs
        try:
            a = np.stack(pd.read_parquet(
                f"{root}/data/chunk-{ch:03d}/episode_{ep_local:06d}.parquet")["action"].values)
            vks = pick_views(info, cfg["views"])
            vrs = [VideoReader(f"{root}/" + info["video_path"].format(
                episode_chunk=ch, episode_index=ep_local, video_key=k)) for k in vks]
        except Exception as e:
            print(f"[label] ep{ep_key} 열기 실패: {e}", flush=True)
            continue
        instr = _instruction(root, ep_local)
        nfr = min(len(a), min(len(v) for v in vrs))
        for f in range(0, max(nfr - cfg["tail"], 0), stride):
            if (ep_key, f) in done:
                continue
            x = desc.descriptors(a, f)
            views = [Image.fromarray(v[f].asnumpy()) for v in vrs]
            r = gate.judge(views, f"{instr}\n{desc.facts_text(x)}", guidance,
                           question=ask, n_ask=len(slots))
            c = r.get("confidences") or [0.0] * len(slots)
            rec = {"ep": ep_key, "f": f,
                   **{k: float(v) for k, v in zip(slots, c)},
                   **desc.computed_risk(x), "speed_mean": x.get("speed_mean", 0.0),
                   "ans": r.get("answer", "")}
            if cfg.get("per_task"):
                rec["task"] = os.path.basename(root)
                rec["ep_local"] = ep_local
            out.write(json.dumps(rec) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"[label] shard{shard}: {n}", flush=True)
                out.flush()
        del vrs
    out.close()
    print(f"[label] shard{shard} 완료 {n} -> {out_path}", flush=True)


def _instruction(root, ep):
    for line in open(f"{root}/meta/episodes.jsonl"):
        d = json.loads(line)
        if d.get("episode_index") == ep:
            c = [t for t in d.get("tasks", [])
                 if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
            return c[0] if c else ""
    return ""


if __name__ == "__main__":
    main()
