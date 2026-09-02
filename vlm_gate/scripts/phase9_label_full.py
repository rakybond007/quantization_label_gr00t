"""phase9's five checks over EVERY frame, decoded from the videos.

The tile directory holds one PNG per 8th frame -- 260,031 of them -- and the
pilot labelled a shuffled 1,898 of those. Labelling every frame through that
directory would mean writing about two million more files into a tree the
infrastructure team already watches, so nothing is written: the three camera
views are decoded straight out of the episode's videos and tiled in memory, the
same way the allex labeller works.

Sharding is by episode so each worker opens each video once and walks it in
order; random access across a whole shard would decode far more than it reads.
Resume is per shard file, so a preemption on the background partition costs the
frames of one episode at most.

    python phase9_label_full.py <port> <shard> <nshard>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase9_checks import ASK, GUIDANCE, NGRADE  # noqa: E402
from robocasa_descriptors import descriptors, facts_text  # noqa: E402
from vlm_gate import VLMGate  # noqa: E402

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
OUT = os.environ.get("PHASE9_OUT", f"{BASE}/output/_gate_distill/phase9_full")
PORT, SHARD, NSHARD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BLOCK = 64                      # frames decoded per read
BATCH = int(os.environ.get('PHASE9_BATCH', 8))   # frames per forward

info = json.load(open(f"{DS}/meta/info.json"))
vks = [k for k in info["features"] if info["features"][k].get("dtype") == "video"]
VK = ([k for k in vks if "left" in k]
      + [k for k in vks if "right" in k and "wrist" not in k]
      + [k for k in vks if "wrist" in k])

instr = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

os.makedirs(OUT, exist_ok=True)
FP = f"{OUT}/labels_s{NSHARD}_{SHARD}.jsonl"
done = set()
if os.path.exists(FP):
    for line in open(FP):
        try:
            r = json.loads(line)
        except Exception:
            continue
        done.add((r["ep"], r["f"]))

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
eps = [e for e in range(7200) if e % NSHARD == SHARD]
print(f"shard {SHARD}/{NSHARD}: {len(eps)} episodes, resuming past {len(done)} frames", flush=True)

json.dump({"batch": BATCH, "shard": SHARD, "nshard": NSHARD, "decode_block": BLOCK,
           "note": "batch width changes borderline answers; rerun with the same "
                   "batch to reproduce this file"},
          open(f"{OUT}/meta_s{NSHARD}_{SHARD}.json", "w"))
fh = open(FP, "a")
nlab = nfull = nempty = 0
for ei, ep in enumerate(eps):
    ch = ep // info["chunks_size"]
    try:
        a = np.stack(pd.read_parquet(
            f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
            episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
    except Exception:
        continue
    n = min(min(len(v) for v in vrs), len(a) - 4)
    todo = [f for f in range(max(n, 0)) if (ep, f) not in done]
    if not todo:
        continue
    ins_ep = instr.get(ep, "")
    for b0 in range(0, len(todo), BLOCK):
        idx = todo[b0:b0 + BLOCK]
        try:
            blocks = [v.get_batch(idx).asnumpy() for v in vrs]
        except Exception:
            break
        # BATCH is part of the label. Each way of running is reproducible on its
        # own -- three single passes agreed 32/32, two batch-8 passes agreed
        # 32/32 -- but single and batch-8 disagree on 2 of 32, because a
        # different batch width picks different kernels and the last bits of
        # bfloat16 move. Those are chunks sitting between two grades either way.
        # The size is recorded in meta.json so a rerun can reproduce this file.
        for b1 in range(0, len(idx), BATCH):
            grp = list(enumerate(idx))[b1:b1 + BATCH]      # (row in blocks, frame)
            payload = [([Image.fromarray(bl[row]) for bl in blocks],
                        f"{ins_ep}\n{facts_text(descriptors(a, f))}") for row, f in grp]
            try:
                rs = gate.judge_batch(payload, GUIDANCE, question=ASK,
                                      n_ask=5, n_grade=NGRADE)
            except Exception as e:
                print(f"  ep{ep} f{grp[0]}+: {type(e).__name__}", flush=True)
                continue
            for (row, f), r in zip(grp, rs):
                # A dead judge answers every call the same; writing rows through
                # it yields a file of the right length and no content, which
                # counting outputs cannot tell from success.
                if r.get("error") or not r.get("text", "").strip():
                    nempty += 1
                    if nempty >= 5 and nlab < 5:
                        raise SystemExit(f"judge not answering: {r.get('error', 'empty')}")
                    continue
                nempty = 0
                picks = r.get("picks") or [None] * 5
                nfull += int(all(p is not None for p in picks))
                fh.write(json.dumps({"ep": ep, "f": f,
                                     **{q: picks[i] for i, q in enumerate("ABCDE")}},
                                    ensure_ascii=False) + "\n")
                nlab += 1
        fh.flush()
    if ei % 20 == 0:
        print(f"  shard{SHARD}: ep {ei}/{len(eps)}  {nlab} frames  full {nfull}", flush=True)

fh.close()
print(f"shard{SHARD} done: {nlab} frames, {nfull} complete -> {FP}")
