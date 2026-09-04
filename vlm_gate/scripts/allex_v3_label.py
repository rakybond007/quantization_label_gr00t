"""Label allex with the v3 checks -- one stage, graded, answered in text.

v2 ran two stages and read teacher-forced YES/NO slots. Both are gone. There is
one question set now (allex_v3_checks), the model WRITES its five digits, and
what comes out is already a ratio -- ceiling_from_checks returns K, so no
confidence, no tau ladder, no rank normalisation.

Nothing is written to the parquet. This writes records.jsonl only, for the
renderer to read.

    python allex_v3_label.py <port> [episodes,comma,separated]
"""
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import TASKS, descriptors, stage2_facts  # noqa: E402
from allex_v3_checks import ASK, GUIDANCE, NGRADE, ceiling_from_checks, snap  # noqa: E402
from vlm_gate import VLMGate  # noqa: E402

DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
OUTDIR = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3"))
os.makedirs(OUTDIR, exist_ok=True)
CHUNK = 16
BATCH = int(os.environ.get("ALLEX_BATCH", 8))

PORT = sys.argv[1]
EPS = [int(e) for e in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
OUT = f"{OUTDIR}/records.jsonl"

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done.add((r["ep"], r["f"]))
        except Exception:
            pass


def grab(ep, frames, side):
    """Decode the requested frame indices from one ego camera."""
    want, got = set(frames), {}
    path = (f"{DS}/videos/chunk-000/observation.images.camera_ego_{side}/"
            f"episode_{ep:06d}.mp4")
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


out = open(OUT, "a")
ntot = nempty = 0
for ep in EPS:
    d = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    starts = [f for f in range(0, len(A) - CHUNK, CHUNK) if (ep, f) not in done]
    if not starts:
        print(f"ep{ep}: 이미 함", flush=True)
        continue
    L, R = grab(ep, starts, "left"), grab(ep, starts, "right")
    for b0 in range(0, len(starts), BATCH):
        grp = starts[b0:b0 + BATCH]
        payload, meta = [], []
        for f in grp:
            x = descriptors(A, WR, WL, f, CHUNK)
            seg = ti[f:f + CHUNK]
            task = TASKS[int(np.bincount(seg, minlength=len(TASKS)).argmax())]
            payload.append(([L[f], R[f]], f"{task}\n{stage2_facts(task, x)}"))
            meta.append((f, task, x))
        try:
            rs = gate.judge_batch(payload, GUIDANCE, question=ASK, n_ask=5, n_grade=NGRADE)
        except Exception as e:
            print(f"  ep{ep} f{grp[0]}+: {type(e).__name__}: {e}", flush=True)
            continue
        for (f, task, x), r in zip(meta, rs):
            # A judge that has died answers every call identically, so a file of
            # the right length with no content is indistinguishable from success
            # unless the empties are counted.
            if r.get("error") or not r.get("text", "").strip():
                nempty += 1
                if nempty >= 5 and ntot < 5:
                    raise SystemExit(f"판정기가 답을 안 함: {r.get('error', 'empty')}")
                continue
            nempty = 0
            picks = r.get("picks") or [None] * 5
            K = ceiling_from_checks(picks)
            rec = {"ep": ep, "f": f, "task": task,
                   **{q: picks[i] for i, q in enumerate("ABCDE")},
                   "K": round(K, 3), "K_snap": snap(K),
                   "text": r.get("text", "").replace("\n", " | "),
                   **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in x.items()}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ntot += 1
        out.flush()
    print(f"ep{ep}: {ntot} chunks", flush=True)
out.close()
print(f"끝: {ntot} chunks -> {OUT}")
