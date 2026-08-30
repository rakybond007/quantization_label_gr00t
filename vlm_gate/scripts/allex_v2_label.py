"""allex v2 - two-stage variable-ratio labelling of the subtask-labelled dataset.

Per 16-step chunk:
  stage 1  general prompt (allex_common_v5 GUIDANCE/ASK) -> base confidence p
  stage 2  task-specific prompt (allex_v2_common)        -> ceiling K_max
  K = snap(1 + p*(K_max-1)) in {1, 2, 2.5, 3}

Both stages are one Cosmos call each: a single image prefill with four
teacher-forced "X) YES/NO" slots, P(YES) read off the {YES,NO} tokens.

  python allex_v2_label.py <PORT> <SHARD> <NSHARDS> [EP_LIST]
"""
import json, os, sys
import numpy as np, pandas as pd, av
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
from allex_common_v5 import GUIDANCE, ASK
from allex_v2_common import (descriptors, facts_text, stage1_confidence,
                             STAGE2_GUIDANCE, STAGE2_ASK, stage2_facts,
                             ceiling_from_stage2, final_ratio, TASKS)

DS = "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin"
OUTDIR = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v2")
os.makedirs(OUTDIR, exist_ok=True)
CHUNK = 16

PORT = sys.argv[1]
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSH = int(sys.argv[3]) if len(sys.argv) > 3 else 1
EPS = [int(e) for e in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
TAG = os.environ.get("TAG", f"s{NSH}_{SHARD}")
OUT = f"{OUTDIR}/labels_{TAG}.jsonl"

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

episodes = []
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l); episodes.append((d["episode_index"], d["length"]))
if EPS is not None:
    episodes = [(e, n) for e, n in episodes if e in EPS]
else:
    episodes = [(e, n) for e, n in episodes if e % NSH == SHARD]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["ep"], r["f"]))
        except Exception:
            pass


def grab(ep, frames, side):
    """Decode the requested frame indices from one ego camera."""
    want = set(frames); got = {}
    path = f"{DS}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


out = open(OUT, "a")
ntot = nerr = 0
for ep, N in episodes:
    d = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    starts = [f for f in range(0, len(A) - CHUNK, CHUNK) if (ep, f) not in done]
    if not starts:
        print(f"ep{ep}: already done", flush=True); continue
    L = grab(ep, starts, "left"); R = grab(ep, starts, "right")
    n0 = ntot
    for f in starts:
        try:
            x = descriptors(A, WR, WL, f, CHUNK)
            seg = ti[f:f + CHUNK]
            task = TASKS[int(np.bincount(seg, minlength=len(TASKS)).argmax())]
            views = [L[f], R[f]]
            # ---- stage 1: general "is this moment safe to compress at all"
            r1 = gate.judge(views, f"{task}\n{facts_text(x)}", GUIDANCE,
                            question=ASK, n_ask=4)
            c1 = r1.get("confidences")
            if not c1 or len(c1) != 4:
                raise ValueError(f"stage1 parse: {r1.get('error','')}")
            p = stage1_confidence(c1, x)
            # ---- stage 2: task-specific ceiling
            r2 = gate.judge(views, f"{task}\n{stage2_facts(task, x)}", STAGE2_GUIDANCE,
                            question=STAGE2_ASK, n_ask=4)
            c2 = r2.get("confidences")
            if not c2 or len(c2) != 4:
                raise ValueError(f"stage2 parse: {r2.get('error','')}")
            K_max = ceiling_from_stage2(task, *c2)
            K_pre = final_ratio(p, K_max)
            rec = {"ep": ep, "f": f, "task": task, "p": float(p), "K_max": float(K_max),
                   "K_pre": float(K_pre),
                   **{f"s1_{k}": float(v) for k, v in zip("ABCD", c1)},
                   **{f"s2_{k}": float(v) for k, v in zip("ABCD", c2)},
                   "ans1": r1.get("answer", ""), "ans2": r2.get("answer", ""),
                   **{k: (int(v) if isinstance(v, bool) else float(v)) for k, v in x.items()}}
            out.write(json.dumps(rec) + "\n"); ntot += 1
            if ntot % 100 == 0:
                out.flush(); print(f"  {TAG}: {ntot} chunks", flush=True)
        except Exception as e:
            nerr += 1
            if nerr <= 5:
                print("ERR", type(e).__name__, str(e)[:160], flush=True)
    out.flush()
    print(f"ep{ep}: {ntot-n0} chunks (total {ntot}, err {nerr})", flush=True)
out.close()
print(f"{TAG} done: {ntot} chunks, {nerr} errors -> {OUT}")
