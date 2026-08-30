"""DexJoCo chunk labeller -- the 1-call design, ported from `cosmos_1call_v6.py`.

Same three layers as RoboCasa:
  2nd layer (computed): `dexjoco_descriptors.descriptors` resolves everything the
      action numbers already answer -- hand opening/closing, direction reversal,
      speed, deceleration, and whether a K=2 merge is even trackable -- and
      `facts_text` states it to the judge as settled fact, never as a question.
  3rd layer (VLM): the two camera views + the evolved guidance + those facts, and
      the five questions the numbers cannot answer, asked in ONE call.
The record carries both, so the aggregation stays a noisy-OR of computed flags
and VLM risk answers.

DexJoCo differences from `cosmos_1call_v6.py` (this is why it is a new file):
  * TWO views, not three -- and click_mouse's base view is `ego_right` while the
    other five use `front`, so the view count comes from the dataset, never a
    hardcoded `range(3)`.
  * Actions are ABSOLUTE 22-D (arm 3+3 rotvec + 16-DoF hand), so the compression
    risk is a skipped intermediate target, not a summed delta.  That reasoning
    lives in `dexjoco_descriptors`; nothing here assumes deltas.
  * SIX datasets with independent episode numbering -- see
    `dexjoco_label_common` for the global `ep` key.
  * Frames are read through `DexjocoDataset`, because the v3.0 layout packs many
    episodes into one parquet; the reader caches whole data files.

Resumability: the output is opened in append mode and every (ep, f) already in
it is skipped, so a requeued preemption re-emits nothing.  `qgate labels` counts
duplicates on exactly that key, which is the check that this holds.

    python dexjoco_label_chunks.py <port> <shard> <n_shards> [--limit N]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate                                    # noqa: E402
from dexjoco_descriptors import descriptors, facts_text, computed_risk  # noqa: E402
from dexjoco_lerobot_reader import DEFAULT_ROOT, DexjocoDataset  # noqa: E402
import dexjoco_label_common as C                                # noqa: E402

PORT = sys.argv[1]
SHARD = int(sys.argv[2])
NSH = int(sys.argv[3])
LIMIT = int(os.environ.get("LIMIT", "0"))

TAG = os.environ.get("TAG", "dexjoco_v1")
OUT = os.environ.get("OUT", f"{C.BASE}/output/_gate_distill/{TAG}_s{NSH}_{SHARD}.jsonl")
TILES = os.environ.get("TILES", C.DEFAULT_TILES)
MANIFEST = os.environ.get("MANIFEST", C.DEFAULT_MANIFEST)
PROMPTS = os.environ.get("PROMPTS", f"{C.BASE}/analysis/_evolver/_dexjoco")
GUIDANCE = os.environ.get("GUIDANCE_FILE", f"{PROMPTS}/dexjoco_guidance_v1.txt")
QUESTIONS = os.environ.get("QUESTIONS_FILE", f"{PROMPTS}/dexjoco_questions_v1.txt")

G = open(GUIDANCE).read().strip()
ASK = open(QUESTIONS).read().strip()
# The answer slots are whatever the questions file actually asks -- counted from
# the "X)" lines, so evolving the question set does not need a code change.
SLOTS = [l.strip()[0] for l in ASK.splitlines()
         if len(l.strip()) > 2 and l.strip()[0].isupper() and l.strip()[1] == ")"]
NQ = len(SLOTS)
assert NQ >= 1, f"no 'X)' question lines found in {QUESTIONS}"

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)

_ds = {}


def DS(task):
    if task not in _ds:
        _ds[task] = DexjocoDataset(os.path.join(DEFAULT_ROOT, task))
    return _ds[task]


_acts = {}


def A(task, ep_local):
    """(T, 22) absolute action chunk of one episode, small LRU over episodes."""
    k = (task, ep_local)
    if k not in _acts:
        try:
            _acts[k] = DS(task).actions(ep_local)
        except Exception:
            _acts[k] = None
        if len(_acts) > 40:
            for kk in list(_acts)[:20]:
                _acts.pop(kk, None)
    return _acts[k]


done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l)
            done.add((r["ep"], r["f"]))
        except Exception:
            pass
print(f"shard{SHARD}/{NSH}: {len(done)} chunks already done in {OUT}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = open(OUT, "a")
n = 0
for line in sorted(set(open(MANIFEST).read().split())):
    task, ep_local, f, nm = C.parse_manifest_line(line)
    ep = C.global_ep(task, ep_local)
    if ep % NSH != SHARD:
        continue
    if (ep, f) in done:
        continue
    a = A(task, ep_local)
    if a is None or f > len(a) - C.CHUNK_N:
        continue
    x = descriptors(a, f, n=C.CHUNK_N, k=2)

    ds = DS(task)
    nv = len(ds.views)                       # 2 here, but never hardcoded
    im = np.array(Image.open(os.path.join(TILES, task, "tiles", nm)).convert("RGB"))
    h, w, _ = im.shape
    views = [Image.fromarray(im[:, k * w // nv:(k + 1) * w // nv]) for k in range(nv)]

    ins = f"{ds.instruction(ep_local)}\n{facts_text(x)}"
    r = gate.judge(views, ins, G, question=ASK, n_ask=NQ)
    c = r.get("confidences") or [0.0] * NQ
    rec = {"ep": ep, "f": f, "task": task, "ep_local": ep_local,
           **{k: float(v) for k, v in zip(SLOTS, c)},
           **computed_risk(x),
           "speed_mean": x["speed_mean"], "skip_excess": x["skip_excess"],
           "ans": r.get("answer", "")}
    out.write(json.dumps(rec) + "\n")
    n += 1
    if n % 200 == 0:
        print(f"shard{SHARD}: {n}", flush=True)
        out.flush()
    if LIMIT and n >= LIMIT:
        break
out.close()
print(f"shard{SHARD} done {n} new rows -> {OUT}")
