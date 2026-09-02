"""One graded question, answered in writing, replacing phase6's five checks.

The five checks each ask whether a RARE state holds right now. Measured over
1,898 chunks, that is why they cannot rank: A answers "borderline" on 98.5% of
chunks and C answers "plainly not" on 98.1%, because those states genuinely are
rare -- grasping occupies about 7% of the data. A question whose honest answer
is the same everywhere carries no ordering, and reading the answer slot's logits
only hides that behind sub-threshold ripple the model never asserted.

So the five axes are not dropped and not split further; they are collapsed into
ONE ordered question that every chunk has a real answer to. The ordering is by
how much of the moment can be thinned out, which is what the pipeline needs:

    1  arriving   -- closing on an object, setting it down, letting go   (was C)
    2  fixed path -- handle, knob, lever, door or drawer edge            (was B)
    3  obstructed -- a rim, edge, shelf or wall has to be cleared        (was D)
    4  travelling -- open space, or reaching toward an untouched object  (was A)

Axis E was base motion. That is measurable from the actions, so it is stated as
a fact rather than asked -- the standing rule that computed quantities are never
put to the model.

The answer is written by the model and parsed. No logits are read.

    python phase8_single_graded.py <port> [n_chunks]
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate  # noqa: E402
from robocasa_descriptors import descriptors, facts_text  # noqa: E402

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
MAN = f"{BASE}/output/_gate_distill/tiles_manifest.txt"
GUID = f"{BASE}/analysis/_evolver/_varkA/robocasa_guidance_phase_v5.txt"
OUT = os.environ.get("PHASE8_OUT", f"{BASE}/output/_gate_distill/phase8_single")
PORT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
LEVELS = 4

ASK = (
    "The measurements above already tell you how the arm and gripper move; do not repeat "
    "them. Answer only from what the cameras show about the MOMENT in front of you.\n"
    "Choose the ONE line below that best matches this moment. Answer on its own line as "
    "\"A) 3\" -- the digit of that line and nothing else.\n"
    "1 = the gripper is arriving: closing onto an object until its weight is taken, "
    "setting an object down, or letting go of it. Where it ends up is the whole point.\n"
    "2 = the gripper is working something fixed in place -- a handle, a knob, a lever, or "
    "the edge of a door or drawer -- so the hinge or slide dictates the path.\n"
    "3 = a rim, an edge, a shelf or a wall is in the way, so the gripper or what it "
    "carries has to be lifted over or taken around it.\n"
    "4 = the gripper is simply travelling: crossing open space, reaching toward something "
    "it has not touched yet, or withdrawing after letting go.\n"
    "Answer:")

info = json.load(open(f"{DS}/meta/info.json"))
_acts = {}


def actions(ep):
    if ep not in _acts:
        ch = ep // info["chunks_size"]
        try:
            _acts[ep] = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            _acts[ep] = None
    return _acts[ep]


instr = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

names = [l.strip()[:-4] if l.strip().endswith(".png") else l.strip()
         for l in open(MAN) if l.strip()]
random.Random(0).shuffle(names)          # same seed as phase7: the SAME chunks
names = names[:N]

G = open(GUID).read().strip()
gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
os.makedirs(OUT, exist_ok=True)

seen = set()
FP = f"{OUT}/single.jsonl"
if os.path.exists(FP):
    for line in open(FP):
        try:
            r = json.loads(line)
        except Exception:
            continue
        seen.add((r["ep"], r["f"]))
print(f"{len(names)} chunks, resuming past {len(seen)}", flush=True)

f_out = open(FP, "a")
done = nfail = 0
for nm in names:
    try:
        ep = int(nm.split("ep")[1].split("_")[0])
        fr = int(nm.split("_f")[1])
    except Exception:
        continue
    if (ep, fr) in seen:
        continue
    a = actions(ep)
    if a is None or fr >= len(a) - 4:
        continue
    try:
        im = np.array(Image.open(f"{TILES}/{nm}.png").convert("RGB"))
        h, w, _ = im.shape
        views = [Image.fromarray(im[:, k * w // 3:(k + 1) * w // 3]) for k in range(3)]
    except Exception:
        continue
    x = descriptors(a, fr)
    ins = f"{instr.get(ep, '')}\n{facts_text(x)}"
    try:
        r = gate.judge(views, ins, G, question=ASK, n_grade=LEVELS, mode="text")
    except Exception as e:
        print(f"  skip {nm}: {type(e).__name__}", flush=True)
        continue
    pick = r.get("pick")
    nfail += int(pick is None)
    f_out.write(json.dumps({"ep": ep, "f": fr, "pick": pick,
                            "text": r.get("text", "")}, ensure_ascii=False) + "\n")
    done += 1
    if done % 200 == 0:
        f_out.flush()
        print(f"  {done}/{len(names)}  파싱실패 {nfail}", flush=True)

f_out.close()
print(f"done {done}, 파싱실패 {nfail} -> {FP}")
