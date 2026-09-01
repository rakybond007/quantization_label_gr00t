"""Is a graded answer per question a better confidence than YES/NO?

phase6's five axes stay exactly as they are. The only thing that changes is what
the model may answer with, so whatever the comparison shows is about the answer
scale and not about the questions. Both arms run off the SAME image prefill on
the SAME chunks, and the aggregation afterwards is identical, so the pilot
isolates one variable.

Three readings come out of it:

  binary    P(YES) read over the {YES,NO} tokens — this is phase6 as shipped.
  grade     the level the model picks, (g-1)/(n-1). This is the pure form of
            "let the model decide the grade": a written answer, nothing else.
  expected  the probability-weighted mean over the levels. The binary reading is
            the two-level case of exactly this, so it is the like-for-like
            comparison; `grade` is what you would get without logits at all.

Run the judge server first, then:
    python phase7_grade_pilot.py <port> [n_chunks]
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
OUT = os.environ.get("PHASE7_OUT", f"{BASE}/output/_gate_distill/phase7_pilot")

PORT = sys.argv[1] if len(sys.argv) > 1 else "8120"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
NGRADE = 5

# phase6's questions, verbatim. Only the answer instruction differs.
_AXES = (
    "A) Is the gripper touching nothing at all, moving through empty space?\n"
    "B) Is the gripper holding a handle, a knob, a lever, or the edge of a door or drawer?\n"
    "C) Is the gripper closing onto an object until its weight is taken, or setting an object down\n"
    "   and letting go of it?\n"
    "D) Is there a rim, an edge, a shelf or a wall in the way, so the gripper or what it carries\n"
    "   would hit it going straight and has to lift over or go around?\n"
    "E) Is the target so far from the arm that the robot has to drive its base while the arm is\n"
    "   still moving?\n"
)
_PRE = ("The measurements above already tell you how the arm and gripper move; do not repeat them. "
        "Answer only what the cameras show about the MOMENT in front of you. ")

ASK_BINARY = (_PRE + "Answer each check on its own line as \"A) YES\" or \"A) NO\", in order, "
              "nothing else. YES and NO refer only to the question asked.\n" + _AXES + "Answer:")

# The anchors describe the same axis at each level, so a grade is a strength and
# not a second question. 5 is the condition plainly holding, 1 is it plainly not.
ASK_GRADED = (
    _PRE + "Answer each check on its own line as \"A) 3\", in order, nothing else — one digit "
    "from 1 to 5 per check, rating how strongly that check holds RIGHT NOW:\n"
    "  5 = it plainly holds, with nothing ambiguous about it\n"
    "  4 = it holds, with some doubt\n"
    "  3 = genuinely borderline, it could be read either way\n"
    "  2 = it mostly does not hold\n"
    "  1 = it plainly does not hold\n"
    "A grade refers only to the check on that line.\n" + _AXES + "Answer:")

info = json.load(open(f"{DS}/meta/info.json"))
_acts = {}


def actions(ep):
    """Episode actions, needed for the computed facts the questions refer to."""
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

# manifest lines carry the .png; the tile stem is what the ids are parsed from
names = [l.strip()[:-4] if l.strip().endswith(".png") else l.strip()
         for l in open(MAN) if l.strip()]
random.Random(0).shuffle(names)
names = names[:N]
print(f"pilot over {len(names)} chunks, {NGRADE} levels", flush=True)

G = open(GUID).read().strip()
gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)
os.makedirs(OUT, exist_ok=True)
# Resumable. The background partition preempts and requeues, and an allocation
# can sit suspended for hours, so a run that restarts from zero throws away
# whatever it had. Chunks already written are skipped and the files are appended.
# A chunk counts as done only when EVERY arm has it. A missing arm file is an
# empty set, not a skip: adding an arm has to re-run the chunks that predate it.
ARMS = ("binary", "graded", "text_binary", "text_graded")
seen = None
for name in ARMS:
    fp = f"{OUT}/{name}.jsonl"
    got = set()
    if os.path.exists(fp):
        for line in open(fp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            got.add((r["ep"], r["f"]))
    seen = got if seen is None else (seen & got)
seen = seen or set()
print(f"resuming past {len(seen)} chunks already labelled", flush=True)

fb = open(f"{OUT}/binary.jsonl", "a")
fg = open(f"{OUT}/graded.jsonl", "a")
# the same two questions, answered by the model in writing instead of by an
# argmax over the answer slot. Same prefill, same prompt, same chunks.
ftb = open(f"{OUT}/text_binary.jsonl", "a")
ftg = open(f"{OUT}/text_graded.jsonl", "a")

done = 0
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
    # The questions open with "the measurements above already tell you..." — so the
    # measurements have to be there. Without them the model is told not to repeat
    # something it was never given.
    x = descriptors(a, fr)
    ins = f"{instr.get(ep, '')}\n{facts_text(x)}"
    try:
        rb = gate.judge(views, ins, G, question=ASK_BINARY, n_ask=5)
        rg = gate.judge(views, ins, G, question=ASK_GRADED, n_ask=5, n_grade=NGRADE)
        rtb = gate.judge(views, ins, G, question=ASK_BINARY, n_ask=5, mode="text")
        rtg = gate.judge(views, ins, G, question=ASK_GRADED, n_ask=5,
                         n_grade=NGRADE, mode="text")
    except Exception as e:
        print(f"  skip {nm}: {type(e).__name__}", flush=True)
        continue
    row = {"ep": ep, "f": fr, "task": ins}
    fb.write(json.dumps({**row, **{q: rb["confidences"][i] for i, q in enumerate("ABCDE")},
                         "ans": rb["answer"]}, ensure_ascii=False) + "\n")
    fg.write(json.dumps({**row,
                         **{q: rg["expected"][i] for i, q in enumerate("ABCDE")},
                         **{f"{q}_pick": rg["grades"][i] for i, q in enumerate("ABCDE")},
                         "ans": rg["answer"]}, ensure_ascii=False) + "\n")
    ftb.write(json.dumps({**row, "picks": rtb.get("picks"),
                          "n_parsed": rtb.get("n_parsed"),
                          "text": rtb.get("text", ""),
                          "logit_ans": rb["answer"]}, ensure_ascii=False) + "\n")
    ftg.write(json.dumps({**row, "picks": rtg.get("picks"),
                          "n_parsed": rtg.get("n_parsed"),
                          "text": rtg.get("text", ""),
                          "logit_ans": rg["answer"]}, ensure_ascii=False) + "\n")
    done += 1
    if done % 100 == 0:
        fb.flush(); fg.flush(); ftb.flush(); ftg.flush()
        print(f"  {done}/{len(names)}", flush=True)

fb.close(); fg.close(); ftb.close(); ftg.close()
print(f"done {done} -> {OUT}/{{binary,graded,text_binary,text_graded}}.jsonl")
