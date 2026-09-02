"""Five questions chosen by what quantization actually broke, answered in writing.

The questions are not invented from the scenes. The 24 robocasa tasks split by
measured K2 damage into a RISK pool (6 tasks where compression cost success) and
a STABLE pool (15 where it was preserved or improved). Candidate phases were read
off the task instructions of each pool, then ranked by how many of that pool's
tasks they cover and how much damage those tasks carry -- with over-coverage of
the opposite pool reported, because a phase that fires on both sides cannot tell
them apart. That is what killed phase6's B: it lumps buttons and knobs (damaged)
together with doors and drawers (improved) and answers both the same way.

  RISK      cover  over    STABLE                    cover  over
  R4 .440   4/6    0/5     S1 .700                   4/15   0/6
  R5 .060   2/6    0/2     S3 .700                   5/15   0/5
                           S4 1.000                  8/15   2/10

B was first chosen to cover only what A misses, and that constraint left it
holding the risk pool's two LEAST damaged tasks (+0.040, +0.020), hence a
damage weight of 0.120 -- it changed the final ranking by 0.019 Spearman.
Overlap is allowed, so B is now the endpoint-tolerance question, which covers
the whole risk pool (0.500) at the cost of 0.200 over-coverage, net 0.300
against A's 0.440. Weights 0.595 / 0.405 instead of 0.880 / 0.120. S4 has the widest reach
and its two over-covers are exactly R5's two tasks -- kept deliberately: at task
level that is a collision, but at chunk level it is a sequence, since putting a
bottle in a cabinet IS crossing open space before it is entering a tight space.
Two questions disagreeing across a chunk boundary is the transition being found.

Each question judges ONE thing and each admits degrees, which is what makes a
grade meaningful rather than a category wearing an ordinal costume.

The grade scale is anchored to the SCENE, not to confidence in the judgement.
The previous scale's negative half was dead -- level 2 ("mostly does not hold")
took 43 of 9,490 slots, because it is not distinguishable from level 1 by
looking. Here level 2 means something visible: not yet, but about to be.

    python phase9_two_sided.py <port> [n_chunks]
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
OUT = os.environ.get("PHASE9_OUT", f"{BASE}/output/_gate_distill/phase9")
PORT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # 0 = every tile
SHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NSHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 1
from phase9_checks import ASK, GUIDANCE, NGRADE, SIGN, WEIGHT  # noqa: E402,F401

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
if N:
    names = names[:N]
# Shard by position in the shuffled list, so every shard sees the same mix of
# tasks rather than one shard drawing a single episode range.
names = names[SHARD::NSHARD]

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
os.makedirs(OUT, exist_ok=True)
FP = f"{OUT}/labels.jsonl" if NSHARD == 1 else f"{OUT}/labels_s{NSHARD}_{SHARD}.jsonl"
seen = set()
if os.path.exists(FP):
    for line in open(FP):
        try:
            r = json.loads(line)
        except Exception:
            continue
        seen.add((r["ep"], r["f"]))
print(f"{len(names)} chunks, resuming past {len(seen)}", flush=True)

fh = open(FP, "a")
done = nfull = nempty = 0
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
    # The measured facts go in, in a shape that is the same length on every frame
    # (facts_text now emits five sentences unconditionally). Measured over the
    # same 1,790 chunks: with them the task-level rho is -0.692 and the checks
    # separate at 0.65; without them -0.642, and D and E collapse into each other
    # at 0.71 because nothing in a still frame tells "carrying across" from
    # "setting down" -- speed and direction do.
    x = descriptors(a, fr)
    ins = f"{instr.get(ep, '')}\n{facts_text(x)}"
    try:
        r = gate.judge(views, ins, GUIDANCE, question=ASK, n_ask=5,
                       n_grade=NGRADE, mode="text")
    except Exception as e:
        print(f"  skip {nm}: {type(e).__name__}", flush=True)
        continue
    # A dead judge answers every request identically, and a run that keeps
    # writing rows through it produces a file of the right length and no
    # content -- which counting outputs cannot distinguish from success.
    if r.get("error") or not r.get("text", "").strip():
        nempty += 1
        if nempty >= 5 and done < 5:
            raise SystemExit(f"판정 서버 응답 없음 ({nempty}건 연속): {r.get('error','빈 응답')}")
        continue
    nempty = 0
    picks = r.get("picks") or [None] * 5
    nfull += int(all(p is not None for p in picks))
    fh.write(json.dumps({"ep": ep, "f": fr,
                         **{q: picks[i] for i, q in enumerate("ABCDE")},
                         "text": r.get("text", "")}, ensure_ascii=False) + "\n")
    done += 1
    if done % 100 == 0:
        fh.flush()
        print(f"  {done}/{len(names)}  5칸 완답 {nfull}", flush=True)

fh.close()
print(f"done {done}, 5칸 완답 {nfull} -> {FP}")
