"""Ask the graded ceiling question again, letting the model WRITE the answer.

The shipped run reads the answer slot's logits over the digit tokens. That was
a scaffold inherited from the binary YES/NO path, not evidence that the model
cannot answer a multiple-choice question in text. This asks the SAME question,
on the SAME chunks, with the SAME prompt, and only changes who names the level:
the model instead of an argmax over logits.

Three things come out of it:
  * how often the model answers in the requested form at all (`pick` is None
    when it does not -- nothing is defaulted, because that rate is the result);
  * whether the text answer agrees with the logit argmax;
  * what the model writes when it disagrees, which the logit path cannot show.

    python allex_graded_text_pilot.py <port> [n_sample]
"""
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_graded_ceiling import (LEVELS, STAGE2_GRADED_ASK,  # noqa: E402
                                  STAGE2_GRADED_GUIDANCE)
from allex_v2_common import descriptors, stage2_facts  # noqa: E402
from vlm_gate import VLMGate  # noqa: E402

PORT = sys.argv[1]
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
DS = os.environ["ALLEX_DS"]
LAB = os.environ["ALLEX_LABELS"]
OUT = os.environ.get("ALLEX_TEXT_OUT", "/tmp/graded_text_pilot.jsonl")
CHUNK = 16

rows = [json.loads(l) for l in open(LAB)]
rows.sort(key=lambda r: (r["ep"], r["f"]))
step = max(1, len(rows) // NSAMP)
samp = rows[::step][:NSAMP]
print(f"{len(rows)} labelled chunks -> sampling {len(samp)}", flush=True)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=600)
tasks = [json.loads(l)["task"] for l in open(f"{DS}/meta/tasks.jsonl")]

by_ep = {}
for r in samp:
    by_ep.setdefault(r["ep"], []).append(r)


def grab(ep, frames, side):
    want, got = set(frames), {}
    path = f"{DS}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


out = open(OUT, "w")
n = nparse = nagree = 0
picks, logit_picks = [], []
for ep in sorted(by_ep):
    d = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    fs = [r["f"] for r in by_ep[ep]]
    L, R = grab(ep, fs, "left"), grab(ep, fs, "right")
    for r in by_ep[ep]:
        f = r["f"]
        x = descriptors(A, WR, WL, f, CHUNK)
        seg = ti[f:f + CHUNK]
        task = tasks[int(np.bincount(seg, minlength=len(tasks)).argmax())]
        res = gate.judge([L[f], R[f]], f"{task}\n{stage2_facts(task, x)}",
                         STAGE2_GRADED_GUIDANCE, question=STAGE2_GRADED_ASK,
                         n_grade=len(LEVELS), mode="text")
        pick = res.get("pick")
        lp = int(LEVELS.index(r["K_max_pick"])) + 1 if r.get("K_max_pick") in LEVELS else None
        n += 1
        if pick is not None:
            nparse += 1
            picks.append(pick)
            logit_picks.append(lp)
            nagree += int(pick == lp)
        out.write(json.dumps({"ep": ep, "f": f, "text_pick": pick,
                              "logit_pick": lp, "text": res.get("text", ""),
                              "n_new": res.get("n_new")}, ensure_ascii=False) + "\n")
        out.flush()
    print(f"ep{ep}: {len(by_ep[ep])} done (total {n}, parsed {nparse})", flush=True)

print(f"\n=== {n} chunks ===")
print(f"형식대로 답한 비율 : {nparse}/{n} = {nparse / max(n,1):.1%}")
if nparse:
    print(f"logit argmax 와 일치: {nagree}/{nparse} = {nagree / nparse:.1%}")
    import collections
    print(f"텍스트 선택 분포   : {dict(sorted(collections.Counter(picks).items()))}")
    print(f"logit 선택 분포    : {dict(sorted(collections.Counter(logit_picks).items()))}")
print(f"\n원문 -> {OUT}")
