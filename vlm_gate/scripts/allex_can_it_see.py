"""판정기가 봉투와 상자를 **구분할 수 있는지** 직접 묻는다.

문항을 더 깎기 전에 알아야 한다. 게이트 문항이 아니라 맨 질문으로 묻는다 --
등급도 계산 사실도 없이, 화면에 무엇이 있는지만.
"""
import json, os, sys
import numpy as np, pandas as pd, av
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import descriptors
from vlm_gate import VLMGate

D = "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4"
PORT = sys.argv[1]
BAG = {56: range(960, 1700), 152: range(1400, 1900), 216: range(560, 1000), 312: range(900, 1100)}

def grab(ep, frames, side):
    want, got = set(frames), {}
    p = f"{D}/videos/chunk-{ep//1000:03d}/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
    with av.open(p) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want): break
    return got

scenes = []
for ep, rng in BAG.items():
    d = pd.read_parquet(f"{D}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values); WL = np.stack(d["action.left_wrist_wrt_base"].values)
    oneh = [f for f in range(0, len(A)-16, 16) if descriptors(A, WR, WL, f, 16)["one_handed"]]
    bag = [f for f in oneh if f in rng][:8]
    box = [f for f in oneh if f not in rng][:8]
    fr = bag + box
    L = grab(ep, fr, "left"); R = grab(ep, fr, "right")
    for f in fr:
        if f in L and f in R:
            scenes.append({"ep": ep, "f": f, "bag": f in bag, "imgs": [L[f], R[f]]})
print(f"[i] 봉투 {sum(s['bag'] for s in scenes)} · 상자 {sum(not s['bag'] for s in scenes)}", flush=True)

G = ("You are looking at two camera views of a robot arm at a parcel conveyor. "
     "Look only at the item the robot is working on right now.")
Q = ("Answer with a single digit and nothing else.\n"
     "1 = the item is a rigid cardboard box or carton\n"
     "2 = the item is a soft bag, plastic parcel or padded envelope\n"
     "3 = there is no item being worked on\n"
     "Answer:")
gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
res = []
for i in range(0, len(scenes), 8):
    b = scenes[i:i+8]
    try:
        rs = gate.judge_batch([(s["imgs"], "") for s in b], G, question=Q, n_ask=1, n_grade=3)
    except Exception as e:
        print("err", e, flush=True); continue
    for s, r in zip(b, rs):
        p = r.get("picks")
        if p and p[0] is not None:
            res.append({"bag": s["bag"], "said": int(p[0]), "ep": s["ep"], "f": s["f"]})
import collections
bagsaid = collections.Counter(r["said"] for r in res if r["bag"])
boxsaid = collections.Counter(r["said"] for r in res if not r["bag"])
print(f"\n실제 봉투 {sum(bagsaid.values())}장 -> 답: {dict(sorted(bagsaid.items()))}")
print(f"실제 상자 {sum(boxsaid.values())}장 -> 답: {dict(sorted(boxsaid.items()))}")
print("  (1=상자 2=봉투 3=없음)")
acc = (bagsaid[2] + boxsaid[1]) / max(1, len(res))
print(f"\n정확도 {acc:.1%}  <- 50% 면 못 보는 것이다")
json.dump(res, open("output/allex_can_it_see.json", "w"))
