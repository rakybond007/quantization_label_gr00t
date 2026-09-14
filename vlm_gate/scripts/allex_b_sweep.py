"""B 문구 후보를 한 번에 재고, 기준을 통과할 때까지 돌린다.

기준(미리 정해 둔다 -- 결과를 보고 고르지 않는다):
  1. 봉투 구간에서 B 가 4등급 이상 20% 넘게 뜬다
  2. conf 가 양손(박스조작, 실측 0.533)에서 낮고 한손에서 높다
  3. A 가 양손에서 뜨고 한손에서 안 뜬다

한 판씩 던지고 기다리는 대신, 후보를 순서대로 같은 청크에 돌려 한 표에 모은다.
"""
import importlib, json, os, sys, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from allex_v2_common import descriptors
from vlm_gate import VLMGate
import av
from PIL import Image

D = os.environ.get("ALLEX_DS", "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
PORT = sys.argv[1]
EPS = [int(x) for x in os.environ.get("ALLEX_EPS", "56 152").split()]
NCHUNK = int(os.environ.get("N_CHUNK", "60"))
CK = importlib.import_module("allex_v4c_checks")

# --- B 후보. 앞서 실패한 것들의 원인을 피한다 --------------------------------
#   움직임/변형을 묻지 않는다(한 장면에 없다) · 색·재질로 좁히지 않는다 ·
#   부정 정의를 쓰지 않는다.
CANDS = [
 ("pouch",
  "Is the item a POUCH OR PACKET -- a soft mailer, a padded envelope, a plastic\n"
  "   parcel -- rather than a rigid carton with square sides?"),
 ("no_height",
  "Does the item have ALMOST NO HEIGHT -- it lies like a sheet or a cushion on the\n"
  "   surface -- rather than standing tall enough to be gripped around its sides?"),
 ("no_side_grip",
  "Would the fingers find NOTHING TO CLOSE AROUND on this item -- no side wall, no\n"
  "   edge to hook -- so the only way to take it is from on top?"),
 ("crushable",
  "Would this item GIVE WAY under the fingers if the hand squeezed harder --\n"
  "   rather than hold its shape however hard it is gripped?"),
 ("flat_press",
  "Is the item FLAT AGAINST THE SURFACE, with the hand coming down onto it from\n"
  "   above -- rather than a solid object standing proud that the hand takes from\n"
  "   the side?"),
 ("not_cardboard",
  "Does the item look like ANYTHING OTHER THAN plain brown or white cardboard --\n"
  "   coloured film, glossy plastic, printed wrapping, a sealed sleeve?"),
]

def grab(ep, frames, side):
    want, got = set(frames), {}
    p = f"{D}/videos/chunk-{ep//1000:03d}/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
    with av.open(p) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got

TASK = json.loads(open(f"{D}/meta/tasks.jsonl").readline())["task"]
gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

# 장면 준비 -- **눈으로 표시한 정답지**를 쓴다.
# 색으로 봉투를 찾으려 했더니 배경의 봉투를 세어 못 쓴다(상자 프레임이 0.637 로
# 가장 높게 나왔다). 손이 실제로 다루는 물체는 사람이 봐야 안다.
EYE = json.load(open(f"{BASE}/_tmp/eye_labels.json")) if os.path.exists(f"{BASE}/_tmp/eye_labels.json") else {}
MANUAL = {56: range(960, 1700), 152: range(1400, 1900),
          216: range(560, 1000), 312: range(900, 1100)}
scenes = []
byep = collections.defaultdict(list)
for k, v in EYE.items():
    if v < 0: continue
    ep, f = k.split("_"); byep[int(ep)].append((int(f), bool(v)))
for ep in MANUAL:
    byep[ep] = []
for ep, want in sorted(byep.items()):
    d = pd.read_parquet(f"{D}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    if not want:                      # 수동 구간 에피
        rng = MANUAL[ep]
        oneh = [f for f in range(0, len(A) - 16, 16)
                if descriptors(A, WR, WL, f, 16)["one_handed"]]
        bag = [(f, True) for f in oneh if f in rng][:6]
        box = [(f, False) for f in oneh if f not in rng][:6]
        want = bag + box
    fr = [f for f, _ in want]
    L = grab(ep, fr, "left"); R = grab(ep, fr, "right")
    for f, isbag in want:
        if f in L and f in R:
            x = descriptors(A, WR, WL, f, 16)
            scenes.append({"ep": ep, "f": f, "x": x, "imgs": [L[f], R[f]], "bag": isbag})
print(f"[i] 장면 {len(scenes)} · 봉투 {sum(s['bag'] for s in scenes)} · "
      f"상자 {sum(not s['bag'] for s in scenes)}", flush=True)

def ask_with(bt):
    axes = CK.ASK.split("B) ")[0] + "B) " + bt + "\n" + \
           "C) " + CK.ASK.split("C) ")[1]
    return axes

out = {}
for name, bt in CANDS:
    ASK = ask_with(bt)
    picks = []
    for i in range(0, len(scenes), 8):
        batch = scenes[i:i+8]
        payload = [(s["imgs"], CK.facts_v3(s["x"], TASK)) for s in batch]
        try:
            rs = gate.judge_batch(payload, CK.GUIDANCE, question=ASK, n_ask=4, n_grade=5)
        except Exception as e:
            print(f"  {name} batch err {e}", flush=True); continue
        for s, r in zip(batch, rs):
            p = r.get("picks")
            if not p or len(p) < 4 or any(v is None for v in p): continue
            picks.append({"ep": s["ep"], "f": s["f"], "one_handed": bool(s["x"]["one_handed"]),
                          "bag": bool(s.get("bag")),
                          "A": int(p[0]), "B": int(p[1]), "C": int(p[2]), "D": int(p[3]),
                          "gp": r.get("grade_probs")})
    out[name] = picks
    bh = np.array([q["B"] for q in picks]); oh = np.array([q["one_handed"] for q in picks])
    bg = np.array([q["bag"] for q in picks])
    # **핵심 지표: 한손 안에서 봉투와 상자를 가르는가.**
    s_bag = bh[bg]; s_box = bh[oh & ~bg]
    sep = (s_bag.mean() - s_box.mean()) if len(s_bag) and len(s_box) else 0.0
    print(f"  {name:12s} n={len(picks):3d}  봉투 {s_bag.mean() if len(s_bag) else 0:.2f} "
          f"vs 한손상자 {s_box.mean() if len(s_box) else 0:.2f}  **차 {sep:+.2f}**  "
          f"양손 {bh[~oh].mean() if (~oh).any() else 0:.2f}  "
          f"분포 {dict(sorted(collections.Counter(bh.tolist()).items()))}", flush=True)
import time as _t
STAMP = os.environ.get("SWEEP_TAG") or _t.strftime("%H%M%S")
json.dump(out, open(f"output/allex_b_sweep_{STAMP}.json", "w"))
print("-> output/allex_b_sweep.json")
