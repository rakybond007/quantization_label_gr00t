"""Burn the v2 variable-ratio labels onto the frames — one combined mp4.

Follows allex_render_labeled_video.py: video on top (both ego views), a panel
underneath carrying the decision for the 16-step chunk the frame belongs to.
What the panel shows, per chunk:
  the subtask name, the stage-1 confidence p, the stage-2 ceiling K_max, the
  final ratio K as a big badge, the computed descriptors, both stages' YES/NO
  answers, and the ratio timeline for the whole segment being played.

  python allex_v2_render.py [out.mp4]
The clip list below is the default: episode 0 end to end (it visits all four
subtasks) followed by one segment per task taken from other episodes.
"""
import json, os, sys, collections
import numpy as np, pandas as pd, av
from PIL import Image, ImageDraw, ImageFont

DS = "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin"
OUTDIR = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v2")
REC = f"{OUTDIR}/records.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/quantization_agent_workspace/assets/videos/allex_v2_variable_ratio.mp4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CHUNK = 16
VW = 448                       # per-view display size (source is 256x256)
W = VW * 2                     # 896: two ego views side by side
PANEL = 312
FT = "/usr/share/fonts/truetype/dejavu"
f_big = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 38)
f_mid = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 19)
f_hd = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 15)
f_sm = ImageFont.truetype(f"{FT}/DejaVuSansMono.ttf", 13)

RCOL = {1.0: (226, 92, 84), 2.0: (232, 176, 84), 2.5: (150, 205, 120), 3.0: (86, 196, 124)}
S1Q = ["A soft/bag", "B reorienting", "C seating pose", "D empty/reaching"]
S2Q = ["A sideways move", "B precise stop", "C two-hand turn", "D soft bag"]

rec = collections.defaultdict(dict)
for l in open(REC):
    r = json.loads(l); rec[r["ep"]][r["f"]] = r

# ---- clip list: ep0 in full, then one segment per task from another episode
def seg_for(task, avoid=(0,), want=170):
    segs = [json.loads(l) for l in open(f"{DS}/meta/subtasks.jsonl")]
    cand = [s for s in segs if s["label"] == task and s["episode_index"] not in avoid
            and s["episode_index"] in rec
            and (s["end_frame"] - s["start_frame"]) >= want]
    if not cand:
        cand = [s for s in segs if s["label"] == task and s["episode_index"] in rec]
    if not cand:
        return None
    s = sorted(cand, key=lambda s: (s["episode_index"], s["start_frame"]))[len(cand) // 2]
    a = (s["start_frame"] // CHUNK) * CHUNK
    b = min(s["end_frame"], a + want)
    return (s["episode_index"], a, b, task)

CLIPS = [(0, 0, max(rec[0]) + CHUNK, "episode 0 — all four subtasks")] if 0 in rec else []
for t in ["Bring Object", "Rotate Box", "Pass Object", "Rotate PolyBag"]:
    s = seg_for(t)
    if s:
        CLIPS.append((s[0], s[1], s[2], f"episode {s[0]} — {t}"))
if not CLIPS:                                   # no ep0 labelled: use what we have
    ep = sorted(rec)[0]
    CLIPS = [(ep, 0, max(rec[ep]) + CHUNK, f"episode {ep}")]


COL1, COL2, COL3 = 16, 320, 612          # computed | stage 1 | stage 2
ROW0 = 132                               # first data row of the three columns


def panel(r, title, ks, pos, ep, f):
    im = Image.new("RGB", (W, PANEL), (18, 18, 22)); d = ImageDraw.Draw(im)
    K = r["K"]; col = RCOL.get(K, (200, 200, 200))
    d.rounded_rectangle([14, 12, 190, 76], 8, fill=col)
    d.text((30, 24), "K = 1" if K == 1.0 else f"K = {K:g}", font=f_big, fill=(15, 15, 18))
    d.text((206, 14), r["task"], font=f_big, fill=(240, 240, 245))
    d.text((206, 58), f"p {r['p']:.3f}    K_max {r['K_max']:.2f}    stages asked K {r['K_pre']:g}"
                      + ("    HARD BLOCK -> 1" if r["blocked"] else ""),
           font=f_mid, fill=(240, 150, 140) if r["blocked"] else (210, 210, 215))
    # confidence bar for p
    d.rectangle([16, 88, W - 16, 102], outline=(70, 70, 78))
    d.rectangle([16, 88, 16 + int((W - 32) * min(1.0, max(0.0, r["p"]))), 102], fill=col)
    # ---- column headers
    d.text((COL1, 110), "computed", font=f_hd, fill=(250, 220, 120))
    d.text((COL2, 110), "stage 1 - safe?", font=f_hd, fill=(250, 220, 120))
    d.text((COL3, 110), "stage 2 - how far?", font=f_hd, fill=(250, 220, 120))
    trend = "closing" if r.get("closing") else ("opening" if r.get("opening") else "steady")
    rows = [f"gap {r['gap_mean']:.2f}m {trend} {r['gap_rate']*1000:.1f}mm/s",
            f"arm {r['arm_speed']:.3f} rad/step" + ("  SLOWING" if r.get("slowing") else ""),
            f"turn {r['wrist_rot']:.0f}deg asym {r['rot_asym']:.0f}"
            + ("  HELD" if r.get("held") else "") + ("  1ARM" if r.get("one_handed") else ""),
            f"skip K2 {r['merge_demand_k2']:.3f} K3 {r['merge_demand_k3']:.3f}",
            f"travel {r['translation']*100:.0f}cm fingers {r['hand_change']:.3f}"]
    for i, t in enumerate(rows):
        d.text((COL1, ROW0 + i * 18), t, font=f_sm, fill=(200, 200, 206))
    for x0c, pre, names, safeq in ((COL2, "s1_", S1Q, "D"), (COL3, "s2_", S2Q, "A")):
        for i, (q, nm) in enumerate(zip("ABCD", names)):
            v = r[pre + q]
            d.text((x0c, ROW0 + i * 18), f"{nm:<17}{'YES' if v >= .5 else 'NO':>3} {v:.2f}",
                   font=f_sm, fill=(240, 150, 140) if (v >= .5 and q != safeq) else
                   ((140, 220, 170) if v >= .5 else (150, 150, 158)))
    # ratio timeline for this clip
    x0, y0, x1, y1 = 16, 236, W - 16, 288
    d.rectangle([x0, y0, x1, y1], outline=(70, 70, 78))
    n = len(ks)
    for j, kk in enumerate(ks):
        xa = x0 + int((x1 - x0) * j / n); xb = x0 + max(xa + 1, int((x1 - x0) * (j + 1) / n))
        hgt = int((y1 - y0) * (kk - 0.5) / 3.0)
        d.rectangle([xa, y1 - hgt, xb, y1], fill=RCOL.get(kk, (150, 150, 150)))
    px = x0 + int((x1 - x0) * pos / max(1, n))
    d.line([px, y0 - 4, px, y1 + 4], fill=(255, 255, 255), width=2)
    d.text((16, 292), f"{title}   frame {f}   chunk {pos+1}/{n}   "
                      f"[timeline: bar height = K, 1 low .. 3 high]",
           font=f_sm, fill=(150, 150, 158))
    return im


PREVIEW = os.environ.get("PREVIEW")       # "ep:frame" -> dump one PNG and exit
if PREVIEW:
    _ep, _f = (int(v) for v in PREVIEW.split(":"))
    _fs = sorted(rec[_ep]); _i = max(i for i, ff in enumerate(_fs) if ff <= _f)
    _im = Image.new("RGB", (W, VW + PANEL))
    _im.paste(panel(rec[_ep][_fs[_i]], f"preview ep{_ep}", [rec[_ep][x]["K"] for x in _fs],
                    _i, _ep, _f), (0, VW))
    _im.save(OUT.replace(".mp4", "_preview.png"))
    print("preview ->", OUT.replace(".mp4", "_preview.png")); raise SystemExit(0)


oc = av.open(OUT, "w")
st = oc.add_stream("libx264", rate=30)
st.width, st.height, st.pix_fmt = W, VW + PANEL, "yuv420p"
st.options = {"crf": "23", "preset": "medium"}
nfr = 0
for ep, a, b, title in CLIPS:
    fs = sorted(k for k in rec[ep] if a <= k < b)
    if not fs:
        continue
    ks = [rec[ep][f]["K"] for f in fs]
    srcs = [av.open(f"{DS}/videos/chunk-000/observation.images.camera_ego_{s}/"
                    f"episode_{ep:06d}.mp4") for s in ("left", "right")]
    gens = [c.decode(video=0) for c in srcs]
    idx = 0
    for i in range(b):
        try:
            imgs = [next(g) for g in gens]
        except StopIteration:
            break
        if i < a:
            continue
        while idx + 1 < len(fs) and i >= fs[idx + 1]:
            idx += 1
        r = rec[ep][fs[idx]]
        canvas = Image.new("RGB", (W, VW + PANEL))
        for k, im in enumerate(imgs):
            canvas.paste(Image.fromarray(im.to_ndarray(format="rgb24")).resize((VW, VW)),
                         (k * VW, 0))
        canvas.paste(panel(r, title, ks, idx, ep, i), (0, VW))
        dd = ImageDraw.Draw(canvas)
        dd.rectangle([0, 0, W - 1, VW - 1], outline=RCOL.get(r["K"], (200, 200, 200)), width=6)
        for p in st.encode(av.VideoFrame.from_ndarray(np.array(canvas), format="rgb24")):
            oc.mux(p)
        nfr += 1
    for c in srcs:
        c.close()
    print(f"clip {title}: through frame {b}, total {nfr}", flush=True)
for p in st.encode():
    oc.mux(p)
oc.close()
print(f"{nfr} frames -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
