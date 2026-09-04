"""Burn the v3 ratios onto the frames — one combined mp4.

Same layout as v2: both ego views on top, a panel underneath for the 16-step
chunk the frame belongs to. What changed is what there is to show. v2 had a
confidence and a ceiling and a rule joining them; v3 has five graded answers
and the ratio they arrive at, so the panel shows each check, the grade the
model wrote, and the ratio that check carries — the whole derivation of K is
on screen.

  python allex_v3_render.py [out.mp4]
The clip list below is the default: episode 0 end to end (it visits all four
subtasks) followed by one segment per task taken from other episodes.
"""
import json, os, sys, collections
import numpy as np, pandas as pd, av
from PIL import Image, ImageDraw, ImageFont

DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
OUTDIR = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v3checks"))
REC = f"{OUTDIR}/records.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/quantization_agent_workspace/assets/videos/allex_v3_ratio.mp4")
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

RCOL = {1.0: (226, 92, 84), 1.5: (231, 130, 84), 2.0: (232, 176, 84),
        2.5: (150, 205, 120), 3.0: (86, 196, 124)}
# name, and the ratio the check carries -- shown so the arithmetic is legible.
QN = [("A limp plastic mailer", 2.0), ("B a new face coming up", 1.5),
      ("C pushed along the plate", 3.0)]

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

# ONLY_TASK renders that subtask's segment alone, so the four can be played
# side by side and compared instead of watched in sequence. WANT sets its length.
ONLY = os.environ.get("ONLY_TASK", "")
if ONLY:
    s = seg_for(ONLY, want=int(os.environ.get("WANT", "170")))
    if not s:
        raise SystemExit(f"no segment for {ONLY!r}")
    CLIPS = [(s[0], s[1], s[2], f"episode {s[0]} — {ONLY}")]
else:
    CLIPS = [(0, 0, max(rec[0]) + CHUNK, "episode 0 — all four subtasks")] if 0 in rec else []
    for t in ["Bring Object", "Rotate Box", "Pass Object", "Rotate PolyBag"]:
        s = seg_for(t)
        if s:
            CLIPS.append((s[0], s[1], s[2], f"episode {s[0]} — {t}"))
if not CLIPS:                                   # no ep0 labelled: use what we have
    ep = sorted(rec)[0]
    CLIPS = [(ep, 0, max(rec[ep]) + CHUNK, f"episode {ep}")]


COL1, COL2, COL3 = 16, 320, 640          # computed | check + grade | ratio
ROW0 = 132                               # first data row of the three columns


def panel(r, title, ks, pos, ep, f):
    im = Image.new("RGB", (W, PANEL), (18, 18, 22)); d = ImageDraw.Draw(im)
    K = r["K_snap"]; col = RCOL.get(K, (200, 200, 200))
    d.rounded_rectangle([14, 12, 190, 76], 8, fill=col)
    d.text((30, 24), "K = 1" if K == 1.0 else f"K = {K:g}", font=f_big, fill=(15, 15, 18))
    d.text((206, 14), r["task"], font=f_big, fill=(240, 240, 245))
    d.text((206, 58), f"raw {r['K']:.2f}  ->  replayable {r['K_snap']:g}"
                      f"      base 2.0, permissive raise then restrictive clamp",
           font=f_mid, fill=(210, 210, 215))
    # where K sits on the candidate line 1 .. 3
    d.rectangle([16, 88, W - 16, 102], outline=(70, 70, 78))
    d.rectangle([16, 88, 16 + int((W - 32) * (min(3.0, max(1.0, r["K"])) - 1.0) / 2.0), 102],
                fill=col)
    # ---- column headers
    d.text((COL1, 110), "computed", font=f_hd, fill=(250, 220, 120))
    d.text((COL2, 110), "checks - grade the model wrote", font=f_hd, fill=(250, 220, 120))
    d.text((COL3, 110), "what it carries", font=f_hd, fill=(250, 220, 120))
    trend = "closing" if r.get("closing") else ("opening" if r.get("opening") else "steady")
    rows = [f"gap {r['gap_mean']:.2f}m {trend} {r['gap_rate']*1000:.1f}mm/s",
            f"arm {r['arm_speed']:.3f} rad/step" + ("  SLOWING" if r.get("slowing") else ""),
            f"turn {r['wrist_rot']:.0f}deg asym {r['rot_asym']:.0f}"
            + ("  HELD" if r.get("held") else "") + ("  1ARM" if r.get("one_handed") else ""),
            f"skip K2 {r['merge_demand_k2']:.3f} K3 {r['merge_demand_k3']:.3f}",
            f"travel {r['translation']*100:.0f}cm fingers {r['hand_change']:.3f}"]
    for i, t in enumerate(rows):
        d.text((COL1, ROW0 + i * 18), t, font=f_sm, fill=(200, 200, 206))
    for i, (q, (nm, carry)) in enumerate(zip("ABC", QN)):
        g = r.get(q)
        w = 0.0 if g is None else (float(g) - 1.0) / 4.0
        # grey until a check actually says something; then coloured by the
        # ratio it carries, so a red row is one pulling K down.
        fill = (150, 150, 158) if w <= 0 else RCOL.get(carry, (200, 200, 200))
        d.text((COL2, ROW0 + i * 18), f"{nm:<22}{'-' if g is None else g}",
               font=f_sm, fill=fill)
        d.text((COL3, ROW0 + i * 18),
               f"{carry:g}x" + ("" if w <= 0 else f"   weight {w:.2f}"),
               font=f_sm, fill=fill)
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
                      f"[timeline: bar height = K, 1 low .. 3 high]   {r['task']}",
           font=f_sm, fill=(150, 150, 158))
    return im


PREVIEW = os.environ.get("PREVIEW")       # "ep:frame" -> dump one PNG and exit
if PREVIEW:
    _ep, _f = (int(v) for v in PREVIEW.split(":"))
    _fs = sorted(rec[_ep]); _i = max(i for i, ff in enumerate(_fs) if ff <= _f)
    _im = Image.new("RGB", (W, VW + PANEL))
    _im.paste(panel(rec[_ep][_fs[_i]], f"preview ep{_ep}", [rec[_ep][x]["K_snap"] for x in _fs],
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
    ks = [rec[ep][f]["K_snap"] for f in fs]
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
        dd.rectangle([0, 0, W - 1, VW - 1], outline=RCOL.get(r["K_snap"], (200, 200, 200)), width=6)
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
