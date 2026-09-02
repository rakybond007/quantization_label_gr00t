"""Render the written `hojin` column back onto the footage, for eyeballing.

Reads the ratio straight out of the parquet -- the column as shipped, ramps and
all -- rather than recomputing it, so what plays is what a consumer of the
dataset would get. The panel carries the subtask the frame is annotated with,
the ceiling that subtask allows, and the whole episode's curve with the current
frame marked, so a moment that looks wrong can be placed in its context.

    python allex_render_subtask_video.py <ep> [out.mp4]
"""
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DS = os.environ.get("ALLEX_DS", "/rlwrld2/home/david/action_quantization/"
                                "replay_evaluation10/replay_evaluation_ee_subtask_hojin")
COL = os.environ.get("ALLEX_COL", "hojin")
EP = int(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(f"~/allex_ep{EP}_{COL}.mp4")
FT = "/usr/share/fonts/truetype/dejavu"
f_big = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 40)
f_mid = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 19)
f_sm = ImageFont.truetype(f"{FT}/DejaVuSansMono.ttf", 15)

d = pd.read_parquet(f"{DS}/data/chunk-000/episode_{EP:06d}.parquet")
K = d[COL].values.astype(float)
ti = d["task_index"].values
TASKS = [json.loads(l)["task"] for l in open(f"{DS}/meta/tasks.jsonl")]
SPEC = {"Bring Object": 2.0, "Rotate Box": 2.0, "Rotate PolyBag": 2.5, "Pass Object": 3.0}
N = len(K)

VW, VH = 400, 300
W, PANEL = VW * 2, 210
HH = VH + PANEL


def frames(side):
    p = f"{DS}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{EP:06d}.mp4"
    with av.open(p) as c:
        for fr in c.decode(video=0):
            yield fr.to_image().resize((VW, VH))


out = av.open(OUT, "w")
st = out.add_stream("libx264", rate=30)
st.width, st.height, st.pix_fmt = W, HH, "yuv420p"
st.options = {"crf": "30", "preset": "veryfast"}

CURVE_X0, CURVE_X1, CURVE_Y0, CURVE_Y1 = 24, W - 24, VH + 96, VH + 186
for i, (L, R) in enumerate(zip(frames("left"), frames("right"))):
    if i >= N:
        break
    im = Image.new("RGB", (W, HH), (16, 16, 18))
    im.paste(L, (0, 0)); im.paste(R, (VW, 0))
    g = ImageDraw.Draw(im)
    k = float(K[i])
    task = TASKS[int(ti[i])] if int(ti[i]) < len(TASKS) else ""
    cap = SPEC.get(task, 3.0)

    # colour runs from held-back (grey) to fully opened up (green)
    t = (k - 1.0) / 2.0
    col = (int(120 + 60 * t), int(120 + 135 * t), int(130 - 40 * t))
    g.text((24, VH + 12), f"{k:.2f}x", font=f_big, fill=col)
    g.text((150, VH + 22), task, font=f_mid, fill=(225, 225, 230))
    g.text((150, VH + 48), f"ceiling for this subtask {cap:.1f}x", font=f_sm, fill=(150, 150, 158))
    g.text((W - 210, VH + 22), f"frame {i}/{N}", font=f_sm, fill=(150, 150, 158))
    g.text((W - 210, VH + 44), f"episode mean {K.mean():.2f}x", font=f_sm, fill=(150, 150, 158))

    # the episode's whole curve, 1x at the bottom and 3x at the top
    def yy(v):
        return CURVE_Y1 - (v - 1.0) / 2.0 * (CURVE_Y1 - CURVE_Y0)
    for lv in (1.0, 2.0, 3.0):
        y = yy(lv)
        g.line([CURVE_X0, y, CURVE_X1, y], fill=(52, 52, 58))
        g.text((4, y - 8), f"{lv:.0f}", font=f_sm, fill=(90, 90, 96))
    pts = [(CURVE_X0 + (CURVE_X1 - CURVE_X0) * j / max(N - 1, 1), yy(K[j])) for j in range(N)]
    g.line(pts, fill=(90, 175, 120), width=2)
    x = CURVE_X0 + (CURVE_X1 - CURVE_X0) * i / max(N - 1, 1)
    g.line([x, CURVE_Y0 - 6, x, CURVE_Y1 + 6], fill=(235, 200, 90), width=2)

    fr = av.VideoFrame.from_image(im)
    for pk in st.encode(fr):
        out.mux(pk)
for pk in st.encode():
    out.mux(pk)
out.close()
print(f"ep{EP}: {N} frames, mean {K.mean():.3f}x -> {OUT} "
      f"({os.path.getsize(OUT)/1e6:.1f} MB)")
