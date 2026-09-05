"""merged_v5tempo 에 붙인 배속을 화면에 태워 mp4 로 만든다.

위에 두 시점, 아래 패널에 그 청크의 결정이 그대로 나온다 -- 칸과 띠, 다섯
문항에 모델이 쓴 등급, 그 등급에서 나온 확신, 띠 안에서 앉은 자리, 그리고
이 청크를 실제로 봤는지 이웃에서 복사해 왔는지.

  ALLEX_CLIPS="ep:from:to:제목;..." python allex_v5tempo_render.py out.mp4
"""
import collections
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v3_checks import ACTIVE, SIGN, TASK_RANGE, DEFAULT_RANGE  # noqa: E402

DS = os.environ.get("ALLEX_DS",
                    "/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo")
OUTDIR = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v5tempo_v3"))
REC = os.environ.get("ALLEX_REC", f"{OUTDIR}/records_full.jsonl")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v5tempo.mp4"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
CHUNK = 16
VW = 384
W = VW * 2
PANEL = 300
CRF = os.environ.get("CRF", "28")
try:
    CH = int(json.load(open(f"{DS}/meta/info.json"))["chunks_size"])
except Exception:
    CH = 1000

FT = "/usr/share/fonts/truetype/dejavu"
f_big = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 34)
f_mid = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 17)
f_hd = ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 14)
f_sm = ImageFont.truetype(f"{FT}/DejaVuSansMono.ttf", 13)

RCOL = {1.0: (226, 92, 84), 1.5: (231, 130, 84), 2.0: (232, 176, 84),
        2.5: (150, 205, 120), 3.0: (86, 196, 124)}
QN = {"CLAMP": "A  두 손 사이에 끼워 듦", "LOOSE": "B  쥔 자리만으로 붙듦",
      "SHOVE": "C  붙들 필요 없는 일", "FLIP": "D  아무 데나 잡아도 되는 것",
      "FREE": "E  아무것에도 안 닿음"}

rec = collections.defaultdict(dict)
for l in open(REC):
    r = json.loads(l)
    rec[r["ep"]][r["f"]] = r

CLIPS = []
for spec in os.environ.get("ALLEX_CLIPS", "0:0:600:episode 0").split(";"):
    if not spec.strip():
        continue
    a = spec.split(":")
    CLIPS.append((int(a[0]), int(a[1]), int(a[2]), a[3] if len(a) > 3 else ""))


def panel(r, title, ks, pos, f):
    im = Image.new("RGB", (W, PANEL), (18, 18, 22))
    d = ImageDraw.Draw(im)
    K = float(r["K"])
    col = RCOL.get(K, (200, 200, 200))
    cell = r.get("cell") or r.get("task")
    lo, hi = TASK_RANGE.get(cell, DEFAULT_RANGE)
    d.rounded_rectangle([14, 12, 176, 70], 8, fill=col)
    d.text((28, 22), f"{K:g}x", font=f_big, fill=(15, 15, 18))
    d.text((192, 12), str(cell), font=f_big, fill=(240, 240, 245))
    filled = "이웃에서 복사" if r.get("filled_from") is not None else "이 청크를 봄"
    d.text((192, 50), f"띠 {lo:g}~{hi:g}   확신 {r.get('conf', 0):.2f}"
                      f"   띠 안 자리 {r.get('K_spread', K):.2f}   {filled}",
           font=f_mid, fill=(210, 210, 215))
    # 띠 위에서 어디쯤인가
    d.rectangle([16, 80, W - 16, 92], outline=(70, 70, 78))
    fr = 0.0 if hi <= lo else (float(r.get("K_spread", K)) - lo) / (hi - lo)
    d.rectangle([16, 80, 16 + int((W - 32) * min(1.0, max(0.0, fr))), 92], fill=col)
    d.text((16, 98), "문항에 모델이 쓴 등급 (1 해당 없음 … 5 지금 일어남)",
           font=f_hd, fill=(250, 220, 120))
    for i, q in enumerate(ACTIVE):
        g = r.get(q)
        mark = "감점" if SIGN.get(q, 1) < 0 else "가점"
        c = (150, 150, 158) if not g or g <= 1 else (
            (232, 130, 120) if SIGN.get(q, 1) < 0 else (130, 205, 150))
        d.text((22, 120 + i * 19), f"{QN.get(q, q):<22} {mark}   {g if g else '-'}",
               font=f_sm, fill=c)
    d.text((470, 98), "계산해서 넣어 준 사실", font=f_hd, fill=(250, 220, 120))
    facts = [f"팔 속도 {r.get('arm_speed', 0):.3f}",
             f"손목 회전 {r.get('wrist_rot', 0):.0f}도",
             f"이동 {r.get('translation', 0)*100:.0f}cm",
             f"손바닥 간격 {r.get('gap_mean', 0):.2f}m",
             f"손가락 움직임 {r.get('hand_change', 0):.3f}"]
    for i, t in enumerate(facts):
        d.text((476, 120 + i * 19), t, font=f_sm, fill=(200, 200, 206))
    x0, y0, x1, y1 = 16, 226, W - 16, 272
    d.rectangle([x0, y0, x1, y1], outline=(70, 70, 78))
    n = max(1, len(ks))
    for j, kk in enumerate(ks):
        xa = x0 + int((x1 - x0) * j / n)
        xb = x0 + max(xa + 1, int((x1 - x0) * (j + 1) / n))
        hgt = int((y1 - y0) * (kk - 1.0) / 2.0)
        d.rectangle([xa, y1 - hgt, xb, y1], fill=RCOL.get(kk, (150, 150, 150)))
    px = x0 + int((x1 - x0) * pos / n)
    d.line([px, y0 - 4, px, y1 + 4], fill=(255, 255, 255), width=2)
    d.text((16, 276), f"{title}   프레임 {f}   청크 {pos+1}/{n}"
                      f"   [막대 높이 = 배속, 1배 낮음 … 3배 높음]",
           font=f_sm, fill=(150, 150, 158))
    return im


oc = av.open(OUT, "w")
st = oc.add_stream("libx264", rate=30)
st.width, st.height, st.pix_fmt = W, VW + PANEL, "yuv420p"
st.options = {"crf": CRF, "preset": "medium"}
nfr = 0
for ep, a, b, title in CLIPS:
    fs = sorted(k for k in rec[ep] if a <= k < b)
    if not fs:
        print(f"ep{ep} {a}-{b}: 라벨 없음")
        continue
    ks = [float(rec[ep][k]["K"]) for k in fs]
    cd = f"chunk-{ep // CH:03d}"
    srcs = [av.open(f"{DS}/videos/{cd}/observation.images.camera_ego_{s}/"
                    f"episode_{ep:06d}.mp4") for s in ("left", "right")]
    gens = [s.decode(video=0) for s in srcs]
    for i in range(b):
        try:
            imgs = [next(g) for g in gens]
        except StopIteration:
            break
        if i < a:
            continue
        j = max(0, sum(1 for k in fs if k <= i) - 1)
        r = rec[ep][fs[j]]
        top = Image.new("RGB", (W, VW))
        for c, im in enumerate(imgs):
            top.paste(Image.fromarray(im.to_ndarray(format="rgb24")).resize((VW, VW)),
                      (c * VW, 0))
        page = Image.new("RGB", (W, VW + PANEL))
        page.paste(top, (0, 0))
        page.paste(panel(r, title, ks, j, i), (0, VW))
        for pk in st.encode(av.VideoFrame.from_image(page)):
            oc.mux(pk)
        nfr += 1
    for s in srcs:
        s.close()
    print(f"{title}: 프레임 {i}까지, 누적 {nfr}")
for pk in st.encode():
    oc.mux(pk)
oc.close()
print(f"{nfr} 프레임 -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
