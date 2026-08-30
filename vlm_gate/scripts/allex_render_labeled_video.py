"""라벨을 영상에 얹어 하나의 mp4로 만든다 — 정성 검토용.

각 프레임 아래 패널에 그 프레임이 속한 16스텝 청크의 판단을 표시한다:
  큰 배지(COMPRESS / KEEP), confidence 막대, 계산값, VLM 문항 답, 그리고
  에피소드 전체 confidence 곡선 위의 현재 위치.
"""
import json, os, sys, numpy as np, pandas as pd, av
from PIL import Image, ImageDraw, ImageFont
H=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
EP=int(sys.argv[1]); TAU=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
V=os.environ.get("LBL","")
OUT=f"{H}/labeled_ep{EP:04d}{V}_tau{str(TAU).replace('.','')}.mp4"
FT="/usr/share/fonts/truetype/dejavu"
f_big=ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 34)
f_mid=ImageFont.truetype(f"{FT}/DejaVuSans-Bold.ttf", 20)
f_sm =ImageFont.truetype(f"{FT}/DejaVuSansMono.ttf", 16)
rec={}
for l in open(f"{H}/labels_ep{EP:04d}{V}.jsonl"):
    r=json.loads(l); rec[r["f"]]=r
starts=sorted(rec)
confs=np.array([rec[f]["conf"] for f in starts])
# 순위정규화 — τ가 곧 차단율이 되도록 (라벨 파이프라인과 동일)
rank=np.argsort(np.argsort(confs))/(len(confs)-1)
# 후처리: 인과적 중앙값 평활 + 히스테리시스 + 하드 차단(실행불가/누적회전)
import sys as _s; _s.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from allex_postprocess import runtime_gate as decide
_dec=decide([rec[f] for f in starts], rank, tau=TAU)
for i,f in enumerate(starts):
    rec[f]["rank"]=float(rank[i]); rec[f]["keep"]=bool(not _dec[i])
W,HH=960,540; PANEL=250
QN=(["A soft/bag","B reorienting","C seating pose","D empty/reaching"] if V else
    ["A soft/bag","B pose matters","C taking/giving","D hands clear"])
def panel(r, idx):
    im=Image.new("RGB",(W,PANEL),(18,18,22)); d=ImageDraw.Draw(im)
    keep = r.get("keep", r["rank"] < TAU)
    col=(226,92,84) if keep else (86,196,124)
    d.rounded_rectangle([14,14,214,74], 8, fill=col)
    d.text((30,26), "KEEP" if keep else "COMPRESS", font=f_big, fill=(15,15,18))
    d.text((230,20), f"rank {r['rank']:.2f}   raw {r['conf']:.3f}   tau {TAU}", font=f_mid, fill=(210,210,215))
    # confidence 막대
    d.rectangle([230,52,930,72], outline=(70,70,78))
    d.rectangle([230,52,230+int(700*r["rank"]),72], fill=col)
    d.line([230+int(700*TAU),46,230+int(700*TAU),78], fill=(250,220,120), width=2)
    # 계산값
    d.text((16,88), "computed", font=f_mid, fill=(250,220,120))
    trend = "closing" if r.get("closing") else ("opening" if r.get("opening") else "steady")
    d.text((16,112), f"wrist gap {r['gap_mean']:.2f} m ({trend}, d{r['gap_change']:.3f})", font=f_sm, fill=(200,200,206))
    d.text((16,132), f"arm speed {r['arm_speed']:.3f} rad/step", font=f_sm, fill=(200,200,206))
    d.text((16,152), f"wrist turn {r.get('wrist_rot',0):.0f} deg (asym {r.get('rot_asym',0):.0f})"
           + ("  HELD" if r.get("held") else ""), font=f_sm,
           fill=(240,180,110) if (r.get("held") and r.get("wrist_rot",0)>10) else (200,200,206))
    mg=r["merge_demand"]
    d.text((16,172), f"merge demand {mg:.3f} rad" + ("  INFEASIBLE" if mg>0.159 else ""),
           font=f_sm, fill=(240,120,110) if mg>0.159 else (200,200,206))
    # VLM 답
    d.text((330,88), "VLM (vision only)", font=f_mid, fill=(250,220,120))
    for i,(q,nm) in enumerate(zip("ABCD",QN)):
        v=r[q]; yes=v>=0.5
        d.text((330,112+i*20), f"{nm:<18} {'YES' if yes else 'NO ':>3}  {v:.2f}",
               font=f_sm, fill=(240,150,140) if (yes and q!='D') else ((140,220,170) if yes else (150,150,158)))
    # 전체 곡선
    x0,y0,x1,y1=620,105,940,195
    d.rectangle([x0,y0,x1,y1], outline=(70,70,78))
    d.line([x0,int(y1-(y1-y0)*TAU),x1,int(y1-(y1-y0)*TAU)], fill=(250,220,120))
    pts=[(x0+int((x1-x0)*i/(len(starts)-1)), int(y1-(y1-y0)*rank[i])) for i in range(len(starts))]
    d.line(pts, fill=(130,170,230), width=1)
    for j,f2 in enumerate(starts):          # 최종 판정을 곡선 아래 띠로
        if rec[f2].get("keep"):
            xx=x0+int((x1-x0)*j/(len(starts)-1)); d.line([xx,y1+1,xx,y1+4], fill=(226,92,84))
    px,py=pts[idx]; d.ellipse([px-4,py-4,px+4,py+4], fill=(255,255,255))
    d.text((x0,y1+6), f"episode {EP}  chunk {idx+1}/{len(starts)}  frame {r['f']}", font=f_sm, fill=(150,150,158))
    return im
src=f"{H}/videos/chunk-000/observation.images.camera_ego_left/episode_{EP:06d}.mp4"
oc=av.open(OUT,"w"); st=oc.add_stream("libx264", rate=30)
st.width,st.height,st.pix_fmt=W,HH+PANEL,"yuv420p"
st.options={"crf":"23","preset":"medium"}
cur=starts[0]; idx=0; n=0
with av.open(src) as c:
    for i,fr in enumerate(c.decode(video=0)):
        if i>=starts[-1]+16: break
        while idx+1<len(starts) and i>=starts[idx+1]: idx+=1
        r=rec[starts[idx]]
        img=Image.fromarray(fr.to_ndarray(format="rgb24")).resize((W,HH))
        canvas=Image.new("RGB",(W,HH+PANEL)); canvas.paste(img,(0,0)); canvas.paste(panel(r,idx),(0,HH))
        # 화면 위 테두리로 판단 표시
        dd=ImageDraw.Draw(canvas)
        dd.rectangle([0,0,W-1,HH-1], outline=(226,92,84) if r["rank"]<TAU else (86,196,124), width=6)
        f2=av.VideoFrame.from_ndarray(np.array(canvas), format="rgb24")
        for p in st.encode(f2): oc.mux(p)
        n+=1
for p in st.encode(): oc.mux(p)
oc.close()
print(f"{n}프레임 -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
