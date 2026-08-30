"""로컬 judge로 실기 타일(PNG) 라벨링 — frontier와 동일 입력·동일 지시문."""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
p=argparse.ArgumentParser()
p.add_argument("--tiles", default="output/_gate_distill/luna_real_full/tiles")
p.add_argument("--judge-url", required=True)
p.add_argument("--guidance", required=True)
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
a=p.parse_args()
g=a.guidance
if g.startswith("@"): g=open(g[1:]).read().strip()
DS="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l)
    c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1]
    instr[d["episode_index"]]=c[0] if c else ""
gate=VLMGate(a.judge_url, timeout=120)
names=sorted(os.listdir(a.tiles))
if a.limit: names=names[:a.limit]
rows=[]
for i,n in enumerate(names):
    ep=int(n[2:5]); f=int(n.split("_f")[1][:3])
    im=np.array(Image.open(f"{a.tiles}/{n}").convert("RGB"))
    h,w,_=im.shape
    views=[Image.fromarray(im[:, :w//2]), Image.fromarray(im[:, w//2:])]  # ext, wrist
    r=gate.judge(views, instr.get(ep,""), g)
    rows.append({"episode_index":ep,"frame_index":f,"task":instr.get(ep,""),
                 "p_yes":float(r.get("confidence", r.get("p_yes",0.0))),"quantize":bool(r.get("quantize",False))})
    if i%200==0:
        print(f"{i}/{len(names)} p_yes평균={np.mean([x['p_yes'] for x in rows]):.3f} qrate={np.mean([x['quantize'] for x in rows]):.2f}", flush=True)
pd.DataFrame(rows).to_parquet(a.out)
print("saved", a.out, len(rows), flush=True)
