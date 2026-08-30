"""로컬 judge로 9k 동일 프레임 라벨링 (±액션수치) — frontier와 공정 대조군"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
p=argparse.ArgumentParser()
p.add_argument("--judge-url", required=True); p.add_argument("--guidance", required=True)
p.add_argument("--out", required=True); p.add_argument("--with-actions", type=int, default=0)
a=p.parse_args()
g=a.guidance
if g.startswith("@"): g=open(g[1:]).read().strip()
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
ref=pd.read_parquet(f"{BASE}/output/_gate_distill/luna_robocasa_strat/labels_cosmos9k.parquet")
gate=VLMGate(a.judge_url, timeout=120)
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
    return acts[ep]
rows=[]
for i,(_,r) in enumerate(ref.iterrows()):
    ep=int(r["episode_index"]); f=int(r["frame_index"])
    pth=f"{TIL}/ep{ep:04d}_f{f:03d}.png"
    if not os.path.exists(pth): continue
    im=np.array(Image.open(pth).convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//3:(k+1)*w//3]) for k in range(3)]
    ins=instr.get(ep,"")
    if a.with_actions:
        aa=A(ep)
        if aa is None or f>=len(aa)-4: continue
        ins += ("\nPlanned actions for the next 16 control steps (12 numbers per step; dims 5-7 = "
                "end-effector delta xyz, last = gripper): " + json.dumps(np.round(aa[f:f+16],2).tolist()))
    res=gate.judge(views, ins, g)
    rows.append({"episode_index":ep,"frame_index":f,"task":instr.get(ep,""),
                 "p_yes":float(res.get("confidence",0.0)),"quantize":bool(res.get("quantize",False))})
    if i%500==0: print(f"{i}/{len(ref)} qrate={np.mean([x['quantize'] for x in rows]):.2f}", flush=True)
pd.DataFrame(rows).to_parquet(a.out); print("saved", a.out, len(rows), flush=True)
