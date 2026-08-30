"""실기: 단일프레임 + 액션수치 + phase-aware 가이던스 + logprobs (배포 가능 조건)"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/_gate_distill/exp_real_api_act"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/paper_prompts/frontier/real_phase_v2.txt").read().strip()
TIL=f"{BASE}/output/_gate_distill/luna_real_full/tiles"
DS="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1]
    instr[d["episode_index"]]=c[0] if c else ""
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
acts={}
def A(ep):
    if ep not in acts:
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
    return acts[ep]
rows=[]
for n in sorted(os.listdir(TIL)):
    ep=int(n[2:5]); f=int(n.split("_f")[1][:3])
    aa=A(ep)
    if aa is None or f>=len(aa)-4: continue
    im=Image.open(f"{TIL}/{n}").convert("RGB")
    if im.width>768: im=im.resize((768, max(1,int(im.height*768/im.width))))
    b=io.BytesIO(); im.save(b,format="JPEG",quality=88)
    txt=(f"Task: {instr.get(ep,'')}\n"
         "You are shown 1 image: LEFT half = external camera, RIGHT half = wrist (eye-in-hand) close-up.\n"
         "You are ALSO given the robot's planned action sequence for the next 16 control steps "
         "(8 numbers per step: 7 joint targets + gripper command).\n"
         "Planned actions:\n"+json.dumps(np.round(aa[f:f+16],3).tolist())+"\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
        "body":{"model":"gpt-5.6-luna","max_completion_tokens":8,"reasoning_effort":"none",
                "logprobs":True,"top_logprobs":5,
                "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                            {"role":"user","content":[
                              {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}},
                              {"type":"text","text":txt}]}]}})
paths=[]
for i in range(0,len(rows),700):
    p=f"{OUT}/part_{i//700:02d}.jsonl"
    with open(p,"w") as fo:
        for r in rows[i:i+700]: fo.write(json.dumps(r)+"\n")
    paths.append(p); print(p, round(os.path.getsize(p)/1e6,1),"MB", flush=True)
json.dump(paths, open(f"{OUT}/files.json","w")); print("요청", len(rows), flush=True)
