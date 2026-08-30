"""stride-16 전 에피소드 라벨링 배치 (frontier + 액션수치 + logprobs)"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
tag=sys.argv[1]; hint = len(sys.argv)>2 and sys.argv[2]=="hint"
OUT=f"{BASE}/output/_gate_distill/exp_{tag}"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
if hint:
    G += ("\n\nCALIBRATION: in this dataset only about 30-35% of moments are safely compressible. "
          "If far more than a third of your answers are YES, you are being too permissive.")
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
view_note=("You are shown 3 camera views (concatenated left-to-right): agentview-left, agentview-right, "
           "and a wrist (eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
           "normally look close in it. Use the wrist view only to spot the actual grasp-closure or fine-insertion instant.")
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
names=sorted(os.listdir(TIL))
sel=[n for n in names if int(n.split("_f")[1][:3]) % 16 == 8]
print("stride16 프레임:", len(sel), flush=True)
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
        if len(acts)>60: 
            for k in list(acts)[:30]: acts.pop(k,None)
    return acts[ep]
paths=[]; CH=1000; buf_rows=[]; idx=0
def flush(rows, idx):
    p=f"{OUT}/part_{idx:03d}.jsonl"
    with open(p,"w") as fo:
        for r in rows: fo.write(json.dumps(r)+"\n")
    paths.append(p); print(p, round(os.path.getsize(p)/1e6,1),"MB", flush=True)
for n in sel:
    ep=int(n[2:6]); f=int(n.split("_f")[1][:3])
    aa=A(ep)
    if aa is None or f>=len(aa)-4: continue
    im=Image.open(f"{TIL}/{n}").convert("RGB")
    b=io.BytesIO(); im.save(b,format="JPEG",quality=90)
    txt=(f"Task: {instr.get(ep,'')}\n{view_note}\n"
         "You are ALSO given the robot's planned action sequence for the next 16 control steps "
         "(12 numbers per step; dims 5-7 = end-effector delta xyz, last = gripper command).\n"
         "Planned actions:\n"+json.dumps(np.round(aa[f:f+16],2).tolist())+"\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    buf_rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
        "body":{"model":"gpt-5.6-luna","max_completion_tokens":8,"reasoning_effort":"none",
                "logprobs":True,"top_logprobs":5,
                "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                            {"role":"user","content":[
                              {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}},
                              {"type":"text","text":txt}]}]}})
    if len(buf_rows)>=CH:
        flush(buf_rows, idx); idx+=1; buf_rows=[]
if buf_rows: flush(buf_rows, idx)
json.dump(paths, open(f"{OUT}/files.json","w"))
print("총 파트:", len(paths), flush=True)
