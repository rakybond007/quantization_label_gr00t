"""9k 동일 프레임에 대한 frontier 라벨링 배치 (액션수치 ± 동작점 힌트)"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
tag=sys.argv[1]; use_act=sys.argv[2]=="act"; rate_hint=sys.argv[3]=="hint"
OUT=f"{BASE}/output/_gate_distill/exp_{tag}"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
if rate_hint:
    G += ("\n\nCALIBRATION: in this dataset only about 30-35% of moments are safely compressible. "
          "If far more than a third of your answers are YES, you are being too permissive — "
          "reserve YES for clear free-space transit.")
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
ref=pd.read_parquet(f"{BASE}/output/_gate_distill/luna_robocasa_strat/labels_cosmos9k.parquet")
have=set(os.listdir(TIL))
view_note=("You are shown 3 camera views (concatenated left-to-right): agentview-left, agentview-right, "
           "and a wrist (eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
           "normally look close in it — general closeness is normal. Use the wrist view only to spot the "
           "actual grasp-closure or fine-insertion instant.")
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
acts={}
def get_actions(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
    return acts[ep]
rows=[]
for _,r in ref.iterrows():
    ep=int(r["episode_index"]); f=int(r["frame_index"])
    nm=f"ep{ep:04d}_f{f:03d}.png"
    if nm not in have: continue
    ax=None
    if use_act:
        A=get_actions(ep)
        if A is None or f>=len(A)-4: continue
        ax=np.round(A[f:f+16],2).tolist()
    rows.append((ep,f,nm,ax))
print("요청 수:", len(rows), flush=True)
paths=[]; CH=800
for i in range(0,len(rows),CH):
    p=f"{OUT}/part_{i//CH:02d}.jsonl"; paths.append(p)
    with open(p,"w") as fo:
        for ep,f,nm,ax in rows[i:i+CH]:
            im=Image.open(f"{TIL}/{nm}").convert("RGB")
            buf=io.BytesIO(); im.save(buf,format="JPEG",quality=90)
            txt=f"Task: {instr.get(ep,'')}\n{view_note}\n"
            if ax is not None:
                txt += ("You are ALSO given the robot's planned action sequence for the next 16 control steps "
                        "(12 numbers per step; dims 5-7 = end-effector delta xyz, last = gripper command). "
                        "Use it to judge motion magnitude, direction changes and gripper transitions.\n"
                        "Planned actions:\n"+json.dumps(ax)+"\n")
            txt += ("Can the next ~1 second of motion be compressed (run at half rate)? "
                    "Answer YES (compress) or NO (needs precise full-rate control).")
            body={"model":"gpt-5.6-luna","max_completion_tokens":8,"reasoning_effort":"none",
                  "logprobs":True,"top_logprobs":5,
                  "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                              {"role":"user","content":[
                                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()}},
                                {"type":"text","text":txt}]}]}
            fo.write(json.dumps({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions","body":body})+"\n")
    print(p, round(os.path.getsize(p)/1e6,1),"MB", flush=True)
json.dump(paths, open(f"{OUT}/files.json","w"))
