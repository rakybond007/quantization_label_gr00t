"""사전점검: 원본 256x256 3뷰 분리 입력 (cosmos가 받은 것과 동일 형식)"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
from decord import VideoReader
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/_gate_distill/exp_pf_fullres"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
vks=[k for k in info["features"] if info["features"][k].get("dtype")=="video"]
VK=[k for k in vks if "left" in k]+[k for k in vks if "right" in k and "wrist" not in k]+[k for k in vks if "wrist" in k]
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
# f9k_act과 동일 프레임 중 1400개
fr=[]
for l in open(f"{BASE}/output/_gate_distill/exp_f9k_act/labels.jsonl"):
    r=json.loads(l); fr.append((r['ep'], r['f']))
fr=sorted(set(fr))[:1400]
byep={}
for ep,f in fr: byep.setdefault(ep,[]).append(f)
view_note=("You are shown 3 camera views as separate images: agentview-left, agentview-right, and a wrist "
           "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects normally look "
           "close in it. Use the wrist view only to spot the actual grasp-closure or fine-insertion instant.")
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
rows=[]
for ep in sorted(byep):
    ch=ep//info["chunks_size"]
    try:
        vrs=[VideoReader(f"{DS}/"+info["video_path"].format(episode_chunk=ch,episode_index=ep,video_key=k)) for k in VK]
        A=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    except Exception: continue
    for f in byep[ep]:
        if f>=min(len(v) for v in vrs) or f>=len(A)-4: continue
        content=[]
        for v in vrs:
            im=Image.fromarray(v[f].asnumpy())
            b=io.BytesIO(); im.save(b,format="JPEG",quality=90)
            content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
        txt=(f"Task: {instr.get(ep,'')}\n{view_note}\n"
             "You are ALSO given the robot's planned action sequence for the next 16 control steps "
             "(12 numbers per step; dims 5-7 = end-effector delta xyz, last = gripper command).\n"
             "Planned actions:\n"+json.dumps(np.round(A[f:f+16],2).tolist())+"\n"
             "Can the next ~1 second of motion be compressed (run at half rate)? "
             "Answer YES (compress) or NO (needs precise full-rate control).")
        content.append({"type":"text","text":txt})
        rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
            "body":{"model":"gpt-5.6-luna","max_completion_tokens":8,"reasoning_effort":"none",
                    "logprobs":True,"top_logprobs":5,
                    "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                                {"role":"user","content":content}]}})
paths=[]
for i in range(0,len(rows),350):
    p=f"{OUT}/part_{i//350:02d}.jsonl"
    with open(p,"w") as fo:
        for r in rows[i:i+350]: fo.write(json.dumps(r)+"\n")
    paths.append(p); print(p, round(os.path.getsize(p)/1e6,1),"MB", flush=True)
json.dump(paths, open(f"{OUT}/files.json","w"))
print("총 요청", len(rows), flush=True)
