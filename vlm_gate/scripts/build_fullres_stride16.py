"""원본 256x256 3뷰 분리, stride16 전 에피소드 (결정 시 사용)"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
from decord import VideoReader
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/_gate_distill/exp_s16_fullres"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
vks=[k for k in info["features"] if info["features"][k].get("dtype")=="video"]
VK=[k for k in vks if "left" in k]+[k for k in vks if "right" in k and "wrist" not in k]+[k for k in vks if "wrist" in k]
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
view_note=("You are shown 3 camera views as separate images: agentview-left, agentview-right, and a wrist "
           "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects normally look "
           "close in it. Use the wrist view only to spot the actual grasp-closure or fine-insertion instant.")
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
paths=[]; rows=[]; idx=0
def flush():
    global rows, idx
    if not rows: return
    p=f"{OUT}/part_{idx:03d}.jsonl"
    with open(p,"w") as fo:
        for r in rows: fo.write(json.dumps(r)+"\n")
    paths.append(p); print(p, round(os.path.getsize(p)/1e6,1),"MB", flush=True)
    json.dump(paths, open(f"{OUT}/files_partial.json","w"))
    rows=[]; idx+=1
for ep in range(info["total_episodes"]):
    ch=ep//info["chunks_size"]
    try:
        vrs=[VideoReader(f"{DS}/"+info["video_path"].format(episode_chunk=ch,episode_index=ep,video_key=k)) for k in VK]
        A=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    except Exception: continue
    n=min(min(len(v) for v in vrs), len(A)-4)
    for f in range(0, n, 16):
        content=[]
        for v in vrs:
            im=Image.fromarray(v[f].asnumpy()); b=io.BytesIO(); im.save(b,format="JPEG",quality=90)
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
        if len(rows)>=350: flush()
    if ep%500==0: print(f"ep {ep}/{info['total_episodes']} parts={len(paths)}", flush=True)
flush()
json.dump(paths, open(f"{OUT}/files.json","w"))
print("완료: 파트", len(paths), flush=True)
