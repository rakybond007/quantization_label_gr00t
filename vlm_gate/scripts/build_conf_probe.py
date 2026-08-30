"""초소규모 confidence 프로토콜 비교: (A) top20 YES/NO 합산, (B) 5단계 언어척도"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
mode=sys.argv[1]           # top20 | scale5
OUT=f"{BASE}/output/_gate_distill/exp_cp_{mode}"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
have=set(os.listdir(TIL))
byep={}
for n in sorted(have): byep.setdefault(int(n[2:6]),[]).append(int(n.split("_f")[1][:3]))
eps=[e for e in sorted(byep) if len(byep[e])>=30][:10]
view_note=("You are shown 3 camera views (concatenated left-to-right): agentview-left, agentview-right, wrist close-up.")
if mode=="scale5":
    ask=("How confident are you that the next ~1 second of motion can be compressed (run at half rate)?\n"
         "Answer with exactly ONE word from this scale:\n"
         "CERTAIN (definitely compressible) / LIKELY / UNSURE / DOUBTFUL / IMPOSSIBLE (definitely needs full rate).")
    K=5
else:
    ask=("Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    K=20
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    return acts[ep]
rows=[]
for ep in eps:
    aa=A(ep)
    for f in sorted(byep[ep])[:30]:
        if f>=len(aa)-4: continue
        im=Image.open(f"{TIL}/ep{ep:04d}_f{f:03d}.png").convert("RGB")
        b=io.BytesIO(); im.save(b,format="JPEG",quality=90)
        txt=(f"Task: {instr.get(ep,'')}\n{view_note}\n"
             "Planned actions for the next 16 control steps (dims 5-7 = EE delta xyz, last = gripper):\n"
             +json.dumps(np.round(aa[f:f+16],2).tolist())+"\n"+ask)
        rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
          "body":{"model":"gpt-5.6-luna","max_completion_tokens":8,"reasoning_effort":"none",
                  "logprobs":True,"top_logprobs":K,
                  "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                              {"role":"user","content":[
                                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}},
                                {"type":"text","text":txt}]}]}})
p=f"{OUT}/part_00.jsonl"
with open(p,"w") as fo:
    for r in rows: fo.write(json.dumps(r)+"\n")
json.dump([p], open(f"{OUT}/files.json","w"))
print(mode, "요청", len(rows), round(os.path.getsize(p)/1e6,1),"MB")
