"""로컬 judge와 동일 조건: SYSTEM 프롬프트 + Task 지시문 + YES/NO logprobs"""
import json, base64, os, random, sys, io, glob
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
dom, motion, eff, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
OUT=f"{BASE}/output/_gate_distill/exp_{tag}"; os.makedirs(OUT, exist_ok=True)
if dom=="real":
    TIL=f"{BASE}/output/_gate_distill/luna_real_full/tiles"; STRIDE=4
    G=open(f"{BASE}/output/_gate_distill/real_gripper_patched_guidance.txt").read().strip()
    DS="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
    instr_of={}
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d=json.loads(l)
        cands=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1]
        instr_of[d["episode_index"]]=cands[0] if cands else ""
    view_note="You are shown 1 image: LEFT half = external camera, RIGHT half = wrist (eye-in-hand) close-up."
    fmt="ep%03d_f%03d.png"
else:
    TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"; STRIDE=8
    G=open(os.environ["GUIDANCE_FILE"]).read().strip()
    DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
    instr_of={}
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d=json.loads(l)
        cands=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
        instr_of[d["episode_index"]]=cands[0] if cands else ""
    view_note=("You are shown 3 camera views (concatenated left-to-right in each image): agentview-left, "
               "agentview-right, and a wrist (eye-in-hand) close-up. The wrist camera is mounted on the "
               "gripper, so objects normally look close in it — general closeness is normal. Use the wrist "
               "view only to spot the actual grasp-closure or fine-insertion instant.")
    fmt="ep%04d_f%03d.png"
have=set(os.listdir(TIL))
byep={}
for n in sorted(have): byep.setdefault(int(n[2:6] if dom!="real" else n[2:5]),[]).append(n)
eps=[e for e in sorted(byep) if len(byep[e])>=20]
random.seed(11); sel=random.sample(eps, min(40,len(eps)))
sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
reqs=[]
for e in sel:
    for f in sorted(int(n.split("_f")[1][:3]) for n in byep[e]):
        fs=[f]+([f+STRIDE,f+2*STRIDE] if motion==3 else [])
        if not all((fmt%(e,x)) in have for x in fs): continue
        reqs.append((e,f,fs))
print("요청 수:", len(reqs))
paths=[]; CH=300 if motion==3 else 800
for i in range(0,len(reqs),CH):
    p=f"{OUT}/part_{i//CH:02d}.jsonl"; paths.append(p)
    with open(p,"w") as fo:
        for e,f,fs in reqs[i:i+CH]:
            content=[]
            for x in fs:
                im=Image.open(f"{TIL}/"+(fmt%(e,x))).convert("RGB")
                if im.width>768: im=im.resize((768,max(1,int(im.height*768/im.width))))
                buf=io.BytesIO(); im.save(buf,format="JPEG",quality=88)
                content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()}})
            mnote=(f"The {len(fs)} images are CONSECUTIVE moments of the SAME episode ({STRIDE/20:.1f}s apart); judge the FIRST one.\n" if motion==3 else "")
            content.append({"type":"text","text":
                f"Task: {instr_of.get(e,'')}\n{view_note}\n{mnote}"
                "Can the next ~1 second of motion be compressed (run at half rate)? "
                "Answer YES (compress) or NO (needs precise full-rate control)."})
            body={"model":"gpt-5.6-luna","max_completion_tokens":2048 if eff!="none" else 8,
                  "reasoning_effort":eff,"logprobs":True,"top_logprobs":5,
                  "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                              {"role":"user","content":content}]}
            fo.write(json.dumps({"custom_id":("ep%04d_f%03d"%(e,f)),"method":"POST","url":"/v1/chat/completions","body":body})+"\n")
    print(p, round(os.path.getsize(p)/1e6,1),"MB")
json.dump(paths, open(f"{OUT}/files.json","w"))
