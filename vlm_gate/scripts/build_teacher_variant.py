"""통합 티처 실험 빌더: 도메인(real/rc) x 모션(1/3프레임) x 액션(on/off) x reasoning"""
import json, base64, os, random, sys, glob, io
from PIL import Image
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
dom, motion, act, eff, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]=="act", sys.argv[4], sys.argv[5]
OUT=f"{BASE}/output/_gate_distill/exp_{tag}"; os.makedirs(OUT, exist_ok=True)
if dom=="real":
    TIL=f"{BASE}/output/_gate_distill/luna_real_full/tiles"; STRIDE=4
    G=open(f"{BASE}/output/_gate_distill/real_gripper_patched_guidance.txt").read().strip()
    DS="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
    scene="a real robot teleoperation episode (pick-and-place). Each image: LEFT half = external camera, RIGHT half = wrist camera."
    def acts(ep):
        d=pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet")
        return np.stack(d["action"].values)
    adesc="8 numbers per step = 7 joint targets + gripper"
else:
    TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"; STRIDE=8
    G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
    DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
    info=json.load(open(f"{DS}/meta/info.json"))
    scene="a RoboCasa simulated kitchen manipulation demo. Each image has 3 panels: left cam, right cam, wrist cam."
    def acts(ep):
        ch=ep//info["chunks_size"]
        d=pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")
        return np.stack(d["action"].values)
    adesc="12 numbers per step; dims 5-7 = end-effector delta xyz (normalized), last = gripper"
have=set(os.listdir(TIL))
byep={}
for n in sorted(have): byep.setdefault(int(n[2:6] if dom!="real" else n[2:5]),[]).append(n)
eps=[e for e in sorted(byep) if len(byep[e])>=20]
random.seed(11); sel=random.sample(eps, min(40,len(eps)))
reqs=[]
for e in sel:
    fr=sorted(int(n.split("_f")[1][:3]) for n in byep[e])
    A=acts(e) if act else None
    for f in fr:
        fs=[f]+([f+STRIDE, f+2*STRIDE] if motion==3 else [])
        pre="ep%04d_f%03d.png" if dom!="real" else "ep%03d_f%03d.png"
        if not all((pre%(e,x)) in have for x in fs): continue
        ax=None
        if act:
            seg=A[f:f+16]
            if len(seg)<8: continue
            ax=np.round(seg,2).tolist()
        reqs.append((e,f,fs,ax))
print("요청 수:", len(reqs))
tail = ("The three images are CONSECUTIVE moments of the SAME episode (t, t+%0.1fs, t+%0.1fs).\nJudge the FIRST image (time t): " % (STRIDE/20, 2*STRIDE/20)) if motion==3 else "Judge this frame: "
tail += ("p_yes in [0,1] = probability the next ~1 second of robot motion can be compressed "
         "(executed at half control rate, merging pairs of actions) without changing the outcome.\n")
if act: tail += f"You are ALSO given the robot's planned action sequence for the next 16 control steps ({adesc}). Use it to judge motion magnitude, direction changes and gripper transitions.\n"
tail += f"Guidance:\n{G}\nOutput ONLY JSON: {{\"p_yes\": <number>}}"
paths=[]; CH=300 if motion==3 else 800
for i in range(0,len(reqs),CH):
    p=f"{OUT}/part_{i//CH:02d}.jsonl"; paths.append(p)
    with open(p,"w") as fo:
        for e,f,fs,ax in reqs[i:i+CH]:
            content=[]
            for x in fs:
                pth=f"{TIL}/"+(("ep%04d_f%03d.png" if dom!="real" else "ep%03d_f%03d.png")%(e,x))
                im=Image.open(pth).convert("RGB")
                if im.width>768: im=im.resize((768, max(1,int(im.height*768/im.width))))
                buf=io.BytesIO(); im.save(buf, format="JPEG", quality=88)
                content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()}})
            txt=f"You are judging {scene}\n"+tail
            if ax is not None: txt += "\nPlanned actions (next 16 steps):\n"+json.dumps(ax)
            content.append({"type":"text","text":txt})
            fo.write(json.dumps({"custom_id":("ep%04d_f%03d"%(e,f)),"method":"POST","url":"/v1/chat/completions",
                "body":{"model":"gpt-5.6-luna","max_completion_tokens":2048 if eff!="none" else 64,
                        "reasoning_effort":eff,"messages":[{"role":"user","content":content}]}})+"\n")
    print(p, round(os.path.getsize(p)/1e6,1),"MB")
json.dump(paths, open(f"{OUT}/files.json","w"))
