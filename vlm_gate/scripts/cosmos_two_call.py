"""cosmos 2회 분리 호출 — 영상 4질문(이미지) + 액션 4질문(이미지+수치), 질문별 확률 수집"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
PORT=sys.argv[1]; OUT=sys.argv[2]
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
VIS=[("A","Is the gripper closing on an object or a handle right now, or opening to release one? Answer YES or NO."),
     ("B","Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle (sink basin, cabinet shelf, microwave, burner)? Answer YES or NO."),
     ("C","Is a door or drawer being PULLED OPEN, with the grasped handle under load? Answer YES or NO."),
     ("D","Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad sweep, or pressing a rigidly mounted button or knob? Answer YES or NO.")]
ACT=[("E","Does the gripper command (last number of each step) change value within the next 16 steps? Answer YES or NO."),
     ("F","Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more than 90 degrees? Answer YES or NO."),
     ("G","Is the gripper closed while the end-effector deltas stay below 0.12 for most of the window? Answer YES or NO."),
     ("H","Do the delta magnitudes decrease steadily and end below 0.15? Answer YES or NO.")]
SCALE=("SCALE REFERENCE: |d| is typically 0.34; 0.12 is slow (10th pct), 0.73 is fast (90th pct); "
       "consecutive steps turn about 11 degrees on average; while carrying an object |d| is still about 0.33.")
gate=VLMGate(f"http://127.0.0.1:{PORT}", timeout=120)
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
frames=[]
for l in open(f"{BASE}/output/_gate_distill/exp_cp_s_vis4/labels.jsonl"):
    r=json.loads(l); frames.append((r['ep'],r['f']))
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
    return acts[ep]
out=open(OUT,"w"); n=0
for ep,f in frames:
    p=f"{TIL}/ep{ep:04d}_f{f:03d}.png"
    a=A(ep)
    if not os.path.exists(p) or a is None or f>=len(a)-4: continue
    im=np.array(Image.open(p).convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//3:(k+1)*w//3]) for k in range(3)]
    rec={"ep":ep,"f":f}
    for key,q in VIS:                       # 1차: 영상만
        r=gate.judge(views, instr.get(ep,""), G, question=q) if "question" in gate.judge.__code__.co_varnames else gate.judge(views, instr.get(ep,""), G)
        rec[key]=float(r.get("confidence",0.0))
    anum=json.dumps(np.round(a[f:f+16,5:],2).tolist())
    for key,q in ACT:                       # 2차: 액션 수치 포함
        ins=f"{instr.get(ep,'')}\n{SCALE}\nPlanned actions (7 numbers/step, gripper last):\n{anum}"
        r=gate.judge(views, ins, G, question=q) if "question" in gate.judge.__code__.co_varnames else gate.judge(views, ins, G)
        rec[key]=float(r.get("confidence",0.0))
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%50==0: print(f"{n}/{len(frames)}", flush=True); out.flush()
out.close(); print("완료", n)
