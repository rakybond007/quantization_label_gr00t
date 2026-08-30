"""cosmos 전량 2회 분리 라벨링 (stride8, 전 에피소드) — 샤드 분할 실행"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
PORT, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT=f"{BASE}/output/_gate_distill/cosmos_full2call_s{NSH}_{SHARD}.jsonl"
PRIOR=[f"{BASE}/output/_gate_distill/cosmos_full2call_shard{i}.jsonl" for i in range(4)]
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
done=set()
for _p in [OUT]+PRIOR:                     # 이전 4샤드 산출물도 재개 기준에 포함
    if not os.path.exists(_p): continue
    for l in open(_p):
        try: r=json.loads(l); done.add((r['ep'],r['f']))
        except Exception: pass
names=sorted(os.listdir(TIL))
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
        if len(acts)>40:
            for k in list(acts)[:20]: acts.pop(k,None)
    return acts[ep]
out=open(OUT,"a"); n=0
for nm in names:
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:3])
    if ep % NSH != SHARD or (ep,f) in done: continue
    a=A(ep)
    if a is None or f>=len(a)-4: continue
    im=np.array(Image.open(f"{TIL}/{nm}").convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//3:(k+1)*w//3]) for k in range(3)]
    rec={"ep":ep,"f":f}
    for key,q in VIS:
        rec[key]=float(gate.judge(views, instr.get(ep,""), G, question=q).get("confidence",0.0))
    anum=json.dumps(np.round(a[f:f+16,5:],2).tolist())
    ins=f"{instr.get(ep,'')}\n{SCALE}\nPlanned actions (7 numbers/step, gripper last):\n{anum}"
    for key,q in ACT:
        rec[key]=float(gate.judge(views, ins, G, question=q).get("confidence",0.0))
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%200==0: print(f"shard{SHARD}: {n} (누적 {len(done)+n})", flush=True); out.flush()
out.close(); print(f"shard{SHARD} 완료 {n}")
