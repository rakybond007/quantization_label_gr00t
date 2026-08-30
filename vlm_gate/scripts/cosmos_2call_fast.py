"""cosmos 2콜 라벨링 — 이미지 프리필 1회로 4문항 confidence를 받는다.

기존 8콜(문항당 프리필 1회)과 동일한 문항·가이던스를 쓰되, 시각 4문항을 한 콜,
액션 4문항을 한 콜로 묶어 프레임당 2콜로 줄인다. 8콜 산출물과의 상관을 검증한 뒤
전량에 적용한다.
"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
PORT=sys.argv[1]; SHARD=int(sys.argv[2]); NSH=int(sys.argv[3])
LIMIT=int(sys.argv[4]) if len(sys.argv)>4 else 0          # >0: 검증용 소규모
STRAT=(len(sys.argv)>5 and sys.argv[5]=="strat")          # 8콜 층화 프레임과 동일 집합
import os as _os
_SYS=_os.environ.get("GATE_SYSTEM","default")
_sfx="" if _SYS=="default" else f"_{_SYS}"
OUT=(f"{BASE}/output/_gate_distill/cosmos_2call_fast_strat{_sfx}.jsonl" if STRAT else
     f"{BASE}/output/_gate_distill/cosmos_2call_fast_val2.jsonl" if LIMIT else
     f"{BASE}/output/_gate_distill/cosmos_fast_s{NSH}_{SHARD}.jsonl")

info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
_GF=("robocasa_cosmos_ttl_best_guidance_aligned.txt"
     if os.environ.get("GATE_SYSTEM")=="aligned" else "robocasa_cosmos_ttl_best_guidance.txt")
# aligned 모드에서는 가이던스도 YES/NO 결속을 제거한 판본을 쓴다 —
# 극성 충돌의 출처가 SYSTEM만이 아니라 가이던스에도 있기 때문.
G=open(f"{BASE}/analysis/_evolver/_varkA/{_GF}").read().strip()

VIS_ASK=("Answer these four checks about the camera views. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
 "B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
 "(sink basin, cabinet shelf, microwave, burner)?\n"
 "C) Is a door or drawer being PULLED OPEN, with the grasped handle under load?\n"
 "D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad sweep, "
 "or pressing a rigidly mounted button or knob?\nAnswer:")
ACT_ASK=("Answer these four checks about the planned action numbers. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Does the gripper command (last number of each step) change value within the next 16 steps?\n"
 "B) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more than 90 degrees?\n"
 "C) Is the gripper closed while the end-effector deltas stay below 0.12 for most of the window?\n"
 "D) Do the delta magnitudes decrease steadily and end below 0.15?\nAnswer:")
SCALE=("SCALE REFERENCE: |d| is typically 0.34; 0.12 is slow (10th pct), 0.73 is fast (90th pct); "
       "consecutive steps turn about 11 degrees on average; while carrying an object |d| is still about 0.33.")
gate=VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)

TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r['ep'],r['f']))
        except Exception: pass
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
        if len(acts)>40:
            for k in list(acts)[:20]: acts.pop(k,None)
    return acts[ep]

WANT=None
if STRAT:                                   # 8콜이 라벨링한 층화 집합 그대로
    WANT=set()
    for l in open(f"{BASE}/output/_gate_distill/cosmos_2call_bits.jsonl"):
        try: r=json.loads(l); WANT.add((r["ep"],r["f"]))
        except Exception: pass
    print(f"층화 프레임 {len(WANT)}개")

out=open(OUT,"a"); n=0
MAN=f"{BASE}/output/_gate_distill/tiles_manifest.txt"
# 26만 항목 디렉터리를 샤드마다 READDIR 하면 NFS에 순간 부하가 걸린다.
# 매니페스트를 한 번 만들어 두고 그걸 읽는다.
_names=(sorted(open(MAN).read().split()) if os.path.exists(MAN) else sorted(os.listdir(TIL)))
for nm in _names:
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:3])
    if WANT is not None and (ep,f) not in WANT: continue
    if WANT is None and ep % NSH != SHARD: continue
    if (ep,f) in done: continue
    a=A(ep)
    if a is None or f>=len(a)-4: continue
    im=np.array(Image.open(f"{TIL}/{nm}").convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//3:(k+1)*w//3]) for k in range(3)]
    rec={"ep":ep,"f":f}
    r1=gate.judge(views, instr.get(ep,""), G, question=VIS_ASK, n_ask=4)
    anum=json.dumps(np.round(a[f:f+16,5:],2).tolist())
    ins=f"{instr.get(ep,'')}\n{SCALE}\nPlanned actions (7 numbers/step, gripper last):\n{anum}"
    r2=gate.judge(views, ins, G, question=ACT_ASK, n_ask=4)
    c1=r1.get("confidences") or [0.0]*4; c2=r2.get("confidences") or [0.0]*4
    for k,v in zip("ABCD", c1): rec[k]=float(v)
    for k,v in zip("EFGH", c2): rec[k]=float(v)
    rec["ans"]=r1.get("answer","")+r2.get("answer","")
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%100==0: print(f"shard{SHARD}: {n}", flush=True); out.flush()
    if LIMIT and not STRAT and n>=LIMIT: break
out.close(); print(f"shard{SHARD} 완료 {n} -> {OUT}")
