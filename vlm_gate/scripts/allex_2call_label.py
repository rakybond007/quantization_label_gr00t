"""allex(양팔 휴머노이드) 2콜 라벨링 — cosmos 또는 API.

로보카사와 다른 점을 전부 반영한다:
  - 카메라 2대(ego left/right)
  - 액션 48차원 관절 절대각(라디안). EE 델타가 아니므로 "델타 합산"이 아니라
    "격 스텝 건너뛰기"가 K2의 의미다.
  - 그리퍼가 0/1이 아니라 손 15관절 x 2. 개폐는 관절 평균의 변화로 본다.
  - 30fps라 16스텝 = 0.53초.
스케일 기준은 이 데이터셋에서 직접 측정한 값(6에피소드 18,873스텝).
"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")
PORT=sys.argv[1]
OUT=f"{T}/allex_cosmos_2call_v2.jsonl"
from allex_common import VIS_ASK, ACT_ASK, SCALE, ARM, RA, LA, RH, LH
from allex_numblock import build as numblock

instr={}; 
for l in open(f"{T}/meta/episodes.jsonl"):
    d=json.loads(l); instr[d["episode_index"]]=(d.get("tasks") or [""])[0]

VIS_ASK=("Answer these four checks about the camera views. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Is either hand closing on an object right now, or opening to release one?\n"
 "B) Is a held object being precisely placed, aligned, or oriented into a target pose\n"
 "   (onto a conveyor, into a fixture, barcode turned to face a reader)?\n"
 "C) Is either arm applying load against a constraint - pushing, pulling, or holding\n"
 "   something that resists?\n"
 "D) Is this plain gross motion - reaching toward something, carrying a firmly held\n"
 "   object, retracting, or repositioning the torso?\nAnswer:")

ACT_ASK=("Answer these four checks about the planned joint trajectory. Answer each on its own\n"
 "line as \"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Do the hand joints change by more than 0.01 rad within the next 16 steps\n"
 "   (a grasp or release)?\n"
 "B) Does the arm motion reverse direction - two consecutive steps both moving more\n"
 "   than 0.024 rad but turning more than 90 degrees in joint space?\n"
 "C) Do the arms stay slower than 0.008 rad/step for most of the window while the\n"
 "   hands are holding a closed pose?\n"
 "D) Does the arm speed decrease steadily and end below 0.010 rad/step?\nAnswer:")


gate=VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r["ep"],r["f"]))
        except Exception: pass
acts={}
wrists={}
def A(ep):
    if ep not in acts:
        d=pd.read_parquet(f"{T}/data/chunk-000/episode_{ep:06d}.parquet")
        acts[ep]=np.stack(d["action"].values)
        wrists[ep]=(np.stack(d["action.right_wrist_wrt_base"].values),
                    np.stack(d["action.left_wrist_wrt_base"].values))
    return acts[ep]
def W(ep):
    A(ep); return wrists[ep]

out=open(OUT,"a"); n=0
for nm in sorted(os.listdir(f"{T}/tiles")):
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:5])
    if (ep,f) in done: continue
    a=A(ep)
    if f>=len(a)-16: continue
    im=np.array(Image.open(f"{T}/tiles/{nm}").convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//2:(k+1)*w//2]) for k in range(2)]
    rec={"ep":ep,"f":f}
    r1=gate.judge(views, instr.get(ep,""), "", question=VIS_ASK, n_ask=4)
    # 관절 궤적은 팔 14 + 손 평균 2개로 압축해서 보여준다(48차원 원본은 토큰 낭비)
    wr,wl=W(ep)
    ins=f"{instr.get(ep,'')}\n{SCALE}\n{numblock(a, wr, wl, f)}"
    r2=gate.judge(views, ins, "", question=ACT_ASK, n_ask=4)
    c1=r1.get("confidences") or [0.0]*4; c2=r2.get("confidences") or [0.0]*4
    for k,v in zip("ABCD",c1): rec[k]=float(v)
    for k,v in zip("EFGH",c2): rec[k]=float(v)
    rec["ans"]=r1.get("answer","")+" | "+r2.get("answer","")
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%50==0: print(f"{n}장", flush=True); out.flush()
out.close(); print(f"완료 {n}장 -> {OUT}")
