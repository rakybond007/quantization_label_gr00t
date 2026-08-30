"""cosmos가 allex에서 왜 판별을 못하는지 격리 진단.

파지 이벤트 양성/음성을 균형 표본으로 뽑아, 단일 문항 경로(검증된 경로)로
 (1) 기존 로보카사 SYSTEM  (2) allex용 SYSTEM
두 조건에서 손 개폐를 묻고 AUC를 비교한다. 프롬프트 문제인지 지각 실패인지 가른다.
"""
import json, os, sys, numpy as np, pandas as pd, glob
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_gate as V
from vlm_gate import VLMGate
A=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")
PORT=sys.argv[1]
RH=slice(14,29); LH=slice(29,44)
acts={}
for f in sorted(glob.glob(f"{A}/data/chunk-000/*.parquet")):
    acts[int(f.split("episode_")[1][:6])]=np.stack(pd.read_parquet(f)["action"].values)
instr={}
for l in open(f"{A}/meta/episodes.jsonl"):
    d=json.loads(l); instr[d["episode_index"]]=(d.get("tasks") or [""])[0]

names=sorted(os.listdir(f"{A}/tiles"))
rows=[]
for nm in names:
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:5]); a=acts[ep]
    if f>=len(a)-16: continue
    w=slice(f,f+16); rh=a[:,RH].mean(1); lh=a[:,LH].mean(1)
    hand=max(rh[w].max()-rh[w].min(), lh[w].max()-lh[w].min())
    rows.append((nm,ep,f,hand))
rows.sort(key=lambda r:-r[3])
pos=rows[:30]; neg=rows[-30:]
print(f"양성 손변화 {pos[0][3]:.3f}~{pos[-1][3]:.3f} / 음성 {neg[0][3]:.4f}~{neg[-1][3]:.4f}")

ALLEX_SYS=("You are a gate deciding whether the next ~0.5 s of a bimanual humanoid robot's motion can run "
 "at HALF the control rate without changing the outcome. The robot has two arms, each ending in a "
 "multi-finger hand. You see two head-mounted ego cameras (left and right eye).\n"
 "Compressing saves time and is preferred for gross motion — reaching, carrying a firmly held object, "
 "retracting, repositioning. It is unsafe only during the brief moments of fine manipulation: fingers "
 "closing on an object or opening to release it, and precise placement or alignment into a target pose.\n"
 "Answer the question asked, in the exact format requested.")
Q="Are the robot's fingers closing on an object right now, or opening to release one? Answer YES or NO."

def run(sys_text, tag):
    V.SYSTEM=sys_text
    g=VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)
    s=[]; y=[]
    for grp,lab in ((pos,1),(neg,0)):
        for nm,ep,f,h in grp:
            im=np.array(Image.open(f"{A}/tiles/{nm}").convert("RGB")); H,W,_=im.shape
            views=[Image.fromarray(im[:, k*W//2:(k+1)*W//2]) for k in range(2)]
            r=g.judge(views, instr.get(ep,""), "", question=Q)
            s.append(float(r.get("confidence",0.0))); y.append(lab)
    s=np.array(s); y=np.array(y).astype(bool)
    p,n=s[y],s[~y]
    auc=float(np.mean((p[:,None]>n[None,:])+0.5*(p[:,None]==n[None,:])))
    print(f"[{tag}] AUC={auc:.3f}  양성평균={p.mean():.3f}  음성평균={n.mean():.3f}  전체std={s.std():.3f}")
    return auc

run(V.SYSTEM, "기존 로보카사 SYSTEM")
run(ALLEX_SYS, "allex 전용 SYSTEM")
