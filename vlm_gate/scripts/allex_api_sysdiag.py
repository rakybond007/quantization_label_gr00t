"""API 쪽 대조: 동일 프레임·동일 질문에 SYSTEM만 바꿔 성능 차이를 잰다.

  (1) system 없음           — 지금까지 API가 받은 조건 (AUC 0.760)
  (2) 로보카사 SYSTEM       — cosmos가 받았던 오염된 조건
  (3) allex 전용 SYSTEM     — 엠보디먼트에 맞춘 조건
cosmos 진단과 같은 균형 표본(파지 양성 30 / 음성 30)을 쓴다.
"""
import base64, io, json, os, sys, glob, threading, queue, urllib.request
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_gate as V
A=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
RH=slice(14,29); LH=slice(29,44)
acts={}
for f in sorted(glob.glob(f"{A}/data/chunk-000/*.parquet")):
    acts[int(f.split("episode_")[1][:6])]=np.stack(pd.read_parquet(f)["action"].values)
instr={}
for l in open(f"{A}/meta/episodes.jsonl"):
    d=json.loads(l); instr[d["episode_index"]]=(d.get("tasks") or [""])[0]
rows=[]
for nm in sorted(os.listdir(f"{A}/tiles")):
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:5]); a=acts[ep]
    if f>=len(a)-16: continue
    w=slice(f,f+16); rh=a[:,RH].mean(1); lh=a[:,LH].mean(1)
    rows.append((nm,ep,f,max(rh[w].max()-rh[w].min(), lh[w].max()-lh[w].min())))
rows.sort(key=lambda r:-r[3])
SAMPLE=[(r,1) for r in rows[:30]]+[(r,0) for r in rows[-30:]]

ALLEX_SYS=("You are a gate deciding whether the next ~0.5 s of a bimanual humanoid robot's motion can run "
 "at HALF the control rate without changing the outcome. The robot has two arms, each ending in a "
 "multi-finger hand. You see two head-mounted ego cameras (left and right eye).\n"
 "Compressing saves time and is preferred for gross motion — reaching, carrying a firmly held object, "
 "retracting, repositioning. It is unsafe only during the brief moments of fine manipulation: fingers "
 "closing on an object or opening to release it, and precise placement or alignment into a target pose.\n"
 "Answer the question asked, in the exact format requested.")
# A문항(손 개폐)은 API 전체 결과에서도 단독 AUC 0.46으로 판별력이 없었다.
# 실제로 신호가 있던 액션 숫자 문항 4개(E~H)를 그대로 쓰고 noisy-OR로 집계해
# SYSTEM 유무가 "판별되는 조건"에서 성능을 얼마나 움직이는지를 잰다.
from allex_common import ACT_ASK, SCALE, ARM, RH as _RH, LH as _LH
Q=ACT_ASK

def call(views, sys_text, txt):
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=88)
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
    content.append({"type":"text","text":txt})
    msgs=([{"role":"system","content":sys_text}] if sys_text else [])+[{"role":"user","content":content}]
    body={"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
          "logprobs":True,"top_logprobs":5,"messages":msgs}
    r=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    d=json.load(urllib.request.urlopen(r,timeout=180))
    YES={"Y","YES"}; NO={"N","NO"}; confs=[]
    for t in d["choices"][0]["logprobs"]["content"]:
        tk=t["token"].strip().upper()
        if tk not in YES|NO: continue
        py=pn=0.0
        for alt in t["top_logprobs"]:
            a=alt["token"].strip().upper()
            if a in YES: py+=np.exp(alt["logprob"])
            elif a in NO: pn+=np.exp(alt["logprob"])
        confs.append(float(py/(py+pn)) if (py+pn)>0 else (1.0 if tk in YES else 0.0))
        if len(confs)==4: break
    return confs if len(confs)==4 else None

def run(sys_text, tag, nw=6):
    q=queue.Queue(); [q.put(x) for x in SAMPLE]
    out=[]; lock=threading.Lock()
    def work():
        while True:
            try: (nm,ep,f,h),lab=q.get_nowait()
            except queue.Empty: return
            try:
                im=np.array(Image.open(f"{A}/tiles/{nm}").convert("RGB")); H,W,_=im.shape
                views=[Image.fromarray(im[:, k*W//2:(k+1)*W//2]) for k in range(2)]
                a=acts[ep]; win=a[f:f+16]
                comp=np.column_stack([win[:,ARM], win[:,_RH].mean(1), win[:,_LH].mean(1)])
                txt=(f"{instr.get(ep,'')}\n{SCALE}\nPlanned joint targets, absolute radians, 16 steps x 16 "
                     f"numbers (14 arm joints, then right-hand mean, then left-hand mean):\n"
                     f"{json.dumps(np.round(comp,3).tolist())}\n\n{Q}")
                c=call(views, sys_text, txt)
                if c is not None:
                    # noisy-OR: E,F,G는 위험 신호, H는 안전 신호
                    risk=1-(1-c[0])*(1-c[1])*(1-c[2])
                    with lock: out.append(((1-risk)*(0.5+0.5*c[3]), lab))
            except Exception as e:
                with lock: pass
    ts=[threading.Thread(target=work) for _ in range(nw)]
    [t.start() for t in ts]; [t.join() for t in ts]
    s=np.array([o[0] for o in out]); y=np.array([o[1] for o in out]).astype(bool)
    p,n=s[y],s[~y]
    auc=float(np.mean((p[:,None]<n[None,:])+0.5*(p[:,None]==n[None,:]))) if len(p) and len(n) else float("nan")
    print(f"[{tag}] n={len(s)} AUC={auc:.3f}  양성평균={p.mean():.3f}  음성평균={n.mean():.3f}  std={s.std():.3f}", flush=True)

run(None, "system 없음(기존 API 조건)")
run(V.SYSTEM, "로보카사 SYSTEM(cosmos가 받던 것)")
run(ALLEX_SYS, "allex 전용 SYSTEM")
