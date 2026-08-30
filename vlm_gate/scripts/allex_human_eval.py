"""육안 라벨 24프레임에 티처를 돌려, 영상 문항이 실제로 작동하는지 본다.

물리 프록시(손관절/팔속도)는 액션 문항과 동어반복이라 영상 문항의 공로를 못 잰다.
사람이 영상만 보고 매긴 라벨로 채점하면 그 순환이 끊긴다.
해상도(224 vs 720p)도 같이 비교한다 — 영상 문항이 죽어 있던 게 해상도 탓인지 본다.
"""
import base64, io, json, os, sys, numpy as np, pandas as pd, av, urllib.request
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os
_V=_os.environ.get('ALLEX_Q','v1')
if _V=='v3':
    from allex_common_v3 import VIS_ASK, ACT_ASK, SCALE, RA, LA, RH, LH
else:
    from allex_common import VIS_ASK, ACT_ASK, SCALE, RA, LA, RH, LH
from allex_numblock import build as numblock
H=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
RES=sys.argv[1] if len(sys.argv)>1 else "hi"      # hi | lo
OUT=f"{H}/human_eval_{RES}_{_V}.jsonl"
lab=json.load(open(f"{H}/human_labels.json"))["labels"]
items=[(int(k),v["ep"],v["f"],v["y"]) for k,v in lab.items() if v["y"]>=0]
instr="Bring the package over, orient barcode up, then place it on the conveyor."

def grab(ep, frames, side):
    want=set(frames); got={}
    p=f"{H}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
    with av.open(p) as c:
        for i,fr in enumerate(c.decode(video=0)):
            if i in want:
                im=Image.fromarray(fr.to_ndarray(format="rgb24"))
                got[i]= im.resize((224,224)) if RES=="lo" else im.resize((896,504))
                if len(got)==len(want): break
    return got

def ask(views, text):
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=90)
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
    content.append({"type":"text","text":text})
    body={"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
          "logprobs":True,"top_logprobs":5,"messages":[{"role":"user","content":content}]}
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

byep={}
for i,ep,f,y in items: byep.setdefault(ep,[]).append(f)
out=open(OUT,"w")
for ep,fs in sorted(byep.items()):
    L=grab(ep,fs,"left"); R=grab(ep,fs,"right")
    d=pd.read_parquet(f"{H}/data/chunk-000/episode_{ep:06d}.parquet")
    a=np.stack(d["action"].values)
    wr=np.stack(d["action.right_wrist_wrt_base"].values); wl=np.stack(d["action.left_wrist_wrt_base"].values)
    for i,e2,f,y in items:
        if e2!=ep: continue
        views=[L[f],R[f]]
        c1=ask(views, f"{instr}\n\n{VIS_ASK}")
        c2=ask(views, f"{instr}\n{SCALE}\n{numblock(a,wr,wl,f)}\n\n{ACT_ASK}")
        if c1 is None or c2 is None:
            print(f"#{i} 파싱 실패", flush=True); continue
        rec={"i":i,"ep":ep,"f":f,"y":y}
        for k,v in zip("ABCD",c1): rec[k]=v
        for k,v in zip("EFGH",c2): rec[k]=v
        out.write(json.dumps(rec)+"\n"); out.flush()
        print(f"#{i} ep{ep} f{f} y={y}  영상 {[round(x,2) for x in c1]}  액션 {[round(x,2) for x in c2]}", flush=True)
out.close(); print(f"저장 -> {OUT}")
