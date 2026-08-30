"""v5 평가 — 기준 문서 기반 프롬프트가 프론티어 라벨을 재현하는가."""
import base64, io, json, os, sys, numpy as np, pandas as pd, av, urllib.request
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_common_v4 import descriptors, facts_text, SCALE
from allex_v5_prompt import SYS, ASK
H=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
USE_SYS = os.environ.get("V5_SYS","1")=="1"
OUT=f"{H}/eval_v5{'' if USE_SYS else '_nosys'}.jsonl"
lab=json.load(open(f"{H}/compressibility_labels.json"))["labels"]
items=[(int(k),v["ep"],v["f"],v["y"]) for k,v in lab.items() if v["y"]>=0]
instr="Bring the package over, orient barcode up, then place it on the conveyor."
def grab(ep, frames, side):
    want=set(frames); got={}
    with av.open(f"{H}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4") as c:
        for i,fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i]=Image.fromarray(fr.to_ndarray(format="rgb24")).resize((896,504))
                if len(got)==len(want): break
    return got
def ask(views, text):
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=90)
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
    content.append({"type":"text","text":text})
    msgs=([{"role":"system","content":SYS}] if USE_SYS else [])+[{"role":"user","content":content}]
    body={"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
          "logprobs":True,"top_logprobs":5,"messages":msgs}
    r=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    d=json.load(urllib.request.urlopen(r,timeout=180))
    YES={"Y","YES"}; NO={"N","NO"}; c=[]
    for t in d["choices"][0]["logprobs"]["content"]:
        tk=t["token"].strip().upper()
        if tk not in YES|NO: continue
        py=pn=0.0
        for alt in t["top_logprobs"]:
            a=alt["token"].strip().upper()
            if a in YES: py+=np.exp(alt["logprob"])
            elif a in NO: pn+=np.exp(alt["logprob"])
        c.append(float(py/(py+pn)) if (py+pn)>0 else (1.0 if tk in YES else 0.0))
        if len(c)==4: break
    return c if len(c)==4 else None
byep={}
for i,ep,f,y in items: byep.setdefault(ep,[]).append(f)
out=open(OUT,"w"); n=0
for ep,fs in sorted(byep.items()):
    L=grab(ep,fs,"left"); R=grab(ep,fs,"right")
    d=pd.read_parquet(f"{H}/data/chunk-000/episode_{ep:06d}.parquet")
    a=np.stack(d["action"].values)
    wr=np.stack(d["action.right_wrist_wrt_base"].values); wl=np.stack(d["action.left_wrist_wrt_base"].values)
    for i,e2,f,y in items:
        if e2!=ep: continue
        desc=descriptors(a,wr,wl,f)
        c=ask([L[f],R[f]], f"{instr}\n{SCALE}\n{facts_text(desc)}\n\n{ASK}")
        if c is None: continue
        rec={"i":i,"ep":ep,"f":f,"y":y,"desc":desc}
        for k,v in zip("ABCD",c): rec[k]=v
        out.write(json.dumps(rec)+"\n"); out.flush(); n+=1
out.close(); print(f"완료 {n}장 -> {OUT}")
