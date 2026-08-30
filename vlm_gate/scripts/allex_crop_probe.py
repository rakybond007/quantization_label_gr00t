"""지각 격리 실험: 같은 접촉 질문을 전체 화면 vs 손 영역 확대로 물어 AUC 비교.

가설 — VLM이 접촉을 못 보는 것은 판단력이 아니라 손이 화면에서 너무 작아서다.
2층(결정적 계산)이 '어디를 볼지'까지 정해주면 3층의 지각이 살아난다는 구조 검증.
"""
import base64, io, json, os, sys, numpy as np, pandas as pd, av, urllib.request
from PIL import Image
H=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
MODE=sys.argv[1]              # full | crop
OUT=f"{H}/crop_probe_{MODE}.jsonl"
lab=json.load(open(f"{H}/human_labels.json"))["labels"]
items=[(int(k),v["ep"],v["f"],v["y"]) for k,v in lab.items() if v["y"]>=0]
Q=("Look at the robot's hands. Answer each check on its own line as \"A) YES\" or \"A) NO\", "
   "in order, nothing else:\n"
   "A) Is either hand actually touching an object - resting on it, pressed against its side, or\n"
   "   squeezing it - as opposed to hovering near it or moving through empty space?\n"
   "B) Is an object held between the two hands right now?\n"
   "C) Is a hand in the middle of taking hold of something or letting go of it?\n"
   "D) Are both hands empty and clear of every object?\nAnswer:")
def grab(ep, frames, side):
    want=set(frames); got={}
    with av.open(f"{H}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4") as c:
        for i,fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i]=Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got)==len(want): break
    return got
def prep(im):
    if MODE=="full": return im.resize((896,504))
    w,h=im.size                                  # 손은 항상 하단 중앙에 나타난다
    return im.crop((int(w*0.12), int(h*0.30), int(w*0.88), h)).resize((896,int(896*(h*0.70)/(w*0.76))))
def ask(views, text):
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=92)
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
out=open(OUT,"w"); n=0
for ep,fs in sorted(byep.items()):
    L=grab(ep,fs,"left"); R=grab(ep,fs,"right")
    for i,e2,f,y in items:
        if e2!=ep: continue
        c=ask([prep(L[f]), prep(R[f])], Q)
        if c is None: continue
        rec={"i":i,"ep":ep,"f":f,"y":y}
        for k,v in zip("ABCD",c): rec[k]=v
        out.write(json.dumps(rec)+"\n"); out.flush(); n+=1
out.close(); print(f"{MODE} 완료 {n}장")
