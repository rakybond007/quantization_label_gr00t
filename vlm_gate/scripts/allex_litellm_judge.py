"""LiteLLM 프록시를 통해 판정한다. 직결 API 판정기와 입력이 같다.

사내 프록시(OpenAI 호환)로 나가므로 모델을 바꿔 끼우기 쉽고, 키 관리가 한 곳이다.
직결(`allex_gemini_judge.py`)과 같은 것을 보낸다 -- 청크 하나의 좌우 두 뷰 +
그 청크의 계산 사실 + GUIDANCE + 문항.

**사고 억제는 `reasoning_effort: "low"`.** 직결의 `thinkingLevel: "low"` 와 같은
자리이고, 실측으로 출력이 999 -> 19 토큰으로 준다. 직결에서 확인했듯 사고를
완전히 끌 수는 없고 줄일 수만 있다.

    키   ~/.config/litellm/key   (chmod 600, 레포에 두지 않는다)
    주소 ~/.config/litellm/base

    python allex_litellm_judge.py <장면목록.json> <출력.jsonl> [모델]
"""
import base64, io, json, os, sys, time, urllib.error, urllib.request
import numpy as np, pandas as pd, av
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
import importlib
from allex_v2_common import descriptors
from allex_facts import facts as _facts
CK = importlib.import_module(os.environ.get("ALLEX_CHECKS_MODULE", "allex_v4c_checks"))

D=os.environ.get("ALLEX_DS","/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
SCENES, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv)>3 else "gemini/gemini-3.8-flash"
EFFORT = os.environ.get("ALLEX_EFFORT", "low")
KEY=open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL=open(os.path.expanduser("~/.config/litellm/base")).read().strip()+"/v1/chat/completions"
Q=tuple(sorted(CK.SIGN))

def b64(im):
    b=io.BytesIO(); im.save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()

def ask(imgs, fx):
    content=[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64(i)}} for i in imgs]
    content.append({"type":"text","text":
        f"{CK.GUIDANCE}\n\nThe two images are the left and right camera views of "
        f"this one moment.\n\n{fx}\n\n{CK.ASK}"})
    body={"model":MODEL,"max_tokens":1024,"temperature":0,
          "messages":[{"role":"user","content":content}]}
    if EFFORT: body["reasoning_effort"]=EFFORT
    req=urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization":f"Bearer {KEY}","content-type":"application/json"})
    for a in range(4):
        try:
            r=json.load(urllib.request.urlopen(req, timeout=180))
            return r["choices"][0]["message"]["content"], r.get("usage",{})
        except urllib.error.HTTPError as e:
            if e.code in (429,500,502,503,529) and a<3: time.sleep(5*(a+1)); continue
            return f"ERR {e.code} {e.read().decode()[:160]}", {}
        except Exception:
            if a<3: time.sleep(5); continue
            return "ERR timeout", {}

def parse(t):
    g={}
    for line in (t or "").splitlines():
        s=line.strip().lstrip("*# ")
        if len(s)>=3 and s[0] in Q and s[1]==")":
            d="".join(c for c in s[2:] if c.isdigit())
            if d: g[s[0]]=max(1,min(CK.NGRADE,int(d[0])))
    return g if len(g)==len(Q) else None

scenes=json.load(open(SCENES)); done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(tuple(json.loads(l)["key"]))
        except Exception: pass
    print(f"[i] 이어서 {len(done)}", flush=True)
cache={}; out=open(OUT,"a"); tin=tout=nbad=n=0; t0=time.time()
for sc in scenes:
    ep,f=sc["ep"],sc["f"]
    if (ep,f) in done: continue
    if ep not in cache:
        d=pd.read_parquet(f"{D}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
        cache[ep]=(np.stack(d["action"].values),
                   np.stack(d["action.right_wrist_wrt_base"].values),
                   np.stack(d["action.left_wrist_wrt_base"].values))
    A,WR,WL=cache[ep]; imgs=[]
    for side in ("left","right"):
        p=f"{D}/videos/chunk-{ep//1000:03d}/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
        try:
            with av.open(p) as c:
                for i,fr in enumerate(c.decode(video=0)):
                    if i==f: imgs.append(fr.to_image().resize((512,512))); break
        except Exception: pass
    if len(imgs)<2: continue
    x=descriptors(A,WR,WL,f,16)
    txt,us=ask(imgs,_facts(x))
    tin+=us.get("prompt_tokens",0) or 0; tout+=us.get("completion_tokens",0) or 0
    g=parse(txt)
    if not g: nbad+=1; continue
    rec={"key":[ep,f],"ep":ep,"f":f,"model":MODEL,"effort":EFFORT,**g,
         "conf":round(0.5*(1+sum(CK.SIGN[q]*CK.WEIGHT[q]*((g[q]-1)/4) for q in Q)),4),
         "one_handed":bool(x["one_handed"]),
         "merge_demand_k2":round(float(x["merge_demand_k2"]),4)}
    out.write(json.dumps(rec)+"\n"); out.flush(); n+=1
    if n%25==0: print(f"  {n} · 입력 {tin} 출력 {tout} · {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"완료 {n} · 형식실패 {nbad} · 입력 {tin} 출력 {tout} · {time.time()-t0:.0f}s -> {OUT}")
