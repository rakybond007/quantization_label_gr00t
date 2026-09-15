"""Gemini 를 판정기로 쓴다. `allex_api_judge.py` 와 같은 입력, 같은 문항.

판정기끼리 비교하려면 입력이 같아야 한다 -- 청크 하나의 좌우 두 뷰 + 그 청크의
계산 사실 + GUIDANCE + 문항. 프레임 목록도 같은 것을 쓴다.

**키는 레포에 두지 않는다.** `~/.config/google/key` (chmod 600).

    python allex_gemini_judge.py <장면목록.json> <출력.jsonl> [모델]
"""
import base64, io, json, os, sys, time, urllib.error, urllib.request
import numpy as np, pandas as pd, av
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from allex_v2_common import descriptors
from allex_facts import facts as _facts
import allex_v4c_checks as CK

D=os.environ.get("ALLEX_DS","/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
SCENES, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv)>3 else "gemini-3.8-flash"
KEY=open(os.path.expanduser("~/.config/google/key")).read().strip()
Q=tuple(sorted(CK.SIGN))

def b64(im):
    b=io.BytesIO(); im.save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()

def ask(imgs, fx):
    parts=[{"inline_data":{"mime_type":"image/png","data":b64(i)}} for i in imgs]
    parts.append({"text": f"{CK.GUIDANCE}\n\nThe two images are the left and right "
                          f"camera views of this one moment.\n\n{fx}\n\n{CK.ASK}"})
    body=json.dumps({"contents":[{"parts":parts}],
                     "generationConfig":{"maxOutputTokens":256,"temperature":0,
                                         "thinkingConfig":{"thinkingBudget":0}}}).encode()
    # **thinking 을 끈다.** Sonnet 호출에는 확장 사고를 켜지 않았으므로
    # 판정기를 비교하려면 이 조건도 같아야 한다. 사고 토큰은 출력으로
    # 과금되고, 끄기 전에는 응답 1토큰에 사고 106토큰이었다.
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    for a in range(4):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(url,data=body,
                headers={"content-type":"application/json"}), timeout=180))
            c=r["candidates"][0]; ps=c.get("content",{}).get("parts") or []
            return ("".join(p.get("text","") for p in ps), r.get("usageMetadata",{}))
        except urllib.error.HTTPError as e:
            if e.code in (429,500,503) and a<3: time.sleep(5*(a+1)); continue
            return (f"ERR {e.code} {e.read().decode()[:160]}", {})
        except Exception as e:
            if a<3: time.sleep(5); continue
            return (f"ERR {type(e).__name__}", {})

def parse(t):
    g={}
    for line in t.splitlines():
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
cache={}; out=open(OUT,"a"); tin=tout=tth=nbad=n=0; t0=time.time()
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
        with av.open(p) as c:
            for i,fr in enumerate(c.decode(video=0)):
                if i==f: imgs.append(fr.to_image().resize((512,512))); break
    if len(imgs)<2: continue
    x=descriptors(A,WR,WL,f,16)
    txt,us=ask(imgs,_facts(x))
    tin+=us.get("promptTokenCount",0); tout+=us.get("candidatesTokenCount",0)
    tth+=us.get("thoughtsTokenCount",0)
    g=parse(txt)
    if not g: nbad+=1; continue
    rec={"key":[ep,f],"ep":ep,"f":f,"model":MODEL,**g,
         "conf":round(0.5*(1+sum(CK.SIGN[q]*CK.WEIGHT[q]*((g[q]-1)/4) for q in Q)),4),
         "one_handed":bool(x["one_handed"]),
         "merge_demand_k2":round(float(x["merge_demand_k2"]),4)}
    out.write(json.dumps(rec)+"\n"); out.flush(); n+=1
    if n%10==0: print(f"  {n} · 입력 {tin} 출력 {tout} 사고 {tth} · {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"완료 {n} · 형식실패 {nbad} · 입력 {tin} 출력 {tout} 사고 {tth} · {time.time()-t0:.0f}s -> {OUT}")
