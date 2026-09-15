"""문항 A 문구 후보를 API 판정기에 돌려 표적 분리도를 잰다.

정답지는 눈으로 표시한 네 범주. A 의 표적은 **양손상자**(실측 0.533, 최악)다.
가중으로는 못 고친다 -- A차는 등급 자체의 분리도이고 가중은 문항 간 배분만
바꾼다. 고칠 수 있는 것은 문구뿐이다.

    python allex_a_sweep_api.py <gemini|sonnet> [청크수]
"""
import base64, io, json, os, sys, time, urllib.error, urllib.request
import numpy as np, pandas as pd, av
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from allex_v2_common import descriptors
from allex_facts import facts as _facts
import allex_v4c_checks as CK

WHICH=sys.argv[1] if len(sys.argv)>1 else "gemini"
NMAX=int(sys.argv[2]) if len(sys.argv)>2 else 40
D=os.environ.get("ALLEX_DS","/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
EYE={tuple(int(x) for x in k.split("_")):v for k,v in
     json.load(open(f"{BASE}/_tmp/eye_labels.json")).items() if v>=0}
Q=sorted(CK.SIGN)

CANDS=[
 ("orig",
  "Are the two hands SQUEEZING something between them to hold it, and is that\n"
  "   thing HARD -- a box, a carton, something that will not squash?"),
 ("both_on_box",
  "Are BOTH grippers on the same object at once, one on each side of it,\n"
  "   holding a stiff box between them?"),
 ("clamped",
  "Is a box CLAMPED BETWEEN THE TWO ARMS -- each arm pressing on an opposite\n"
  "   face, so the box would fall if either arm eased off?"),
 ("two_sided",
  "Is the object held FROM TWO SIDES AT ONCE rather than by a single hand --\n"
  "   with a rigid body caught between the opposing grippers?"),
 ("no_fingers_round",
  "Is a rigid box being held by PRESSURE FROM BOTH SIDES, with neither gripper's\n"
  "   fingers wrapped around or hooked under it?"),
]

def imgs_for(ep,f,cache):
    if ep not in cache:
        d=pd.read_parquet(f"{D}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
        cache[ep]=(np.stack(d["action"].values),
                   np.stack(d["action.right_wrist_wrt_base"].values),
                   np.stack(d["action.left_wrist_wrt_base"].values))
    ims=[]
    for side in ("left","right"):
        p=f"{D}/videos/chunk-{ep//1000:03d}/observation.images.camera_ego_{side}/episode_{ep:06d}.mp4"
        with av.open(p) as c:
            for i,fr in enumerate(c.decode(video=0)):
                if i==f: ims.append(fr.to_image().resize((512,512))); break
    return ims, cache[ep]

def b64(im):
    b=io.BytesIO(); im.save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()

if WHICH=="gemini":
    KEY=open(os.path.expanduser("~/.config/google/key")).read().strip()
    MODEL="gemini-3.8-flash"
    def call(ims, prompt):
        parts=[{"inline_data":{"mime_type":"image/png","data":b64(i)}} for i in ims]
        parts.append({"text":prompt})
        body=json.dumps({"contents":[{"parts":parts}],
            "generationConfig":{"maxOutputTokens":256,"temperature":0,
                                "thinkingConfig":{"thinkingBudget":0}}}).encode()
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
        r=json.load(urllib.request.urlopen(urllib.request.Request(url,data=body,
            headers={"content-type":"application/json"}), timeout=180))
        ps=r["candidates"][0].get("content",{}).get("parts") or []
        return "".join(p.get("text","") for p in ps)
else:
    KEY=open(os.path.expanduser("~/.config/anthropic/key")).read().strip()
    MODEL="claude-sonnet-4-5"
    def call(ims, prompt):
        content=[{"type":"image","source":{"type":"base64","media_type":"image/png","data":b64(i)}} for i in ims]
        content.append({"type":"text","text":prompt})
        body=json.dumps({"model":MODEL,"max_tokens":64,
                         "messages":[{"role":"user","content":content}]}).encode()
        r=json.load(urllib.request.urlopen(urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key":KEY,"anthropic-version":"2023-06-01",
                     "content-type":"application/json"}), timeout=180))
        return r["content"][0]["text"]

def ask_text(at):
    return CK.ASK.split("A) ")[0] + "A) " + at + "\n" + "B) " + CK.ASK.split("B) ")[1]

def parse(t):
    g={}
    for line in t.splitlines():
        s=line.strip().lstrip("*# ")
        if len(s)>=3 and s[0] in Q and s[1]==")":
            d="".join(c for c in s[2:] if c.isdigit())
            if d: g[s[0]]=max(1,min(5,int(d[0])))
    return g if len(g)==len(Q) else None

# 표적(양손상자) 과 나머지를 균형 있게
two=[k for k,v in EYE.items() if v==2][:NMAX//2]
oth=[k for k,v in EYE.items() if v!=2][:NMAX//2]
ks=two+oth
print(f"[i] {WHICH} · {len(ks)}청크 (양손상자 {len(two)} · 나머지 {len(oth)})", flush=True)
cache={}; prep={}
for ep,f in ks:
    ims,(A,WR,WL)=imgs_for(ep,f,cache)
    prep[(ep,f)]=(ims, _facts(descriptors(A,WR,WL,f,16)))
for name,at in CANDS:
    ASK=ask_text(at); res={}
    for k in ks:
        ims,fx=prep[k]
        prompt=f"{CK.GUIDANCE}\n\nThe two images are the left and right camera views of this one moment.\n\n{fx}\n\n{ASK}"
        try: g=parse(call(ims,prompt))
        except Exception: g=None
        if g: res[k]=g["A"]
    if len(res)<len(ks)*0.6: print(f"  {name:16s} 응답 부족 {len(res)}/{len(ks)}"); continue
    a=np.array([res[k] for k in res if EYE[k]==2]); b=np.array([res[k] for k in res if EYE[k]!=2])
    import collections
    print(f"  {name:16s} 양손 {a.mean():.2f} vs 나머지 {b.mean():.2f} · **차 {a.mean()-b.mean():+.2f}** · "
          f"양손4+ {np.mean(a>=4):.0%} · 분포 {dict(sorted(collections.Counter(np.concatenate([a,b]).tolist()).items()))}", flush=True)
