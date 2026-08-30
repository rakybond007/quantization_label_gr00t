"""allex 2콜 라벨링 — OpenAI(gpt-5.6-luna) 직접 호출판.

cosmos 판과 동일한 질문·스케일 기준을 쓴다. 문항별 confidence는 답 문자열
"YNNY"의 각 위치 top_logprobs에서 P(Y) vs P(N)으로 뽑는다(로컬 판정기의
슬롯 채점과 같은 것을 API 수단으로).
"""
import base64, io, json, os, sys, threading, queue, urllib.request, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_common import VIS_ASK, ACT_ASK, SCALE, ARM, RA, LA, RH, LH   # 질문 정의 공용
from allex_numblock import build as numblock
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")
OUT=f"{T}/allex_api_2call_v2.jsonl"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
NW=int(sys.argv[1]) if len(sys.argv)>1 else 6

instr={}
for l in open(f"{T}/meta/episodes.jsonl"):
    d=json.loads(l); instr[d["episode_index"]]=(d.get("tasks") or [""])[0]

def ask4(views, text):
    """4문항 -> [P(Y) x4]. 답은 YNNY 형식 4글자."""
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=88)
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
    content.append({"type":"text","text":text})
    body={"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
          "logprobs":True,"top_logprobs":5,
          "messages":[{"role":"user","content":content}]}
    req=urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    r=json.load(urllib.request.urlopen(req,timeout=180))
    lp=r["choices"][0]["logprobs"]["content"]
    confs=[]
    # "NNNY"를 요구하면 토크나이저가 'NN','NY'로 묶어버려 문항별 확률이 사라진다.
    # 라벨 줄 형식이면 ' YES'/' NO'가 문항마다 독립 토큰으로 나온다.
    YES={"Y","YES"}; NO={"N","NO"}
    for tokinfo in lp:
        tk=tokinfo["token"].strip().upper()
        if tk not in YES|NO: continue
        py=pn=0.0
        for alt in tokinfo["top_logprobs"]:
            a=alt["token"].strip().upper()
            if a in YES: py+=np.exp(alt["logprob"])
            elif a in NO: pn+=np.exp(alt["logprob"])
        confs.append(float(py/(py+pn)) if (py+pn)>0 else (1.0 if tk in YES else 0.0))
        if len(confs)==4: break
    if len(confs)<4: raise ValueError(f"문항 {len(confs)}개만 파싱됨: {r['choices'][0]['message']['content']!r}")
    return confs, r["choices"][0]["message"]["content"].strip()

# cosmos와 동일한 라벨 줄 형식을 그대로 쓴다 — 형식이 같아야 두 티처가 비교 가능하고,
# 토큰도 문항마다 분리된다.
VIS4=VIS_ASK; ACT4=ACT_ASK

done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r["ep"],r["f"]))
        except Exception: pass
names=[n for n in sorted(os.listdir(f"{T}/tiles"))
       if (int(n[2:6]), int(n.split("_f")[1][:5])) not in done]
q=queue.Queue(); [q.put(n) for n in names]
lock=threading.Lock(); out=open(OUT,"a"); stat={"ok":0,"err":0}
acts={}; wrists={}; alock=threading.Lock()
def A(ep):
    with alock:
        if ep not in acts:
            d=pd.read_parquet(f"{T}/data/chunk-000/episode_{ep:06d}.parquet")
            acts[ep]=np.stack(d["action"].values)
            wrists[ep]=(np.stack(d["action.right_wrist_wrt_base"].values),
                        np.stack(d["action.left_wrist_wrt_base"].values))
        return acts[ep]
def W(ep):
    A(ep); return wrists[ep]
def work():
    while True:
        try: nm=q.get_nowait()
        except queue.Empty: return
        try:
            ep=int(nm[2:6]); f=int(nm.split("_f")[1][:5])
            a=A(ep)
            if f>=len(a)-16: continue
            im=np.array(Image.open(f"{T}/tiles/{nm}").convert("RGB")); h,w,_=im.shape
            views=[Image.fromarray(im[:, k*w//2:(k+1)*w//2]) for k in range(2)]
            c1,a1=ask4(views, VIS4)
            wr,wl=W(ep)
            txt=f"{instr.get(ep,'')}\n{SCALE}\n{numblock(a, wr, wl, f)}\n\n{ACT4}"
            c2,a2=ask4(views, txt)
            rec={"ep":ep,"f":f,"ans":a1+" | "+a2}
            for k,v in zip("ABCD",c1): rec[k]=v
            for k,v in zip("EFGH",c2): rec[k]=v
            with lock:
                out.write(json.dumps(rec)+"\n"); out.flush(); stat["ok"]+=1
                if stat["ok"]%25==0: print(f"{stat['ok']}장 (실패 {stat['err']})", flush=True)
        except Exception as e:
            with lock:
                stat["err"]+=1
                if stat["err"]<=3: print("오류:",type(e).__name__,str(e)[:160], flush=True)
ts=[threading.Thread(target=work) for _ in range(NW)]
[t.start() for t in ts]; [t.join() for t in ts]
out.close(); print(f"완료 성공 {stat['ok']} / 실패 {stat['err']} -> {OUT}")
