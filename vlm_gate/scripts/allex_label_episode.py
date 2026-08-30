"""allex 에피소드를 런타임 단위(16스텝 청크)로 전부 라벨링한다.

로보카사 v6b와 같은 구조:
  계산층 — 손목 간격·추세, 팔 속도, 손가락 변화, K2 병합 실현가능성을 정확히 계산
  VLM층 — 계산으로 알 수 없는 것만 질문 (물체 종류, 자세 확정 여부, 접촉 국면)
정성 검토용이므로 프레임을 건너뛰지 않고 청크 경계마다 전부 라벨한다.
"""
import base64, io, json, os, sys, threading, queue, urllib.request
import numpy as np, pandas as pd, av
from PIL import Image
H=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
EP=int(sys.argv[1]); NW=int(sys.argv[2]) if len(sys.argv)>2 else 6
OUT=f"{H}/labels_ep{EP:04d}.jsonl"
instr="Bring the package over, orient barcode up, then place it on the conveyor."
from allex_common_v5 import descriptors as _desc, facts_text, GUIDANCE, ASK, MERGE_LIMIT
OUT=f"{H}/labels_ep{EP:04d}_v5.jsonl"

d=pd.read_parquet(f"{H}/data/chunk-000/episode_{EP:06d}.parquet")
A=np.stack(d["action"].values)
WR=np.stack(d["action.right_wrist_wrt_base"].values); WL=np.stack(d["action.left_wrist_wrt_base"].values)
N=len(A)
def descriptors(f, n=16): return _desc(A, WR, WL, f, n)
def facts(x): return facts_text(x)

def grab(frames, side):
    want=set(frames); got={}
    with av.open(f"{H}/videos/chunk-000/observation.images.camera_ego_{side}/episode_{EP:06d}.mp4") as c:
        for i,fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i]=Image.fromarray(fr.to_ndarray(format="rgb24")).resize((896,504))
                if len(got)==len(want): break
    return got

def ask(views, text):
    content=[]
    for v in views:
        b=io.BytesIO(); v.save(b,format="JPEG",quality=88)
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}})
    content.append({"type":"text","text":text})
    body={"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
          "logprobs":True,"top_logprobs":5,"messages":[{"role":"user","content":content}]}
    r=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    dd=json.load(urllib.request.urlopen(r,timeout=180))
    YES={"Y","YES"}; NO={"N","NO"}; c=[]
    for t in dd["choices"][0]["logprobs"]["content"]:
        tk=t["token"].strip().upper()
        if tk not in YES|NO: continue
        py=pn=0.0
        for alt in t["top_logprobs"]:
            a=alt["token"].strip().upper()
            if a in YES: py+=np.exp(alt["logprob"])
            elif a in NO: pn+=np.exp(alt["logprob"])
        c.append(float(py/(py+pn)) if (py+pn)>0 else (1.0 if tk in YES else 0.0))
        if len(c)==4: break
    return (c, dd["choices"][0]["message"]["content"].strip()) if len(c)==4 else (None,None)

starts=list(range(0, N-16, 16))
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["f"])
        except Exception: pass
starts=[f for f in starts if f not in done]
print(f"ep{EP}: {N}프레임, 청크 {len(starts)}개 라벨링")
L=grab(starts,"left"); R=grab(starts,"right")
q=queue.Queue(); [q.put(f) for f in starts]
out=open(OUT,"a"); lock=threading.Lock(); cnt={"ok":0,"err":0}
def work():
    while True:
        try: f=q.get_nowait()
        except queue.Empty: return
        try:
            x=descriptors(f)
            c,ans=ask([L[f],R[f]], f"{instr}\n{GUIDANCE}\n\n{facts(x)}\n\n{ASK}")
            if c is None: raise ValueError("파싱 실패")
            # VLM 위험: B 재배향, C 자세 확정 / A 변형체는 가중 / D 미접촉은 안전
            v_risk=1-(1-c[1])*(1-c[2])
            soft=c[0]; safe=0.5+0.5*c[3]
            # 계산 위험: 실행 불가 + 파지 중 회전(측정상 고회전 청크의 49%가 실행 불가) + 파지 전이
            infeas=float(x["merge_demand"]>MERGE_LIMIT)
            rot_hold=float(x["held"] and x["wrist_rot"]>10.0)
            # 누적 회전은 직전 청크들이 필요 — 2패스로 처리하므로 여기선 원값만 남긴다
            grip_tr=min(1.0, x["hand_change"]/0.02)
            c_risk=1-(1-infeas)*(1-rot_hold)*(1-grip_tr)
            total=1-(1-v_risk)*(1-soft*0.5)*(1-c_risk)
            conf=(1-total)*safe
            rec={"ep":EP,"f":f,"conf":float(conf),"ans":ans,
                 **{k:float(v) for k,v in zip("ABCD",c)}, **{k:(float(v) if not isinstance(v,bool) else int(v)) for k,v in x.items()}}
            with lock:
                out.write(json.dumps(rec)+"\n"); out.flush(); cnt["ok"]+=1
                if cnt["ok"]%40==0: print(f"  {cnt['ok']}/{len(starts)}", flush=True)
        except Exception as e:
            with lock:
                cnt["err"]+=1
                if cnt["err"]<=3: print("오류:",type(e).__name__,str(e)[:100], flush=True)
ts=[threading.Thread(target=work) for _ in range(NW)]
[t.start() for t in ts]; [t.join() for t in ts]
out.close(); print(f"ep{EP} 완료: 성공 {cnt['ok']} 실패 {cnt['err']} -> {OUT}")
