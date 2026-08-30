"""8비트 가중치 학습 — 비트별 상·하한 + 모달리티 그룹 하한 (영상/액션 둘 다 반영 보장)"""
import json, numpy as np, pandas as pd, os, re, sys, urllib.request
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
LO,HI = 0.05, 0.30          # 비트별 버퍼
VIS_MIN, ACT_MIN = 0.40, 0.40   # 영상(ABC)·액션(EFGH) 그룹 최소 비중
info=json.load(open(f"{DS}/meta/info.json")); cache={}
def A_(ep):
    if ep not in cache:
        ch=ep//info["chunks_size"]
        try: cache[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: cache[ep]=None
    return cache[ep]
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip(); H={"Authorization":f"Bearer {KEY}"}
tag=sys.argv[1]
rows=[]
for bid in dict.fromkeys(re.findall(r"batch_[a-z0-9]+", open(f"{BASE}/output/_gate_distill/exp_{tag}/run.log").read())):
    s=json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/batches/{bid}",headers=H),timeout=300))
    if s["status"]!="completed" or not s.get("output_file_id"): continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{s['output_file_id']}/content",headers=H),timeout=900).read().decode()
    for line in raw.splitlines():
        try:
            r=json.loads(line); c=r["response"]["body"]["choices"][0]["message"]["content"]
            ys=[ch for ch in (c or "").strip().upper() if ch in "YN"][:8]
            if len(ys)!=8: continue
            nm=r["custom_id"]; ep=int(nm[2:6]); f=int(nm[8:11]); a=A_(ep)
            if a is None or f>=len(a): continue
            g=a[:,-1]; gd=np.abs(np.diff(g,prepend=g[0])); grip=gd[f:f+16].max()>0.5
            d=a[:,5:8]; n=np.linalg.norm(d,axis=1)
            ca=np.sum(d[:-1]*d[1:],axis=1)/((n[:-1]+1e-9)*(n[1:]+1e-9)); big=(n[:-1]>0.10)&(n[1:]>0.10)
            rev=((ca<0)&big)[f:min(f+16,len(ca))].any()
            fine=((g>0.5)&(n<0.12))[f:f+16].mean()>0.5
            rows.append(dict(ep=ep,bits=[int(x=="Y") for x in ys],risk=int(grip or rev or fine)))
        except Exception: pass
df=pd.DataFrame(rows)
if len(df)<200: print("데이터 부족", len(df)); raise SystemExit
B=np.array(df.bits.tolist(),float); y=df.risk.values.astype(float)
RISK=[0,1,2,4,5,6,7]; GROSS=3
VIS=[0,1,2]; ACT=[4,5,6,7]
idx={b:i for i,b in enumerate(RISK)}
X=B[:,RISK]
def project(w):
    w=np.clip(w,LO,HI)
    for _ in range(300):
        w=np.clip(w,LO,HI); w=w/w.sum()
        vis=sum(w[idx[i]] for i in VIS); act=sum(w[idx[i]] for i in ACT)
        if vis<VIS_MIN:
            for i in VIS: w[idx[i]]*=VIS_MIN/max(vis,1e-9)
        if act<ACT_MIN:
            for i in ACT: w[idx[i]]*=ACT_MIN/max(act,1e-9)
        w=np.clip(w,LO,HI); w=w/w.sum()
        vis=sum(w[idx[i]] for i in VIS); act=sum(w[idx[i]] for i in ACT)
        if vis>=VIS_MIN-1e-6 and act>=ACT_MIN-1e-6 and abs(w.sum()-1)<1e-6: break
    return w
w=project(np.ones(len(RISK))/len(RISK))
for _ in range(4000):
    z=(X@w)*6-2; p=1/(1+np.exp(-z))
    w=project(w-0.3*(X*((p-y)*6)[:,None]).mean(0))
risk=X@w; conf=1-risk
names=["A(그리퍼시각)","B(정밀삽입)","C(문당김)","E(그리퍼수치)","F(반전)","G(닫힘저속)","H(감속)"]
print(f"n={len(df)} 위험 기저율={y.mean():.3f}")
print("가중치:", {n_: round(v,3) for n_,v in zip(names,w)})
print(f"  영상 그룹 합={sum(w[idx[i]] for i in VIS):.2f}  액션 그룹 합={sum(w[idx[i]] for i in ACT):.2f}")
print(f"\n{'τ':>6s}{'qrate':>8s}{'위험검출':>9s}{'차단정확도':>10s}")
for t in [0.95,0.9,0.85,0.8,0.7,0.6]:
    bl=conf<t
    print(f"{t:6.2f}{1-bl.mean():8.2f}{(y.astype(bool)&bl).sum()/max(y.sum(),1):9.1%}{(y.astype(bool)&bl).sum()/max(bl.sum(),1):10.1%}")
json.dump({"weights":{n_: float(v) for n_,v in zip(names,w)},"bit_lo":LO,"bit_hi":HI,
           "group_min":{"visual":VIS_MIN,"action":ACT_MIN},
           "note":"evolve/RL 조정 대상 — τ와 함께 폐루프 성공률·스텝 기준으로 탐색"},
          open(f"{BASE}/analysis/bit_weights8.json","w"), ensure_ascii=False, indent=1)
print("\n저장: analysis/bit_weights8.json")
