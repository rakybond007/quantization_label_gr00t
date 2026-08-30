"""비트 가중치 학습 — 심플렉스 + 하한/상한 버퍼 제약 (evolve/RL 대상 파라미터)"""
import json, numpy as np, pandas as pd, os, re, sys, urllib.request
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
FLOOR, CAP = 0.12, 0.45     # 버퍼: 어떤 비트도 12% 미만/45% 초과 비중 금지
info=json.load(open(f"{DS}/meta/info.json")); cache={}
def A_(ep):
    if ep not in cache:
        ch=ep//info["chunks_size"]
        try: cache[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: cache[ep]=None
    return cache[ep]
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip(); H={"Authorization":f"Bearer {KEY}"}
tag=sys.argv[1] if len(sys.argv)>1 else "cp_u_bits4"
rows=[]
for bid in dict.fromkeys(re.findall(r"batch_[a-z0-9]+", open(f"{BASE}/output/_gate_distill/exp_{tag}/run.log").read())):
    s=json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/batches/{bid}",headers=H),timeout=300))
    if s["status"]!="completed" or not s.get("output_file_id"): continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{s['output_file_id']}/content",headers=H),timeout=900).read().decode()
    for line in raw.splitlines():
        try:
            r=json.loads(line); c=r["response"]["body"]["choices"][0]["message"]["content"]
            ys=[ch for ch in (c or "").strip().upper() if ch in "YN"][:4]
            if len(ys)!=4: continue
            nm=r["custom_id"]; ep=int(nm[2:6]); f=int(nm[8:11]); a=A_(ep)
            if a is None or f>=len(a): continue
            g=a[:,-1]; gd=np.abs(np.diff(g,prepend=g[0])); grip=gd[f:f+16].max()>0.5
            d=a[:,5:8]; n=np.linalg.norm(d,axis=1)
            ca=np.sum(d[:-1]*d[1:],axis=1)/((n[:-1]+1e-9)*(n[1:]+1e-9)); big=(n[:-1]>0.10)&(n[1:]>0.10)
            rev=((ca<0)&big)[f:min(f+16,len(ca))].any()
            fine=((g>0.5)&(n<0.12))[f:f+16].mean()>0.5
            rows.append(dict(ep=ep,bits=[int(x=="Y") for x in ys],risk=int(grip or rev or fine)))
        except Exception: pass
df=pd.DataFrame(rows); X=np.array(df.bits.tolist(),float); y=df.risk.values.astype(float)
def project(w):
    w=np.maximum(w, 1e-6)
    for _ in range(200):                      # 클립↔정규화 반복 투영 (상·하한 동시 만족)
        w=np.clip(w, FLOOR, CAP)
        s_=w.sum()
        if abs(s_-1)<1e-9: break
        w=w/s_
    return np.clip(w, FLOOR, CAP)
w=project(np.ones(4)/4)
for it in range(4000):                       # 제약 하에서 로지스틱 손실 최소화
    z=(X@w)*6-2; p=1/(1+np.exp(-z))
    grad=(X*((p-y)*6)[:,None]).mean(0)
    w=project(w-0.3*grad)
risk=X@w; conf=1-risk
print(f"n={len(df)} 위험 기저율={y.mean():.3f}")
print("제약 하 가중치(합=1, 하한 {:.2f} 상한 {:.2f}):".format(FLOOR,CAP),
      {k: round(v,3) for k,v in zip("ABCD", w)})
print(f"confidence 분포: 유니크={len(np.unique(np.round(conf,3)))} std={conf.std():.3f}")
print(f"\n{'τ':>6s}{'qrate':>8s}{'위험검출':>9s}{'차단정확도':>10s}")
for t in [0.95,0.85,0.75,0.6,0.5]:
    bl=conf<t
    print(f"{t:6.2f}{1-bl.mean():8.2f}{(y.astype(bool)&bl).sum()/max(y.sum(),1):9.1%}{(y.astype(bool)&bl).sum()/max(bl.sum(),1):10.1%}")
json.dump({"weights":dict(zip("ABCD", map(float,w))),"floor":FLOOR,"cap":CAP,
           "note":"evolve/RL 조정 대상. τ와 함께 폐루프 성공률·스텝을 목표로 탐색할 것"},
          open(f"{BASE}/analysis/bit_weights.json","w"), ensure_ascii=False, indent=1)
print("\n저장: analysis/bit_weights.json")
