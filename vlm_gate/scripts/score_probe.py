"""프로브 채점 (가드: 양성<50이면 판정 거부, CI 필수 출력)"""
import json, sys, numpy as np, pandas as pd
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json")); cache={}
def acts(ep):
    if ep not in cache:
        ch=ep//info["chunks_size"]
        try: cache[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: cache[ep]=None
    return cache[ep]
def auc_ci(y,s,B=2000):
    y=np.asarray(y,float); s=np.asarray(s,float)
    def a(y,s):
        o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
        n1=y.sum(); n0=len(y)-n1
        return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1*n0>0 else np.nan
    base=a(y,s); rng=np.random.default_rng(0); bs=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))>1: bs.append(a(y[i],s[i]))
    return base, np.percentile(bs,2.5), np.percentile(bs,97.5)
for tag in sys.argv[1:]:
    d={}
    for l in open(f"output/_gate_distill/exp_{tag}/labels.jsonl"):
        try: r=json.loads(l); d[(r['ep'],r['f'])]=r['p_yes']
        except: pass
    v=np.array(list(d.values()))
    y=[];s=[];a=[];b=[]
    for (ep,f),x in d.items():
        A=acts(ep)
        if A is None or f>=len(A): continue
        gd=np.abs(np.diff(A[:,-1],prepend=A[0,-1]))
        y.append(int(gd[f:f+16].max()>0.5)); s.append(1-x)   # 다음 16스텝 내 그리퍼 변화
        q=d.get((ep,f+8))
        if q is not None: a.append(x); b.append(q)
    y=np.array(y); npos=int(y.sum())
    A_,lo,hi=auc_ci(y,s) if npos>=10 else (np.nan,np.nan,np.nan)
    raw=np.corrcoef(a,b)[0,1] if len(a)>100 else np.nan
    verdict = "판정가능" if npos>=50 else f"판정보류(양성 {npos}<50)"
    print(f"{tag:16s} n={len(v):5d} std={v.std():.3f} qrate={np.mean(v>=0.5):.2f} "
          f"이웃상관={raw:+.3f} 그리퍼AUC={A_:.3f} [{lo:.2f},{hi:.2f}] 양성={npos} → {verdict}")
