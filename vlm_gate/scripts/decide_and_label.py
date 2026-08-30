"""사전점검 결과로 전량 라벨링 설정을 자동 결정하고 기록."""
import json, os, glob
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
def load(p):
    d={}
    for l in open(p):
        try: r=json.loads(l); d[(r['ep'],r['f'])]=r['p_yes']
        except: pass
    return d
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s); o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1*n0>0 else float('nan')
cache={}
def sig(ep):
    if ep in cache: return cache[ep]
    ch=ep//info["chunks_size"]
    try: a=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    except Exception: cache[ep]=None; return None
    grip=a[:,-1]; gd=np.abs(np.diff(grip, prepend=grip[0]))
    near=np.zeros(len(a),bool)
    for t in np.where(gd>0.5)[0]: near[max(0,t-8):t+9]=True
    cache[ep]={"grip":near,"n":len(a)}
    return cache[ep]
def metrics(d):
    y=[];s=[]
    for (ep,f),p in d.items():
        g=sig(ep)
        if g is None or f>=g["n"]: continue
        y.append(int(g["grip"][f])); s.append(1-p)
    byep={}
    for (ep,f),p in d.items(): byep.setdefault(ep,{})[f]=p
    res={}
    for ep,v in byep.items():
        if len(v)<5: continue
        m=np.mean(list(v.values()))
        for f,p in v.items(): res[(ep,f)]=p-m
    a=[];b=[]
    for (ep,f),p in res.items():
        q=res.get((ep,f+8))
        if q is not None: a.append(p); b.append(q)
    v=np.array(list(d.values()))
    return dict(n=len(d), qrate=float(np.mean(v>=0.5)), gripAUC=auc(y,s),
                consist=float(np.corrcoef(a,b)[0,1]) if len(a)>50 else float('nan'))
tiles=load(f"{BASE}/output/_gate_distill/exp_f9k_act/labels.jsonl")
full=load(f"{BASE}/output/_gate_distill/exp_pf_fullres/labels.jsonl")
common=set(tiles)&set(full)
mt=metrics({k:tiles[k] for k in common}); mf=metrics({k:full[k] for k in common})
print("동일 프레임", len(common))
print("타일(384x128):   ", mt)
print("원본3뷰(256x256):", mf)
gain = (mf["gripAUC"]-mt["gripAUC"])
decision = "fullres_stride16" if gain >= 0.05 else "tiles_stride8"
out={"common":len(common),"tiles":mt,"fullres":mf,"gripAUC_gain":gain,"decision":decision}
json.dump(out, open(f"{BASE}/output/_gate_distill/PREFLIGHT_DECISION.json","w"), indent=1, default=float)
print("\n결정:", decision, f"(그리퍼AUC 이득 {gain:+.3f}; 임계 +0.05)")
