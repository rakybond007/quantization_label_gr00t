import json, sys, glob, os
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
def load(p):
    d={}
    if not os.path.exists(p): return d
    for l in open(p):
        try: r=json.loads(l); d[(r['ep'],r['f'])]=r['p_yes']
        except: pass
    return d
def resid_g8(d, stride):
    byep={}
    for (ep,f),p in d.items(): byep.setdefault(ep,{})[f]=p
    res={}; bm=[]; wi=[]
    for ep,v in byep.items():
        if len(v)<5: continue
        m=np.mean(list(v.values())); bm.append(m)
        for f,p in v.items(): res[(ep,f)]=p-m; wi.append(p-m)
    a=[];b=[]
    for (ep,f),p in res.items():
        q=res.get((ep,f+stride))
        if q is not None: a.append(p); b.append(q)
    bv=np.var(bm) if bm else 0; wv=np.var(wi) if wi else 1
    return (np.corrcoef(a,b)[0,1] if len(a)>100 else float('nan')), bv/(bv+wv)
# real ground-truth proxy
DSR="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
_gt=None
def gt_real():
    global _gt
    if _gt is None:
        _gt={}
        for f in sorted(glob.glob(f"{DSR}/data/chunk-000/episode_*.parquet")):
            ep=int(f.split("_")[-1].split(".")[0])
            st=np.stack(pd.read_parquet(f)["observation.state"].values)
            g=(st[:,7]>0.5).astype(int); tr=np.where(np.diff(g)!=0)[0]
            pr=np.zeros(len(g),bool)
            for t in tr: pr[max(0,t-5):t+6]=True
            _gt[ep]=pr
    return _gt
def auc(y,s):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1*n0>0 else float('nan')
_cos=None
def cosmos():
    global _cos
    if _cos is None:
        c=pd.read_parquet("/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet")
        _cos=dict(zip(zip(c['episode_index'].astype(int),c['frame_index'].astype(int)), c['p_yes']))
    return _cos
for tag in sys.argv[1:]:
    p=f"{BASE}/output/_gate_distill/exp_{tag}/labels.jsonl"
    d=load(p)
    if not d: print(f"{tag:20s} (아직 결과 없음)"); continue
    dom="real" if "real" in tag else "rc"
    stride=4 if dom=="real" else 8
    r8, betw = resid_g8(d, stride)
    v=np.array(list(d.values()))
    line=f"{tag:20s} n={len(d):5d} std={v.std():.3f} qrate={np.mean(v>=0.5):.2f} 잔차g8={r8:.3f} ep間={betw:.2f}"
    if dom=="real":
        g=gt_real(); y=[];s=[]
        for (ep,f),pv in d.items():
            pr=g.get(ep)
            if pr is None or f>=len(pr): continue
            y.append(int(pr[f])); s.append(1-pv)
        if len(y)>50: line+=f" | 그리퍼AUC={auc(np.array(y),np.array(s)):.3f}"
    else:
        ck=cosmos(); ks=[k for k in d if k in ck]
        if len(ks)>50:
            x=np.array([d[k] for k in ks]); yv=np.array([ck[k] for k in ks])
            line+=f" | cosmos일치={((x>=0.5)==(yv>=0.5)).mean():.2f}"
    print(line, flush=True)
