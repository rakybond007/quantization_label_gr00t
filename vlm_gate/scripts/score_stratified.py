"""층화 표본 채점 — evolve 가이던스에 부합하는 위험 정의 사용
   위험: PnP 파지·해제 순간, 문/서랍 '당겨 열기' 중, 정밀 삽입(닫힘+저속)
   안전: 노브·버튼 조작(Turn*/Coffee*), 문/서랍 '닫기', 자유 이동·운반"""
import json, sys, os, re, numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json")); cache={}
envof={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l)
    c=[t for t in d.get("tasks",[]) if isinstance(t,str) and re.fullmatch(r"[A-Z][A-Za-z]+",t) and t!="Valid"]
    if c: envof[d["episode_index"]]=c[0]
def act(ep):
    if ep not in cache:
        ch=ep//info["chunks_size"]
        try: cache[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: cache[ep]=None
    return cache[ep]
def risk_of(ep,f):
    a=act(ep)
    if a is None or f>=len(a): return None
    cls=envof.get(ep,"?")
    g=a[:,-1]; gd=np.abs(np.diff(g,prepend=g[0]))
    d=a[:,5:8]; n=np.linalg.norm(d,axis=1)
    trans = gd[f:f+16].max()>0.5
    fine = ((g>0.5)&(n<0.12))[f:f+16].mean()>0.5
    if cls.startswith(("Turn","Coffee")):            # 고정 기구 조작 → 가이던스상 안전
        return 0
    if cls.startswith("Close"):                       # 닫기/밀기 → 안전
        return 0
    if cls.startswith("Open"):                        # 당겨 열기 → 하중 구간 위험
        return int(trans or (g[f]>0.5 and n[f:f+16].mean()>0.05))
    return int(trans or fine)                         # PnP: 파지·해제·정밀배치
def load(p):
    d={}
    for l in open(p):
        try: r=json.loads(l); d[(r['ep'],r['f'])]=r['p_yes']
        except: pass
    return d
for tag in sys.argv[1:]:
    p=f"{BASE}/output/_gate_distill/exp_{tag}/labels.jsonl"
    if not os.path.exists(p) or os.path.getsize(p)<100: print(f"{tag:16s} (수집 중)"); continue
    d=load(p); rows=[]
    for (ep,f),v in d.items():
        r=risk_of(ep,f)
        if r is None: continue
        rows.append((v,r,envof.get(ep,"?")))
    df=pd.DataFrame(rows,columns=["p","risk","cls"])
    v=df.p.values; bl=df.p<0.5
    print(f"{tag:16s} n={len(df):5d} 태스크={df.cls.nunique()}종 conf평균={v.mean():.3f} std={v.std():.3f} "
          f"| 위험기저율={df.risk.mean():.3f} 위험검출={(df.risk.astype(bool)&bl).sum()/max(df.risk.sum(),1):5.1%} "
          f"차단정확도={(df.risk.astype(bool)&bl).sum()/max(bl.sum(),1):5.1%}")
