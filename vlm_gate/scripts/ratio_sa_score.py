"""배속 문항 채점. 문항이 낸 배속이 실측 최대 안전 배율과 맞는가.

    python ratio_sa_score.py robocasa|libero
"""
import csv, importlib, json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)

def expected(gp, n):
    return [sum((i+1)*float(p) for i,p in enumerate(r)) for r in gp] if gp else None

def robocasa():
    C=importlib.import_module("ratio_checks_sa1")
    R=[1.0,1.5,2.0,2.5,2.67,4.0]
    L=json.load(open(f"{BASE}/analysis/eval_results/LADDERS.json"))["ladders"]["robocasa/uniform"]["rungs"]
    tgt={}
    for t in L[0]["tasks"]:
        b=L[0]["tasks"][t]["success"]
        if b<=0.05: continue
        mx=1.0
        for i,r in enumerate(R):
            if L[i]["tasks"][t]["success"]/b>=0.85: mx=r
            else: break
        tgt[t]=mx
    d=pd.DataFrame([json.loads(l) for l in open(f"{BASE}/analysis/ratio_prompt/ratio_sa1.jsonl")])
    d=d.drop_duplicates("tile").reset_index(drop=True)
    Q=sorted(C.SIGN)
    eg=[expected(g,C.NGRADE) if isinstance(g,list) else None for g in d.get("gp",[None]*len(d))]
    for i,q in enumerate(Q): d["e"+q]=[(r[i] if r else float(d[q].iloc[j])) for j,r in enumerate(eg)]
    d["sub"]=[C.subaction({q:r["e"+q] for q in Q}) for _,r in d.iterrows()]
    d["ratio"]=[C.LIMITS[s] for s in d["sub"]]
    return d, tgt, "task"

def libero():
    C=importlib.import_module("libero_ratio_sa1")
    inst={}; new={}
    with open(f"{BASE}/analysis/libero_retention/naive_K23_clipoff.csv") as f:
        for r in csv.DictReader(f):
            k=f"{r['suite']}_{r['task_id']}"; inst[k]=r["task"]; new[k]=int(r["successes"])/50
    L=json.load(open(f"{BASE}/analysis/eval_results/LADDERS.json"))["ladders"]["libero/naive"]["rungs"]
    T=[{k.replace("/","_"):v for k,v in r["tasks"].items()} for r in L]
    base=T[0]; R=[1.0,1.667,2.0,2.4]
    tgt={}
    for t in base:
        if base[t]["success"]<=0.05 or t not in new: continue
        mx=1.0
        for i,r in enumerate(R):
            s=new[t] if i==3 else T[i][t]["success"]
            if s/base[t]["success"]>=0.85: mx=r
            else: break
        tgt[t]=mx
    d=pd.DataFrame([json.loads(l) for l in open(f"{BASE}/_tmp/libero/lb_sa1.jsonl")])
    d=d.drop_duplicates("tile").reset_index(drop=True)
    Q=sorted(C.NAME)
    if "eg" in d.columns:
        for i,q in enumerate(Q): d["e"+q]=[r[i] if isinstance(r,list) and len(r)>i else None for r in d.eg]
    if "picks" in d:
        for i,q in enumerate(Q): d[q]=[p[i] if isinstance(p,list) and len(p)>i else None for p in d.picks]
    eg=[expected(g,C.NGRADE) if isinstance(g,list) else None for g in d.get("gp",[None]*len(d))]
    for i,q in enumerate(Q): d["e"+q]=[(r[i] if r else float(d[q].iloc[j])) for j,r in enumerate(eg)]
    d["sub"]=[C.subaction({q:r["e"+q] for q in Q}) for _,r in d.iterrows()]
    d["ratio"]=[C.LIMITS[s] for s in d["sub"]]
    src = "task" if "task" in d.columns else "cell"
    d["task"]=[str(t).replace("/","_") for t in d[src]]
    return d, tgt, "task"

def main():
    d,tgt,key = robocasa() if sys.argv[1]=="robocasa" else libero()
    print(f"청크 {len(d):,} · 서브액션 분포")
    for s,n in d["sub"].value_counts().items(): print(f"  {s:16s} {n:6d} ({n/len(d):5.1%})")
    g=d[d[key].isin(tgt)].groupby(key).ratio.mean()
    y=np.array([tgt[t] for t in g.index]); p=g.to_numpy()
    rho,pv=spearmanr(p,y)
    print(f"\n태스크 {len(g)} · 문항 배속 대 실측 상한")
    print(f"  순위상관 {rho:+.3f} (p={pv:.4f}) · RMSE {np.sqrt(np.mean((p-y)**2)):.3f}"
          f" · 상수 {np.sqrt(np.mean((y.mean()-y)**2)):.3f}")
    o=np.argsort(y)
    print(f"\n{'태스크':26s} {'실측':>5s} {'문항':>5s}")
    for i in o: print(f"  {g.index[i][:24]:24s} {y[i]:5.2f} {p[i]:5.2f}")

if __name__=="__main__": main()
