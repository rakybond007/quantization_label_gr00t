"""VLM 관용도 + 계산 요구량 -> 배율. 판정 둘로 채점한다.

    (a) 태스크 최대 배율을 맞히는가   -- 태스크 하나씩 빼고(LOO)
    (b) 국면 순서가 물리에 맞는가     -- 파지·놓기 < 운반·접근

(b) 가 필요한 이유: 계산만 쓰면 파지·놓기가 천천히 움직여 변위가 작다는 이유로
가장 빠르게 나온다(파지 2.11 > 접근 1.76). 정밀도 요구는 변위가 아니라 상대의
성질에서 오고, 그건 화면에만 있다.

    ratio = max { r : demand_r <= G * exp(beta * (tol - tol_mean)) }

    python ratio_combine_score.py <v5.jsonl> [...]
"""
import importlib, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
RATIOS=[1.0,1.5,2.0,2.5,2.67,4.0]; R=np.array(RATIOS)
TIGHT=("파지","놓기","해제")            # 물리적으로 빡빡해야 하는 국면
LOOSE=("접근","운반")

def tolerance(d, C):
    Q=sorted(C.SIGN)
    eg=[]
    for _,row in d.iterrows():
        gp=row.get("gp")
        if isinstance(gp,list) and len(gp)==len(Q):
            eg.append([sum((i+1)*float(p) for i,p in enumerate(r)) for r in gp])
        else: eg.append([float(row[q]) for q in Q])
    E=np.array(eg)
    g=(E-1.0)/(C.NGRADE-1)
    risk=sum(C.WEIGHT[q]*g[:,i] for i,q in enumerate(Q) if C.SIGN[q]<0)
    safe=sum(C.WEIGHT[q]*g[:,i] for i,q in enumerate(Q) if C.SIGN[q]>0)
    return (1.0+safe-risk)/2.0          # 클수록 관대

def main():
    L=json.load(open(f"{BASE}/analysis/eval_results/LADDERS.json"))["ladders"]["robocasa/uniform"]["rungs"]
    target={}
    for t in L[0]["tasks"]:
        b=L[0]["tasks"][t]["success"]
        if b<=0.05: continue
        mx=1.0
        for i,r in enumerate(RATIOS):
            if L[i]["tasks"][t]["success"]/b>=0.85: mx=r
            else: break
        target[t]=mx
    DEM=pd.read_parquet(f"{BASE}/analysis/robocasa_ladder/chunk_demand.parquet")
    for path in sys.argv[1:]:
        meta=json.load(open(path.replace(".jsonl","_meta.json")))
        C=importlib.import_module(meta["checks"])
        d=pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("tile",keep="last")
        d=d[d.task.isin(target)].reset_index(drop=True)
        # 같은 장면 집합이라 순서를 맞춰 요구량을 붙인다
        key=DEM.assign(k=DEM.task+"|"+DEM.phase).groupby("k").cumcount()
        DEM2=DEM.copy(); DEM2["kk"]=DEM.task+"|"+DEM.phase+"|"+key.astype(str)
        d["kk"]=d.task+"|"+d.phase+"|"+d.groupby(["task","phase"]).cumcount().astype(str)
        m=d.merge(DEM2[["kk"]+[f"d{i}" for i in range(6)]],on="kk",how="inner")
        if len(m)<200:
            print(f"{os.path.basename(path)}: 요구량 붙은 행 {len(m)} -- 건너뜀"); continue
        tol=tolerance(m,C); Dm=m[[f"d{i}" for i in range(6)]].to_numpy()
        tasks=sorted(target); ti=m.task.map({t:i for i,t in enumerate(tasks)}).to_numpy()
        y=np.array([target[t] for t in tasks]); t0=tol.mean()
        def khat(z,use_vlm=True):
            lim=np.exp(z[0])*(np.exp(z[1]*(tol-t0)) if use_vlm else 1.0)
            ok=Dm<=lim[:,None] if use_vlm else Dm<=np.exp(z[0])
            ok=np.atleast_2d(ok); ok[:,0]=True
            return R[(ok*np.arange(6)).max(1)]
        def fit(mask,use_vlm):
            def Lf(z):
                k=khat(z,use_vlm); pr=np.array([k[ti==i].mean() for i in range(len(tasks))])
                return float(np.mean((pr[mask]-y[mask])**2))
            return minimize(Lf,[np.log(np.median(Dm)),1.0],method="Nelder-Mead",
                            options={"maxiter":4000,"fatol":1e-7}).x
        print(f"\n{'='*66}\n{os.path.basename(path)}  ({len(m)}청크)")
        for nm,use in (("계산만",False),("계산+VLM 관용도",True)):
            pred=np.zeros(len(tasks))
            for i in range(len(tasks)):
                msk=np.ones(len(tasks),bool); msk[i]=False
                z=fit(msk,use); k=khat(z,use); pred[i]=k[ti==i].mean()
            z=fit(np.ones(len(tasks),bool),use); k=khat(z,use)
            rho=spearmanr(pred,y)[0]
            tight=np.mean([k[m.phase.to_numpy()==p].mean() for p in TIGHT if (m.phase==p).any()])
            loose=np.mean([k[m.phase.to_numpy()==p].mean() for p in LOOSE if (m.phase==p).any()])
            okb = tight < loose
            print(f"  {nm:16s} (a) LOO 상관 {rho:+.3f} · RMSE {np.sqrt(np.mean((pred-y)**2)):.3f}"
                  f"   (b) 빡빡 {tight:.2f} 대 느슨 {loose:.2f}  {'통과' if okb else '**실패**'}")
            if use:
                print("     국면별: " + " · ".join(
                    f"{p} {k[m.phase.to_numpy()==p].mean():.2f}" for p in sorted(set(m.phase))))
        print(f"  (상수 예측기 RMSE {np.sqrt(np.mean((y.mean()-y)**2)):.3f})")

if __name__=="__main__": main()
