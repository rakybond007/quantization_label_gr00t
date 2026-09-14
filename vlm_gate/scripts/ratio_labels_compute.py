"""계산만으로 청크마다 배율 라벨을 만든다. GPU 불필요.

    K̂(청크) = max { r : demand_r(청크) <= limit[국면(청크)] }

국면은 액션에서 유도하고(정지 화면으로는 그리퍼 개폐가 안 보인다), 요구량은
델타 누적으로 계산하며, 국면별 한계는 실측 사다리에 적합한 값이다.
홀드아웃(태스크 하나씩 제외) 순위상관 +0.485.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
import phase_scorecard as ps
from ratio_fit_robocasa import demand_curve, RATIOS

LIMITS={"놓기":1.1892,"해제":1.2086,"파지":1.3612,"운반":1.5026,"접근":1.8411,
        "전이(둘다)":1.2086}
R=np.array(RATIOS)
OUT=os.environ.get("OUT", f"{BASE}/output/_gate_distill/robocasa_ratio_labels.parquet")
SRC=f"{BASE}/_tmp/hf_upload/labels/robocasa_v7.parquet"

def main():
    d=pd.read_parquet(SRC, columns=["episode_index","frame_index","task","conf_vlm"])
    print(f"{len(d):,}행 · 에피 {d.episode_index.nunique():,}", flush=True)
    out=[]
    for ep,g in d.groupby("episode_index", sort=True):
        try: a=ps.get(int(ep))
        except Exception: continue
        for r in g.itertuples():
            f=int(r.frame_index)
            try: ph=ps.ph6(int(ep),f)
            except Exception: ph="접근"
            dm=demand_curve(a,f)
            lim=LIMITS.get(ph,1.5)
            ok=[x<=lim for x in dm]; ok[0]=True
            k=float(R[max(i for i,v in enumerate(ok) if v)])
            out.append((int(ep),f,r.task,ph,k,float(r.conf_vlm),*[round(x,4) for x in dm]))
        if len(out)%50000<200: print(f"  {len(out):,}", flush=True)
    C=["episode_index","frame_index","task","phase","ratio","conf_vlm"]+[f"demand_{x}" for x in RATIOS]
    D=pd.DataFrame(out,columns=C)
    D.to_parquet(OUT,index=False,compression="zstd")
    print(f"\n{len(D):,}행 -> {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB)")
    print("배율 분포: " + " · ".join(f"{r}x {(D.ratio==r).mean():.1%}" for r in RATIOS))
    print("국면별 평균 배율: " + " · ".join(
        f"{p} {D.ratio[D.phase==p].mean():.2f}" for p in sorted(D.phase.unique())))

if __name__=="__main__": main()
