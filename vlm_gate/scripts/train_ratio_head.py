"""배속 소형 모듈. 청크마다 배율 4단계를 낸다.

게이트가 이진(압축/비압축)이던 자리를 배율 사다리로 바꾼 것이다. 출력이
{1.0, 2.0, 2.67, 4.0} 위의 분포이고, 배포 시점에는 기대값이나 argmax 를 쓴다.

**순서가 있는 목표다.** 4를 1로 틀리는 것과 2로 틀리는 것은 손해가 다르므로
일반 교차엔트로피가 아니라 누적 이진(ordinal) 로 푼다 -- K-1 개의 "이 배율보다
큰가" 이진 문제로 쪼갠다. 단조성이 구조로 보장된다.

    python train_ratio_head.py <라벨.parquet> [출력.pt]
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, torch.nn as nn

RATIOS=[1.0,2.0,2.67,4.0]
PHASES=["접근","운반","파지","해제","놓기","전이(둘다)"]

class RatioHead(nn.Module):
    """작은 MLP. 입력은 청크에서 계산되는 것 + (있으면) VLM conf."""
    def __init__(self, d_in, hidden=64, n_cut=len(RATIOS)-1):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(d_in,hidden), nn.SiLU(),
                             nn.Linear(hidden,hidden), nn.SiLU())
        self.score=nn.Linear(hidden,1)
        # 누적 절단점: 단조 증가하도록 소프트플러스 누적
        self.cuts=nn.Parameter(torch.zeros(n_cut))
    def forward(self,x):
        s=self.score(self.f(x))                      # (B,1) 클수록 빠르게 가도 됨
        c=torch.cumsum(torch.nn.functional.softplus(self.cuts),0)
        return s-c                                   # (B,K-1) 로짓: "이 절단점 초과"

def featurize(d, use_conf=True):
    X=[np.eye(len(PHASES))[[PHASES.index(p) if p in PHASES else 0 for p in d.phase]]]
    dm=d[[f"demand_{r}" for r in RATIOS]].to_numpy(np.float32)
    X.append(np.log1p(dm))
    if use_conf and "conf_vlm" in d: X.append(d[["conf_vlm"]].to_numpy(np.float32))
    return np.concatenate(X,1).astype(np.float32)

def main():
    src=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "output/ratio_head.pt"
    d=pd.read_parquet(src)
    use_conf=os.environ.get("USE_CONF","1")=="1"
    X=featurize(d,use_conf); y=np.searchsorted(RATIOS,d.ratio.to_numpy())
    tasks=d.task.to_numpy(); uq=sorted(set(tasks))
    # 태스크로 가른다: 안 본 태스크에서도 되는지 봐야 한다
    rng=np.random.default_rng(11); ho=set(rng.choice(uq,max(2,len(uq)//5),replace=False))
    tr=~np.isin(tasks,list(ho))
    print(f"학습 {tr.sum():,} · 홀드아웃 {(~tr).sum():,} (태스크 {len(ho)}개 제외)")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=RatioHead(X.shape[1]).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    Xt=torch.tensor(X[tr],device=dev); yt=torch.tensor(y[tr],device=dev)
    Xv=torch.tensor(X[~tr],device=dev); yv=torch.tensor(y[~tr],device=dev)
    K=len(RATIOS)-1
    tgt=(yt[:,None]>torch.arange(K,device=dev)[None,:]).float()
    tgv=(yv[:,None]>torch.arange(K,device=dev)[None,:]).float()
    lossf=nn.BCEWithLogitsLoss()
    best=9e9
    for ep in range(300):
        m.train(); opt.zero_grad()
        l=lossf(m(Xt),tgt); l.backward(); opt.step()
        if ep%25==0 or ep==299:
            m.eval()
            with torch.no_grad():
                lv=lossf(m(Xv),tgv).item()
                pv=(torch.sigmoid(m(Xv))>0.5).sum(1).cpu().numpy()
                pt=(torch.sigmoid(m(Xt))>0.5).sum(1).cpu().numpy()
            acc=(pv==y[~tr]).mean(); mae=np.abs(np.array(RATIOS)[pv]-np.array(RATIOS)[y[~tr]]).mean()
            print(f"  ep{ep:3d} train {l.item():.4f} · hold {lv:.4f} · 정확도 {acc:.3f} · MAE {mae:.3f}")
            best=min(best,lv)
    torch.save({"state":m.state_dict(),"d_in":X.shape[1],"ratios":RATIOS,
                "phases":PHASES,"use_conf":use_conf}, out)
    npar=sum(p.numel() for p in m.parameters())
    print(f"\n파라미터 {npar:,} ({npar*4/1024:.1f}KB) -> {out}")

if __name__=="__main__": main()
