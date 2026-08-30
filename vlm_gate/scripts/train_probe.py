"""인코더·풀링 비교 프로브. 헤드를 동일하게 두고 앞단만 바꾼다.

전역평균(mean) vs attention 풀링을 같은 특징 위에서 비교하므로, B 변형의 성능 저하가
백본 탓인지 masked-mean 풀링 탓인지 가려낸다. 홀드아웃은 에피소드 단위.
"""
import argparse, os, numpy as np, torch, torch.nn as nn
WS=os.path.expanduser("~/quantization_agent_workspace")

class Head(nn.Module):
    """풀링된 768차원 -> P(quantize). 모든 팔에서 동일."""
    def __init__(self, d=768):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

class MeanPool(nn.Module):
    def forward(self,x): return x.mean(1)            # (B,48,768)->(B,768)

class AttnPool(nn.Module):
    """학습되는 질의 하나로 48개 토큰(3뷰x4x4)에 attention. 공간 정보를 살린다."""
    def __init__(self,d=768,nh=6):
        super().__init__()
        self.q=nn.Parameter(torch.randn(1,1,d)*0.02)
        self.att=nn.MultiheadAttention(d,nh,batch_first=True)
        self.pos=nn.Parameter(torch.randn(1,48,d)*0.02)
    def forward(self,x):
        x=x+self.pos
        o,_=self.att(self.q.expand(x.shape[0],-1,-1), x, x)
        return o[:,0]

def auc(s,y):
    s=np.asarray(s); y=np.asarray(y); p=s[y>=0.5]; n=s[y<0.5]
    if len(p)==0 or len(n)==0: return float("nan")
    r=np.argsort(np.argsort(np.concatenate([p,n])))
    return (r[:len(p)].sum()-len(p)*(len(p)-1)/2)/(len(p)*len(n))

class SmallGate(nn.Module):
    """현재 쓰는 학생과 동일 구조. 같은 부분집합·같은 분할에서의 공정한 기준선."""
    def __init__(self, ch=32):
        super().__init__()
        blk=lambda i,o: nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        self.net=nn.Sequential(blk(9,ch),blk(ch,ch*2),blk(ch*2,ch*4),blk(ch*4,ch*4),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(ch*4,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.head(self.net(x).flatten(1))


_TILES=None
def _load_tiles(names, idx):
    """미리 구운 uint8 memmap 에서 읽는다 (PNG 디코딩 없음)."""
    global _TILES
    if _TILES is None:
        _TILES=np.load(f"{WS}/assets/probe_features/tiles_u8.npy", mmap_mode="r")
    return np.asarray(_TILES[idx], dtype=np.float32)/255.


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--pool", required=True, choices=["mean","attn"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    a=ap.parse_args()
    SG = (a.enc=="smallgate")
    d=f"{WS}/assets/probe_features/{'dinov3' if SG else a.enc}"
    X=None if SG else np.load(f"{d}/feat.npy", mmap_mode="r")
    M=np.load(f"{d}/meta.npz"); ep=M["ep"]; y=M["y"].astype(np.float32)
    names=[str(x) for x in M["names"]]
    ue=np.unique(ep); rng=np.random.default_rng(0); rng.shuffle(ue)
    val_ep=set(ue[:int(len(ue)*0.25)].tolist())
    vm=np.array([e in val_ep for e in ep]); tr=np.where(~vm)[0]; va=np.where(vm)[0]
    print(f"[probe] {a.enc}/{a.pool}  train {len(tr)} / val {len(va)}  (에피소드 {len(ue)-len(val_ep)}/{len(val_ep)})", flush=True)
    dev="cuda"
    if SG:
        model=SmallGate().to(dev); params=list(model.parameters())
        fwd=lambda idx: model(torch.from_numpy(_load_tiles(names,idx)).to(dev))
    else:
        Dm=X.shape[-1]
        pool=(MeanPool() if a.pool=="mean" else AttnPool(Dm)).to(dev); head=Head(Dm).to(dev)
        params=list(pool.parameters())+list(head.parameters())
        def fwd(idx):
            xb=torch.from_numpy(np.asarray(X[idx],dtype=np.float32)).to(dev).reshape(len(idx),48,Dm)
            return head(pool(xb))
    opt=torch.optim.AdamW(params, lr=3e-4)
    lossf=nn.BCEWithLogitsLoss()
    best=-1
    for e in range(a.epochs):
        (model if SG else pool).train(); (model if SG else head).train(); rng.shuffle(tr); tl=0; n=0
        for i in range(0,len(tr),a.bs):
            idx=np.sort(tr[i:i+a.bs])
            yb=torch.from_numpy(y[idx]).to(dev).unsqueeze(1)
            opt.zero_grad(); l=lossf(fwd(idx),yb); l.backward(); opt.step()
            tl+=l.item()*len(idx); n+=len(idx)
        (model if SG else pool).eval(); (model if SG else head).eval(); S=[]
        with torch.no_grad():
            for i in range(0,len(va),a.bs):
                idx=np.sort(va[i:i+a.bs])
                S+=torch.sigmoid(fwd(idx)).cpu().numpy().ravel().tolist()
        A=auc(S,y[np.sort(va)]); best=max(best,A)
        print(f"  epoch {e+1}/{a.epochs} loss={tl/n:.4f} val_AUC={A:.4f}", flush=True)
    print(f"[RESULT] {a.enc} {a.pool} best_val_AUC={best:.4f}", flush=True)

if __name__=="__main__": main()
