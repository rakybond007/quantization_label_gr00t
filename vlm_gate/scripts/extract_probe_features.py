"""인코더 후보별 특징 추출 — 아키텍처 비교 프로브용.

타일(384x128 = 128x128 뷰 3장)을 인코더에 넣고 뷰마다 4x4 공간 그리드로 풀링해 저장한다.
전역 평균은 이 그리드에서 다시 만들 수 있으므로, 같은 특징으로
'전역 평균 풀링 vs 공간 보존 + attention 풀링'을 비교할 수 있다.
B 변형의 성능 저하가 백본 탓인지 풀링 탓인지 가르는 것이 목적.
"""
import os, sys, argparse, numpy as np
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from PIL import Image

WS=os.path.expanduser("~/quantization_agent_workspace")
BASE=f"{WS}/vlm_gate"
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
MAN=f"{BASE}/output/_gate_distill/tiles_manifest.txt"
ENC={"dinov3s":"facebook/dinov3-vits16-pretrain-lvd1689m",
     "dinov3":"facebook/dinov3-vitb16-pretrain-lvd1689m",
     "dinov2":"facebook/dinov2-with-registers-base"}
GRID=4

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENC))
    ap.add_argument("--index", default=f"{WS}/assets/probe_features/index.npz")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="")
    a=ap.parse_args()
    out=a.out or f"{WS}/assets/probe_features/{a.encoder}"
    os.makedirs(out, exist_ok=True)

    ix=np.load(a.index, allow_pickle=False)
    names=[str(x) for x in ix["names"]]
    print(f"[probe] 라벨과 매칭된 타일 {len(names)}장 (인코더 {a.encoder})", flush=True)

    from transformers import AutoModel
    dev="cuda"
    m=AutoModel.from_pretrained(ENC[a.encoder], dtype=torch.half).to(dev).eval()
    ps=getattr(m.config,"patch_size",16)
    mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1).half()
    std =torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1).half()

    D=m.config.hidden_size
    feats=np.lib.format.open_memmap(f"{out}/feat.npy", mode="w+", dtype=np.float16,
                                    shape=(len(names), 3, GRID*GRID, D))
    meta=[]
    buf=[]
    def flush(idx0, imgs):
        x=torch.from_numpy(np.stack(imgs)).to(dev).half().permute(0,3,1,2)/255.
        x=torch.nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        x=(x-mean)/std
        with torch.no_grad():
            h=m(pixel_values=x).last_hidden_state          # (B, 1+reg+P, D)
        P=(224//ps)**2
        tok=h[:, -P:, :]                                   # 패치 토큰만
        s=int(P**0.5)
        tok=tok.transpose(1,2).reshape(tok.shape[0], D, s, s)
        g=torch.nn.functional.adaptive_avg_pool2d(tok, GRID)      # (B,D,4,4)
        g=g.flatten(2).transpose(1,2)                             # (B,16,D)
        g=g.reshape(-1, 3, GRID*GRID, D)                          # 3뷰 묶기
        feats[idx0:idx0+g.shape[0]]=g.cpu().numpy().astype(np.float16)

    imgs=[]; idx0=0
    for i,nm in enumerate(names):
        im=np.array(Image.open(f"{TIL}/{nm}").convert("RGB"))
        h_,w_,_=im.shape
        for k in range(3): imgs.append(im[:, k*w_//3:(k+1)*w_//3])

        if len(imgs)>=a.bs*3:
            flush(idx0, imgs); idx0+=len(imgs)//3; imgs=[]
            if idx0 % 5000 < a.bs: print(f"  {idx0}/{len(names)}", flush=True)
    if imgs: flush(idx0, imgs)
    np.savez(f"{out}/meta.npz", names=ix["names"], ep=ix["ep"], fr=ix["fr"], y=ix["y"])
    print(f"[probe] 저장 {out}  feat{feats.shape}", flush=True)

if __name__=="__main__": main()
