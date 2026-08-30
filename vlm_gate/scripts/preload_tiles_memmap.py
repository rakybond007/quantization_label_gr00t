"""프로브 부분집합 타일을 uint8 memmap 으로 한 번에 굽는다.

학습 루프 안에서 PNG 를 디코딩하면 GPU 가 놀고 디스크만 때린다. 실제 학생 학습이
프레임 캐시 memmap 을 쓰는 것과 같은 방식으로 맞춘다 (인코더 종류와 무관한 문제).
"""
import os, sys, numpy as np
from multiprocessing import Pool
from PIL import Image
WS=os.path.expanduser("~/quantization_agent_workspace")
TIL=f"{WS}/vlm_gate/output/_gate_distill/luna_robocasa_full/tiles"
IDX=f"{WS}/assets/probe_features/index.npz"
OUT=f"{WS}/assets/probe_features/tiles_u8.npy"

def one(nm):
    im=np.asarray(Image.open(f"{TIL}/{nm}").convert("RGB"))
    h,w,_=im.shape
    return np.concatenate([im[:, k*w//3:(k+1)*w//3] for k in range(3)],axis=2).transpose(2,0,1)

if __name__=="__main__":
    names=[str(x) for x in np.load(IDX)["names"]]
    m=np.lib.format.open_memmap(OUT, mode="w+", dtype=np.uint8, shape=(len(names),9,128,128))
    with Pool(16) as p:
        for i,arr in enumerate(p.imap(one, names, chunksize=64)):
            m[i]=arr
            if i%5000==0: print(f"  {i}/{len(names)}", flush=True)
    m.flush(); print(f"[preload] {OUT}  {m.shape}  {os.path.getsize(OUT)/1e9:.1f}GB")
