"""allex 데모 데이터셋: 2뷰(ego left/right) 가로 결합 타일 생성 (stride 지정)."""
import json, os, sys, numpy as np
from PIL import Image
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")
OUT=f"{T}/tiles"; os.makedirs(OUT,exist_ok=True)
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 60
VIEWS=["observation.images.camera_ego_left","observation.images.camera_ego_right"]
def frames(mp4, want):
    import av; out={}; want=set(want)
    with av.open(mp4) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i in want:
                out[i]=f.to_ndarray(format="rgb24")
                if len(out)==len(want): break
    return out
eps={}
for l in open(f"{T}/meta/episodes.jsonl"):
    d=json.loads(l); eps[d["episode_index"]]=d["length"]
made=0
for ep in sorted(eps):
    p=f"{T}/data/chunk-000/episode_{ep:06d}.parquet"
    if not os.path.exists(p): continue
    n=eps[ep]; want=list(range(0,n-16,STRIDE))
    got=[frames(f"{T}/videos/chunk-000/{v}/episode_{ep:06d}.mp4", want) for v in VIEWS]
    for f in want:
        ims=[g.get(f) for g in got]
        if any(x is None for x in ims): continue
        Image.fromarray(np.concatenate(ims,axis=1)).save(f"{OUT}/ep{ep:04d}_f{f:05d}.png")
        made+=1
    print(f"ep{ep}: {len(want)}프레임", flush=True)
print(f"타일 {made}장 -> {OUT}")
