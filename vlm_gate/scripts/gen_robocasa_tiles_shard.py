import sys, json, numpy as np
from decord import VideoReader
from PIL import Image
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
vks=[k for k in info["features"] if info["features"][k].get("dtype")=="video"]
VK=[k for k in vks if "left" in k]+[k for k in vks if "right" in k and "wrist" not in k]+[k for k in vks if "wrist" in k]
out="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/output/_gate_distill/luna_robocasa_full"
shard=int(sys.argv[1]); nsh=int(sys.argv[2])
eps=[e for e in range(7200) if e%nsh==shard]
man=[]
for i,ep in enumerate(eps):
    ch=ep//info["chunks_size"]
    try: vrs=[VideoReader(f"{DS}/"+info["video_path"].format(episode_chunk=ch,episode_index=ep,video_key=k)) for k in VK]
    except Exception: continue
    n=min(len(v) for v in vrs)
    for fi in range(0,n,8):
        p=f"{out}/tiles/ep{ep:04d}_f{fi:03d}.png"
        t=np.concatenate([v[fi].asnumpy() for v in vrs],axis=1)
        Image.fromarray(t).resize((t.shape[1]//2,t.shape[0]//2)).save(p)
        man.append({"ep":ep,"f":fi,"path":p})
    if i%200==0: print(f"shard{shard}: {i}/{len(eps)} eps, {len(man)} tiles",flush=True)
json.dump(man,open(f"{out}/manifest_shard{shard}.json","w"))
print(f"shard{shard} done: {len(man)}")
