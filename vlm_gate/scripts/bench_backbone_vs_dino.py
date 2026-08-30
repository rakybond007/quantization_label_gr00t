"""B 아키텍처(GR00T Eagle 백본)와 DINOv3 소형판의 크기·순전파 시간 비교."""
import os, sys, json, time, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def bench(fn,n=30,warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

MODE=sys.argv[1]
if MODE=="eagle":
    from extract_gate_backbone_features import load_policy, VIEW_KEYS, VIDEO_KEYS, STATE_KEYS, LANG_KEY
    DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
    CK=os.path.expanduser("~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000")
    mod=json.load(open(f"{DS}/meta/modality.json"))["state"]
    policy=load_policy(CK)
    bb=policy.model.backbone
    npar=sum(p.numel() for p in bb.parameters())
    obs={}
    for dk in VIDEO_KEYS: obs[dk]=np.zeros((1,1,256,256,3), np.uint8)
    for sk in STATE_KEYS:
        sl=mod[sk.split("state.")[1]]; obs[sk]=np.zeros((1,1,sl["end"]-sl["start"]), np.float64)
    obs[LANG_KEY]=[["pick up the mug"]]
    with torch.no_grad():
        norm=policy.apply_transforms(obs); bi,_=policy.model.prepare_input(norm)
        ms=bench(lambda: policy.model.backbone(bi))
    print(f"B (GR00T Eagle 백본, 3뷰 1샘플): 파라미터 {npar/1e6:.1f}M  순전파 {ms:.2f}ms")
else:
    from transformers import AutoModel
    for name in ["facebook/dinov3-vits16-pretrain-lvd1689m","facebook/dinov3-vitb16-pretrain-lvd1689m"]:
        m=AutoModel.from_pretrained(name, dtype=torch.half).to("cuda").eval()
        v=torch.randn(3,3,224,224,device="cuda",dtype=torch.half)
        with torch.no_grad(): ms=bench(lambda: m(pixel_values=v))
        print(f"{name.split('/')[1]}: 파라미터 {sum(p.numel() for p in m.parameters())/1e6:.1f}M  순전파(3뷰) {ms:.2f}ms")
        del m; torch.cuda.empty_cache()
