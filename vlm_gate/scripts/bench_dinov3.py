import os, time, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModel
import transformers; print("transformers", transformers.__version__)
dev="cuda"
def bench(fn,n=50,warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for name in ["facebook/dinov3-vitb16-pretrain-lvd1689m","facebook/dinov2-with-registers-base"]:
    try:
        m=AutoModel.from_pretrained(name, torch_dtype=torch.half).to(dev).eval()
        v=torch.randn(3,3,224,224,device=dev,dtype=torch.half)
        with torch.no_grad():
            o=m(pixel_values=v); ms=bench(lambda: m(pixel_values=v))
        hs=o.last_hidden_state
        print(f"{name}\n   파라미터 {sum(p.numel() for p in m.parameters())/1e6:.1f}M  순전파(3뷰) {ms:.2f}ms  "
              f"토큰 {tuple(hs.shape)}")
        del m; torch.cuda.empty_cache()
    except Exception as e: print(f"{name} 실패: {type(e).__name__}: {str(e)[:160]}")
