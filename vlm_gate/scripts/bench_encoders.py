"""게이트 인코더 후보의 순수 GPU 순전파 시간. 전송 오버헤드 없이 연산만 격리한다."""
import time, torch, torch.nn as nn, os
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
dev="cuda"; torch.backends.cudnn.benchmark=True

def bench(fn, n=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

class SmallGate(nn.Module):
    def __init__(self, ch=32, temb=384):
        super().__init__()
        blk=lambda i,o: nn.Sequential(nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU())
        self.net=nn.Sequential(blk(9,ch),blk(ch,ch*2),blk(ch*2,ch*4),blk(ch*4,ch*4),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(ch*4+temb,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x,t): return self.head(torch.cat([self.net(x).flatten(1),t],1))

res={}
m=SmallGate().to(dev).eval().half()
x=torch.randn(1,9,128,128,device=dev,dtype=torch.half); t=torch.randn(1,384,device=dev,dtype=torch.half)
with torch.no_grad(): res["SmallGate(현재) 128x128x9"]=(bench(lambda: m(x,t)), sum(p.numel() for p in m.parameters()))

# 같은 구조를 4배 폭으로 (용량만 키운 대조군)
m2=SmallGate(ch=128).to(dev).eval().half()
with torch.no_grad(): res["SmallGate x4폭 128x128x9"]=(bench(lambda: m2(x,t)), sum(p.numel() for p in m2.parameters()))

from transformers import AutoModel
for name,path in [("DINOv3 ViT-B/16 224x224 x3뷰","facebook/dinov3-vitb16-pretrain-lvd1689m"),
                  ("DINOv2-reg ViT-B/14 224x224 x3뷰","facebook/dinov2-with-registers-base")]:
    try:
        enc=AutoModel.from_pretrained(path, torch_dtype=torch.half).to(dev).eval()
        v=torch.randn(3,3,224,224,device=dev,dtype=torch.half)   # 3뷰 배치
        with torch.no_grad(): res[name]=(bench(lambda: enc(pixel_values=v)), sum(p.numel() for p in enc.parameters()))
        del enc; torch.cuda.empty_cache()
    except Exception as e: res[name]=(float('nan'), 0); print(f"  [{name}] 로드 실패: {type(e).__name__}: {str(e)[:120]}")

print(f"\n{'후보':38s} {'파라미터':>10s} {'순전파':>10s}")
for k,(ms,p) in res.items(): print(f"{k:38s} {p/1e6:9.2f}M {ms:9.2f}ms")
print(f"\n참고: 실측 gate 왕복 p50 ≈ 60ms (HTTP+base64+PIL 포함)")
