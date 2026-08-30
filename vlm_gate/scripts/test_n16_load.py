"""n1.6 모델이 이 클러스터의 기존 환경(torch 2.5.1+cu124, transformers 4.51.3,
flash_attn 2.7.1)에서 로드·순전파 되는지 확인. 새 torch 를 빌드하기 전 관문."""
import os, sys, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
print("torch", torch.__version__, "| cuda", torch.version.cuda)
import transformers; print("transformers", transformers.__version__)
import gr00t; print("gr00t from", os.path.dirname(gr00t.__file__))
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from transformers import AutoConfig
CK="nvidia/GR00T-N1.6-3B"
try:
    m=Gr00tN1d6.from_pretrained(CK, torch_dtype=torch.bfloat16)
    bb=m.backbone
    print("로드 성공")
    print(f"  백본 파라미터 {sum(p.numel() for p in bb.parameters())/1e6:.0f}M")
    print(f"  select_layer = {getattr(bb,'select_layer','?')}")
    n_layers=len(bb.model.language_model.model.layers)
    print(f"  남은 LLM 레이어 수 = {n_layers}")
    tr=[n for n,p in bb.named_parameters() if p.requires_grad]
    print(f"  백본 학습가능 파라미터 텐서 {len(tr)}개")
    if tr: print("   예:", tr[:3])
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"실패: {type(e).__name__}")
