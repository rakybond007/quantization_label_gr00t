"""n1.7 이 이 클러스터(cu124 드라이버)에서 로드·순전파 되는지 확인.
동시에 게이트 모듈을 붙일 지점(중간 레이어 + image_mask)이 실제로 잡히는지 본다."""
import os, sys, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
print("torch", torch.__version__, "| cuda", torch.version.cuda)
import transformers; print("transformers", transformers.__version__)
import gr00t; print("gr00t from", os.path.dirname(gr00t.__file__))
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
CK="nvidia/GR00T-N1.7-3B"
m=Gr00tN1d7.from_pretrained(CK, torch_dtype=torch.bfloat16)
bb=m.backbone
print("로드 성공")
print(f"  백본 파라미터 {sum(p.numel() for p in bb.parameters())/1e6:.0f}M")
print(f"  select_layer = {getattr(bb,'select_layer','?')}")
try:
    L=len(bb.model.language_model.layers); print(f"  남은 LLM 레이어 수 = {L}")
except Exception as e: print("  레이어 수 확인 실패", e)
tr=[n for n,p in bb.named_parameters() if p.requires_grad]
print(f"  백본 학습가능 텐서 {len(tr)}개" + (f" 예: {tr[:2]}" if tr else " (전부 동결 — tune_top_llm_layers 기본 0)"))
# 상위 4층 unfreeze 가 동작하는지
if hasattr(bb, "set_trainable_parameters"):
    bb.set_trainable_parameters(False, False, 4)
    tr2=[n for n,p in bb.named_parameters() if p.requires_grad]
    print(f"  tune_top_llm_layers=4 적용 후 학습가능 텐서 {len(tr2)}개")
    if tr2: print(f"   예: {tr2[0]}")
