"""n1.7 공동 파인튜닝 학습 스텝 스모크.

실제 배치가 프로세서·콜레이터를 거쳐 모델을 통과하며
  action_loss (기존) 와 gate_loss (우리가 추가) 가 함께 나오고 역전파되는지 확인한다.
라벨 조인이 실제로 배치에 실려 손실에 반영되는지도 여기서 드러난다.
"""
import os, sys, torch, numpy as np

WS = os.path.expanduser("~/quantization_agent_workspace")
DS = os.environ.get("DS", f"{WS}/assets/datasets/robocasa_n17_mirror")
LABELS = os.environ.get("LABELS", f"{WS}/assets/labels/robocasa/v6b_phase5_1call_full.parquet")
TAG = "robocasa_panda_gripper"
EMB = "new_embodiment"
GATE_LAYER = int(os.environ.get("GATE_LAYER", "14"))
TUNE_TOP = int(os.environ.get("TUNE_TOP", "4"))
BS = int(os.environ.get("BS", "2"))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
from gr00t.data.dataset.quant_gate_labels import GateLabelLookup, patch_dataset_gate_labels
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

print("① 데이터셋")
mc = MODALITY_CONFIGS[TAG]
ds = ShardedSingleStepDataset(
    dataset_path=DS, embodiment_tag=EmbodimentTag(EMB), modality_configs=mc,
    video_backend="decord", shard_size=2**10, episode_sampling_rate=0.1,
    seed=0, allow_padding=True)
print(f"   샤드 {len(ds)}개")

print("② 프로세서")
proc = Gr00tN1d7Processor(
    modality_configs={EMB: mc},
    statistics={EMB: ds.get_dataset_statistics()},
    embodiment_id_mapping={EMB: 0},
    # n1.7 모델 설정 기본값 (gr00t/configs/model/gr00t_n1d7.py)
    image_crop_size=(230, 230),
    image_target_size=(256, 256),
)
ds.processor = proc

print("③ 게이트 라벨 조인")
patch_dataset_gate_labels(ds, GateLabelLookup(LABELS))

print("④ 배치 구성")
shard = ds.get_shard(0)
coll = Gr00tN1d7DataCollator(model_name="nvidia/Cosmos-Reason2-2B")
batch = coll(shard[:BS])["inputs"]
lab = batch.get("gate_label"); val = batch.get("gate_valid")
print(f"   배치 키 {len(batch)}개 · gate_label {tuple(lab.shape)} 값 {lab.tolist()}")
print(f"   gate_valid {val.tolist()}  (1=교사 라벨 있음)")

print("⑤ 모델")
m = Gr00tN1d7.from_pretrained("nvidia/GR00T-N1.7-3B", torch_dtype=torch.bfloat16).cuda()
m.backbone.set_trainable_parameters(tune_llm=False, tune_visual=False, tune_top_llm_layers=TUNE_TOP)
m.backbone.to(torch.bfloat16)
m.attach_quant_gate(gate_layer=GATE_LAYER, loss_weight=1.0)

print("⑥ forward + backward")
out = m(batch)
al, gl, tot = out.get("action_loss_only"), out.get("gate_loss"), out["loss"]
print(f"   action_loss {float(al):.4f} · gate_loss {float(gl):.4f} · total {float(tot):.4f}")
assert abs(float(tot) - (float(al) + float(gl))) < 1e-3, "결합 손실 불일치"
tot.backward()

gh = sum(1 for p in m.quant_gate.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
bb = {}
for n, p in m.backbone.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0 or ".layers." not in n: continue
    bb.setdefault(int(n.split(".layers.")[1].split(".")[0]), 0)
    bb[int(n.split(".layers.")[1].split(".")[0])] += 1
ah = sum(1 for p in m.action_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"   그래디언트 — 게이트헤드 {gh} · 액션헤드 {ah} · 백본 레이어별 {dict(sorted(bb.items()))}")
print("   통과 조건: 셋 다 >0, 백본은 학습 지정 범위 안에서만")
