"""Quick sanity: can we load MoE model + dataset + run 1 forward pass via policy.get_action?"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe
from gr00t.data.embodiment_tags import EmbodimentTag

GT_ROOT = Path("/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300")
CKPT_REPO = "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"

print("[1/3] load model")
cfg = DATA_CONFIG_MAP["single_panda_gripper"]
policy = Gr00tPolicyFairMoe(
    model_path=CKPT_REPO,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(backbone_model_type="eagle"),
    denoising_steps=4,
)
policy.model.action_head.discrete_action_dims = [4, 11]
print(f"  model loaded OK; head moe_num_experts={policy.model.action_head.moe_num_experts}")

print("[2/3] inspect modality config keys (what obs should contain)")
mc = cfg.modality_config()
for sec, m in mc.items():
    print(f"  {sec}: {m.modality_keys}")

print("[3/3] load one episode parquet + 1 frame")
import pyarrow.parquet as pq
import imageio.v2 as imageio
ep_idx = 0
data_path = GT_ROOT / f"data/chunk-{ep_idx // 1000:03d}/episode_{ep_idx:06d}.parquet"
df = pq.read_table(str(data_path)).to_pandas()
print(f"  ep {ep_idx}: len={len(df)} columns={list(df.columns)[:8]}...")
print(f"  state shape: {np.asarray(df.iloc[0]['observation.state']).shape}")
print(f"  action shape: {np.asarray(df.iloc[0]['action']).shape}")

vk = "observation.images.wrist_view"
video_path = GT_ROOT / f"videos/chunk-{ep_idx // 1000:03d}/{vk}/episode_{ep_idx:06d}.mp4"
reader = imageio.get_reader(str(video_path))
frame = reader.get_data(50)
reader.close()
print(f"  wrist frame at t=50: shape={frame.shape}")

print("OK smoke complete")
