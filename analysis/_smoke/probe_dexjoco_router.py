"""Probe DexJoCo MoE no_balance router pick distribution + full prob values.

Probe both:
  - moe4_v1 no_balance (abs, K=4: raw16, m8, m4, n8) — eval'd checkpoint (60k)

Reports argmax pick AND raw softmax probs per query so we know whether the
collapse is sharp (probs e.g. [0,0,1,0]) or soft (e.g. [0.1, 0.25, 0.5, 0.15]).
"""
import sys, numpy as np, torch
sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
from collections import Counter

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe

dc = DATA_CONFIG_MAP["dexjoco_single_arm_multi_horizon"]
CKPT = "ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance/checkpoint-60000"

print(f"{'variant':<20} | {'picks (k=)':<48} | fraction")
print("-" * 100)
policy = Gr00tPolicyFairMoe(
    model_path=CKPT,
    modality_config=dc.modality_config(),
    modality_transform=dc.transform("eagle"),
    embodiment_tag="new_embodiment", denoising_steps=4, backbone_model_type="eagle",
)
policy.inference_head = "moe"

picks, all_probs = [], []
N = 30
for q in range(N):
    torch.manual_seed(q); np.random.seed(q)
    quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32) + np.random.randn(1, 4).astype(np.float32) * 0.05
    obs = {
        "video.front": (np.random.rand(1, 640, 640, 3) * 255).astype(np.uint8),
        "video.wrist": (np.random.rand(1, 640, 640, 3) * 255).astype(np.uint8),
        "state.arm_pos": np.random.randn(1, 3).astype(np.float32) * 0.1,
        "state.arm_rot": quat,
        "state.hand": np.random.randn(1, 16).astype(np.float32) * 0.05,
        "annotation.human.action.task_description": ["hammer the nail"],
    }
    try:
        out = policy.get_action(obs)
        p = int(np.asarray(out["_moe_picked"]).flatten()[0])
        probs = np.asarray(out["_moe_probs"]).flatten()
        picks.append(p); all_probs.append(probs)
    except Exception as e:
        print(f"  query {q}: err {type(e).__name__}: {e}")
        continue

if picks:
    c = Counter(picks)
    ks = sorted(c)
    cs = " ".join(f"k={k}:{c[k]}" for k in ks)
    fs = " ".join(f"k={k}:{c[k]/len(picks):.2f}" for k in ks)
    print(f"{'abs no_balance':<20} | {cs:<48} | {fs}")
    probs_arr = np.stack(all_probs)
    print(f"  per-expert prob: mean={probs_arr.mean(axis=0).round(3).tolist()}  "
          f"std={probs_arr.std(axis=0).round(3).tolist()}  K={probs_arr.shape[1]}")
    print(f"  first 5 queries:")
    for i in range(min(5, len(all_probs))):
        print(f"    q={i}: probs={all_probs[i].round(3).tolist()}  pick=k={picks[i]}")

print()
print("Legend: k=0 raw16  k=1 m8  k=2 m4  k=3 n8")
print("DEXJOCO_PROBE_DONE")
