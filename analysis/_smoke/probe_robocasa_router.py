"""Probe robocasa MoE (no_balance) router pick distribution + full prob values.

Goal: compare against DexJoCo no_balance (k=0:0%, k=2:71%) to determine if router
collapse is abs-action-specific or just a general training dynamic.

If robocasa no_balance ALSO collapses to k=2 → training dynamic, balance term alone
may fix DexJoCo too.
If robocasa no_balance is spread → DexJoCo abs is uniquely collapsing → need
something beyond balance.

Prints per-query probs (after softmax with moe_inference_temp) AND argmax pick.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
from collections import Counter

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe

dc = DATA_CONFIG_MAP["single_panda_gripper_multi_horizon"]

ROBOCASA_CKPTS = [
    ("delta balance0.05", "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"),
    ("delta no_balance", "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-no-balance-60k"),
]

print(f"{'variant':<20} | {'picks (k=)':<48} | fraction")
print("-" * 100)
for label, ckpt in ROBOCASA_CKPTS:
    policy = Gr00tPolicyFairMoe(
        model_path=ckpt,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform("eagle"),
        embodiment_tag="new_embodiment", denoising_steps=4, backbone_model_type="eagle",
    )
    policy.inference_head = "moe"

    picks, all_probs = [], []
    N = 30
    for q in range(N):
        torch.manual_seed(q); np.random.seed(q)
        # rotation_relative is QUATERNION (4-dim) per data_config target_rotations.
        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # identity
        obs = {
            "video.left_view":  (np.random.rand(1, 256, 256, 3) * 255).astype(np.uint8),
            "video.right_view": (np.random.rand(1, 256, 256, 3) * 255).astype(np.uint8),
            "video.wrist_view": (np.random.rand(1, 256, 256, 3) * 255).astype(np.uint8),
            "state.end_effector_position_relative": np.random.randn(1, 3).astype(np.float32) * 0.1,
            "state.end_effector_rotation_relative": quat + np.random.randn(1, 4).astype(np.float32) * 0.05,
            "state.gripper_qpos": np.random.randn(1, 2).astype(np.float32) * 0.01,
            "state.base_position": np.zeros((1, 3), dtype=np.float32),
            "state.base_rotation": quat.copy(),
            "annotation.human.action.task_description": ["pick up the object"],
        }
        try:
            out = policy.get_action(obs)
            p = int(np.asarray(out["_moe_picked"]).flatten()[0])
            probs = np.asarray(out["_moe_probs"]).flatten()
            picks.append(p); all_probs.append(probs)
        except Exception as e:
            print(f"  query {q}: err {type(e).__name__}: {e}")
            continue
    if not picks:
        continue
    c = Counter(picks)
    ks = sorted(c)
    cs = " ".join(f"k={k}:{c[k]}" for k in ks)
    fs = " ".join(f"k={k}:{c[k]/len(picks):.2f}" for k in ks)
    print(f"{label:<20} | {cs:<48} | {fs}")
    probs_arr = np.stack(all_probs)  # (N, K)
    print(f"  per-expert prob: mean={probs_arr.mean(axis=0).round(3).tolist()}  "
          f"std={probs_arr.std(axis=0).round(3).tolist()}  K={probs_arr.shape[1]}")
    print(f"  first 5 queries (probs k=0..K):")
    for i in range(min(5, len(all_probs))):
        print(f"    q={i}: probs={all_probs[i].round(3).tolist()}  pick=k={picks[i]}")
    del policy; torch.cuda.empty_cache()

print()
print("Legend: k=0 raw16  k=1 m8  k=2 m4  k=3 n8")
print("ROBOCASA_PROBE_DONE")
