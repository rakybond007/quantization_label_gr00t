"""Probe dexjoco MoE router probabilities on REAL GT observations.

Loads first frame of N episodes + corresponding state from dexjoco lerobot data,
feeds through Gr00tPolicyFairMoe, prints _moe_probs per timestep across a few
timesteps per episode. Shows true input-dependent variation (vs. synthetic).
"""
import sys, glob, json
sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
import numpy as np
import pandas as pd
import imageio.v3 as iio

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe

DATA_ROOT = "/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20"
CKPT = "ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_balance/checkpoint-30000"

TASKS = ["hammer_nail", "click_mouse", "fold_glasses", "water_plant"]
TIMESTEPS_PER_EP = 6            # sample 6 timesteps per episode (start / quarter / mid ...)
N_EP_PER_TASK = 3               # use 3 episodes per task

dc = DATA_CONFIG_MAP["dexjoco_single_arm_multi_horizon"]
policy = Gr00tPolicyFairMoe(
    model_path=CKPT,
    modality_config=dc.modality_config(),
    modality_transform=dc.transform("eagle"),
    embodiment_tag="new_embodiment", denoising_steps=4, backbone_model_type="eagle",
)
policy.inference_head = "moe"

records = []
print(f"{'task':<13} | {'ep':<6} | {'t':<4} | probs (k=0 raw16, k=1 m8, k=2 m4, k=3 n8)   | pick")
print("-" * 110)
for task in TASKS:
    parquets = sorted(glob.glob(f"{DATA_ROOT}/{task}/data/chunk-000/episode_*.parquet"))[:N_EP_PER_TASK]
    for pf in parquets:
        ep_id = pf.split("episode_")[-1].split(".")[0]
        front_mp4 = f"{DATA_ROOT}/{task}/videos/chunk-000/observation.images.front/episode_{ep_id}.mp4"
        wrist_mp4 = f"{DATA_ROOT}/{task}/videos/chunk-000/observation.images.wrist/episode_{ep_id}.mp4"
        try:
            front_vid = iio.imread(front_mp4)  # (T, H, W, 3) uint8
            wrist_vid = iio.imread(wrist_mp4)
        except Exception as e:
            print(f"  skip ep {ep_id}: video read fail {e}")
            continue
        df = pd.read_parquet(pf)
        state = np.stack(df["observation.state"].values)   # (T, 23)
        prompt = df["annotation.human.action.task_description"].iloc[0]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            prompt = prompt[0]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        T = min(len(df), front_vid.shape[0], wrist_vid.shape[0])
        # sample evenly distributed timesteps
        ts = np.linspace(0, T - 1, TIMESTEPS_PER_EP, dtype=int)
        for t in ts:
            s = state[t]   # (23,) = [pos3, quat4, hand16]
            obs = {
                "video.front": front_vid[t:t+1].astype(np.uint8),    # (1, H, W, 3)
                "video.wrist": wrist_vid[t:t+1].astype(np.uint8),
                "state.arm_pos": s[:3].astype(np.float32)[None],     # (1, 3)
                "state.arm_rot": s[3:7].astype(np.float32)[None],    # (1, 4) quat
                "state.hand":    s[7:23].astype(np.float32)[None],   # (1, 16)
                "annotation.human.action.task_description": [prompt],
            }
            try:
                out = policy.get_action(obs)
                p = int(np.asarray(out["_moe_picked"]).flatten()[0])
                probs = np.asarray(out["_moe_probs"]).flatten()
                records.append({"task": task, "ep": ep_id, "t": int(t),
                                "probs": probs.tolist(), "pick": p})
                probs_s = " ".join(f"{x:.3f}" for x in probs)
                print(f"{task:<13} | {ep_id:<6} | {t:<4} | [{probs_s}] | k={p}")
            except Exception as e:
                print(f"  err task={task} ep={ep_id} t={t}: {type(e).__name__}: {e}")

# Aggregate
print()
print("=" * 110)
all_probs = np.array([r["probs"] for r in records])    # (N, K)
print(f"Total samples: {len(records)}")
print(f"Per-expert prob   mean = {all_probs.mean(axis=0).round(3).tolist()}")
print(f"Per-expert prob   std  = {all_probs.std(axis=0).round(3).tolist()}")
print(f"Per-expert prob   min  = {all_probs.min(axis=0).round(3).tolist()}")
print(f"Per-expert prob   max  = {all_probs.max(axis=0).round(3).tolist()}")
from collections import Counter
picks = Counter(r["pick"] for r in records)
print(f"Argmax pick counts: {dict(sorted(picks.items()))}")
# Per-task breakdown
print()
print("Per-task prob mean:")
for task in TASKS:
    p = np.array([r["probs"] for r in records if r["task"] == task])
    if len(p) == 0: continue
    print(f"  {task:<13} n={len(p):3d}  mean={p.mean(axis=0).round(3).tolist()}  std={p.std(axis=0).round(3).tolist()}")

with open("/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/analysis/_smoke/_logs/dexjoco_router_real_probs_balance30k.json", "w") as f:
    json.dump(records, f, indent=2)
print("\nDEXJOCO_REAL_PROBE_DONE")
