"""Offline probe: load MoE policy, sample real DexJoCo env states per task,
call get_action with head=moe, tally _moe_picked distribution.

Per-task this gives the router-pick frequency under realistic obs.
Compare across tasks to see if certain tasks force compressed picks more often.
"""
import os
import sys
import numpy as np
import yaml
import torch
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/external_dependencies/dexjoco/dexjoco")
sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/external_dependencies/dexjoco/openpi/packages/openpi-client/src")

os.environ.setdefault("MUJOCO_GL", "egl")

from openpi_client import image_tools
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

# Patch native-resolution obs (matches our server adapter)
def _process_obs_native(self, env_obs: dict):
    obs_dict = {}
    for policy_key, env_key in self.camera_mapping.items():
        obs_dict[policy_key] = image_tools.convert_to_uint8(env_obs[env_key])
    state = env_obs["state"][:23]
    obs_dict["state"] = state
    obs_dict["prompt"] = self.prompt
    return obs_dict
DexJoCoOpenPIEnv._process_obs = _process_obs_native

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe

dc = DATA_CONFIG_MAP["dexjoco_single_arm_multi_horizon"]
policy = Gr00tPolicyFairMoe(
    model_path="/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance/checkpoint-60000",
    modality_config=dc.modality_config(), modality_transform=dc.transform("eagle"),
    embodiment_tag="new_embodiment", denoising_steps=4, backbone_model_type="eagle",
)
policy.inference_head = "moe"

TASKS = ["hammer_nail", "click_mouse", "pick_bucket", "pinch_tongs", "fold_glasses", "water_plant"]
DEXJOCO_REPO = "/sjw_alinlab2/home/hojin2/multigpu_workspace/external_dependencies/dexjoco"

N_QUERIES = 8   # router queries per task (different env states via random steps)

print(f"{'task':<15}  {'picks':<60}  {'distribution'}")
print("-" * 110)
all_picks = []
for task in TASKS:
    cfg = yaml.safe_load(open(f"{DEXJOCO_REPO}/configs/rand_obj/{task}.yaml"))
    env = DexJoCoOpenPIEnv(
        env_name=cfg["env_name"], camera_mapping=cfg["camera_mapping"],
        seed=42, rand_full=False, randomize_dynamics=False,
        dual_arm=(cfg["robot_type"]=="dual_arm"), prompt=cfg["prompt"],
        render_mode="rgb_array", pad_state_dim46=False,
        password=cfg.get("password", None),
    )
    env.start()
    env.reset()

    picks = []
    # query at reset state, and at several states obtained by stepping with policy actions
    for q in range(N_QUERIES):
        obs = env.get_obs()
        # Convert to GR00T model input format (same as our serve adapter does)
        state = np.asarray(obs["state"], dtype=np.float32)
        gr00t_obs = {
            "video.front": np.asarray(obs["base"])[None],
            "video.wrist": np.asarray(obs["wrist"])[None],
            "state.arm_pos": state[:3][None],
            "state.arm_rot": state[3:7][None],
            "state.hand":    state[7:23][None],
            "annotation.human.action.task_description": [str(obs["prompt"])],
        }
        out = policy.get_action(gr00t_obs)
        picked = int(np.asarray(out.get("_moe_picked", [0])).flatten()[0])
        picks.append(picked)
        # advance env a few steps using actions from this output for variety
        arm_pos = np.asarray(out["action.arm_pos"])
        arm_rot = np.asarray(out["action.arm_rot"])
        hand    = np.asarray(out["action.hand"])
        action_chunk = np.concatenate([arm_pos, arm_rot, hand], axis=-1).astype(np.float32)
        for a in action_chunk[: min(5, len(action_chunk))]:
            if env.is_done: break
            env.step(a)
        if env.is_done: env.reset()
    env.close()

    cnt = Counter(picks)
    pick_str = " ".join(f"k={k}:{cnt[k]}" for k in sorted(cnt))
    all_picks.extend(picks)
    print(f"{task:<15}  {pick_str:<60}  {[(k, round(cnt[k]/len(picks),2)) for k in sorted(cnt)]}")

# Overall summary (k=0:raw16, k=1:m8, k=2:m4, k=3:n8 for legacy MoE4 v1)
print()
print(f"{'ALL':<15}  total={len(all_picks)}  {dict(Counter(all_picks))}")
print()
print("Legend (MoE4 v1 K=4 legacy layout, factors [1,2,2,1]):")
print("  k=0 = raw 16-step  | k=1 = m8 (sum-pair from 16, factor 2)")
print("  k=2 = m4 (sum-pair from 8, factor 2) | k=3 = n8 (raw 8-step, factor 1)")
print("PICKS_DONE")
