"""Probe MoE router pick distribution per task.

Server: start serve_policy_dexjoco.py with --head moe (gr00t env).
Client: this script (dexjoco env). For each task, reset env, call infer N
times advancing env with returned actions, count _moe_picked.
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tyro
import yaml
from openpi_client import image_tools, websocket_client_policy

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


def _process_obs_native(self, env_obs: dict):
    obs = {pk: image_tools.convert_to_uint8(env_obs[ek]) for pk, ek in self.camera_mapping.items()}
    obs["state"] = env_obs["state"][:46 if self.dual_arm else 23]
    obs["prompt"] = self.prompt
    return obs


DexJoCoOpenPIEnv._process_obs = _process_obs_native


def main(
    port: int = 8000,
    host: str = "127.0.0.1",
    n_queries: int = 12,
    dexjoco_repo: str = "/sjw_alinlab2/home/hojin2/multigpu_workspace/external_dependencies/dexjoco",
):
    os.environ.setdefault("MUJOCO_GL", "egl")
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    TASKS = ["hammer_nail", "click_mouse", "pick_bucket", "pinch_tongs", "fold_glasses", "water_plant"]
    print(f"{'task':<15}  {'picks (k=)':<48}  fraction (k=)")
    print("-" * 100)
    grand = []
    for task in TASKS:
        cfg = yaml.safe_load(open(f"{dexjoco_repo}/configs/rand_obj/{task}.yaml"))
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
        for _ in range(n_queries):
            if env.is_done:
                env.reset()
            result = client.infer(env.get_obs())
            if "_moe_picked" not in result:
                print(f"{task:<15}  (server didn't return _moe_picked)")
                break
            picks.append(int(np.asarray(result["_moe_picked"]).flatten()[0]))
            # advance env with the returned chunk (so subsequent queries see varied states)
            for a in result["actions"]:
                if env.is_done:
                    break
                env.step(a)
        env.close()
        cnt = Counter(picks)
        ks = sorted(cnt)
        cnt_str = " ".join(f"k={k}:{cnt[k]}" for k in ks)
        frac_str = " ".join(f"k={k}:{cnt[k]/len(picks):.2f}" for k in ks)
        grand.extend(picks)
        print(f"{task:<15}  {cnt_str:<48}  {frac_str}")
    print()
    gc = Counter(grand)
    ks = sorted(gc)
    print(f"{'TOTAL':<15}  " + "  ".join(f"k={k}: {gc[k]} ({gc[k]/len(grand):.2%})" for k in ks))
    print()
    print("Legend (MoE4 v1 K=4 legacy; speed factor [1,2,2,1]):")
    print("  k=0 raw 16-step  (s=1)")
    print("  k=1 m8 (sum-pair from 16) 8-step  (s=2)")
    print("  k=2 m4 (sum-pair from 8) 4-step   (s=2)")
    print("  k=3 n8 raw 8-step                  (s=1)")
    print("PICKS_DONE")


if __name__ == "__main__":
    tyro.cli(main)
