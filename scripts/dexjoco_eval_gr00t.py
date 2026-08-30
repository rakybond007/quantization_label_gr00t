"""Run DexJoCo eval against a GR00T policy server.

Thin wrapper around `dexjoco_openpi_client.eval_dexjoco_openpi.main` that
monkey-patches `DexJoCoOpenPIEnv._process_obs` to send camera frames at their
native simulator resolution (640x640) instead of the pi0.5-style 224x224.
Our GR00T `dexjoco_single_arm_multi_horizon` data-config registers the dataset's
640x640 video resolution and applies VideoCrop + VideoResize internally, so
pre-resizing on the client breaks the VideoToTensor resolution check.

State / prompt translation stays in the server-side adapter
(`scripts/serve_policy_dexjoco.py`).

Run inside the `dexjoco` conda env, alongside the GR00T server.
"""

import numpy as np
import tyro
from openpi_client import image_tools

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from dexjoco_openpi_client.eval_dexjoco_openpi import main


def _process_obs_native(self, env_obs: dict):
    """Identical to upstream _process_obs except the camera frames keep their
    native (640x640) resolution."""
    obs_dict = {}
    for policy_key, env_key in self.camera_mapping.items():
        img = env_obs[env_key]
        obs_dict[policy_key] = image_tools.convert_to_uint8(img)
    if self.dual_arm:
        state = env_obs["state"][:46]
    else:
        state = env_obs["state"][:23]
        if self.pad_state_dim46:
            state = np.concatenate([state, np.zeros(46 - len(state))])
    obs_dict["state"] = state
    obs_dict["prompt"] = self.prompt
    return obs_dict


DexJoCoOpenPIEnv._process_obs = _process_obs_native
print("[patch] DexJoCoOpenPIEnv._process_obs sends native-resolution frames", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
