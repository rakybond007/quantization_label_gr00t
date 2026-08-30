#!/bin/bash
# Ablate3: async client with receive_actions patched to take entire chunk
# verbatim (no stale-drop, no per-action slice, no blend). Preserves rollouts.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"; CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_moe4_v1_no_balance/checkpoint-60000"
DEXJOCO_REPO="$HOME/multigpu_workspace/external_dependencies/dexjoco"
ROOT="$BASE_DIR/analysis/_smoke/dexjoco_moe_ablate3"; PORT=8102
mkdir -p "$ROOT/no_slice"

cat > "$ROOT/wrapper.py" << 'PY'
import os, numpy as np, tyro
from pathlib import Path
from queue import Empty
from openpi_client import image_tools
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from dexjoco_openpi_client import eval_dexjoco_openpi as upstream

def _process_obs_native(self, env_obs):
    obs = {pk: image_tools.convert_to_uint8(env_obs[ek]) for pk, ek in self.camera_mapping.items()}
    obs["state"] = env_obs["state"][:46 if self.dual_arm else 23]
    obs["prompt"] = self.prompt
    return obs
DexJoCoOpenPIEnv._process_obs = _process_obs_native

def receive_actions_full(action_queue, actions_buffer, now_timestamp, dual_arm):
    Action = upstream.Action
    while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
        actions_buffer.popleft()
    while True:
        try:
            ac = action_queue.get_nowait()
        except Empty:
            break
        start = (actions_buffer[-1].timestamp + 1) if actions_buffer else now_timestamp
        for i, a in enumerate(ac.action):
            actions_buffer.append(Action(action=a, timestamp=start + i))

upstream.receive_actions = receive_actions_full

def main(config: Path, port: int, host: str = "127.0.0.1",
         output: Path | None = None, episodes: int = 2,
         seed: int = 0, pad_state_dim46: bool = False):
    upstream.main(config=config, port=port, host=host, output=output,
                  episodes=episodes, seed=seed, pad_state_dim46=pad_state_dim46,
                  replan_ratio=0.8, randomize_dynamics=False, rand_full=False,
                  render_mode="rgb_array")

if __name__ == "__main__":
    tyro.cli(main)
PY

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"; export NO_ALBUMENTATIONS_UPDATE=1
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/serve_policy_dexjoco.py" \
    --port "$PORT" --model-path "$CKPT" \
    --data-config dexjoco_single_arm_multi_horizon --embodiment-tag new_embodiment \
    --head moe --denoising-steps 4 > "$ROOT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 36); do
    if grep -qE "server listening|Listening on|serve_forever|local_ip" "$ROOT/server.log" 2>/dev/null; then break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "ABLATE3 FAIL (server died)"; tail -20 "$ROOT/server.log"; exit 1; fi
    sleep 5
done
echo "[i] server ready"
MUJOCO_GL=egl "$CONDA_PATH/envs/dexjoco/bin/python" -u "$ROOT/wrapper.py" \
    --config "$DEXJOCO_REPO/configs/rand_obj/hammer_nail.yaml" \
    --port "$PORT" --host 127.0.0.1 --episodes 2 \
    --output "$ROOT/no_slice" > "$ROOT/no_slice/eval.log" 2>&1
sr=$(grep "Success rate" "$ROOT/no_slice/eval.log" | tail -1)
echo "ABLATE3 [no_slice + no_drop + no_blend]  $sr"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "ABLATE3_DONE"
