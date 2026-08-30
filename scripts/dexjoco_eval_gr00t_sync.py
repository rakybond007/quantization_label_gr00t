"""Synchronous DexJoCo eval (robocasa/libero pattern): get_action -> execute
ALL chunk actions -> next get_action. No async inference, no interpolation,
no buffer/timestamp games. Lets us isolate whether the async path is what
makes short MoE m8 chunks behave as frozen.

Uses the same DexJoCoOpenPIEnv with the native-resolution _process_obs patch.
"""
import os
import time
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import tyro
from openpi_client import image_tools, websocket_client_policy
from scipy.spatial.transform import Rotation as R

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


def _process_obs_native(self, env_obs: dict):
    """Native-resolution variant (matches our async wrapper). Camera frames
    keep dataset resolution (e.g. 640x640) so the GR00T VideoToTensor accepts."""
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


def main(
    config: Path,
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    port: int = 8000,
    host: str = "127.0.0.1",
    output: Path | None = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    episodes: int = 50,
    pad_state_dim46: bool = False,
    max_episode_steps: int = 1500,
    compress_k: int = 1,
):
    import yaml
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(seed)

    cfg = yaml.safe_load(open(config))
    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    dual_arm = cfg["robot_type"] == "dual_arm"
    prompt = cfg["prompt"]

    output_dir = output or Path("outputs") / f"{env_name}_sync_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DexJoCoOpenPIEnv(
        env_name=env_name, camera_mapping=camera_mapping, seed=seed,
        rand_full=rand_full, randomize_dynamics=randomize_dynamics,
        dual_arm=dual_arm, prompt=prompt, render_mode=render_mode,
        pad_state_dim46=pad_state_dim46, password=cfg.get("password", None),
    )
    env.start()
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    num_success = 0
    # Resume: count already-completed eps and skip them. An episode is "done"
    # iff its final dir (episode_NN_success or episode_NN_failure) exists.
    already_succ = 0
    for ep_done in range(episodes):
        if (output_dir / f"episode_{ep_done:02d}_success").is_dir():
            already_succ += 1
    for ep in range(episodes):
        succ_dir = output_dir / f"episode_{ep:02d}_success"
        fail_dir = output_dir / f"episode_{ep:02d}_failure"
        if succ_dir.is_dir() or fail_dir.is_dir():
            print(f"Episode {ep + 1}/{episodes}: SKIP (already done)", flush=True)
            continue
        print(f"Episode {ep + 1}/{episodes}", flush=True)
        # Clean any leftover _temp from a previous crashed run for this ep idx.
        leftover = output_dir / f"episode_{ep:02d}_temp"
        if leftover.is_dir():
            import shutil
            shutil.rmtree(leftover, ignore_errors=True)
        video_dir = output_dir / f"episode_{ep:02d}_temp"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_writers = {
            cam: imageio.get_writer(video_dir / f"{cam}.mp4", fps=30)
            for cam in camera_mapping.values()
        }
        env.reset()
        # save reset frame
        for cam, w in video_writers.items(): w.append_data(env.get_raw_images()[cam])

        steps = 0
        # Pre-roll for click_mouse (same as async client)
        if env_name == "click_mouse":
            preroll = np.array([-4.4294e-01, 1.3729e-06, 1.5170e00,
                                -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0.263, 0, 0, 0])
            for _ in range(30):
                env.step(preroll)
                for cam, w in video_writers.items(): w.append_data(env.get_raw_images()[cam])
                steps += 1

        # Synchronous control loop
        while steps < max_episode_steps:
            result = client.infer(env.get_obs())
            chunk = result["actions"]               # (H, 22)
            # Fixed-K compression for absolute actions: take last value of each
            # K-block. If H not divisible by K, leading num_full*K actions form
            # K-blocks (block-last), remaining (H - num_full*K) executed raw.
            # K=1 -> no compression.
            if compress_k > 1:
                H_in = chunk.shape[0]
                num_full = H_in // compress_k
                out = []
                for bi in range(num_full):
                    block = chunk[bi * compress_k:(bi + 1) * compress_k]
                    out.append(block[-1])           # block-last (abs)
                for ji in range(num_full * compress_k, H_in):
                    out.append(chunk[ji])           # raw remainder
                chunk = np.stack(out)
            for a in chunk:
                # Copy to writable np.ndarray — websocket deserialization can
                # hand back read-only arrays which break scipy R.from_rotvec.
                env.step(np.array(a))
                for cam, w in video_writers.items(): w.append_data(env.get_raw_images()[cam])
                steps += 1
                if env.is_done or steps >= max_episode_steps:
                    break
            if env.is_done:
                break

        for w in video_writers.values(): w.close()
        result_suffix = "success" if env.is_success else "failure"
        final_dir = output_dir / f"episode_{ep:02d}_{result_suffix}"
        video_dir.rename(final_dir)
        # Record per-episode result + action_steps (robocasa prediction.txt format)
        # so step counts can be aggregated directly instead of via ffprobe.
        with open(output_dir / "prediction.txt", "a") as pf:
            flag = "True" if env.is_success else "False"
            pf.write(f"episode {ep} is_success: [{flag:>5}] action_steps: {steps}\n")
        if env.is_success:
            num_success += 1
            print("Success!", flush=True)
        else:
            print("Failed", flush=True)

    # Recount: this run only sees its own num_success; total = pre-existing + new.
    total_succ = num_success + already_succ
    print(f"\nSuccess rate (incl. resumed): {total_succ}/{episodes} ({100 * total_succ / episodes:.1f}%)", flush=True)
    (output_dir / f"success_rate_{total_succ}_{episodes}.txt").touch()
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
