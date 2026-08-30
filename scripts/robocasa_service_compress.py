"""Fixed-ratio action compression eval client for robocasa.

Inference-time intervention only (no retraining): receive a 16-step chunk from
the standard GR00T-N1.5 baseline server (head=main), aggregate the chunk into
blocks of size K, and execute the compressed chunk step-by-step.

Aggregator (per action key):
  - continuous keys: block-sum (delta actions compose by sum)
  - discrete keys  : block-last (gripper / control_mode latch)

If 16 is not divisible by K (e.g. K=3 -> 16 = 5*3 + 1), the first 15 actions
form 5 compressed blocks and the last action is kept raw, yielding 6 actions
to execute. K=2 -> 8 actions; K=3 -> 6 actions.
"""
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

from robocasa_service_selective import (
    DISCRETE_KEYS, add_to, flatten, _to_ntd, step_action,
)


def compress_chunk(chunk_dict, K, discrete_keys=DISCRETE_KEYS):
    """Apply fixed-K block aggregation per action key.

    chunk_dict: {key: (T, D)} where T = base horizon (16).
    Returns {key: (T_out, D)} with T_out = (T // K) + (T mod K) raw tail.
    """
    out = {}
    for key, v in chunk_dict.items():
        v = np.asarray(v)
        if v.ndim < 2:
            v = v[..., None]
        T = v.shape[0]
        num_full = T // K
        rem = T - num_full * K
        is_disc = key in discrete_keys
        agg_blocks = []
        for i in range(num_full):
            block = v[i * K:(i + 1) * K]
            if is_disc:
                agg_blocks.append(block[-1])  # latch
            else:
                agg_blocks.append(block.sum(axis=0))  # delta sum
        if rem > 0:
            for j in range(num_full * K, T):
                agg_blocks.append(v[j])  # raw remainder
        out[key] = np.stack(agg_blocks)
    return out


def collect_chunk(action_dict):
    """Pull main chunk from server reply. Server returns single-sample chunks."""
    out = {}
    for k, v in action_dict.items():
        if not k.startswith("action."):
            continue
        if k.endswith("_m8"):  # in case server returns m8 companion (we ignore)
            continue
        v = np.asarray(v)
        # Normalize to (T, D)
        if v.ndim == 1:
            v = v[..., None]
        elif v.ndim == 3:
            v = v[0]  # if N samples returned, take sample 0
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--video_dir", type=str, default="./videos")
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--robots", nargs="+", type=str, default="PandaOmron")
    p.add_argument("--controller", type=str, default=None)
    p.add_argument("--layout", type=int, nargs="+", default=-1)
    p.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--compress-k", type=int, default=2,
                   help="Block size K for fixed-ratio compression (K=2 -> 8 actions, K=3 -> 6 actions).")
    args = p.parse_args()

    client = SimulationInferenceClient(host=args.host, port=args.port)
    print("modality config keys:", list(client.get_modality_config().keys()))

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    env = load_robocasa_gym_env(
        args.env_name, seed=args.seed,
        robots=args.robots, camera_widths=256, camera_heights=256,
        render_onscreen=False, obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False, layout_ids=args.layout, style_ids=args.style,
        collect_data=False,
    )
    print(f"Env {args.env_name} loaded.")
    env = RoboCasaWrapper(env)

    stats = defaultdict(list)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                if "is_success:" not in line:
                    continue
                tail = line.strip().split("is_success:", 1)[1]
                tok = tail.strip().split()[0].strip("[],")
                add_to(stats, flatten({"is_success": tok}))

    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env, Path(args.video_dir), disable_logger=True,
            episode_trigger=lambda t: True, fps=20,
            name_prefix=f"{args.env_name}",
        )

    env = MultiStepWrapper(
        env, video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1), n_action_steps=1,
    )

    K = int(args.compress_k)
    assert K >= 1
    chunk_lens = []

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            continue
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"]  = np.flip(obs["video.left_view"],  axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)

            raw = client.get_action(obs)
            chunk_raw = collect_chunk(raw)            # {key: (16, D)}
            if K > 1:
                chunk = compress_chunk(chunk_raw, K)  # {key: (T_out, D)}
            else:
                chunk = chunk_raw
            H_exec = next(iter(chunk.values())).shape[0]
            chunk_lens.append(H_exec)
            for j in range(H_exec):
                obs, reward, terminated, truncated, info = step_action(chunk, j, env)
                done = terminated or truncated
                t += 1
                pbar.update(1)
                if done or t >= args.max_episode_steps:
                    break
        pbar.close()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, exec_chunk_len={H_exec}, K={K}")

    def _is_true(x):
        arr = np.asarray(x).ravel()
        return bool(arr[0]) if arr.size else False
    succ_arr = [1 if _is_true(x) else 0 for x in stats["is_success"]]
    succ_rate = float(np.mean(succ_arr)) if succ_arr else 0.0
    with open(pred_path, "a") as f:
        f.write(f"is_success: {succ_rate:.4f}\n")
        f.write(f"compress_k: {K}\n")
        f.write(f"exec_chunk_len_mean: {np.mean(chunk_lens) if chunk_lens else 0:.2f}\n")
    print(f"[done] succ={succ_rate:.4f}  K={K}  exec_chunk_len_mean={np.mean(chunk_lens) if chunk_lens else 0:.2f}")


if __name__ == "__main__":
    main()
