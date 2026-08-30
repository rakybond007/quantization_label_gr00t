# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Client with adjacent-action merging: server returns 16-step delta chunks, client
# merges adjacent pairs into 8-step chunks before executing in env.
# Continuous dims: sum adjacent pairs (equivalent total displacement for delta actions).
# Discrete dims (gripper_close, control_mode): use the LATER value of each pair
# (more recent decision; summing binary signals is meaningless).
#
# Mirrors robocasa_service.py so it works with the same server (inference_service.py
# --server) running the baseline GR00T-N1.5 checkpoint.

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env


DISCRETE_KEYS = {"action.gripper_close", "action.control_mode"}


def merge_adjacent_actions(action_chunk: dict) -> dict:
    """Merge adjacent action steps: (H, D) -> (H//2, D)."""
    merged = {}
    for k, v in action_chunk.items():
        v = np.array(v)
        if v.ndim < 1 or v.shape[0] < 2:
            merged[k] = v
            continue
        H = v.shape[0]
        H2 = H // 2
        even = v[0 : 2 * H2 : 2]
        odd = v[1 : 2 * H2 : 2]
        if k in DISCRETE_KEYS:
            merged[k] = odd
        else:
            merged[k] = even + odd
    return merged


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Server-side chunk length before client-side merging.")
    parser.add_argument("--env_name", type=str, default="<ENV_NAME>")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--video_dir", type=str, default="./videos")
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--max_episode_steps", type=int, default=1440)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--robots", nargs="+", type=str, default="PandaOmron")
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--obj_groups", type=str, nargs="+", default=None)
    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--collect_data", type=bool, default=False)
    args = parser.parse_args()

    if not args.client:
        raise ValueError("This script is client-only. Run server via scripts/inference_service.py.")

    simulation_client = SimulationInferenceClient(host=args.host, port=args.port)
    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_name = args.env_name
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "generative_textures": "100p",
    }
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config
    if env_name in ["Lift"]:
        if args.obj_groups is not None:
            print("Specifying 'obj_groups' in non-kitchen environment does not have an effect.")
    else:
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        config["obj_instance_split"] = "A"
    env_info = json.dumps(config)

    env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        robots=args.robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False,
        layout_ids=args.layout,
        style_ids=args.style,
        collect_data=args.collect_data,
    )
    print(f"Environment {args.env_name} loaded successfully.")

    env = RoboCasaWrapper(env)

    stats = defaultdict(list)
    if os.path.exists(f"{args.video_dir}/prediction.txt"):
        with open(f"{args.video_dir}/prediction.txt", "r") as f:
            for line in f:
                success = line.strip().split(":")[-1].strip()
                add_to(stats, flatten({"is_success": success}))

    if args.video_dir is not None:
        video_base_path = Path(args.video_dir)
        trigger = len(stats["is_success"]) if "is_success" in stats else 1
        print(f"Recording videos from episode {trigger}")
        episode_trigger = lambda t: t % trigger == 0  # noqa
        env = RecordVideo(
            env,
            video_base_path,
            disable_logger=True,
            episode_trigger=episode_trigger,
            fps=20,
            name_prefix=f"{args.env_name}",
        )

    MERGED_HORIZON = args.action_horizon // 2
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=MERGED_HORIZON,
    )
    print(f"Client-side merge: server horizon={args.action_horizon} -> exec horizon={MERGED_HORIZON}")

    def postprocess_action(action):
        new_action = {}
        for k, v in action.items():
            if v.ndim == 1:
                new_action[k] = v[..., None]
            else:
                new_action[k] = v
        return new_action

    print(f"Starting evaluation for {args.env_name} with {args.n_episodes} episodes...")
    for i in trange(args.n_episodes):
        pbar = tqdm(
            total=args.max_episode_steps,
            desc=f"Episode {i} / {env.unwrapped.get_ep_meta()['lang']}",
            leave=False,
        )
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            print(f"Skipping episode {i} as it has already been evaluated.")
            continue
        done = False
        step = 0
        while not done:
            obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)

            action = simulation_client.get_action(obs)
            merged = merge_adjacent_actions(action)
            post_action = postprocess_action(merged)

            obs, reward, terminated, truncated, info = env.step(post_action)
            done = terminated or truncated
            step += MERGED_HORIZON
            pbar.update(MERGED_HORIZON)

        add_to(stats, flatten({"is_success": info["is_success"]}))
        with open(f"{args.video_dir}/prediction.txt", "a") as f:
            f.write(f"episode {i} is_success: {info['is_success']} action_steps: {step}\n")
        pbar.close()

    env.close()
    for k, v in stats.items():
        stats[k] = np.mean(v)
        with open(f"{args.video_dir}/prediction.txt", "a") as f:
            f.write(f"{k}: {stats[k]} \n")
    print(stats)
