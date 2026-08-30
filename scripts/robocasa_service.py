# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
from collections import defaultdict
from collections import deque
from pathlib import Path

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.robocasa_simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy

from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
import cv2

def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
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
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="<EMBODIMENT_TAG>",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        default="<ENV_NAME>",  # change this to your environment name
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument(
        "--video_dir", type=str, help="Directory to save videos.", default="./videos"
    )
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="Maximum number of steps per episode.", default=1440
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )

    # Robocasa env parameters
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    # Data collection parameters
    parser.add_argument(
        "--collect_data",
        type=bool,
        default=False,
        help="Whether to collect data",
    )
    parser.add_argument(
        "--data_collection_path",
        type=str,
        default=None,
        help="Path to save the data collection",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1
    )
    parser.add_argument(
        "--backbone_model_type",
        type=str,
        default="eagle",
        choices=["eagle", "qwen2_5_vl", "paligemma"],
        help="The backbone model type to use for the policy.",
    )


    args = parser.parse_args()

    if args.server:
        # Create a policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        # Create a simulation client
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

        # ROBOCASA ENV SETUP
        # load robocasa env
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots if isinstance(args.robots, str) else args.robots[0],
        )

        env_name = args.env_name
        # Create argument configuration
        config = {
            "env_name": env_name,
            "robots": args.robots,
            "controller_configs": controller_config,
            "generative_textures": "100p",
        }

        # Check if we're using a multi-armed environment and use env_configuration argument if so
        if "TwoArm" in env_name:
            config["env_configuration"] = args.config

        # Mirror actions if using a kitchen environment
        if env_name in ["Lift"]:  # add other non-kitchen tasks here
            if args.obj_groups is not None:
                print(
                    "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
                )
        else:
            config["layout_ids"] = args.layout
            config["style_ids"] = args.style
            ### update config for kitchen envs ###
            if args.obj_groups is not None:
                config.update({"obj_groups": args.obj_groups})

            # by default use obj instance split A
            config["obj_instance_split"] = "A"
            # config["obj_instance_split"] = None
            # config["obj_registries"] = ("aigen",)

        # Grab reference to controller config and convert it to json-encoded string
        env_info = json.dumps(config)

        env = load_robocasa_gym_env(
            args.env_name,
            seed=args.seed,
            # robosuite-related configs
            robots=args.robots,
            camera_widths=256,
            camera_heights=256,
            render_onscreen=False,
            # robocasa-related configs
            obj_instance_split="A",
            generative_textures="100p" if args.generative_textures else None,
            randomize_cameras=False,
            layout_ids=args.layout,
            style_ids=args.style,
            # data collection configs
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

        record_video = args.video_dir is not None
        if record_video:
            video_base_path = Path(args.video_dir)
            # video_base_path.mkdir(parents=True, exist_ok=True)
            trigger = len(stats['is_success']) if 'is_success' in stats else 1
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

        env = MultiStepWrapper(
            env,
            video_delta_indices=np.arange(1),
            state_delta_indices=np.arange(1),
            n_action_steps=args.action_horizon,
        )

        # postprocess function of action, to handle the case where number of dimensions are not the same
        def postprocess_action(action):
            new_action = {}
            for k, v in action.items():
                if v.ndim == 1:
                    new_action[k] = v[..., None]
                else:
                    new_action[k] = v
            return new_action

        print(f"Starting evaluation for {args.env_name} with {args.n_episodes} episodes...")
        # main evaluation loop
        for i in trange(args.n_episodes):
            pbar = tqdm(
                total=args.max_episode_steps, desc=f"Episode {i} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
            )
            obs, info = env.reset()
            if i < len(stats['is_success']):
                print(f"Skipping episode {i} as it has already been evaluated.")
                continue
            done = False
            step = 0
                
            while not done:
                obs['video.left_view'] = np.flip(obs['video.left_view'], axis=1)
                obs['video.right_view'] = np.flip(obs['video.right_view'], axis=1)
                obs['video.wrist_view'] = np.flip(obs['video.wrist_view'], axis=1)

                #####
                # Save current frame images as PNG files
                # Ensure directory exists
                # save_dir = Path(args.video_dir) if args.video_dir else Path("./collected_data")
                # save_dir.mkdir(parents=True, exist_ok=True)
                
                # # Create episode-specific directory
                # episode_dir = save_dir / f"episode_{i}"
                # episode_dir.mkdir(exist_ok=True)
                
                # # Save the current frame images
                # left_img = obs['video.left_view'][0]
                # right_img = obs['video.right_view'][0]
                # wrist_img =  obs['video.wrist_view'][0]
                # print(f"Shape of left_img: {left_img.shape}, right_img: {right_img.shape}, wrist_img: {wrist_img.shape}")
                
                # cv2.imwrite(str(episode_dir / f"left_view_step_{step}.png"), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(str(episode_dir / f"right_view_step_{step}.png"), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(str(episode_dir / f"wrist_view_step_{step}.png"), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

                #####

                action = simulation_client.get_action(obs)
                post_action = postprocess_action(action)

                # step = env.step(post_action)
                # print(f"Step: {step}, Action: {post_action}")

                # (next_obs, obs_seq), reward, terminated, truncated, info = env.step(post_action)
                obs, reward, terminated, truncated, info = env.step(post_action)
                done = terminated or truncated
                step += args.action_horizon
                
                pbar.update(args.action_horizon)
            add_to(stats, flatten({"is_success": info["is_success"]}))
            with open(f"{args.video_dir}/prediction.txt", "a") as f:
                f.write(f"episode {i} is_success: {info['is_success']} action_steps: {step}\n")
            pbar.close()

        env.close()

        for k, v in stats.items():
            stats[k] = np.mean(v)
            with open(f"{args.video_dir}/prediction.txt", "a") as f:
                f.write(f'{k}: {stats[k]} \n')
        print(stats)
        
        exit()

    else:
        raise ValueError("Please specify either --server or --client")
