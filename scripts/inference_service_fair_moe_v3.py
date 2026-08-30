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

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_fair_moe_v3 import Gr00tPolicyFairMoeV3 as Gr00tPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    backbone_model_type: str = "eagle"
    """The type of backbone model to use, e.g., 'eagle', 'qwen2_5_vl', etc."""

    head: str = "main"
    """Which decoder head to use at inference. 'main' = standard 16-step.
    'f2' / 'f4' = aux heads (each output step represents 2 / 4 env-step deltas).
    'ensemble' = LS combine of main+f2+f4. 'ensemble_fix' = LS but with
    discrete dims overwritten by main (gripper, control_mode).
    Only meaningful when the checkpoint was trained with multi-horizon loss."""

    discrete_action_dims: tuple[int, ...] = ()
    """Indices of discrete action dimensions. Used by 'ensemble_fix' head
    (overwrites those dims with main head). For single_panda_gripper:
    --discrete-action-dims 6 11  (gripper_close, control_mode)."""

    n_samples: int = 1
    """Number of main samples to draw per request when head='selective'.
    1 = single sample (equivalent to main_and_m8). >1 enables entropy estimation
    over per-pair variance across samples (selective compression Score 1)."""

    beta_sample_t: bool = False
    """When True, the flow-matching denoising chain samples its timestep schedule
    from the action head's training Beta(noise_beta_alpha, noise_beta_beta) at
    inference, instead of using the deterministic uniform schedule. Adds
    inter-call stochasticity on top of the existing per-sample initial-noise
    diversity used by AAC cliff scoring."""

#####################################################################################


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform(backbone_model_type=args.backbone_model_type)

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            backbone_model_type=args.backbone_model_type,
        )

        # Apply head selection (used by Gr00tPolicy._get_action_from_normalized_input)
        policy.inference_head = args.head
        # Override discrete_action_dims if given
        if args.discrete_action_dims:
            policy.model.action_head.discrete_action_dims = list(args.discrete_action_dims)
            print(f"[Server] discrete_action_dims = {list(args.discrete_action_dims)}")
        # Wire N samples for head=selective
        if args.n_samples > 1:
            policy.model.action_head._n_samples_at_inference = int(args.n_samples)
            print(f"[Server] head=selective N samples = {args.n_samples}")
        if args.beta_sample_t:
            policy.model.action_head._beta_sample_t_at_inference = True
            ah = policy.model.action_head
            print(f"[Server] Beta-sampled timestep schedule enabled at inference "
                  f"(noise_beta_alpha={float(ah.config.noise_beta_alpha)}, "
                  f"noise_beta_beta={float(ah.config.noise_beta_beta)})")
        print(f"[Server] Using inference head: {args.head}")

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        import time

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # Create a policy wrapper
        policy_client = RobotInferenceClient(host=args.host, port=args.port)

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        time_start = time.time()
        action = policy_client.get_action(obs)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
