import dataclasses
import enum
import logging
import socket

import tyro

import os, sys
ROOT = "/virtual_lab/sjw_alinlab/taeyoung/workspace/Isaac-GR00T"
sys.path.append(os.path.abspath(ROOT))

from gr00t.model.policy_fair_moe_v3 import Gr00tPolicyFairMoeV3 as Gr00tPolicy
from gr00t.model.policy import BasePolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.serving import websocket_policy_server
from typing import Literal


@dataclasses.dataclass
class Args:
    port: int = 8000

    model_path: str = "TODO"
    backbone_model_type: Literal["eagle", "qwen2_5_vl", "qwen2_5_vl_7b", "paligemma"] = "eagle"
    embodiment_tag: str = "libero"

    # Inference head selection: "main" (16-step), "m8" (8-step compressed),
    # "ensemble" / "ensemble_fix" (WLS combine of main+aux heads). Requires the
    # checkpoint to have been trained with the matching heads.
    head: str = "main"
    # Indices of discrete action dims (only used by head="ensemble_fix").
    discrete_action_dims: tuple[int, ...] = ()

    # N samples for head="selective" / "moe_selective" (entropy / hybrid score).
    # 1 = fast path (single main sample); >=2 enables N-batched main forward.
    n_samples: int = 1


def main(args: Args) -> None:
    data_config = DATA_CONFIG_MAP['libero']
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform(backbone_model_type=args.backbone_model_type)

    policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=4,
            backbone_model_type = args.backbone_model_type
        )
    # Apply head selection (consumed by Gr00tPolicy._get_action_from_normalized_input)
    policy.inference_head = args.head
    if args.discrete_action_dims:
        policy.model.action_head.discrete_action_dims = list(args.discrete_action_dims)
        logging.info(f"discrete_action_dims = {list(args.discrete_action_dims)}")
    if args.n_samples > 1:
        policy.model.action_head._n_samples_at_inference = int(args.n_samples)
        logging.info(f"head=selective N samples = {args.n_samples}")
    logging.info(f"Using inference head: {args.head}")
    policy_metadata = None

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    logging.info(f"policy path: {args.model_path}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
