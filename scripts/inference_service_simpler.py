"""Simpler-only inference server (msgpack/zmq) for GR00T MoE.

Same Gr00tPolicy as scripts/inference_service.py, but the server uses the
RLDX-1-style PolicyServer (msgpack codec, no torch on the wire). Required
because the SAPIEN-based simpler client cannot tolerate a TorchSerializer
peer (importing torch on the client side segfaults SAPIEN renderer init).
"""
from dataclasses import dataclass
import logging
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.eval.policy_server import PolicyServer
from gr00t.model.policy import Gr00tPolicy


@dataclass
class Args:
    model_path: str = "TODO"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "simplerenv_bridge"
    backbone_model_type: Literal["eagle", "qwen2_5_vl", "qwen2_5_vl_7b", "paligemma"] = "eagle"
    head: str = "main"
    discrete_action_dims: tuple[int, ...] = ()
    n_samples: int = 1
    denoising_steps: int = 4
    host: str = "0.0.0.0"
    port: int = 5555


def main(args: Args):
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
    policy.inference_head = args.head
    if args.discrete_action_dims:
        policy.model.action_head.discrete_action_dims = list(args.discrete_action_dims)
        print(f"[Server] discrete_action_dims = {list(args.discrete_action_dims)}", flush=True)
    if args.n_samples > 1:
        policy.model.action_head._n_samples_at_inference = int(args.n_samples)
    print(f"[Server] Using inference head: {args.head}", flush=True)
    print(f"[Server] policy path: {args.model_path}", flush=True)

    def _modality_keys():
        return list(modality_config.keys())

    def _get_action(observations):
        return policy.get_action(observations)

    server = PolicyServer(host=args.host, port=args.port)
    server.register_endpoint("get_action", _get_action)
    server.register_endpoint("get_modality_config", _modality_keys, requires_input=False)
    server.register_endpoint("reset", lambda: {"ok": True}, requires_input=False)
    server.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
