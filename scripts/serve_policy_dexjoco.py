"""Serve a GR00T policy for DexJoCo eval via the OpenPI websocket protocol.

DexJoCo client (`external_dependencies/dexjoco/.../eval_dexjoco_openpi.py`)
sends raw obs in its own format and expects a flat 22-dim action chunk under
`result["actions"]`. This adapter wraps `Gr00tPolicy` to translate:

    client                          ->  GR00T model input keys
    obs["base"]    HxWx3 uint8         video.front (1,H,W,3)
    obs["wrist"]   HxWx3 uint8         video.wrist (1,H,W,3)
    obs["state"]   (23,)               state.arm_pos (1,3) + state.arm_rot (1,4 quat)
                                       + state.hand (1,16)
    obs["prompt"]  str                 annotation.human.action.task_description [str]

    GR00T model output                ->  client expected
    action.arm_pos (H,3)               concat to actions (H,22)
    action.arm_rot (H,3) rotvec
    action.hand    (H,16)

Run alongside DexJoCo's eval client (same conda env split as our other
benchmarks: this server uses `gr00t`, the client uses `dexjoco`).
"""

import dataclasses
import json
import logging
import os
import socket
from typing import Any, Dict, Literal

import numpy as np
import tyro

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe
from gr00t.serving import websocket_policy_server


def _select_policy_class(model_path: str):
    """Pick Gr00tPolicy vs Gr00tPolicyFairMoe based on the ckpt's config.json.

    Loading a fair_moe ckpt with the base Gr00tPolicy silently downgrades the
    MoE routing/decoder logic ("type gr00t_n1_5_fair_moe to instantiate type
    gr00t_n1_5") and produces near-zero actions at inference, which manifests
    as a robot that never moves in the eval rollout.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        model_type = json.load(open(cfg_path)).get("model_type", "")
        if "fair_moe" in model_type:
            logging.info(f"[policy] detected MoE ckpt (model_type={model_type}); using Gr00tPolicyFairMoe")
            return Gr00tPolicyFairMoe
    logging.info("[policy] using base Gr00tPolicy")
    return Gr00tPolicy


class DexJoCoPolicyAdapter(BasePolicy):
    """Translate DexJoCo client obs/action format <-> GR00T modality keys."""

    def __init__(self, base_policy, dual_arm: bool = False):
        self.base = base_policy
        self.dual_arm = dual_arm

    def get_modality_config(self):
        # Delegate to wrapped policy so the websocket server can report
        # the actual modality contract on /metadata.
        return self.base.get_modality_config()

    def _get_action_single(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state = np.asarray(obs["state"], dtype=np.float32)
        gr00t_obs = {
            "video.front": np.asarray(obs["base"])[None],   # (1, H, W, 3) uint8
            "video.wrist": np.asarray(obs["wrist"])[None],
            "state.arm_pos": state[:3][None],               # (1, 3)
            "state.arm_rot": state[3:7][None],              # (1, 4) quaternion
            "state.hand":    state[7:23][None],             # (1, 16)
            "annotation.human.action.task_description": [str(obs["prompt"])],
        }
        result = self.base.get_action(gr00t_obs)
        actions = np.concatenate(
            [
                np.asarray(result["action.arm_pos"]),   # (H, 3)
                np.asarray(result["action.arm_rot"]),   # (H, 3) rotvec
                np.asarray(result["action.hand"]),      # (H, 16)
            ],
            axis=-1,
        )  # (H, 22) — [xyz(3), rotvec(3), hand(16)]
        out = {"actions": actions.astype(np.float32)}
        # Pass through MoE router pick if present (used by debug clients).
        if "_moe_picked" in result:
            out["_moe_picked"] = np.asarray(result["_moe_picked"], dtype=np.int32)
        return out

    def _get_action_dual(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # state(46): R[pos 0:3, rot 3:7 quat, hand 7:23] L[pos 23:26, rot 26:30 quat, hand 30:46]
        state = np.asarray(obs["state"], dtype=np.float32)
        gr00t_obs = {
            "video.front":       np.asarray(obs["base"])[None],
            "video.wrist_left":  np.asarray(obs["wrist_left"])[None],
            "video.wrist_right": np.asarray(obs["wrist_right"])[None],
            "state.right_arm_pos": state[0:3][None],
            "state.right_arm_rot": state[3:7][None],     # quaternion
            "state.right_hand":    state[7:23][None],
            "state.left_arm_pos":  state[23:26][None],
            "state.left_arm_rot":  state[26:30][None],   # quaternion
            "state.left_hand":     state[30:46][None],
            "annotation.human.action.task_description": [str(obs["prompt"])],
        }
        result = self.base.get_action(gr00t_obs)
        # action(44): R[pos 3, rot 3 rotvec, hand 16] L[pos 3, rot 3 rotvec, hand 16]
        actions = np.concatenate(
            [
                np.asarray(result["action.right_arm_pos"]),
                np.asarray(result["action.right_arm_rot"]),
                np.asarray(result["action.right_hand"]),
                np.asarray(result["action.left_arm_pos"]),
                np.asarray(result["action.left_arm_rot"]),
                np.asarray(result["action.left_hand"]),
            ],
            axis=-1,
        )  # (H, 44)
        out = {"actions": actions.astype(np.float32)}
        if "_moe_picked" in result:
            out["_moe_picked"] = np.asarray(result["_moe_picked"], dtype=np.int32)
        return out

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.dual_arm:
            return self._get_action_dual(obs)
        return self._get_action_single(obs)


@dataclasses.dataclass
class Args:
    port: int = 8000
    model_path: str = "TODO"
    backbone_model_type: Literal[
        "eagle", "qwen2_5_vl", "qwen2_5_vl_7b", "paligemma"
    ] = "eagle"
    embodiment_tag: str = "new_embodiment"
    data_config: str = "dexjoco_single_arm_multi_horizon"
    head: str = "main"
    # For head="moe_selective": N samples for entropy-cliff scoring.
    n_samples: int = 1
    denoising_steps: int = 4
    # Force uniform-random expert pick (head="moe"): ignore router probs, sample
    # one of K experts uniformly at random. Sanity check: if uniform performs
    # similarly to argmax, the trained router is not adding value.
    moe_force_uniform: bool = False
    # Stochastic expert pick (head="moe"): sample expert by trained router probs
    # instead of argmax. Useful when router probs are soft (no single expert
    # dominant) so argmax locks the model into a single expert by tiny margins.
    moe_stochastic: bool = False
    # Force a specific expert index k in [0, K) (head="moe"). Useful for
    # eval'ing single-decoder policies (e.g., raw16-only from MoE ckpt). -1 = disabled.
    moe_force_pick: int = -1
    # Confidence-gated stochastic: if max(probs) >= threshold, use argmax;
    # otherwise multinomial sample. Only meaningful when --moe-stochastic is set.
    # 0.0 = pure stochastic (no gate). Paper uses 0.7.
    moe_confidence_threshold: float = 0.0


def main(args: Args) -> None:
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform(backbone_model_type=args.backbone_model_type)

    PolicyCls = _select_policy_class(args.model_path)
    base_policy = PolicyCls(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        backbone_model_type=args.backbone_model_type,
    )
    base_policy.inference_head = args.head
    if args.n_samples > 1:
        base_policy.model.action_head._n_samples_at_inference = int(args.n_samples)
        logging.info(f"head={args.head} N samples = {args.n_samples}")
    if args.moe_force_uniform:
        base_policy.model.action_head.moe_force_uniform = True
        logging.info(f"head={args.head} MoE force_uniform=True (random expert pick)")
    if args.moe_stochastic:
        base_policy.model.action_head.moe_inference_stochastic = True
        logging.info(f"head={args.head} MoE stochastic=True (sample by router probs)")
    if args.moe_confidence_threshold > 0:
        base_policy.model.action_head.moe_confidence_threshold = float(args.moe_confidence_threshold)
        logging.info(
            f"head={args.head} MoE confidence-gated stochastic: tau_c={args.moe_confidence_threshold}"
        )
    if args.moe_force_pick >= 0:
        base_policy.model.action_head.moe_force_pick = int(args.moe_force_pick)
        logging.info(f"head={args.head} MoE force_pick={args.moe_force_pick}")
    logging.info(f"Using inference head: {args.head}")

    dual_arm = "dual_arm" in args.data_config
    if dual_arm:
        logging.info("[adapter] dual-arm mode (3 cams, 46-dim state, 44-dim action)")
    policy = DexJoCoPolicyAdapter(base_policy, dual_arm=dual_arm)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    logging.info(f"policy path: {args.model_path}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
