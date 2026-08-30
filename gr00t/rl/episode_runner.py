"""Run a single RoboCasa episode where every chunk has a binary merge gate
deciding 16-step (no-merge) vs 8-step (merge) execution.

We bypass MultiStepWrapper to keep n_action_steps dynamic per chunk and to
collect per-env-step renders for video logging.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from robosuite.controllers import load_composite_controller_config

from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.rl.policy_with_features import policy_forward


DISCRETE_KEYS = {"action.gripper_close", "action.control_mode"}


def merge_action_chunk(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """16 -> 8: continuous dims sum adjacent pairs, discrete dims take latter."""
    merged = {}
    for k, v in action.items():
        v = np.asarray(v)
        if v.ndim < 1 or v.shape[0] < 2:
            merged[k] = v
            continue
        H = v.shape[0]
        H2 = H // 2
        even = v[0:2 * H2:2]
        odd = v[1:2 * H2:2]
        merged[k] = odd if k in DISCRETE_KEYS else (even + odd)
    return merged


def postprocess_action(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in action.items():
        if v.ndim == 1:
            out[k] = v[..., None]
        else:
            out[k] = v
    return out


@dataclass
class ChunkRecord:
    chunk_idx: int
    decision: int           # 0 = no_merge (16 steps), 1 = merge (8 steps)
    log_prob: float
    prob: float
    env_steps: int
    vl_features: torch.Tensor   # (T, D) on CPU, kept for re-evaluating logp
    frame_indices: List[int]    # which env-step indices this chunk produced


@dataclass
class EpisodeResult:
    success: bool
    total_env_steps: int
    chunks: List[ChunkRecord] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)   # (H,W,3) uint8 per env-step
    task_lang: str = ""
    seed: int = 0


def make_env(env_name: str, seed: int, generative_textures: bool = True,
             style_ids=None, layout_ids=-1):
    if style_ids is None:
        style_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    env = load_robocasa_gym_env(
        env_name=env_name,
        seed=seed,
        robots="PandaOmron",
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        obj_instance_split="A",
        generative_textures="100p" if generative_textures else None,
        randomize_cameras=False,
        layout_ids=layout_ids,
        style_ids=style_ids,
        collect_data=False,
    )
    env = RoboCasaWrapper(env)
    return env


def _flip_views(obs):
    obs["video.left_view"] = np.flip(obs["video.left_view"], axis=0)
    obs["video.right_view"] = np.flip(obs["video.right_view"], axis=0)
    obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=0)
    return obs


def _add_time_axis(obs):
    """Convert raw RoboCasaWrapper obs (H,W,C) / (D,) to the (T=1,...) format
    that the policy expects (normally produced by MultiStepWrapper)."""
    out = {}
    for k, v in obs.items():
        if k.startswith("video.") or k.startswith("state."):
            out[k] = np.expand_dims(np.asarray(v), axis=0)
        else:
            out[k] = v
    return out


def run_episode(env, policy, gate, device: torch.device, seed: int,
                max_env_steps: int = 1500, sample_gate: bool = True,
                capture_frames: bool = False) -> EpisodeResult:
    """Run a single episode. Returns full trajectory record."""
    obs, info = env.reset(seed=seed)
    task_lang = env.unwrapped.get_ep_meta().get("lang", "")

    result = EpisodeResult(success=False, total_env_steps=0, task_lang=task_lang, seed=seed)
    done = False

    while not done and result.total_env_steps < max_env_steps:
        obs = _flip_views(obs)
        policy_obs = _add_time_axis(obs)

        # Frame-level: build single-step obs dict for the policy (it expects T=1)
        action_dict, vl_feats = policy_forward(policy, policy_obs)
        # vl_feats shape: (1, T, D)
        vl_feats = vl_feats.squeeze(0)  # (T, D)

        # Gate decision
        with torch.no_grad():
            decision_t, log_prob_t, prob_t = gate.sample(
                vl_feats.unsqueeze(0).to(device), deterministic=not sample_gate,
            )
        decision = int(decision_t.item())
        log_prob = float(log_prob_t.item())
        prob = float(prob_t.item())

        # Apply merge if decided
        chunk = merge_action_chunk(action_dict) if decision == 1 else action_dict
        chunk = postprocess_action(chunk)

        H = next(iter(chunk.values())).shape[0]
        chunk_frame_idx = []
        for step in range(H):
            single_step = {k: v[step, :] for k, v in chunk.items()}
            obs, reward, terminated, truncated, info = env.step(single_step)
            if capture_frames:
                # render returns (H,W,3); see RoboCasaWrapper.render
                try:
                    frame = env.render()
                except Exception:
                    frame = np.zeros((256, 256, 3), dtype=np.uint8)
                result.frames.append(frame)
                chunk_frame_idx.append(len(result.frames) - 1)
            result.total_env_steps += 1
            if terminated or truncated or result.total_env_steps >= max_env_steps:
                done = True
                break

        result.chunks.append(ChunkRecord(
            chunk_idx=len(result.chunks),
            decision=decision,
            log_prob=log_prob,
            prob=prob,
            env_steps=H,
            vl_features=vl_feats.cpu(),
            frame_indices=chunk_frame_idx,
        ))

    is_success = bool(np.any(info.get("is_success", False)))
    result.success = is_success
    return result
