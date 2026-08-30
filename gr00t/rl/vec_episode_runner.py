"""Vectorised RL rollout: run K parallel RoboCasa envs (subprocess workers via
gym.vector.AsyncVectorEnv) sharing one batched policy forward on the main GPU.

Per-chunk flow:
  1. Collect K active obs.
  2. Build a single batched obs dict and run policy_forward once.
  3. gate.sample(vl_batch) → K decisions.
  4. Each env executes its own chunk (8 if merge, 16 if not) by inner-stepping
     the vec_env up to max(chunk_lens) times. Envs that finished early are
     padded with zero-delta actions (continuous dims = 0, discrete dims = -1
     to keep gripper open / control_mode unchanged) until the slowest env
     finishes the chunk.
  5. Envs that terminate are masked out and dropped from subsequent chunks.

We collect K independent EpisodeResult objects suitable for the existing
GRPO trainer.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import gymnasium as gym
import numpy as np
import torch

from gr00t.rl.episode_runner import (
    DISCRETE_KEYS,
    ChunkRecord,
    EpisodeResult,
    _add_time_axis,
    _flip_views,
    make_env,
    merge_action_chunk,
    postprocess_action,
)
from gr00t.rl.policy_with_features import policy_forward


def _stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Batch a list of obs dicts (each with leading T=1) into one dict with
    leading B dimension."""
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        if k.startswith("video.") or k.startswith("state."):
            out[k] = np.stack([o[k] for o in obs_list], axis=0)  # (B, T, ...)
        else:
            # annotations: list per env; we keep nested list
            out[k] = [o[k] for o in obs_list]
    return out


def _split_action_batch(action_batch: Dict[str, np.ndarray], k: int) -> List[Dict[str, np.ndarray]]:
    """Split (B, H, D) action dict into list of B single-env (H, D) dicts."""
    out = [{} for _ in range(k)]
    for key, val in action_batch.items():
        # val shape: (B, H, D)  — Gr00tPolicy produces this when batch input given
        for i in range(k):
            out[i][key] = val[i]
    return out


def _zero_step_action(template_chunk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build a single-step 'do nothing' action matching the schema of
    `template_chunk` (taking step 0 shapes). Continuous dims = 0, discrete = -1."""
    act = {}
    for k, v in template_chunk.items():
        # v: (H, D) or (H,)
        single = np.zeros_like(v[0]) if v.ndim >= 1 else np.array(0.0)
        if k in DISCRETE_KEYS:
            single = np.full_like(single, -1)
        act[k] = single
    return act


def make_env_fn(env_name: str, seed: int, generative_textures: bool = True):
    """Return a thunk that AsyncVectorEnv can call to construct one env."""
    def _thunk():
        return make_env(env_name=env_name, seed=seed, generative_textures=generative_textures)
    return _thunk


def run_vec_episodes(env_name: str, seeds: List[int], policy, gate, device: torch.device,
                     max_env_steps: int = 1500, sample_gate: bool = True,
                     capture_idx: int = -1) -> List[EpisodeResult]:
    """Run K = len(seeds) parallel episodes. Returns one EpisodeResult per env.

    capture_idx: index of the env whose render frames to record (-1 = none).
    """
    K = len(seeds)
    fns = [make_env_fn(env_name, s) for s in seeds]
    # MuJoCo isn't fork-safe (GL state corrupts after fork); spawn each worker
    # in a fresh interpreter so each gets its own EGL context.
    vec_env = gym.vector.AsyncVectorEnv(fns, shared_memory=False, context="spawn")

    # Per-env reset
    obs_batched, info = vec_env.reset(seed=seeds)
    # AsyncVectorEnv returns obs as dict-of-arrays with leading B dim
    # i.e. video.left_view shape (B, H, W, C); state.X shape (B, D)
    # but our per-env wrapper returns single (H,W,C) / (D,), so vec wraps adds B.

    # Reconstruct per-env obs lists
    def _per_env_obs(batched):
        out = []
        for i in range(K):
            d = {}
            for k, v in batched.items():
                # v shape: (B, ...) — take env i
                if isinstance(v, (list, tuple)):
                    d[k] = v[i]
                else:
                    d[k] = v[i]
            out.append(d)
        return out

    obs_list = _per_env_obs(obs_batched)
    active = np.ones(K, dtype=bool)
    results = [EpisodeResult(success=False, total_env_steps=0,
                             task_lang="", seed=seeds[i]) for i in range(K)]

    while active.any() and max(r.total_env_steps for r in results) < max_env_steps:
        # 1. Collect active envs' obs (flipped + with T=1 axis) for policy
        active_idx = np.where(active)[0]
        flipped = [_add_time_axis(_flip_views(obs_list[i].copy())) for i in active_idx]
        batched_obs = _stack_obs(flipped)

        # 2. Batched policy forward → (B, H=16, D) per action key
        action_batch, vl_batch = policy_forward(policy, batched_obs)
        # vl_batch shape: (B, T, D)

        # 3. Gate decision per env
        with torch.no_grad():
            decisions_t, log_probs_t, probs_t = gate.sample(
                vl_batch.to(device), deterministic=not sample_gate,
            )
        decisions = decisions_t.cpu().numpy().astype(int)
        log_probs = log_probs_t.cpu().numpy()
        probs = probs_t.cpu().numpy()

        # 4. Per-env action chunks
        per_env_actions = _split_action_batch(action_batch, len(active_idx))
        chunks_per_env = []
        chunk_lens = []
        for i, raw in enumerate(per_env_actions):
            if decisions[i] == 1:
                merged = merge_action_chunk(raw)
                merged = postprocess_action(merged)
            else:
                merged = postprocess_action(raw)
            chunks_per_env.append(merged)
            chunk_lens.append(next(iter(merged.values())).shape[0])
        max_chunk_len = max(chunk_lens)

        # 5. Inner step loop with padding for short-chunk envs
        # vec_env.step requires actions for ALL K envs (including currently
        # inactive ones); we send zero-delta actions for those.
        step_terminate = np.zeros(K, dtype=bool)
        chunk_frame_indices = [[] for _ in active_idx]
        for t in range(max_chunk_len):
            full_actions = {}
            # Build per-key (K, D) batch
            template = chunks_per_env[0]
            for key in template.keys():
                # Default = zero (or -1 for discrete) for inactive / past-chunk-end envs
                sample_step = template[key][0]
                if key in DISCRETE_KEYS:
                    fill = np.full_like(sample_step, -1)
                else:
                    fill = np.zeros_like(sample_step)
                # Build (K, D)
                full = np.tile(fill[None, ...], (K,) + (1,) * sample_step.ndim)
                for j, env_i in enumerate(active_idx):
                    if t < chunk_lens[j]:
                        full[env_i] = chunks_per_env[j][key][t]
                full_actions[key] = full

            obs_batched, reward, terminated, truncated, info = vec_env.step(full_actions)

            # Per-env is_success may live in the info dict (per-env keys).
            is_success_per_env = info.get("is_success", [False] * K) if isinstance(info, dict) else [False] * K

            # Update per-env step counters / termination
            for j, env_i in enumerate(active_idx):
                if t < chunk_lens[j]:
                    results[env_i].total_env_steps += 1
                    if env_i == capture_idx:
                        chunk_frame_indices[j].append(results[env_i].total_env_steps - 1)
                    # AsyncVectorEnv returns per-env arrays
                    if bool(terminated[env_i]) or bool(truncated[env_i]):
                        step_terminate[env_i] = True
                    # Track success (sticky once True)
                    try:
                        if bool(np.asarray(is_success_per_env[env_i]).any()):
                            results[env_i].success = True
                    except Exception:
                        pass

            # Refresh per-env obs from the latest step
            obs_list = _per_env_obs(obs_batched)

        # 6. Record chunks per active env
        for j, env_i in enumerate(active_idx):
            results[env_i].chunks.append(ChunkRecord(
                chunk_idx=len(results[env_i].chunks),
                decision=int(decisions[j]),
                log_prob=float(log_probs[j]),
                prob=float(probs[j]),
                env_steps=chunk_lens[j],
                vl_features=vl_batch[j].cpu(),
                frame_indices=chunk_frame_indices[j],
            ))

        # 7. Mark terminated envs as inactive
        for env_i in range(K):
            if step_terminate[env_i] or results[env_i].total_env_steps >= max_env_steps:
                active[env_i] = False

    # Determine success flags from final infos
    # We'd need to track is_success per env across the loop; do a final check.
    # AsyncVectorEnv exposes success in the info on termination step.
    # We approximate: any time a step's info reported is_success True for env_i, set success.
    # (Tracked inside the inner loop above? Add it.)

    vec_env.close()
    return results
