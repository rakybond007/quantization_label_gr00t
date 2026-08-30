"""SimplerEnv eval client for GR00T MoE — RLDX-1 rollout_policy.py shape:
env factory + SyncVectorEnv + msgpack/zmq client. The client process never
imports torch, so SAPIEN renderer init does not race with libcuda."""
import argparse
import faulthandler
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

faulthandler.enable()

import numpy as np
import gymnasium as gym
import imageio
from tqdm import tqdm, trange


# Same VIDEO_KEY mapping as gr00t/eval/sim/SimplerEnv/simpler_env.py.
WIDOWX_VIDEO_KEY = "video.image_0"
GOOGLE_VIDEO_KEY = "video.image"


def _video_key(env_name: str) -> str:
    return WIDOWX_VIDEO_KEY if "widowx" in env_name else GOOGLE_VIDEO_KEY

# msgpack/zmq client (no torch import on the client side, so SAPIEN's GPU
# Vulkan does not race with libcuda from torch). Server is
# scripts/inference_service_simpler.py (PolicyServer + GR00T model wrap).
from gr00t.eval.policy_server import PolicyClient


def get_simpler_env_fn(env_name: str):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
        register_simpler_envs()

        # WidowXBridgeEnv / GoogleRobotEnv internally call simpler_env.make,
        # which doesn't forward renderer_kwargs. Monkey-patch simpler_env.make
        # to route SAPIEN onto a different GPU (server holds GPU 0; pick GPU 1).
        import simpler_env
        if not getattr(simpler_env, "_make_pinned", False):
            _orig_make = simpler_env.make
            _RENDER_KW = {"offscreen_only": True, "device": "cuda:1"}
            def _patched_make(task_name):
                env_spec, kwargs = simpler_env.ENVIRONMENT_MAP[task_name]
                kwargs = dict(kwargs)
                kwargs["prepackaged_config"] = True
                kwargs.setdefault("renderer_kwargs", _RENDER_KW)
                return gym.make(env_spec, obs_mode="rgbd", **kwargs)
            simpler_env.make = _patched_make
            simpler_env._make_pinned = True

        return gym.make(env_name)
    return env_fn


def make_eval_env(env_name: str):
    # No MultiStepWrapper: we already pop one chunk-step per `env.step` call,
    # and MultiStepWrapper's `value[step, :]` indexing is incompatible with
    # the scalar-shaped action keys (action_space = Box(shape=())).
    return get_simpler_env_fn(env_name)()


def step_action(chunk_dict, idx, H):
    """Pull one action at index idx from a chunk dict. The shapes match the
    base env's action_space exactly: a per-step value matches `space.shape`.

    For SimplerEnv WidowX/Google envs, all action keys are `Box(shape=())`,
    so every per-step value is a 0-D numpy scalar. Skip keys that are not
    chunked along axis 0 (e.g. moe metadata like `_moe_picked`)."""
    one = {}
    for k, v in chunk_dict.items():
        v = np.asarray(v)
        if v.ndim == 0 or v.shape[0] != H:
            continue
        # v.shape = (H,) + space_shape. v[idx] has shape `space_shape`.
        one[k] = v[idx]
    return one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--video_dir", type=str, default="./videos")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[client] env_name={args.env_name}", flush=True)

    env = gym.vector.SyncVectorEnv([partial(make_eval_env, args.env_name)])
    print("[client] SyncVectorEnv ready", flush=True)

    client = PolicyClient(host=args.host, port=args.port)
    print(f"[client] connected. modality keys: {client.call_endpoint('get_modality_config', requires_input=False)}", flush=True)

    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        os.remove(pred_path)

    def unbatch(obs):
        """SyncVectorEnv adds an n_envs leading axis we strip, then we add a
        T=1 leading axis for video/state to match the GR00T transform contract
        (libero/robocasa client uses the same `np.array([img])` pattern).
        Annotation strings get wrapped in a list."""
        out = {}
        for k, v in obs.items():
            if k.startswith("annotation."):
                v0 = v[0] if hasattr(v, "__len__") and len(v) > 0 else v
                if isinstance(v0, np.ndarray):
                    v0 = v0.item()
                out[k] = [str(v0)]
                continue
            arr = np.asarray(v)[0]              # drop n_envs axis
            if arr.ndim == 0:                    # scalar state → (T=1,)
                out[k] = np.asarray([arr])
            else:                                # video (H, W, C) → (1, H, W, C)
                out[k] = arr[None, ...]
        return out

    pick_counts = defaultdict(int)
    succ_list = []
    chunk_lens = []
    vkey = _video_key(args.env_name)

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset(seed=args.seed + i)
        obs = unbatch(obs)
        done = False; t = 0
        last_info = info
        # Collect frames for the rollout video (matches libero/robocasa pattern).
        frames = [np.asarray(obs[vkey])[0]]  # drop T axis → (H, W, 3)
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            action = client.call_endpoint("get_action", {"observations": obs})
            # H = chunk length. Skip 0-d / scalar metadata (e.g. _moe_picked).
            H = next(np.asarray(v).shape[0] for v in action.values()
                     if np.asarray(v).ndim >= 1)
            pick_counts[H] += 1
            chunk_lens.append(H)
            for j in range(H):
                a = step_action(action, j, H)
                # SyncVectorEnv batches across n_envs; MultiStepWrapper batches
                # across n_action_steps=1. Both add a leading axis, so each
                # value ends up at (1, 1, ...) — but we only need (1, ...) and
                # MultiStepWrapper unpacks the n_action_steps axis with
                # value[step, :], which expects ndim>=2.
                batched = {k: v[None, ...] for k, v in a.items()}
                next_obs, _, terms, truncs, infos = env.step(batched)
                done = bool(terms[0]) or bool(truncs[0])
                final_info = infos.get("final_info")
                if final_info is not None and final_info[0] is not None:
                    last_info = final_info[0]
                else:
                    last_info = {k: (v[0] if hasattr(v, "__len__") else v)
                                 for k, v in infos.items()
                                 if k not in ("final_info", "_final_info")}
                obs = unbatch(next_obs)
                frames.append(np.asarray(obs[vkey])[0])
                t += 1
                pbar.update(1)
                if done or t >= args.max_episode_steps:
                    break
        pbar.close()
        succ = bool(last_info.get("success", False)) if isinstance(last_info, dict) else False
        succ_list.append(succ)
        suffix = "success" if succ else "failure"
        seg = args.env_name.replace("/", "_")
        try:
            imageio.mimwrite(
                f"{args.video_dir}/rollout_{seg}_{i}_{suffix}.mp4",
                [np.asarray(f) for f in frames], fps=20,
            )
        except Exception as e:
            print(f"  [warn] video save failed for ep {i}: {e}", flush=True)
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: [{succ}] action_steps: {t}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, mean exec len so far={np.mean(chunk_lens):.1f}",
              flush=True)

    env.close()
    sr = float(np.mean(succ_list)) if succ_list else 0.0
    mean_len = float(np.mean(chunk_lens)) if chunk_lens else 0.0
    print(f"\nFinal: success rate = {sr:.4f} ({len(succ_list)} ep)")
    print(f"Mean chunk exec length = {mean_len:.2f}")
    print(f"Router pick distribution (chunks): {dict(pick_counts)}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {sr:.4f}\n")
        f.write(f"mean_exec_chunk_len: {mean_len:.4f}\n")
        f.write(f"router_picks: {dict(pick_counts)}\n")


if __name__ == "__main__":
    main()
