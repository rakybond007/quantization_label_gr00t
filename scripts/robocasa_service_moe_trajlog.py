"""Robocasa eval client for MoE inference (variable-horizon expert outputs).

Server returns chunks of variable length (16 / 8 / 4) depending on which expert
the router picks. This client steps env one action at a time and consumes the
WHOLE chunk before replanning (matches robocasa baseline convention — no
separate replan_steps; the action chunk length itself determines replan cadence).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

def add_to(d, single):
    for k, v in single.items():
        d[k].append(v)


def flatten(d, parent="", sep="."):
    out = []
    for k, v in d.items():
        nk = f"{parent}{sep}{k}" if parent else k
        if hasattr(v, "items"):
            out.extend(flatten(v, nk, sep=sep).items())
        else:
            out.append((nk, v))
    return dict(out)


def step_action(action_chunk_dict, idx, env, H):
    """Pull a single 1-step action from a chunk dict and call env.step.
    Skip MoE metadata keys (`_moe_picked`, `moe_picked`, `moe_probs` ...)
    that are not chunked along axis 0; they collide with the env's action
    keys when iterating per-step."""
    one = {}
    for k, v in action_chunk_dict.items():
        if k.startswith("_moe") or k.startswith("moe_"):
            continue
        v = np.asarray(v)
        if v.ndim == 0 or v.shape[0] != H:
            continue
        if v.ndim == 1:        # (H,)
            one[k] = np.asarray([[v[idx]]])           # → (1, 1)
        else:                  # (H, D)
            one[k] = np.asarray([v[idx]])             # → (1, D)
    return env.step(one)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--video_dir", type=str, default="./videos")
    p.add_argument("--trajlog_dir", type=str, default=None,
                   help="If set, dump per-step state to <dir>/traj_ep<NN>.jsonl")
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--robots", nargs="+", type=str, default="PandaOmron")
    p.add_argument("--controller", type=str, default=None)
    p.add_argument("--layout", type=int, nargs="+", default=-1)
    p.add_argument("--style", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--no_record_video", action="store_true",
                   help="Skip RecordVideo wrap (keeps prediction.txt + logs in "
                        "video_dir). Workaround for tasks that hit "
                        "mujoco.FatalError 'framebuffer not complete' in "
                        "env.reset() between episodes — disabling per-ep video "
                        "capture removes the offscreen-render call that "
                        "triggers the GL state issue. Default off.")
    args = p.parse_args()

    client = SimulationInferenceClient(host=args.host, port=args.port)
    print("modality config keys:", list(client.get_modality_config().keys()))

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        robots=args.robots,
        camera_widths=256, camera_heights=256,
        render_onscreen=False,
        obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False,
        layout_ids=args.layout,
        style_ids=args.style,
        collect_data=False,
    )
    print(f"Env {args.env_name} loaded.")
    env = RoboCasaWrapper(env)

    # Resume from any existing prediction.txt (matches scripts/robocasa_service.py
    # baseline). Episodes already recorded are skipped at the loop head.
    stats = defaultdict(list)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                if "is_success:" not in line:
                    continue
                # line format: "episode {i} is_success: {bool} action_steps: {t}"
                tail = line.strip().split("is_success:", 1)[1]
                tok = tail.strip().split()[0].strip("[],")
                add_to(stats, flatten({"is_success": tok}))

    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        if not args.no_record_video:
            # Record EVERY episode (no sparse triggering).
            env = RecordVideo(
                env, Path(args.video_dir),
                disable_logger=True,
                episode_trigger=lambda t: True,
                fps=20,
                name_prefix=f"{args.env_name}",
            )

    # Wrap with MultiStepWrapper so obs has the expected shape (B=1 frame).
    # n_action_steps=1 lets us call env.step(single-action) per call manually.
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=1,
    )

    # Track router pick distribution
    pick_counts = defaultdict(int)

    traj_dir = Path(args.trajlog_dir) if args.trajlog_dir else None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)
    for i in trange(args.n_episodes, desc=args.env_name):
        np.random.seed(args.seed + i)
        import random as _random
        _random.seed(args.seed + i)
        try:
            env.unwrapped.seed(args.seed + i)
        except Exception:
            pass
        obs, info = env.reset(seed=args.seed + i)
        if i < len(stats["is_success"]):
            continue
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        traj_fh = open(traj_dir / f"traj_ep{i:02d}.jsonl", "w") if traj_dir else None
        # Log initial post-reset state as step 0 (same physical pose as baseline
        # under matched per-episode seeding).
        if traj_fh is not None:
            eef0 = obs.get("state.end_effector_position_absolute")
            eef0_rel = obs.get("state.end_effector_position_relative")
            traj_fh.write(json.dumps({
                "step": 0, "chunk_H": 0, "moe_picked": -1,
                "eef_abs": np.asarray(eef0[0]).tolist() if eef0 is not None else None,
                "eef_rel": np.asarray(eef0_rel[0]).tolist() if eef0_rel is not None else None,
            }) + "\n"); traj_fh.flush()
        while not done and t < args.max_episode_steps:
            obs["video.left_view"]  = np.flip(obs["video.left_view"],  axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
            action = client.get_action(obs)
            H = next(np.asarray(v).shape[0]
                     for k, v in action.items()
                     if not k.startswith(("moe_", "_moe"))
                     and np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] > 1)
            picked = int(np.asarray(action.get("_moe_picked", [-1])).flatten()[0]) \
                     if "_moe_picked" in action else -1
            pick_counts[H] += 1
            for j in range(H):
                obs, reward, terminated, truncated, info = step_action(action, j, env, H)
                done = terminated or truncated
                t += 1
                pbar.update(1)
                if traj_fh is not None:
                    eef = obs.get("state.end_effector_position_absolute")
                    eef_rel = obs.get("state.end_effector_position_relative")
                    rec = {
                        "step": int(t),
                        "chunk_H": int(H),
                        "moe_picked": picked,
                        "eef_abs": np.asarray(eef[0]).tolist() if eef is not None else None,
                        "eef_rel": np.asarray(eef_rel[0]).tolist() if eef_rel is not None else None,
                    }
                    traj_fh.write(json.dumps(rec) + "\n")
                    traj_fh.flush()
                if done or t >= args.max_episode_steps:
                    break
        if traj_fh is not None:
            traj_fh.close()
        pbar.close()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, router picks={dict(pick_counts)}")

    env.close()
    print(f"\nFinal: success rate = {np.mean(stats['is_success']):.4f}  ({len(stats['is_success'])} ep)")
    print(f"Router pick distribution (chunks): {dict(pick_counts)}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {np.mean(stats['is_success']):.4f}\n")
        f.write(f"router_picks: {dict(pick_counts)}\n")


if __name__ == "__main__":
    main()
