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
    p.add_argument("--judge-url", type=str, default="",
                   help="If set, query this VLM gate per chunk; when its confidence "
                        ">= --judge-threshold, add --bias to the MoE router logits of "
                        "the compressed decoders (m8/m4) via obs['_compress_bias'].")
    p.add_argument("--judge-threshold", type=float, default=0.5)
    p.add_argument("--bias", type=float, default=2.0,
                   help="router logit boost SCALE added to m8/m4 (see --bias-mode)")
    p.add_argument("--bias-mode", type=str, default="onesided",
                   choices=["onesided", "signed"],
                   help="onesided: +bias to m8/m4 when conf>=threshold, else 0. "
                        "signed: bidirectional bias = bias*(2*conf-1) -> conf=1:+bias "
                        "(compress), conf=0:-bias (push to raw/n8), conf=0.5:0.")
    p.add_argument("--judge-guidance", type=str, default="",
                   help="guidance text or @path injected into the VLM gate prompt")
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

    # Optional VLM gate -> MoE router bias toward compressed decoders (m8/m4).
    gate = None; gate_guidance = ""; gate_yes = 0; gate_total = 0; gate_log = None
    if args.judge_url:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vlm_gate import VLMGate
        gate = VLMGate(args.judge_url)
        if args.judge_guidance:
            g = args.judge_guidance
            gate_guidance = open(g[1:]).read() if (g.startswith("@") and os.path.exists(g[1:])) else g
        gate_log = open(f"{args.video_dir}/gate_router.csv", "w")
        gate_log.write("episode,step,conf,bias,H\n")
        print(f"[gate] VLM router-bias ON url={args.judge_url} tau={args.judge_threshold} "
              f"bias={args.bias} ({len(gate_guidance)}c guidance)", flush=True)

    def _instr_from_obs(o, fb):
        v = o.get("annotation.human.action.task_description")
        while isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            v = v[0]
        return str(v) if isinstance(v, str) and v else fb

    # Track router pick distribution
    pick_counts = defaultdict(int)

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            # Already evaluated in a prior run; skip ahead.
            continue
        ep_instruction = _instr_from_obs(obs, args.env_name)
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"]  = np.flip(obs["video.left_view"],  axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
            conf = bias = 0.0
            if gate is not None:
                views = [obs["video.left_view"], obs["video.right_view"], obs["video.wrist_view"]]
                conf = float(gate.judge(views, ep_instruction, gate_guidance).get("confidence", 0.0))
                if args.bias_mode == "signed":
                    bias = args.bias * (2.0 * conf - 1.0)   # conf 0..1 -> -bias..+bias
                else:
                    bias = args.bias if conf >= args.judge_threshold else 0.0
                obs["_compress_bias"] = bias
                gate_yes += int(bias > 0); gate_total += 1
            action = client.get_action(obs)
            # Use the WHOLE chunk before replanning (matches robocasa convention).
            # Skip MoE metadata keys when picking H.
            H = next(np.asarray(v).shape[0]
                     for k, v in action.items()
                     if not k.startswith(("moe_", "_moe"))
                     and np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] > 1)
            pick_counts[H] += 1
            if gate_log is not None:
                gate_log.write(f"{i},{t},{conf:.4f},{bias:.2f},{H}\n"); gate_log.flush()
            for j in range(H):
                obs, reward, terminated, truncated, info = step_action(action, j, env, H)
                done = terminated or truncated
                t += 1
                pbar.update(1)
                if done or t >= args.max_episode_steps:
                    break
        pbar.close()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, router picks={dict(pick_counts)}")

    if gate_log is not None:
        gate_log.close()
    env.close()
    print(f"\nFinal: success rate = {np.mean(stats['is_success']):.4f}  ({len(stats['is_success'])} ep)")
    if gate is not None and gate_total:
        print(f"VLM gate fired bias on {gate_yes}/{gate_total} = {100*gate_yes/gate_total:.0f}% of chunks")
    print(f"Router pick distribution (chunks): {dict(pick_counts)}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {np.mean(stats['is_success']):.4f}\n")
        f.write(f"router_picks: {dict(pick_counts)}\n")


if __name__ == "__main__":
    main()
