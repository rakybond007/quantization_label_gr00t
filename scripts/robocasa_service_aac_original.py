"""Original AAC (paper) eval client for robocasa.

Implements the paper's dynamic-horizon recipe on the plain GR00T-N1.5 baseline:
  1. Server (--head main_n --n_samples N) returns N main 16-step samples.
  2. Compute per-step entropy across the N samples (per_step_entropy_aac).
  3. h* = max(argmax_h(avg_E[0..h] - avg_E[0..h-1]), xi) over steps 1..16.
  4. Execute first h* steps of sample 0; re-query the server; repeat.

No compression, no merged-8 head needed — this is the unmodified AAC paper
recipe applied as a wrapper around any single-head diffusion policy.
"""
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

from robocasa_service_selective import (
    DISCRETE_KEYS, add_to, flatten, _to_ntd,
    per_step_entropy_aac, aac_cliff_h_star, step_action,
)


def collect_main_samples(action_dict):
    """Pull main N-sample dict from server reply (no m8 expected for head=main_n).

    Returns {key: (N, T, D)} for keys starting with 'action.' that are not
    the _m8 companion (which main_n never emits, but we still filter just in case).
    """
    out = {}
    for k, v in action_dict.items():
        if not k.startswith("action."):
            continue
        if k.endswith("_m8"):
            continue
        out[k] = _to_ntd(v, has_N=True)
    return out


def sample0_chunk(samples_main_dict):
    """Drop the N axis: take sample[0] for execution. Output {key: (T, D)}."""
    return {k: v[0] for k, v in samples_main_dict.items()}


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
    p.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--aac-xi", type=int, default=1,
                   help="AAC paper ξ: minimum execution length (steps). h* >= xi enforced.")
    args = p.parse_args()

    client = SimulationInferenceClient(host=args.host, port=args.port)
    print("modality config keys:", list(client.get_modality_config().keys()))

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    env = load_robocasa_gym_env(
        args.env_name, seed=args.seed,
        robots=args.robots, camera_widths=256, camera_heights=256,
        render_onscreen=False, obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False, layout_ids=args.layout, style_ids=args.style,
        collect_data=False,
    )
    print(f"Env {args.env_name} loaded.")
    env = RoboCasaWrapper(env)

    stats = defaultdict(list)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                if "is_success:" not in line:
                    continue
                tail = line.strip().split("is_success:", 1)[1]
                tok = tail.strip().split()[0].strip("[],")
                add_to(stats, flatten({"is_success": tok}))

    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env, Path(args.video_dir), disable_logger=True,
            episode_trigger=lambda t: True, fps=20,
            name_prefix=f"{args.env_name}",
        )

    env = MultiStepWrapper(
        env, video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1), n_action_steps=1,
    )

    h_star_log = []
    all_step_E = []

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            continue
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"]  = np.flip(obs["video.left_view"],  axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)

            raw = client.get_action(obs)
            samples_main = collect_main_samples(raw)
            if not samples_main:
                raise RuntimeError("server returned no action.* keys")
            N = next(iter(samples_main.values())).shape[0]
            if N < 2:
                raise RuntimeError(
                    f"original AAC needs N>=2 samples; got N={N}. Start server with --n_samples >= 2."
                )

            step_E = per_step_entropy_aac(samples_main)            # (16,)
            all_step_E.append(step_E.tolist())
            h_star = aac_cliff_h_star(step_E, xi=max(1, args.aac_xi))   # in steps, 1..16
            h_star_log.append(int(h_star))

            chunk = sample0_chunk(samples_main)
            T_total = next(iter(chunk.values())).shape[0]
            H_exec = min(int(h_star), T_total)

            for j in range(H_exec):
                obs, reward, terminated, truncated, info = step_action(chunk, j, env)
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
        print(f"  ep {i}: success={succ}, action_steps={t}, mean h*={np.mean(h_star_log):.2f}")

    # Summary
    def _is_true(x):
        arr = np.asarray(x).ravel()
        return bool(arr[0]) if arr.size else False
    succ_arr = [1 if _is_true(x) else 0 for x in stats["is_success"]]
    succ_rate = float(np.mean(succ_arr)) if succ_arr else 0.0
    h_hist = Counter(h_star_log)
    with open(pred_path, "a") as f:
        f.write(f"is_success: {succ_rate:.4f}\n")
        f.write(f"h_star_hist: {dict(sorted(h_hist.items()))}\n")
        f.write(f"h_star_mean: {np.mean(h_star_log) if h_star_log else 0:.2f}\n")
        f.write(f"n_chunks: {len(h_star_log)}\n")
    print(f"[done] succ={succ_rate:.4f}  h*_mean={np.mean(h_star_log) if h_star_log else 0:.2f}  "
          f"h*_hist={dict(sorted(h_hist.items()))}")


if __name__ == "__main__":
    main()
