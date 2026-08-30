"""Robocasa eval client with SAIL-style faster-than-demo execution (Tier A,
inference-only approximation; no retraining, no controller swap).

Background
----------
SAIL (Arachchige et al., 2025) accelerates imitation policies via (4.2) a
controller-invariant *reached-pose* target, (4.3) *adaptive speed* that slows
down for critical actions, (4.1) error-adaptive guidance [EAG], and (4.4) action
scheduling. RACE (ICLR 2026) shows EAG (which needs a conditionally-trained
model) can be dropped and is the smallest-impact component. We therefore
implement an inference-only approximation of the two highest-impact pieces on the
*existing* GR00T-N1.5 robocasa base model:

  * Reached-pose target (approx): the policy emits *delta* eef poses. Summing
    consecutive deltas reconstructs the displacement toward a reached pose
    several steps ahead, which the high-gain OSC (kp=150, ``input_type=delta``)
    then tracks in a single control step -> fewer env steps for the same path.
    This is exactly SAIL's "Aggregated Actions" rule (Alg. 3): keep accumulating
    aligned deltas while ``||sum|| < mag_thresh`` and ``dot(a, sum) >= dot_thresh``;
    otherwise flush the group. The ``mag_thresh`` cap equals the OSC delta output
    bound (0.05 m) so the aggregated target never gets clipped.

  * Adaptive speed via gripper events (SAIL 4.3): a step is *critical* when the
    predicted gripper command changes (open<->close). We force a group boundary
    at gripper changes and execute critical steps one-by-one (c_slow = 1), so
    grasp/release precision is preserved while free-space motion is accelerated.

When ``--sail`` is OFF this client is behaviorally identical to
``robocasa_service_moe`` / baseline: every chunk action is executed 1-by-1
(group size 1). This keeps the file a drop-in, option-selectable variant.

Speedup is measured directly: ``action_steps`` = number of env.step() calls to
finish the task. SAIL reduces it; success rate (SR) should be preserved.
Connects to a base-model server (e.g. ``robocasa_service.py --server`` or
``serve_policy.py``) via SimulationInferenceClient.
"""
import argparse
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

# Action keys (see gr00t/eval/wrappers/robocasa_wrapper.py:131-144).
POS_KEY = "action.end_effector_position"   # (3,) delta -> SUM within a group
ROT_KEY = "action.end_effector_rotation"   # (3,) delta -> SUM within a group
GRIP_KEY = "action.gripper_close"          # (1,) binary -> LAST in group; change=critical
BASE_KEY = "action.base_motion"            # (4,) delta -> SUM within a group
MODE_KEY = "action.control_mode"           # (1,) binary -> LAST in group
SUM_KEYS = (POS_KEY, ROT_KEY, BASE_KEY)
LAST_KEYS = (GRIP_KEY, MODE_KEY)


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


def _chunk_len(action):
    """Infer chunk length H from the first chunked action key."""
    for k, v in action.items():
        if k.startswith(("moe_", "_moe")):
            continue
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] > 1:
            return v.shape[0]
    return 1


def _grip_sign(action, j):
    g = np.asarray(action[GRIP_KEY])
    val = g[j] if g.ndim == 1 else g[j].reshape(-1)[0]
    return 1 if val > 0 else -1


def build_groups(action, H, args):
    """Return a list of (start, end) index ranges [start, end) to be each
    executed as ONE aggregated env step.

    SAIL off  -> singleton groups (identical to 1-by-1 baseline execution).
    SAIL on   -> SAIL Alg.3 aggregation on the position delta, with a forced
                 group boundary at gripper changes (adaptive-speed critical
                 gating) and a rotation-magnitude clip guard.
    """
    if not args.sail:
        return [(j, j + 1) for j in range(H)]

    pos = np.asarray(action[POS_KEY]).reshape(H, -1)
    rot = np.asarray(action[ROT_KEY]).reshape(H, -1)
    grip = np.array([_grip_sign(action, j) for j in range(H)])

    # Critical step = gripper command changes vs previous step (+/- window).
    critical = np.zeros(H, dtype=bool)
    if args.gripper_gate:
        for j in range(1, H):
            if grip[j] != grip[j - 1]:
                lo = max(0, j - args.gripper_window)
                hi = min(H, j + args.gripper_window + 1)
                critical[lo:hi] = True

    groups = []
    j = 0
    while j < H:
        start = j
        cur_pos = pos[j].astype(np.float64).copy()
        cur_rot = rot[j].astype(np.float64).copy()
        j += 1
        # Critical steps are never aggregated (c_slow = 1).
        if not critical[start]:
            while j < H and not critical[j] and (j - start) < args.max_group:
                a = pos[j].astype(np.float64)
                nxt = cur_pos + a
                # Per-DIMENSION cap = OSC delta output bound (0.05/dim, 0.5/dim
                # for rotation): merge only while every axis stays within the
                # controller's range, so the aggregated target is never clipped.
                if np.max(np.abs(nxt)) > args.agg_mag_thresh:
                    break
                if np.max(np.abs(cur_rot + rot[j])) > args.agg_rot_thresh:
                    break
                # SAIL Alg.3 direction gate: stop if the next delta diverges from
                # the accumulated direction (keeps fine motion un-merged).
                cn, an = np.linalg.norm(cur_pos), np.linalg.norm(a)
                if cn > 1e-8 and an > 1e-8 and float(cur_pos @ a) / (cn * an) < args.agg_dot_thresh:
                    break
                cur_pos = nxt
                cur_rot = cur_rot + rot[j]
                j += 1
        groups.append((start, j))
    return groups


def step_group(action, start, end, env, H):
    """Aggregate chunk[start:end) into one action dict and call env.step once.
    SUM_KEYS are summed; LAST_KEYS take the value at end-1. Output matches the
    MultiStepWrapper(n_action_steps=1) per-step format: (1, D) / (1, 1)."""
    one = {}
    for k, v in action.items():
        if k.startswith(("moe_", "_moe")):
            continue
        v = np.asarray(v)
        if v.ndim == 0 or v.shape[0] != H:
            continue
        if k in SUM_KEYS:
            agg = v[start:end].astype(np.float64).sum(axis=0)
        else:  # LAST_KEYS and anything else: take last in group
            agg = v[end - 1]
        if v.ndim == 1:        # (H,) -> (1, 1)
            one[k] = np.asarray([[agg]])
        else:                  # (H, D) -> (1, D)
            one[k] = np.asarray([agg])
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
    p.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--no_record_video", action="store_true")

    # ---- SAIL options ----
    p.add_argument("--sail", action="store_true",
                   help="Enable SAIL-style aggregation/re-timing. Off => 1-by-1 baseline.")
    p.add_argument("--agg_mag_thresh", type=float, default=0.05,
                   help="Per-dim cap on summed pos delta (= OSC delta bound 0.05/dim, avoids clip).")
    p.add_argument("--agg_dot_thresh", type=float, default=0.25,
                   help="Min cosine alignment to keep aggregating (SAIL Alg.3).")
    p.add_argument("--agg_rot_thresh", type=float, default=0.5,
                   help="Per-dim cap on summed rot delta (= OSC rot bound 0.5/dim).")
    p.add_argument("--max_group", type=int, default=8,
                   help="Hard cap on steps merged into one group.")
    p.add_argument("--gripper_gate", action="store_true", default=True,
                   help="Force group boundary + 1-by-1 around gripper changes (adaptive speed).")
    p.add_argument("--no_gripper_gate", dest="gripper_gate", action="store_false")
    p.add_argument("--gripper_window", type=int, default=1,
                   help="Steps around a gripper change marked critical.")
    args = p.parse_args()

    client = SimulationInferenceClient(host=args.host, port=args.port)
    print("modality config keys:", list(client.get_modality_config().keys()))
    print(f"[SAIL] enabled={args.sail} mag={args.agg_mag_thresh} dot={args.agg_dot_thresh} "
          f"rot={args.agg_rot_thresh} max_group={args.max_group} "
          f"gripper_gate={args.gripper_gate} window={args.gripper_window}")

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
        if not args.no_record_video:
            env = RecordVideo(
                env, Path(args.video_dir),
                disable_logger=True,
                episode_trigger=lambda t: True,
                fps=20,
                name_prefix=f"{args.env_name}",
            )

    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=1,
    )

    group_sizes = defaultdict(int)  # histogram of executed group sizes

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            continue
        done = False
        t = 0          # env.step calls (= action_steps; the speedup metric)
        n_infer = 0    # number of chunk inferences
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
            action = client.get_action(obs)
            n_infer += 1
            H = _chunk_len(action)
            groups = build_groups(action, H, args)
            for (s, e) in groups:
                group_sizes[e - s] += 1
                obs, reward, terminated, truncated, info = step_group(action, s, e, env, H)
                done = terminated or truncated
                t += 1
                pbar.update(1)
                if done or t >= args.max_episode_steps:
                    break
        pbar.close()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t} n_infer: {n_infer}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, n_infer={n_infer}, "
              f"group_sizes={dict(group_sizes)}")

    env.close()
    print(f"\nFinal: success rate = {np.mean(stats['is_success']):.4f}  ({len(stats['is_success'])} ep)")
    print(f"Group-size histogram: {dict(group_sizes)}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {np.mean(stats['is_success']):.4f}\n")
        f.write(f"sail: {args.sail} group_sizes: {dict(group_sizes)}\n")


if __name__ == "__main__":
    main()
