# SPDX-License-Identifier: Apache-2.0
"""RoboCasa rollout client for an F-level *ratio* checkpoint.

Counterpart of `gr00t/eval/libero/eval_taskwise_gr00t_flevel.py`, which only
covers LIBERO.  The server (`serve_policy_flevel.py`) picks a speed per chunk and
ships it as `_flevel_level` / `_flevel_k`; the decoder for that level emits rows
that are ALREADY merged -- one row of a K=2 stream is two fine actions summed.

    docs/ratio_flevel/README.md: "디코더 출력을 역정규화한 뒤 다시 압축하면
                                  이중 압축이므로 그대로 실행해야 한다."

So this client must NOT aggregate.  Every returned row is one env step, and the
chunk length varies with the level (16 / 11 / 8 / 7 for K = 1 / 1.5 / 2 / 2.5).
That is the whole difference from the fixed-K client, which merges client-side.

Writes `prediction.txt` in the same format the RoboCasa analysis scripts already
parse, plus a per-chunk `flevel_chunks.csv` ledger for the level distribution and
replan accounting the README asks for.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

# Same split the labelling and training side uses: everything else is a delta
# that was summed inside a block, these two latch to the block's last value.
SUM_KEYS = {"action.end_effector_position", "action.end_effector_rotation",
            "action.base_motion"}
LAST_KEYS = {"action.gripper_close", "action.control_mode"}
RATIO_GRID = (1.0, 1.5, 2.0, 2.5)


def add_to(d, single):
    for k, v in single.items():
        d[k].append(v)


def flatten(d, parent="", sep="."):
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def chunk_rows(action):
    """Number of rows the server actually sent -- this is the executed length."""
    n = 0
    for k, v in action.items():
        if not k.startswith("action."):
            continue
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] > n:
            n = v.shape[0]
    return n


def block_sizes(horizon, k):
    """Same rule as flare/action_head_flevel.py:block_sizes -- the client cannot
    import flare, so it is restated here and asserted against the horizon."""
    k = float(k)
    if k == 1.0:
        sizes = [1] * horizon
    elif k == 1.5:
        sizes = [1 if i % 2 == 0 else 2 for i in range(horizon)]
        sizes = _trim(sizes, horizon)
    elif k == 2.0:
        sizes = _trim([2] * horizon, horizon)
    elif k == 2.5:
        sizes = _trim([2 if i % 2 == 0 else 3 for i in range(horizon)], horizon)
    else:
        raise ValueError(f"unsupported K {k}")
    assert sum(sizes) == horizon, (k, sizes)
    return sizes


def _trim(sizes, horizon):
    out, tot = [], 0
    for s in sizes:
        if tot >= horizon:
            break
        s = min(s, horizon - tot)
        out.append(s); tot += s
    return out


def merge_rows(action, start, end, H):
    """Sum the delta axes over [start, end) and latch the discrete ones.

    Only for the ablation arm, where the policy has a single K=1 decoder and the
    speed decision arrives as `_flevel_k`.  A flevel checkpoint's own decoder
    already emits merged rows and must NOT go through here -- that would compress
    twice.
    """
    one = {}
    for k, v in action.items():
        if not k.startswith("action."):
            continue
        v = np.asarray(v)
        if v.ndim == 0 or v.shape[0] != H:
            continue
        agg = (v[start:end].astype(np.float64).sum(axis=0) if k in SUM_KEYS else v[end - 1])
        one[k] = np.asarray([[agg]]) if v.ndim == 1 else np.asarray([agg])
    return one


def step_row(action, j, env, H):
    """Execute row j as one env step.  No aggregation: the decoder already did it."""
    one = {}
    for k, v in action.items():
        if not k.startswith("action."):
            continue
        v = np.asarray(v)
        if v.ndim == 0 or v.shape[0] != H:
            continue
        row = v[j]
        one[k] = np.asarray([[row]]) if v.ndim == 1 else np.asarray([row])
    return env.step(one)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--video_dir", type=str, default="./videos")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--robots", nargs="+", type=str, default="PandaOmron")
    p.add_argument("--controller", type=str, default=None)
    p.add_argument("--layout", type=int, nargs="+", default=-1)
    p.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--no_record_video", action="store_true")
    p.add_argument("--merge-by-k", action="store_true",
                   help="Ablation arm: the server returns all 16 fine rows plus the chosen "
                        "K, and the client merges them here.  Off (default) means the "
                        "server's decoder already emitted merged rows -- merging again "
                        "would compress twice.")
    p.add_argument("--clip-scale", dest="clip_scale", type=float, default=8.0,
                   help="Scale controller clip bounds.  A merged row of a K>1 stream is "
                        "several fine actions summed and routinely leaves +-1; that range "
                        "is a normalisation artefact, not a hardware limit, so clipping it "
                        "only creates OOD.  Matches the DemoSpeedup/ISR arms.")
    args = p.parse_args()

    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    client = SimulationInferenceClient(host=args.host, port=args.port)
    print("modality config keys:", list(client.get_modality_config().keys()), flush=True)

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    env = load_robocasa_gym_env(
        args.env_name, seed=args.seed, robots=args.robots,
        camera_widths=256, camera_heights=256, render_onscreen=False,
        obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False, layout_ids=args.layout, style_ids=args.style,
        collect_data=False,
    )
    print(f"Env {args.env_name} loaded.", flush=True)
    env = RoboCasaWrapper(env)

    if args.clip_scale != 1.0:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "vlm_gate", "scripts"))
        from robocasa_service_compress import patch_clip_bounds
        n_patched = patch_clip_bounds(env, args.clip_scale)
        assert n_patched > 0, "clip_scale requested but no controller bounds were patched"
        print(f"[clip] controller clip bounds x{args.clip_scale} ({n_patched} controllers patched)",
              flush=True)

    stats = defaultdict(list)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                if not line.startswith("episode "):
                    continue
                tail = line.strip().split("is_success:", 1)[1]
                add_to(stats, flatten({"is_success": tail.strip().split()[0].strip("[],")}))

    if args.video_dir and not args.no_record_video:
        env = RecordVideo(env, Path(args.video_dir), disable_logger=True,
                          episode_trigger=lambda t: True, fps=20,
                          name_prefix=f"{args.env_name}")

    env = MultiStepWrapper(env, video_delta_indices=np.arange(1),
                           state_delta_indices=np.arange(1), n_action_steps=1)

    ledger_path = Path(args.video_dir) / "flevel_chunks.csv"
    new_ledger = not ledger_path.exists()
    ledger = open(ledger_path, "a", newline="")
    lw = csv.writer(ledger)
    if new_ledger:
        lw.writerow(["episode", "chunk", "step", "level", "k", "readout_level",
                     "rows", "rows_executed", "ratio_probs"])

    level_hist = defaultdict(int)

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            continue
        done = False
        t = 0          # env.step calls -- the speed metric
        n_infer = 0    # chunk inferences -- the replan count
        ep_levels = defaultdict(int)
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
            action = client.get_action(obs)
            n_infer += 1

            level = int(np.asarray(action.get("_flevel_level", 1)).ravel()[0])
            # The server sends the grid it was built with; the module constant is
            # only the 4-decoder default and is wrong for a 2-decoder checkpoint
            # (level 2 there is K=2 or 2.5, not RATIO_GRID[1]=1.5).
            grid = np.asarray(action.get("_flevel_ratio_grid", RATIO_GRID)).ravel()
            k_exec = float(np.asarray(action.get("_flevel_k", grid[level - 1])).ravel()[0])
            readout = int(np.asarray(action.get("_flevel_readout_level", level)).ravel()[0])
            probs = np.asarray(action.get("_flevel_ratio_probs", [])).ravel().tolist()
            H = chunk_rows(action)
            assert H > 0, "server returned no action rows"
            level_hist[k_exec] += 1
            ep_levels[k_exec] += 1

            if args.merge_by_k:
                assert H == 16, f"--merge-by-k expects the fine 16-row chunk, got {H}"
                spans, pos = [], 0
                for sz in block_sizes(H, k_exec):
                    spans.append((pos, pos + sz)); pos += sz
            else:
                spans = [(j, j + 1) for j in range(H)]

            executed = 0
            for (s0, s1) in spans:
                one = merge_rows(action, s0, s1, H) if args.merge_by_k else None
                if one is None:
                    obs, reward, terminated, truncated, info = step_row(action, s0, env, H)
                else:
                    obs, reward, terminated, truncated, info = env.step(one)
                done = terminated or truncated
                t += 1
                executed += 1
                pbar.update(1)
                if done or t >= args.max_episode_steps:
                    break
            lw.writerow([i, n_infer - 1, t, level, k_exec, readout, H, executed,
                         ";".join(f"{x:.4f}" for x in probs)])
        pbar.close()
        ledger.flush()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t} n_infer: {n_infer}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, n_infer={n_infer}, "
              f"levels={dict(sorted(ep_levels.items()))}", flush=True)

    env.close()
    ledger.close()
    total = sum(level_hist.values()) or 1
    mean_k = sum(k * n for k, n in level_hist.items()) / total
    print(f"\nFinal: success rate = {np.mean(stats['is_success']):.4f}  "
          f"({len(stats['is_success'])} ep)")
    print(f"Level histogram (K -> chunks): {dict(sorted(level_hist.items()))}")
    print(f"Mean selected K = {mean_k:.3f}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {np.mean(stats['is_success']):.4f}\n")
        f.write(f"flevel_levels: {dict(sorted(level_hist.items()))} mean_k: {mean_k:.4f}\n")


if __name__ == "__main__":
    main()
