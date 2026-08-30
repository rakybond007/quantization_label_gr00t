"""DexJoCo eval client with K-block action compression + quantizability gate.

DexJoCo counterpart of `robocasa_service_compress.py`.  Same contract:

  * fixed-K compression of every 16-step GR00T action chunk,
  * an optional per-(sub)chunk gate (`--judge-url`, `--judge-threshold`) that
    decides whether THIS chunk may run compressed,
  * per-episode `prediction.txt` in exactly the RoboCasa format
        `episode N is_success: [ True] action_steps: M`
    plus the same trailing summary keys, so the existing analysis scripts work
    unchanged,
  * `gate_conf.csv` with the same `episode,step,conf,quantize,called,instruction`
    header.

Runs in the `dexjoco` conda env and talks to a GR00T policy server started from
`Isaac-GR00T/scripts/serve_policy_dexjoco.py` over the OpenPI websocket
protocol.  Single-arm tasks only.

===============================================================================
ACTIONS ARE ABSOLUTE TARGETS, **NOT** DELTAS  ->  COMPRESS BY SKIPPING
===============================================================================
A DexJoCo single-arm action is

    [ arm_pos(3) | arm_rot(3, rotation vector) | hand(16 joint targets) ]

and every one of those groups is an ABSOLUTE servo target in the robot base
frame, not an increment.  Three independent pieces of evidence:

 1. The LeRobot metadata shipped with the GR00T checkpoint
    (`ckpt/.../experiment_cfg/metadata.json`) marks each action group
        "action": {"arm_pos": {"absolute": true, ...},
                   "arm_rot": {"absolute": true, ...},
                   "hand":    {"absolute": true, ...}}
 2. `DexJoCoOpenPIEnv.stay()` (dexjoco_openpi_env.py) holds the current pose by
    re-sending the CURRENT STATE as an action:
        xyz = arm[:3]; rotvec = R.from_quat(arm[3:7]).as_rotvec()
        action = concat([xyz, rotvec, hand])
    A delta controller would hold still with a zero action; an absolute one has
    to be told where it already is.  This is decisive.
 3. `DexJoCoOpenPIEnv._process_action` converts the rotvec back to a quaternion
    and passes the pose straight to the operational-space controller, which
    servos toward it.

Therefore adjacent actions MUST NOT be summed.  Summing two absolute poses is
physically meaningless (it would command roughly twice the world coordinate).
K-compression here keeps the LAST target of each K-block and drops the ones in
between ("block-last skipping"), which is the correct absolute-space analogue of
RoboCasa's delta summation: both make the arm cover K steps of motion in one
control tick.  The ragged tail (T mod K steps) is executed raw, exactly like
`robocasa_service_compress.compress_chunk`.

There is also no binary gripper dimension to latch: the 16-DoF hand is fully
continuous and is skipped like everything else.
"""

import argparse
import json
import re
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# `vlm_gate` and `dexjoco_descriptors` live next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dexjoco_descriptors import action_rule_block, descriptors, facts_text  # noqa: E402

DEFAULT_CONFIG_ROOT = os.environ.get(
    "DEXJOCO_CONFIG_ROOT",
    "/sjw_alinlab/home/hojin2/multigpu_workspace/external_dependencies/dexjoco/configs",
)

# Single-arm task ids (robot_type: single_arm in the DexJoCo configs).
SINGLE_ARM_TASKS = [
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
]

# click_mouse needs the same 30-step pre-roll the upstream DexJoCo eval client
# applies, to align the arm with the dataset's starting pose.
CLICK_MOUSE_PREROLL = np.array([
    -4.4294e-01, 1.3729e-06, 1.5170e00,
    -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.263, 0, 0, 0,
], dtype=np.float32)


# -----------------------------------------------------------------------------
# Compression
# -----------------------------------------------------------------------------

def compress_chunk(chunk, K, return_blocks=False):
    """Block-last K-compression of an ABSOLUTE action chunk.

    chunk: (T, D) absolute targets.  Returns (T//K + T%K, D).

    Blocks of K consecutive targets collapse to the block's LAST target (the
    intermediate way-points are simply never commanded); the T mod K leftover
    steps at the end are executed raw.  NEVER sum -- see the module docstring.
    """
    v = np.asarray(chunk, dtype=np.float32)
    if v.ndim == 1:
        v = v[None]
    T = v.shape[0]
    if K <= 1:
        blocks = [(i, i + 1) for i in range(T)]
        return (v, blocks) if return_blocks else v
    num_full = T // K
    out, blocks = [], []
    for i in range(num_full):
        out.append(v[(i + 1) * K - 1])            # block-last (absolute skip)
        blocks.append((i * K, (i + 1) * K))
    for j in range(num_full * K, T):
        out.append(v[j])                          # raw remainder
        blocks.append((j, j + 1))
    out = np.stack(out) if out else v
    return (out, blocks) if return_blocks else out


def merge_jump_stats(sub_raw, sub_exec, prev_target):
    """How far each executed (possibly merged) command asks the arm to jump.

    Absolute-space analogue of RoboCasa's clip-excess accounting: instead of
    "does the summed delta exceed the controller limit" we measure "how big is
    the one-tick pose jump the controller is asked to service".
    """
    from dexjoco_descriptors import JUMP_LIMIT_POS
    seq = np.asarray(sub_exec, dtype=float)[:, :3]
    if prev_target is not None:
        seq = np.concatenate([np.asarray(prev_target, dtype=float)[None, :3], seq], axis=0)
    if len(seq) < 2:
        return 0, 0, 0.0
    jumps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    over = jumps - JUMP_LIMIT_POS
    n_over = int((over > 1e-9).sum())
    return n_over, int(len(jumps)), float(over[over > 0].sum())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # --- env / rollout ------------------------------------------------------
    p.add_argument("--env_name", type=str, required=True,
                   help=f"DexJoCo single-arm task id, one of {SINGLE_ARM_TASKS}")
    p.add_argument("--config", type=str, default="",
                   help="explicit DexJoCo eval yaml; default "
                        "<DEXJOCO_CONFIG_ROOT>/<config_family>/<env_name>.yaml")
    p.add_argument("--config-family", type=str, default="rand_obj",
                   choices=["rand_obj", "rand_full", "multi_task"],
                   help="rand_obj is the single-arm GR00T default: it maps "
                        "cameras to {base, wrist}, which is what the GR00T "
                        "adapter expects (multi_task maps wrist_left/right and "
                        "will NOT work single-arm)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--video_dir", type=str, default="./videos")
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rand-full", type=int, default=0, help="scene randomization")
    p.add_argument("--randomize-dynamics", type=int, default=0)
    p.add_argument("--save-video", type=int, default=1)
    p.add_argument("--dump-actions", type=str, default="",
                   help="optional .npz path: dump every raw policy chunk "
                        "(used to calibrate the descriptor thresholds)")
    # --- compression --------------------------------------------------------
    p.add_argument("--compress-k", type=int, default=2)
    # --- gate ---------------------------------------------------------------
    p.add_argument("--judge-url", type=str, default="",
                   help="empty = always compress with K (naive-K baseline)")
    p.add_argument("--judge-threshold", type=float, default=0.5)
    p.add_argument("--judge-guidance", type=str, default="",
                   help="literal text, or @path to a text file")
    p.add_argument("--judge-instruction", type=str, default="",
                   help="override the task prompt sent to the judge")
    p.add_argument("--judge-facts", type=int, default=0,
                   help="append dexjoco_descriptors.facts_text() to the "
                        "instruction sent to the judge")
    p.add_argument("--judge-actions", type=int, default=0,
                   help="append the raw planned action numbers to the "
                        "instruction sent to the judge")
    p.add_argument("--action-rules", type=int, default=0,
                   help="hard deterministic veto ANDed with the gate")
    p.add_argument("--gate-k3-threshold", type=float, default=0.0)
    p.add_argument("--gate-subchunk", type=int, default=0,
                   help="0 = one gate decision per 16-step chunk")
    p.add_argument("--gate-ttl-max", type=int, default=0)
    p.add_argument("--gate-ttl-lo", type=float, default=0.15)
    p.add_argument("--gate-ttl-hi", type=float, default=0.30)
    p.add_argument("--gate-hand-trigger", type=int, default=1,
                   help="always re-ask the judge when the planned hand motion "
                        "contains a finger transition (DexJoCo analogue of "
                        "RoboCasa's --gate-gripper-trigger)")
    return p


def main():
    args = build_parser().parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(args.seed)

    if args.env_name not in SINGLE_ARM_TASKS:
        print(f"[warn] {args.env_name!r} is not in the single-arm task list "
              f"{SINGLE_ARM_TASKS}; this client only supports single-arm.",
              flush=True)

    cfg_path = args.config or os.path.join(
        DEFAULT_CONFIG_ROOT, args.config_family, f"{args.env_name}.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    if cfg.get("robot_type") != "single_arm":
        raise SystemExit(f"[fatal] {cfg_path} is robot_type={cfg.get('robot_type')}; "
                         "this client is single-arm only.")
    cam_map = cfg["camera_mapping"]
    if set(cam_map.keys()) != {"base", "wrist"}:
        raise SystemExit(
            f"[fatal] camera_mapping keys {sorted(cam_map)} != ['base','wrist']. "
            "The GR00T single-arm adapter reads obs['base'] and obs['wrist']; "
            "use --config-family rand_obj (or rand_full).")
    prompt = cfg["prompt"]

    # Imports that need the dexjoco env; done after arg parsing so --help is cheap.
    from openpi_client import websocket_client_policy
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
    from openpi_client import image_tools

    def _process_obs_native(self, env_obs):
        """Send camera frames at native (640x640) resolution.

        The GR00T `dexjoco_single_arm_multi_horizon` data config registers the
        dataset's 640x640 video resolution and applies VideoCrop+VideoResize
        itself, so the upstream client-side resize to 224x224 breaks the
        VideoToTensor resolution check.  Same patch as
        Isaac-GR00T/scripts/dexjoco_eval_gr00t.py.
        """
        obs_dict = {}
        for policy_key, env_key in self.camera_mapping.items():
            obs_dict[policy_key] = image_tools.convert_to_uint8(env_obs[env_key])
        obs_dict["state"] = env_obs["state"][:23]
        obs_dict["prompt"] = self.prompt
        return obs_dict

    DexJoCoOpenPIEnv._process_obs = _process_obs_native

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    K = args.compress_k

    # --- gate setup ---------------------------------------------------------
    gate = None
    gate_log = None
    gate_guidance = args.judge_guidance
    if gate_guidance.startswith("@") and os.path.exists(gate_guidance[1:]):
        gate_guidance = open(gate_guidance[1:]).read()
    if args.judge_url:
        from vlm_gate import VLMGate
        gate = VLMGate(args.judge_url)
        print(f"[gate] VLM gate ON url={args.judge_url} K_quantize={K} "
              f"threshold={args.judge_threshold} ttl_max={args.gate_ttl_max}", flush=True)
        gc = video_dir / "gate_conf.csv"
        gc_new = (not gc.exists()) or gc.stat().st_size == 0
        gate_log = open(gc, "a")
        if gc_new:
            gate_log.write("episode,step,conf,quantize,called,instruction\n")
    else:
        print(f"[gate] no judge: every chunk compressed with K={K}", flush=True)

    # --- resume (same semantics as robocasa_service_compress) ---------------
    pred_path = video_dir / "prediction.txt"
    done_flags = []
    if pred_path.exists():
        # RoboCasa writes the numpy repr of a 1-element bool array, i.e.
        # "is_success: [ True]" (leading space!) or "is_success: [False]".
        # A naive `split()[0].strip("[],")` yields "[" for the True case and
        # silently drops every successful episode from the resume set, so match
        # the token with a bracket/space-tolerant regex instead. The trailing
        # summary line ("is_success: 0.5000") does not match and is skipped.
        pat = re.compile(r"is_success:\s*\[?\s*(True|False)\s*\]?")
        for line in open(pred_path):
            m = pat.search(line)
            if m:
                done_flags.append(m.group(1) == "True")
    if done_flags:
        print(f"[resume] {len(done_flags)} episode(s) already in {pred_path}", flush=True)

    env = DexJoCoOpenPIEnv(
        env_name=cfg["env_name"], camera_mapping=cam_map, seed=args.seed,
        rand_full=bool(args.rand_full), randomize_dynamics=bool(args.randomize_dynamics),
        dual_arm=False, prompt=prompt, render_mode="rgb_array",
        pad_state_dim46=False, password=None,
    )
    env.start()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    if args.save_video:
        import imageio

    cam_order = [cam_map["base"], cam_map["wrist"]]

    succ_flags = list(done_flags)
    chunk_lens, gate_confs = [], []
    gate_yes = gate_total = gate_calls = 0
    rule_blocks = 0
    rule_reasons = {}
    merge_over = merge_steps = 0
    merge_excess = 0.0
    policy_ms = gate_ms = 0.0
    dumped = []

    try:
        for i in range(args.n_episodes):
            env.reset()
            if i < len(done_flags):
                print(f"ep {i}: SKIP (already in prediction.txt)", flush=True)
                continue

            writers = None
            if args.save_video:
                vd = video_dir / f"episode_{i:02d}_temp"
                if vd.is_dir():
                    import shutil
                    shutil.rmtree(vd, ignore_errors=True)
                vd.mkdir(parents=True, exist_ok=True)
                writers = {c: imageio.get_writer(vd / f"{c}.mp4", fps=30) for c in cam_order}

            def _frame():
                if writers is None:
                    return
                raw = env.get_raw_images()
                for c, w in writers.items():
                    w.append_data(raw[c])

            _frame()

            t = 0
            g_ttl = 0
            g_last_conf = g_last_q = None
            prev_target = None
            done = False

            if cfg["env_name"] == "click_mouse":
                for _ in range(30):
                    env.step(CLICK_MOUSE_PREROLL.copy())
                    _frame()
                    t += 1

            while not done and t < args.max_episode_steps:
                t0 = time.time()
                res = client.infer(env.get_obs())
                policy_ms += (time.time() - t0) * 1000.0
                chunk = np.asarray(res["actions"], dtype=np.float32)   # (T, 22)
                if args.dump_actions:
                    dumped.append(chunk.copy())
                T = chunk.shape[0]
                S = args.gate_subchunk if (gate is not None and args.gate_subchunk > 0) else T

                pos = 0
                while pos < T and not done and t < args.max_episode_steps:
                    sub = chunk[pos:pos + S]
                    k_eff = K
                    conf, q, call = None, True, False

                    if gate is not None:
                        gate_total += 1
                        # TTL: reuse the previous decision unless the planned
                        # hand motion contains a finger transition.
                        call = True
                        if args.gate_ttl_max > 0 and g_last_q is not None:
                            hand_evt = False
                            if args.gate_hand_trigger:
                                hand_evt = bool(descriptors(sub, 0, len(sub), k=K)["hand_change"])
                            if not hand_evt and g_ttl > 0:
                                call = False
                        if call:
                            gate_calls += 1
                            raw = env.get_raw_images()
                            views = [raw[c] for c in cam_order]
                            instr = args.judge_instruction or prompt
                            if args.judge_facts:
                                instr = instr + "\n" + facts_text(descriptors(sub, 0, len(sub), k=K))
                            if args.judge_actions:
                                instr = (f"{instr}\nPlanned ABSOLUTE targets for the next "
                                         f"{len(sub)} control steps "
                                         "[x,y,z, rotvec(3), 16 hand joints]:\n"
                                         + json.dumps(np.round(sub, 3).tolist()))
                            t0 = time.time()
                            jr = gate.judge(views, instr, gate_guidance)
                            gate_ms += (time.time() - t0) * 1000.0
                            # Fail-safe: a judge error dict has no "confidence",
                            # so conf=0 -> q=False -> run raw.
                            conf = float(jr.get("confidence", 0.0))
                            q = conf >= args.judge_threshold
                            if args.gate_ttl_max > 0:
                                d = abs(conf - args.judge_threshold)
                                g_ttl = (0 if d < args.gate_ttl_lo
                                         else 1 if d < args.gate_ttl_hi
                                         else args.gate_ttl_max)
                            g_last_conf, g_last_q = conf, q
                        else:
                            conf, q = g_last_conf, g_last_q
                            g_ttl -= 1

                        if args.action_rules:
                            blk, why = action_rule_block(sub, k=K)
                            if blk and q:
                                q = False
                                rule_blocks += 1
                                rule_reasons[why] = rule_reasons.get(why, 0) + 1

                        gate_confs.append(conf)
                        gate_yes += int(bool(q))
                        k_eff = K if q else 1
                        if q and args.gate_k3_threshold > 0 and conf >= args.gate_k3_threshold:
                            k_eff = 3
                        if gate_log is not None:
                            gate_log.write(f"{i},{t},{conf:.4f},{int(bool(q))},"
                                           f"{int(call)},{prompt!r}\n")
                            gate_log.flush()

                    sub_exec = compress_chunk(sub, k_eff)
                    H_exec = sub_exec.shape[0]
                    chunk_lens.append(H_exec)
                    if k_eff > 1:
                        no, ns, ex = merge_jump_stats(sub, sub_exec, prev_target)
                        merge_over += no
                        merge_steps += ns
                        merge_excess += ex

                    for j in range(H_exec):
                        # np.array copy: websocket deserialization can hand back
                        # read-only arrays, which break scipy R.from_rotvec.
                        a = np.array(sub_exec[j], dtype=np.float64)
                        env.step(a)
                        prev_target = a
                        _frame()
                        t += 1
                        done = env.is_done
                        if done or t >= args.max_episode_steps:
                            break
                    pos += S

            if writers is not None:
                for w in writers.values():
                    w.close()
                suffix = "success" if env.is_success else "failure"
                (video_dir / f"episode_{i:02d}_temp").rename(
                    video_dir / f"episode_{i:02d}_{suffix}")

            succ = bool(env.is_success)
            succ_flags.append(succ)
            flag = "True" if succ else "False"
            with open(pred_path, "a") as f:
                f.write(f"episode {i} is_success: [{flag:>5}] action_steps: {t}\n")
            print(f"  ep {i}: success={succ}, action_steps={t}, K={K}", flush=True)

    finally:
        env.close()
        if gate_log is not None:
            gate_log.close()

    rate = float(np.mean(succ_flags)) if succ_flags else 0.0
    with open(pred_path, "a") as f:
        f.write(f"is_success: {rate:.4f}\n")
        f.write(f"compress_k: {K}\n")
        f.write(f"exec_chunk_len_mean: {np.mean(chunk_lens) if chunk_lens else 0:.2f}\n")
        f.write(f"merge_jump_over: {merge_over}/{merge_steps} "
                f"merge_jump_excess: {merge_excess:.4f}\n")
        if args.action_rules:
            f.write(f"action_rule_blocks: {rule_blocks} {rule_reasons}\n")
        if gate is not None:
            f.write(f"gate_quantize_rate: {gate_yes / max(gate_total, 1):.4f} "
                    f"({gate_yes}/{gate_total})\n")
            f.write(f"gate_threshold: {args.judge_threshold}\n")
            f.write(f"gate_call_rate: {gate_calls / max(gate_total, 1):.4f} "
                    f"({gate_calls}/{gate_total})\n")
            if args.gate_ttl_max > 0:
                f.write(f"gate_ttl: max={args.gate_ttl_max} lo={args.gate_ttl_lo} "
                        f"hi={args.gate_ttl_hi} hand_trigger={args.gate_hand_trigger}\n")
            cs = np.asarray([c for c in gate_confs if c is not None], dtype=float)
            if cs.size:
                f.write(f"gate_conf_mean: {cs.mean():.4f} min: {cs.min():.4f} "
                        f"max: {cs.max():.4f} p50: {np.median(cs):.4f}\n")
    if args.dump_actions and dumped:
        np.savez_compressed(args.dump_actions, chunks=np.stack(dumped))
        print(f"[dump] {len(dumped)} chunks -> {args.dump_actions}", flush=True)
    print(f"[latency] policy {policy_ms:.0f} ms total, gate {gate_ms:.0f} ms total", flush=True)
    print(f"\nSuccess rate: {sum(succ_flags)}/{len(succ_flags)} ({100 * rate:.1f}%)", flush=True)
    (video_dir / f"success_rate_{sum(succ_flags)}_{len(succ_flags)}.txt").touch()


if __name__ == "__main__":
    main()
