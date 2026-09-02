import os
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    # Naive fixed-K action quantization applied client-side to the 16-step chunk:
    # continuous (eef pos/rot delta, dims 0:6) are block-summed, the gripper
    # (dim 6, latching) takes the block's last value. K=1 disables quantization.
    compress_k: int = 1

    # ---- Optional VLM gate (per-chunk quantize decision) -------------------
    # If judge_url is set, query the gate per action chunk with the current
    # camera views + task language + guidance; quantize with compress_k when
    # P(YES) >= judge_threshold, else run that chunk raw (K=1). gate_out_dir,
    # when set, also writes an evolver-compatible prediction.txt + gate_conf.csv
    # under <gate_out_dir>/<suite>_<task_idx>/ for the self-evolve loop.
    judge_url: str = ""
    judge_threshold: float = 0.5
    judge_guidance: str = ""      # raw text, or @/path/to/guidance.txt
    # Q1-(1) masking test: what the JUDGE sees as the instruction.
    # "" = real task description (default); "__empty__" = blank;
    # any other string = that literal (irrelevant-task test).
    judge_instruction_override: str = ""
    gate_out_dir: str = ""
    # ---- TTL skip policy (fewer judge calls; the async 2-GPU setting already
    # hides per-call latency, this cuts the number of calls). At each judge
    # opportunity: a gripper open<->close transition inside the fresh action
    # chunk (a *predictive* free signal - the VLA has already forecast the next
    # 16 steps) always forces a real call; otherwise while ttl>0 the previous
    # decision is reused. After a real call: ttl=0 if |conf-tau|<ttl_lo
    # (ambiguous -> re-judge next time), 1 if <ttl_hi, else ttl_max.
    # gate_ttl_max=0 disables the policy (call every opportunity, as before).
    # Replay-calibrated defaults: gemma lo/hi=0.15/0.30, cosmos 0.05/0.15.
    gate_ttl_max: int = 0
    gate_ttl_lo: float = 0.15
    gate_ttl_hi: float = 0.30
    gate_gripper_trigger: bool = True
    # ---- magnitude-aware VARIABLE-K quantization. The robosuite OSC controller
    # clips each env-step action to [-1,1] per dim (mapping to <=5cm / 0.5rad),
    # so a fixed-K block-sum beyond that bound silently loses displacement.
    # With vark_bound>0, consecutive deltas are merged GREEDILY while the running
    # block sum stays within vark_bound on every continuous dim (and the block
    # stays <= compress_k steps); a gripper transition always breaks the block.
    # 0 disables (fixed-K behaviour). Recommended 0.95 (safety margin under 1.0).
    vark_bound: float = 0.0
    # floor-2 variant: always merge pairs (K2 floor), extend to 3..kmax only
    # within vark_bound -> steps <= fixed-K2 by construction.
    vark_floor2: int = 0
    # ---- EXPERIMENTAL: scale the controller's action clip bounds (input_max/min
    # AND output_max/min) by this factor, raising the per-env-step displacement
    # cap (default 5cm/0.5rad) so K-sums need not clip. CHANGES BENCHMARK
    # PHYSICS — results form a separate "modified-controller" track, not
    # comparable to standard numbers. 1.0 = untouched.
    clip_scale: float = 1.0
    # dynamics unlock: torque caps x this, kp x this (kd x sqrt). 1.0 = off.
    dyn_scale: float = 1.0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    task_idx: int = -1  # Task index to run

    seed: int = 7  # Random Seed (for reproducibility)


def _patch_clip_bounds(env, scale):
    """Scale the robosuite controller clip bounds by `scale` (both input and
    output ranges, so per-unit displacement is unchanged but the per-step cap
    rises). Works for both classic single controllers (robot.controller) and
    composite part controllers. Returns number of controllers patched."""
    base = env
    for _ in range(8):
        if hasattr(base, "robots"):
            break
        base = getattr(base, "env", base)
    n = 0
    for r in getattr(base, "robots", []):
        ctrls = []
        c = getattr(r, "controller", None)
        if c is not None:
            ctrls.append(c)
        cc = getattr(r, "composite_controller", None)
        if cc is not None:
            pc = getattr(cc, "part_controllers", None) or getattr(cc, "controllers", None) or {}
            ctrls += list(pc.values() if hasattr(pc, "values") else pc)
        for c in ctrls:
            hit = False
            for a in ("input_max", "input_min", "output_max", "output_min"):
                if hasattr(c, a) and getattr(c, a) is not None:
                    setattr(c, a, getattr(c, a) * scale)
                    hit = True
            n += int(hit)
    return n



def _patch_dynamics(env, torque_scale, kp_scale):
    """EXPERIMENTAL round-2: raise the robot's DYNAMICS limits — actuator torque
    caps (controller actuator_min/max + mujoco forcerange) x torque_scale, and
    OSC stiffness kp x kp_scale (kd x sqrt for near-critical damping). This is
    the second clip layer + responsiveness bound behind the command clip."""
    import numpy as _np
    base = env
    for _ in range(10):
        if hasattr(base, "robots"):
            break
        base = getattr(base, "env", base)
    n = 0
    for r in getattr(base, "robots", []):
        ctrls = []
        c = getattr(r, "controller", None)
        if c is not None:
            ctrls.append(c)
        cc = getattr(r, "composite_controller", None)
        if cc is not None:
            pc = getattr(cc, "part_controllers", None) or getattr(cc, "controllers", None) or {}
            ctrls += list(pc.values() if hasattr(pc, "values") else pc)
        for c in ctrls:
            hit = False
            for a in ("actuator_min", "actuator_max"):
                if hasattr(c, a) and getattr(c, a) is not None:
                    setattr(c, a, getattr(c, a) * torque_scale); hit = True
            if hasattr(c, "kp") and getattr(c, "kp") is not None:
                c.kp = c.kp * kp_scale; hit = True
            if hasattr(c, "kd") and getattr(c, "kd") is not None:
                c.kd = c.kd * (kp_scale ** 0.5); hit = True
            n += int(hit)
    sim = getattr(base, "sim", None)
    if sim is not None and hasattr(sim, "model") and hasattr(sim.model, "actuator_forcerange"):
        fr = sim.model.actuator_forcerange
        fr[:] = fr * torque_scale
        n += 1
    return n


def _vark_compress(chunk, kmax, bound, floor2=False):
    """Greedy magnitude-aware variable-K block merge of a (T,7) action chunk.

    Merges consecutive steps while (a) the block has < kmax steps, (b) the
    running sum of the 6 continuous delta dims stays within `bound` (the OSC
    controller clips at 1.0 -> merged displacement would be lost beyond it),
    and (c) no gripper transition is crossed. Continuous dims are summed,
    the gripper (dim 6) latches to the block's last value. Total displacement
    is preserved exactly; only the time partition changes.
    """
    blocks, i, T = [], 0, chunk.shape[0]
    while i < T:
        acc = chunk[i].copy()
        j = i + 1
        if floor2 and j < T:
            # floor-2: merge the pair UNCONDITIONALLY (exact fixed-K2 semantics,
            # per-dim clipping accepted downstream, gripper latches) -> the
            # block count can never exceed fixed-K2's, guaranteeing steps <= K2.
            acc[:6] = acc[:6] + chunk[j][:6]
            acc[6] = chunk[j][6]
            j += 1
        while j < T and (j - i) < kmax:
            if abs(chunk[j][6] - chunk[j - 1][6]) > 0.5:
                break                      # gripper event: never merge across it
            cand = acc[:6] + chunk[j][:6]
            if np.abs(cand).max() > bound:
                break                      # would clip -> stop merging here
            acc[:6] = cand
            acc[6] = chunk[j][6]           # gripper latch
            j += 1
        blocks.append(acc)
        i = j
    return np.stack(blocks)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, idx: {args.task_idx}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    logging.info(f"Number of trials per task: {args.num_trials_per_task}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial" or args.task_suite_name == "libero_spatial_new" or args.task_suite_name == "libero_spatial_ood":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object" or args.task_suite_name == "libero_object_ood":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal" or args.task_suite_name == "libero_goal_new" or args.task_suite_name == "libero_goal_ood":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    ep_records = []  # (episode_idx, success_bool, action_steps)

    # ---- Optional VLM gate setup ------------------------------------------
    gate = None
    gate_guidance = ""
    gate_rows = []          # (episode_idx, step, conf, quantize, called)
    gate_yes = 0
    gate_tot = 0
    gate_calls = 0          # real VLM calls (gate_tot includes TTL reuses)
    if args.judge_url:
        import sys as _sys
        _sys.path.insert(0, os.path.expanduser("~/quantization_agent_workspace/vlm_gate/scripts"))
        from vlm_gate import VLMGate
        gate = VLMGate(args.judge_url)
        g = args.judge_guidance
        if g.startswith("@"):
            try:
                gate_guidance = open(g[1:]).read()
            except Exception:
                gate_guidance = ""
        else:
            gate_guidance = g
        logging.info(f"[gate] VLM gate ON url={args.judge_url} K={args.compress_k} tau={args.judge_threshold}")
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.task_idx != -1 and task_id != args.task_idx:
            continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        if args.clip_scale != 1.0:
            _npatched = _patch_clip_bounds(env, args.clip_scale)
            logging.info(f"[clip] controller clip bounds x{args.clip_scale} ({_npatched} controllers patched)")
        if args.dyn_scale != 1.0:
            _nd = _patch_dynamics(env, args.dyn_scale, args.dyn_scale)
            logging.info(f"[dyn] torque/kp x{args.dyn_scale} ({_nd} targets patched)")

        # Start episodes
        task_episodes, task_successes = 0, 0
        # ---- Preemption-safe resume (background partition requeues jobs) -------
        # On requeue the replay videos may already exist; the old code skipped
        # those episodes WITHOUT re-recording their per-episode metrics, so a
        # fully-resumed task wrote an EMPTY prediction.txt (lost steps + gate
        # quant decisions). We persist each episode's record to ep_records.jsonl
        # the instant it finishes and reload it here, so resume reconstructs every
        # aggregate exactly once and prediction.txt is always complete.
        import json as _json
        sidecar = None
        done_idx = set()
        # DIAG_LOG=1: per-env-step commanded-vs-achieved EE displacement sidecar
        # (recovery-vs-clipping causal check for the K-ladder saturation).
        _diag = None
        if args.gate_out_dir and args.task_idx != -1:
            _td0 = pathlib.Path(args.gate_out_dir) / f"{args.task_suite_name}_{args.task_idx}"
            _td0.mkdir(parents=True, exist_ok=True)
            sidecar = _td0 / "ep_records.jsonl"
            if os.environ.get("DIAG_LOG") == "1":
                _diag = open(_td0 / "diag_steps.jsonl", "a", buffering=1)
            if sidecar.exists():
                for _line in open(sidecar):
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _r = _json.loads(_line)
                    except Exception:
                        continue
                    _i = int(_r["idx"])
                    if _i in done_idx:
                        continue
                    done_idx.add(_i)
                    ep_records.append((_i, bool(_r["ok"]), int(_r["steps"])))
                    for _gr in _r.get("rows", []):
                        # rows are [step, conf, q] (pre-TTL) or [step, conf, q, called]
                        gate_rows.append((_i, _gr[0], _gr[1], _gr[2],
                                          _gr[3] if len(_gr) > 3 else 1))
                    gate_yes += int(_r.get("qyes", 0)); gate_tot += int(_r.get("qtot", 0))
                    gate_calls += int(_r.get("qcalls", _r.get("qtot", 0)))
                    total_episodes += 1; task_episodes += 1
                    if bool(_r["ok"]):
                        total_successes += 1; task_successes += 1
                if done_idx:
                    logging.info(f"[resume] loaded {len(done_idx)} episodes from {sidecar.name}")
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            task_segment = task_description.replace(" ", "_")
            # Resume skip is driven by the durable sidecar (loaded above), NOT by
            # video existence: a video can exist without its per-episode record
            # having been flushed, in which case we re-run that single episode.
            if episode_idx in done_idx:
                logging.info(f"Episode {episode_idx} already recorded, skipping...")
                continue
            if client is None:
                client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

            # Reset environment
            env.reset()
            # robosuite reset() RE-CREATES robots/controllers, wiping any
            # clip/dynamics patches applied at env construction — re-apply
            # them after every reset or the scales silently do nothing.
            if args.clip_scale != 1.0:
                _patch_clip_bounds(env, args.clip_scale)
            if args.dyn_scale != 1.0:
                _patch_dynamics(env, args.dyn_scale, args.dyn_scale)
            action_plan = collections.deque()
            # Per-episode TTL-policy state (fresh each episode)
            _g_ttl = 0
            _g_last_conf = None
            _g_last_q = None
            _g_prev_grip = None

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])


            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # No need to resize for GR00T 
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )

                    #"""  # Save images for debugging
                    if t == args.num_steps_wait and episode_idx == 0:
                        task_segment = task_description.replace(" ", "_")
                        image_prefix = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}"
                        image_prefix.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(f"{image_prefix}_img.png", img)
                        imageio.imwrite(f"{image_prefix}_wrist.png", wrist_img)
                    #"""

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        _blk_spans = None
                        # Prepare observations dict

                        # element = {
                        #     "observation/image": img,
                        #     "observation/wrist_image": wrist_img,
                        #     "observation/state": np.concatenate(
                        #         (
                        #             obs["robot0_eef_pos"],
                        #             _quat2axisangle(obs["robot0_eef_quat"]),
                        #             obs["robot0_gripper_qpos"],
                        #         )
                        #     ),
                        #     **({"previous_actions": prev_actions_vec,
                        #         "previous_state": prev_state_vec,
                        #         "observation/previous_image": prev_img,
                        #         "observation/previous_wrist_image": prev_wrist_img} if args.use_previous_info else {}),
                        #     "prompt": str(task_description),
                        # }
                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"], # GR00T requries [horizon, state_dim]
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]),
                            "state.gripper_close": obs["robot0_gripper_qpos"],
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        # Query model to get action
                        action_chunk = client.infer(element)
                        action_chunk = np.concatenate([
                            action_chunk['action.eef_pos_delta'],
                            action_chunk['action.eef_rot_delta'],
                            action_chunk['action.gripper_close'].reshape(-1, 1)
                        ], axis=-1)
                        # Gate decision: quantize with compress_k when the VLM
                        # gate says YES (conf>=tau), else run this chunk raw.
                        # With gate_ttl_max>0 the TTL skip policy reuses the
                        # previous decision instead of calling the VLM, unless a
                        # gripper transition inside the fresh chunk forces a call.
                        K_eff = args.compress_k
                        if gate is not None:
                            call = True
                            if args.gate_ttl_max > 0 and _g_last_q is not None:
                                grip_evt = False
                                if args.gate_gripper_trigger:
                                    _gseq = action_chunk[:, 6]
                                    if _g_prev_grip is not None:
                                        _gseq = np.concatenate([[_g_prev_grip], _gseq])
                                    grip_evt = bool(np.abs(np.diff(_gseq)).max() > 0.5)
                                if not grip_evt and _g_ttl > 0:
                                    call = False
                            if call:
                                ji = str(task_description)
                                if args.judge_instruction_override == "__empty__":
                                    ji = ""
                                elif args.judge_instruction_override:
                                    ji = args.judge_instruction_override
                                res = gate.judge([img, wrist_img], ji, gate_guidance)
                                conf = float(res.get("confidence", 0.0))
                                q = conf >= args.judge_threshold
                                _g_last_conf, _g_last_q = conf, q
                                if args.gate_ttl_max > 0:
                                    _d = abs(conf - args.judge_threshold)
                                    _g_ttl = (0 if _d < args.gate_ttl_lo
                                              else 1 if _d < args.gate_ttl_hi
                                              else args.gate_ttl_max)
                            else:
                                conf, q = _g_last_conf, _g_last_q
                                _g_ttl -= 1
                            _g_prev_grip = float(action_chunk[-1, 6])
                            gate_rows.append((episode_idx, t, conf, int(q), int(call)))
                            gate_yes += int(q); gate_tot += 1; gate_calls += int(call)
                            K_eff = args.compress_k if q else 1
                        # Naive fixed-K block quantization (client-side). Continuous
                        # dims 0:6 are delta -> block-sum; dim 6 (gripper) latches ->
                        # block-last. Remainder steps (T mod K) are kept raw.
                        if K_eff > 1 and args.vark_bound > 0:
                            action_chunk = _vark_compress(action_chunk, K_eff, args.vark_bound, floor2=args.vark_floor2 > 0)
                        elif K_eff > 1:
                            K = K_eff
                            T = action_chunk.shape[0]
                            nfull = T // K
                            blocks = []
                            for i in range(nfull):
                                blk = action_chunk[i * K:(i + 1) * K]
                                agg = blk.sum(axis=0)
                                agg[6] = blk[-1, 6]   # gripper latch
                                blocks.append(agg)
                            spans = [K] * nfull
                            for j in range(nfull * K, T):
                                blocks.append(action_chunk[j])  # raw tail
                                spans.append(1)
                            action_chunk = np.stack(blocks)
                            _blk_spans = spans
                        # replan_steps counts RAW timesteps, not compressed blocks.
                        # Slicing the compressed chunk directly let one block stand for
                        # K raw steps, so K2 replanned every 10 raw steps and K4 every
                        # 20 -- the horizon grew with the compression instead of staying
                        # at LIBERO's 5, and what was measured as the cost of compression
                        # was partly the cost of replanning less often. Take whole blocks
                        # until their raw span reaches replan_steps, always at least one.
                        spans = _blk_spans if _blk_spans else [1] * len(action_chunk)
                        rs, raw = 0, 0
                        for sp in spans:
                            if rs and raw + sp > args.replan_steps:
                                break
                            rs += 1; raw += sp
                        action_plan.extend(action_chunk[: rs])

                    action = action_plan.popleft()

                    # Execute action in environment
                    if _diag is not None:
                        _pre = np.asarray(obs["robot0_eef_pos"], dtype=float)
                    obs, reward, done, info = env.step(action.tolist())
                    if _diag is not None:
                        _post = np.asarray(obs["robot0_eef_pos"], dtype=float)
                        _diag.write(_json.dumps({
                            "ep": episode_idx, "t": t,
                            "cmd": [round(float(v), 5) for v in action[:3]],
                            "ach": [round(float(v), 5) for v in (_post - _pre)],
                            "pos": [round(float(v), 5) for v in _post],
                        }) + "\n")
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    assert False
                    break

            task_episodes += 1
            total_episodes += 1
            action_steps = max(0, t - args.num_steps_wait)
            ep_records.append((episode_idx, bool(done), int(action_steps)))
            # Durably record this episode (preemption-safe) BEFORE the video write
            # so a requeue never loses its steps / gate-quant decisions.
            if sidecar is not None:
                _erows = [[gr[1], gr[2], gr[3], gr[4]] for gr in gate_rows if gr[0] == episode_idx]
                with open(sidecar, "a") as _sf:
                    _sf.write(_json.dumps({
                        "idx": int(episode_idx), "ok": bool(done), "steps": int(action_steps),
                        "qyes": int(sum(r[2] for r in _erows)), "qtot": int(len(_erows)),
                        "qcalls": int(sum(r[3] for r in _erows)),
                        "rows": _erows,
                    }) + "\n")
                    _sf.flush()

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=30,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    succ_steps = [s for (_, ok, s) in ep_records if ok]
    succ_only_mean = (sum(succ_steps) / len(succ_steps)) if succ_steps else 0.0
    result_save_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    with open(result_save_path, "w") as f:
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Success-only mean steps: {succ_only_mean:.2f} (over {len(succ_steps)} ep)\n")
        f.write("Per-ep records (idx, success, action_steps):\n")
        for rec in ep_records:
            f.write(f"  {rec[0]}\t{rec[1]}\t{rec[2]}\n")
    logging.info(f"Results saved to {result_save_path}")

    # Evolver-compatible per-task output (prediction.txt + gate_conf.csv), so the
    # self-evolve loop can consume LIBERO runs the same way as robocasa.
    if args.gate_out_dir and args.task_idx != -1:
        import statistics as _st
        td = pathlib.Path(args.gate_out_dir) / f"{args.task_suite_name}_{args.task_idx}"
        td.mkdir(parents=True, exist_ok=True)
        sr = (float(total_successes) / float(total_episodes)) if total_episodes else 0.0
        with open(td / "prediction.txt", "w") as f:
            for (idx, ok, stp) in ep_records:
                f.write(f"episode {idx} is_success: [{' True' if ok else 'False'}] action_steps: {stp}\n")
            f.write(f"is_success: {sr:.4f}\n")
            f.write(f"compress_k: {args.compress_k}\n")
            if args.vark_bound > 0:
                f.write(f"vark_bound: {args.vark_bound}\n")
            if args.vark_floor2:
                f.write(f"vark_floor2: 1\n")
            if args.clip_scale != 1.0:
                f.write(f"clip_scale: {args.clip_scale}\n")
            if args.dyn_scale != 1.0:
                f.write(f"dyn_scale: {args.dyn_scale}\n")
            if gate is not None and gate_tot:
                f.write(f"gate_quantize_rate: {gate_yes / gate_tot:.4f} ({gate_yes}/{gate_tot})\n")
                f.write(f"gate_threshold: {args.judge_threshold}\n")
                f.write(f"gate_call_rate: {gate_calls / gate_tot:.4f} ({gate_calls}/{gate_tot})\n")
                if args.gate_ttl_max > 0:
                    f.write(f"gate_ttl: max={args.gate_ttl_max} lo={args.gate_ttl_lo} "
                            f"hi={args.gate_ttl_hi} gripper_trigger={int(args.gate_gripper_trigger)}\n")
                cs = [r[2] for r in gate_rows]
                if cs:
                    f.write(f"gate_conf_mean: {_st.mean(cs):.4f} min: {min(cs):.4f} "
                            f"max: {max(cs):.4f} p50: {_st.median(cs):.4f}\n")
        with open(td / "gate_conf.csv", "w") as f:
            f.write("episode,step,conf,quantize,called,instruction\n")
            for (idx, stp, conf, q, called) in gate_rows:
                f.write(f"{idx},{stp},{conf:.4f},{q},{called},{str(task_description)!r}\n")
        logging.info(f"Evolver-format output saved to {td}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
    # Force-exit. The libero OffScreenRenderEnv / openpi websocket client can keep
    # this process alive after all results are written (every output uses a `with`
    # block, so it is already flushed+closed to disk by here). Without this, the
    # process lingers and the parent eval script's `wait` blocks forever, leaving
    # the sbatch job RUNNING until the time limit. All output is on disk, so exit hard.
    os._exit(0)
