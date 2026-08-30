"""LIBERO eval client for MoE per_expert checkpoints.

Supports 4 decision modes (selectable via --args.decision_mode):
  - "as_is":              router pick used directly (head=moe server)
  - "aac_cliff":          server head=moe_selective + N samples;
                          when router picks main, compute step-level AAC h*
                          and compress prefix [0, h*) of main pairs
  - "aac_chunk_binary":   server head=moe_selective + N samples;
                          when main picked, compute h* and binarize against K
                          → use m8 (compressed) iff h*/2 >= K
  - "self_agree_chunk_binary": server head=moe_selective N=1;
                          when main picked, compute per-pair self_agree score
                          (||main_pair_sum - m8||) and compare chunk-mean vs τ
                          → use m8 iff mean(score) < τ

For non-main expert picks (m8/m4/n8), the chunk is used as-is regardless of mode.

replan_steps mapping (action-token units; both compressed decoders are 2x):
  raw 16-step (main)                          → 5  (1 token = 1 env-step)
  raw 8-step (n8)                             → 5  (1 token = 1 env-step, cap at len)
  2x compressed 8-step (m8, sum-pair of 16)   → 2  (1 token = 2 env-steps)
  2x compressed 4-step (m4, sum-pair of [:8]) → 2  (1 token = 2 env-steps)
  AAC cliff variable mix                      → env-step coverage rule
"""
import collections
import dataclasses
import logging
import math
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    # Decision mode (see file docstring)
    decision_mode: str = "as_is"

    # replan_steps per chunk type (in action-token units).
    # Both m8 and m4 are 2x sum-pair compression: 1 token = 2 env-steps. m8 is
    # sum-pair of the 16-step chunk (8 tokens, covers 16), m4 is sum-pair of the
    # first 8 (4 tokens, covers 8). So both compressed decoders use 2 tokens (=4
    # env-steps) per replan; the two raw decoders (1 token = 1 env-step) use 5.
    replan_steps_main: int = 5     # raw 16-step (1 action = 1 env-step)
    replan_steps_m8: int = 2       # 2x compressed (1 action = 2 env-step)
    replan_steps_m4: int = 2       # 2x compressed (1 action = 2 env-step)
    replan_steps_n8: int = 5       # raw 8-step (1 action = 1 env-step, capped by len)
    aac_target_envstep_coverage: int = 5  # for cliff mix mode

    # AAC params (used by aac_cliff / aac_chunk_binary)
    aac_xi: int = 1                # cliff: min compress count; binary: K threshold
    # Where the compressed (8-step) actions come from in cliff / binary modes:
    #   "m8"          : m8 decoder output (trained on sum-pair target). Needs the
    #                   server's extra m8 forward.
    #   "main_sumpair": sum-pair the same main sample we measured entropy on
    #                   (no m8 forward needed, trajectory-consistent).
    # Does NOT affect self_agree (m8 decoder is intrinsic there as the compare ref).
    aac_compress_src: str = "m8"
    # self_agree params
    self_agree_tau: float = 0.5    # chunk_binary threshold on mean disagreement

    # alpha for hybrid (unused for now; here for forward-compat)
    alpha: float = 0.5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "/tmp/libero/videos"
    task_idx: int = 0
    seed: int = 7


def _per_step_entropy_from_samples(samples: np.ndarray, gripper_idx: int = 6) -> np.ndarray:
    """samples: (N, 16, D). Returns (16,) per-step entropy.
       Continuous dims: Gaussian differential entropy; discrete (gripper): Shannon entropy.
       Per AAC paper formulas, summed across keys.
    """
    N, T, D = samples.shape
    LN_2PIE = math.log(2.0 * math.pi * math.e)
    out = np.zeros(T, dtype=np.float64)
    cont_dims = [d for d in range(D) if d != gripper_idx]
    for t in range(T):
        # continuous: cov over N samples at this timestep
        if cont_dims:
            X = samples[:, t, cont_dims]   # (N, len(cont_dims))
            d = X.shape[1]
            if N > d:
                cov = np.cov(X, rowvar=False, ddof=1) + 1e-9 * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = float(np.sum(np.log(np.abs(np.diag(cov)) + 1e-12)))
                out[t] += 0.5 * (d * LN_2PIE + logdet)
            else:
                out[t] += float(np.sum(np.log(X.var(axis=0, ddof=1) + 1e-12)))
        # discrete (gripper)
        col = samples[:, t, gripper_idx]
        # quantize binary: > 0 = open
        binary = (col > 0).astype(np.int32)
        p1 = float(binary.mean()); p0 = 1.0 - p1
        e = 0.0
        for p in (p0, p1):
            if p > 1e-12:
                e -= p * math.log(p)
        out[t] += e
    return out


def _aac_h_star(values: np.ndarray, xi: int = 1) -> int:
    """h* = max(argmax_h(avg[0..h] - avg[0..h-1]), xi)."""
    s = np.asarray(values, dtype=np.float64)
    H = len(s)
    avgs = np.cumsum(s) / np.arange(1, H + 1)
    diffs = avgs[1:] - avgs[:-1]
    if len(diffs) == 0:
        return max(0, xi)
    return max(int(np.argmax(diffs)) + 1, xi)


def _sum_pair_compress(arr16: np.ndarray, gripper_idx: int = 6) -> np.ndarray:
    """(16, D) → (8, D). Continuous: sum-pair. Discrete (gripper): later value."""
    out = np.empty((8, arr16.shape[1]), dtype=arr16.dtype)
    for k in range(8):
        a, b = arr16[2 * k], arr16[2 * k + 1]
        out[k] = a + b
        out[k, gripper_idx] = b[gripper_idx]   # later value for gripper
    return out


def _per_pair_self_agree(main16: np.ndarray, m8: np.ndarray, gripper_idx: int = 6) -> np.ndarray:
    """(8,) per-pair disagreement: continuous L2 + discrete mismatch."""
    cont = [d for d in range(main16.shape[1]) if d != gripper_idx]
    s = np.zeros(8, dtype=np.float32)
    for k in range(8):
        a, b = main16[2 * k], main16[2 * k + 1]
        s[k] += float(np.linalg.norm(a[cont] + b[cont] - m8[k, cont]))
        s[k] += float(b[gripper_idx] != m8[k, gripper_idx])
    return s


def _build_chunk_aac_cliff(samples_main: np.ndarray, m8: np.ndarray,
                           xi: int, gripper_idx: int = 6, compress_src: str = "m8"):
    """Step-level AAC h* on samples_main (N, 16, D); compress prefix h*/2 pairs.
       Returns (chunk, n_compressed_pairs, n_raw_actions).
       compress_src: 'm8' uses the m8 decoder output; 'main_sumpair' sum-pairs
       the same main sample (no m8 forward needed)."""
    step_E = _per_step_entropy_from_samples(samples_main, gripper_idx=gripper_idx)
    h_step = _aac_h_star(step_E, xi=2 * xi)
    h_pair = h_step // 2
    h_pair = max(min(h_pair, 8), 0)
    main = samples_main[0]  # (16, D)
    comp8 = m8 if compress_src == "m8" else _sum_pair_compress(main, gripper_idx)
    if h_pair == 0:
        return main, 0, 16
    if h_pair == 8:
        return comp8.copy(), 8, 0
    # mix: first h_pair pairs compressed, remaining (8 - h_pair) pairs raw from main
    pieces = [comp8[:h_pair]]
    pieces.append(main[2 * h_pair:])
    chunk = np.concatenate(pieces, axis=0)
    return chunk, h_pair, 16 - 2 * h_pair


def _build_chunk_aac_binary(samples_main: np.ndarray, m8: np.ndarray, K: int,
                            compress_src: str = "m8", gripper_idx: int = 6):
    """Compress all-or-nothing iff h* (pair) >= K.
       compress_src: 'm8' or 'main_sumpair' (sum-pair the main sample)."""
    step_E = _per_step_entropy_from_samples(samples_main)
    h_step = _aac_h_star(step_E, xi=0)
    h_pair = h_step // 2
    comp8 = m8 if compress_src == "m8" else _sum_pair_compress(samples_main[0], gripper_idx)
    if h_pair >= K:
        return comp8.copy(), True
    return samples_main[0], False


def _build_chunk_self_agree(main16: np.ndarray, m8: np.ndarray, tau: float):
    s = _per_pair_self_agree(main16, m8)
    if float(s.mean()) < tau:
        return m8.copy(), True
    return main16, False


def _normalize01(s: np.ndarray) -> np.ndarray:
    """Per-chunk min-max normalize to [0, 1] for hybrid combination."""
    s = np.asarray(s, dtype=np.float32)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def _hybrid_pair_scores(samples_main: np.ndarray, m8: np.ndarray,
                        alpha: float, gripper_idx: int = 6) -> np.ndarray:
    """(8,) hybrid per-pair score = α·norm(entropy_pair) + (1-α)·norm(self_agree).
       entropy_per_step is averaged within each adjacent pair."""
    step_E = _per_step_entropy_from_samples(samples_main, gripper_idx=gripper_idx)
    ent_pair = 0.5 * (step_E[0::2] + step_E[1::2])           # (8,)
    sa_pair = _per_pair_self_agree(samples_main[0], m8, gripper_idx=gripper_idx)  # (8,)
    return alpha * _normalize01(ent_pair) + (1.0 - alpha) * _normalize01(sa_pair)


def _build_chunk_aac_cliff_hybrid(samples_main: np.ndarray, m8: np.ndarray,
                                  xi: int, alpha: float, gripper_idx: int = 6,
                                  compress_src: str = "m8"):
    """AAC cliff over hybrid (entropy + self_agree) per-pair score.
       Hybrid score always needs m8 (self_agree term); compress_src only
       controls where the compressed action values come from."""
    pair_score = _hybrid_pair_scores(samples_main, m8, alpha, gripper_idx)
    h_pair = max(_aac_h_star(pair_score, xi=xi), 0)
    h_pair = min(h_pair, 8)
    main = samples_main[0]
    comp8 = m8 if compress_src == "m8" else _sum_pair_compress(main, gripper_idx)
    if h_pair == 0:
        return main, 0, 16
    if h_pair == 8:
        return comp8.copy(), 8, 0
    pieces = [comp8[:h_pair], main[2 * h_pair:]]
    chunk = np.concatenate(pieces, axis=0)
    return chunk, h_pair, 16 - 2 * h_pair


def _build_chunk_aac_binary_hybrid(samples_main: np.ndarray, m8: np.ndarray,
                                   K: int, alpha: float, gripper_idx: int = 6,
                                   compress_src: str = "m8"):
    """All-or-nothing on hybrid pair score's h* >= K."""
    pair_score = _hybrid_pair_scores(samples_main, m8, alpha, gripper_idx)
    h_pair = _aac_h_star(pair_score, xi=0)
    comp8 = m8 if compress_src == "m8" else _sum_pair_compress(samples_main[0], gripper_idx)
    if h_pair >= K:
        return comp8.copy(), True
    return samples_main[0], False


def _decide_replan(chunk_kind: str, args: Args, chunk_len: int) -> int:
    if chunk_kind == "main_raw":      return min(args.replan_steps_main, chunk_len)
    if chunk_kind == "m8_compressed": return min(args.replan_steps_m8, chunk_len)
    if chunk_kind == "n8_raw":        return min(args.replan_steps_n8, chunk_len)
    if chunk_kind == "m4_compressed": return min(args.replan_steps_m4, chunk_len)
    if chunk_kind == "aac_cliff_mix" or chunk_kind.startswith("per_quad_mix"):
        # Walk forward, accumulate env-step coverage; stop ~target.
        return min(max(2, args.aac_target_envstep_coverage), chunk_len)
    return min(args.replan_steps_main, chunk_len)


def _per_quad_replan_actions(picks, target_envstep: int) -> tuple:
    """Given per_quad_mask picks (length Q ints, c=0 main / c=1 m8 / c=2 m4),
    walk through actions in chunk order, accumulating env-step coverage. Return
    (n_actions_to_execute, kind_label).

    Per-quad action counts × per-action env-step coverage:
      c=0 (main 4-raw): 4 actions × 1 env-step
      c=1 (m8 2-comp ): 2 actions × 2 env-steps
      c=2 (m4 1-deep ): 1 action  × 4 env-steps
    """
    coverage_per_action = []
    for c in picks:
        c = int(c)
        if c == 0:   coverage_per_action.extend([1] * 4)
        elif c == 1: coverage_per_action.extend([2] * 2)
        else:        coverage_per_action.extend([4] * 1)  # c=2
    cum = 0
    n_use = 0
    for cov in coverage_per_action:
        cum += cov
        n_use += 1
        if cum >= target_envstep:
            break
    counts = [int(sum(1 for c in picks if int(c) == k)) for k in (0, 1, 2)]
    kind = f"per_quad_mix_main{counts[0]}_m8{counts[1]}_m4{counts[2]}"
    return n_use, kind


def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_suite_name == "libero_spatial":     max_steps = 220
    elif args.task_suite_name == "libero_object":    max_steps = 280
    elif args.task_suite_name == "libero_goal":      max_steps = 300
    elif args.task_suite_name == "libero_10":        max_steps = 520
    elif args.task_suite_name == "libero_90":        max_steps = 400
    else: raise ValueError(f"Unknown {args.task_suite_name}")

    client = None
    total_episodes = total_successes = 0
    pick_counts = collections.Counter()
    chunk_kind_counts = collections.Counter()
    ep_records = []  # list of (episode_idx, success_bool, action_steps)

    for task_id in [args.task_idx]:
        if task_id >= num_tasks_in_suite:
            break
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        logging.info(f"Task {task_id}: {task_description}")
        task_episodes = task_successes = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if client is None:
                client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"],
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]),
                            "state.gripper_close": obs["robot0_gripper_qpos"],
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        out = client.infer(element)

                        # Detect router pick. `_moe_picked` is now a length-1 array
                        # for single_pick and length-Q array for per_quad_mask.
                        picked = None
                        picked_arr = None
                        if "_moe_picked" in out:
                            picked_arr = np.asarray(out["_moe_picked"]).flatten()
                            if picked_arr.size == 1:
                                picked = int(picked_arr[0])
                        pick_counts[picked] += 1

                        # Helper to assemble chunk dict → (T, 7) array
                        def _flat(prefix=""):
                            ep = out.get(f"action.eef_pos_delta{prefix}")
                            er = out.get(f"action.eef_rot_delta{prefix}")
                            gp = out.get(f"action.gripper_close{prefix}")
                            if ep is None: return None
                            ep = np.asarray(ep); er = np.asarray(er); gp = np.asarray(gp)
                            if gp.ndim == 1: gp = gp.reshape(-1, 1)
                            # Handle batched (N, T, D) shape from selective head's main key
                            if ep.ndim == 3:
                                # (N, T, 3) — keep batch dim, will be (N, T, 7) after stack
                                return np.concatenate([ep, er, gp[..., None] if gp.ndim == 2 else gp], axis=-1)
                            return np.concatenate([ep, er, gp], axis=-1)

                        main_block = _flat("")
                        m8_block = _flat("_m8")

                        # Branch on router pick + decision mode
                        chunk = None
                        kind = "main_raw"
                        if picked_arr is not None and picked_arr.size > 1:
                            # per_quad_mask: chunk is a mix of slices from main/m8/m4
                            # decoders. Each action has a different env-step coverage,
                            # so replan_steps must be computed in env-step units.
                            chunk = main_block if main_block.ndim == 2 else main_block[0]
                            n_use, kind = _per_quad_replan_actions(
                                picked_arr, args.aac_target_envstep_coverage)
                            chunk_kind_counts[kind] += 1
                            action_plan.extend(chunk[:n_use])
                            # Skip the standard decision_mode flow; per_quad_mask is
                            # self-contained and the replan target is env-step based.
                            action = action_plan.popleft()
                            obs, reward, done, info = env.step(action.tolist())
                            if done:
                                task_successes += 1; total_successes += 1
                                break
                            t += 1
                            continue
                        if picked is not None and picked != 0:
                            # Non-main expert picked → use as-is
                            chunk = main_block if main_block.ndim == 2 else main_block[0]
                            T = chunk.shape[0]
                            if picked == 1:   kind = "m8_compressed"
                            elif picked == 2: kind = "m4_compressed"
                            elif picked == 3: kind = "n8_raw"
                            else:             kind = "main_raw"
                        else:
                            # Main picked or non-MoE: apply decision mode
                            if main_block.ndim == 3:
                                samples_main = main_block       # (N, 16, 7)
                                main16 = samples_main[0]
                            else:
                                samples_main = main_block[None]
                                main16 = main_block

                            if args.decision_mode == "as_is":
                                chunk = main16
                                kind = "main_raw"
                            elif args.decision_mode == "aac_cliff":
                                if samples_main.shape[0] < 2:
                                    chunk = main16; kind = "main_raw"
                                else:
                                    if m8_block is None: m8_block = _sum_pair_compress(main16)
                                    chunk, n_comp, n_raw = _build_chunk_aac_cliff(samples_main, m8_block, args.aac_xi, compress_src=args.aac_compress_src)
                                    if n_comp == 0:    kind = "main_raw"
                                    elif n_raw == 0:   kind = "m8_compressed"
                                    else:              kind = "aac_cliff_mix"
                            elif args.decision_mode == "aac_chunk_binary":
                                if samples_main.shape[0] < 2 or m8_block is None:
                                    chunk = main16; kind = "main_raw"
                                else:
                                    chunk, used_m8 = _build_chunk_aac_binary(samples_main, m8_block, args.aac_xi, compress_src=args.aac_compress_src)
                                    kind = "m8_compressed" if used_m8 else "main_raw"
                            elif args.decision_mode == "self_agree":
                                if m8_block is None:
                                    chunk = main16; kind = "main_raw"
                                else:
                                    chunk, used_m8 = _build_chunk_self_agree(main16, m8_block, args.self_agree_tau)
                                    kind = "m8_compressed" if used_m8 else "main_raw"
                            elif args.decision_mode == "aac_cliff_hybrid":
                                if samples_main.shape[0] < 2 or m8_block is None:
                                    chunk = main16; kind = "main_raw"
                                else:
                                    chunk, n_comp, n_raw = _build_chunk_aac_cliff_hybrid(
                                        samples_main, m8_block, args.aac_xi, args.alpha, compress_src=args.aac_compress_src)
                                    if n_comp == 0:    kind = "main_raw"
                                    elif n_raw == 0:   kind = "m8_compressed"
                                    else:              kind = "aac_cliff_mix"
                            elif args.decision_mode == "aac_chunk_binary_hybrid":
                                if samples_main.shape[0] < 2 or m8_block is None:
                                    chunk = main16; kind = "main_raw"
                                else:
                                    chunk, used_m8 = _build_chunk_aac_binary_hybrid(
                                        samples_main, m8_block, args.aac_xi, args.alpha, compress_src=args.aac_compress_src)
                                    kind = "m8_compressed" if used_m8 else "main_raw"
                            else:
                                raise ValueError(f"Unknown decision_mode {args.decision_mode}")

                        # Pick replan steps based on chunk kind & length
                        chunk_kind_counts[kind] += 1
                        rs = _decide_replan(kind, args, chunk.shape[0])
                        action_plan.extend(chunk[:rs])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1; total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    raise

            task_episodes += 1
            total_episodes += 1
            action_steps = max(0, t - args.num_steps_wait)
            ep_records.append((episode_idx, bool(done), int(action_steps)))
            suffix = "success" if done else "failure"
            seg = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{seg}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images], fps=30,
            )
            logging.info(f"  Episode {episode_idx}: {'OK' if done else 'FAIL'}; steps={action_steps}; {total_successes}/{total_episodes}")

        logging.info(f"Task success: {task_successes}/{task_episodes}")

    logging.info(f"Total: {total_successes}/{total_episodes} = {total_successes/max(total_episodes,1):.4f}")
    logging.info(f"Pick counts: {dict(pick_counts)}")
    logging.info(f"Chunk kind counts: {dict(chunk_kind_counts)}")

    succ_steps = [s for (_, ok, s) in ep_records if ok]
    succ_only_mean = (sum(succ_steps) / len(succ_steps)) if succ_steps else 0.0
    rp = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    with open(rp, "w") as f:
        f.write(f"Total success rate: {total_successes/max(total_episodes,1):.4f}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Decision mode: {args.decision_mode}\n")
        f.write(f"Pick counts: {dict(pick_counts)}\n")
        f.write(f"Chunk kind counts: {dict(chunk_kind_counts)}\n")
        f.write(f"Success-only mean steps: {succ_only_mean:.2f} (over {len(succ_steps)} ep)\n")
        f.write("Per-ep records (idx, success, action_steps):\n")
        for rec in ep_records:
            f.write(f"  {rec[0]}\t{rec[1]}\t{rec[2]}\n")
    logging.info(f"Results saved to {rp}")
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
