"""Robocasa eval client with SELECTIVE pair-wise compression.

Server returns (main 16-step, m8 8-step) via head=main_and_m8.
Client decides PER-PAIR whether to compress (sum-pair → 1 exec action)
or keep raw (2 exec actions) → variable-length chunk between 8 and 16 actions
covering 16 env steps' worth of trajectory.

Compression criterion (CLI --score-mode):
  - self_agree : per-pair disagreement = ||main_pair_sum_k - m8_k||
                 (low = heads agree → safe to compress)
  - entropy    : per-pair variance over N main samples (REQUIRES head=main_n
                 server, not yet implemented — placeholder for Phase B)
  - hybrid     : alpha * normalized(entropy) + (1-alpha) * normalized(self_agree)

Selection (CLI --K): compress the K pairs with LOWEST score (most agreeable).
K=0 = no compression (= main 16). K=8 = full compression (= m8 8).

Uses head=main_and_m8 server. Works on any ckpt with both main + m8_action_decoder
(mh_m8/multi_horizon ckpts AND per_expert MoE ckpts both qualify — MoE router
is bypassed in this initial version).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env


DISCRETE_KEYS = {"action.gripper_close", "action.control_mode"}
M8_SUFFIX = "_m8"


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


def _to_ntd(v, has_N, T_expected=16):
    """Normalize to (N, T, D) if has_N else (T, D). 1D actions get a trailing dim.

    np.squeeze removes ALL length-1 dims, so for 1D actions (gripper, control_mode)
    we may receive shapes like (T,) or (N, T) instead of (1, T, 1) / (N, T, 1).
    Disambiguate using the known T_expected (default 16 for our action_horizon).

    Possible incoming shapes:
      (T,)        1D action, N=1                         -> (1, T, 1) or (T, 1)
      (T, D)      multi-dim action, N=1                  -> (1, T, D) or (T, D)
      (N, T)      1D action, N>1 (D=1 squeezed away)     -> (N, T, 1)
      (N, T, D)   multi-dim action, N>1                  -> (N, T, D)
    """
    v = np.asarray(v)
    if has_N:
        if v.ndim == 3:
            # (N, T, D) — use as-is (assume T == T_expected)
            return v
        if v.ndim == 2:
            # Either (T, D) with T == T_expected → N=1 was squeezed
            # Or (N, T) with T == T_expected → 1D action, D=1 was squeezed
            if v.shape[0] == T_expected:
                v = v[None, ...]            # (T, D) → (1, T, D)
            elif v.shape[1] == T_expected:
                v = v[..., None]            # (N, T) → (N, T, 1)
            else:
                raise ValueError(
                    f"can't disambiguate 2D shape {v.shape} vs T={T_expected}"
                )
            return v
        if v.ndim == 1:                     # (T,) — both N and D squeezed
            return v[None, :, None]
        raise ValueError(f"unexpected main shape {v.shape}")
    else:
        # Goal: (T, D)
        if v.ndim == 1:
            v = v[..., None]
        return v


def split_main_m8(action_dict):
    """Split dict with mixed main/main_m8 keys into (main_samples, m8_dict).

    main_samples: dict where each value is normalized to (N, T, D).
    m8_dict:      dict where each value is normalized to (T, D).
    """
    main, m8 = {}, {}
    for k, v in action_dict.items():
        if not k.startswith("action."):
            continue
        if k.endswith(M8_SUFFIX):
            m8[k[: -len(M8_SUFFIX)]] = _to_ntd(v, has_N=False)
        else:
            main[k] = _to_ntd(v, has_N=True)
    return main, m8


def get_moe_picked(action_dict):
    """Extract _moe_picked (router pick) from server response. None if absent."""
    v = action_dict.get("_moe_picked", None)
    if v is None:
        return None
    return int(np.asarray(v).flatten()[0])


def normalize_score(s):
    """Per-chunk min-max normalize to [0, 1] for hybrid combination."""
    s = np.asarray(s, dtype=np.float32)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def per_pair_disagreement(samples_main_dict, m8_dict, discrete_keys=DISCRETE_KEYS):
    """Per-pair (8,) self-agree disagreement using sample[0] of main vs m8.

    Continuous: ||main[2k] + main[2k+1] - m8[k]||  (L2)
    Discrete:   1.0 if any-dim of main[2k+1] != m8[k]

    Returns ABSOLUTE summed scores (not min-max normalized) so an absolute
    threshold τ has meaning across chunks. Different keys have different scales
    — interpret τ relative to the population of observed scores (calibrate by
    running --dump-scores-only first).
    """
    scores = np.zeros(8, dtype=np.float32)
    for key, v in samples_main_dict.items():
        v_main = v[0]                           # (16, D)
        if v_main.shape[0] < 16:
            continue
        if key not in m8_dict or m8_dict[key].shape[0] < 8:
            continue
        v_m8 = m8_dict[key]
        is_disc = key in discrete_keys
        for k in range(8):
            a, b = v_main[2 * k], v_main[2 * k + 1]
            if is_disc:
                scores[k] += float(not np.allclose(b, v_m8[k]))
            else:
                scores[k] += float(np.linalg.norm(a + b - v_m8[k]))
    return scores


def per_step_entropy_aac(samples_main_dict, discrete_keys=DISCRETE_KEYS):
    """Per-STEP (16,) entropy following AAC paper (Liang et al. 2026).

    For each of the 16 steps in the chunk, compute:
      Continuous keys: Gaussian differential entropy
            E = ½ log[(2πe)^d det(Σ)]
        where Σ is the d×d covariance over N samples at that step.
      Discrete keys: Shannon entropy
            E = -Σ p log p
        where p is the empirical probability over N samples (per dim, summed).

    Per-step total = sum over all keys (continuous + discrete entropies).
    Returns shape (16,) — one entropy per step.
    """
    H = 16
    step_E = np.zeros(H, dtype=np.float64)
    LN_2PIE = np.log(2.0 * np.pi * np.e)
    for key, v in samples_main_dict.items():
        # v shape: (N, T, D)
        if v.shape[1] < H:
            continue
        N = v.shape[0]
        if N < 2:
            continue
        D = v.shape[2]
        is_disc = key in discrete_keys
        for i in range(H):
            samples_i = v[:, i, :]                          # (N, D)
            if is_disc:
                # Shannon entropy per dim, summed
                e = 0.0
                for d in range(D):
                    col = samples_i[:, d]
                    _, counts = np.unique(col, return_counts=True)
                    p = counts.astype(np.float64) / N
                    e += -float(np.sum(p * np.log(p + 1e-12)))
                step_E[i] += e
            else:
                # Gaussian differential entropy: ½ log[(2πe)^D det(Σ)]
                # Using sample covariance with ddof=1 (unbiased)
                if N <= D:
                    # Σ singular when N <= D — fall back to per-dim sum of log-variance
                    log_det = float(np.sum(np.log(samples_i.var(axis=0, ddof=1) + 1e-12)))
                else:
                    cov = np.cov(samples_i, rowvar=False, ddof=1)   # (D, D)
                    cov = cov + 1e-9 * np.eye(D)                    # regularize
                    sign, log_det = np.linalg.slogdet(cov)
                    if sign <= 0:
                        log_det = float(np.sum(np.log(np.abs(np.diag(cov)) + 1e-12)))
                step_E[i] += 0.5 * (D * LN_2PIE + log_det)
    return step_E.astype(np.float32)


def per_pair_entropy_continuous(samples_main_dict, discrete_keys=DISCRETE_KEYS):
    """Pair-level (8,) entropy = sum of two adjacent step entropies (AAC paper
    formulas). For pair k: E_k = step_E[2k] + step_E[2k+1].
    """
    step_E = per_step_entropy_aac(samples_main_dict, discrete_keys)
    pair_E = np.zeros(8, dtype=np.float32)
    for k in range(8):
        pair_E[k] = step_E[2 * k] + step_E[2 * k + 1]
    return pair_E


def per_pair_curvature(samples_main_dict, exec_idx=0, discrete_keys=DISCRETE_KEYS):
    """(8,) per-pair sum-pair geometric loss from the executed sample.

    For pair k = (2k, 2k+1) with continuous deltas a, b:
        cost_k = ||a|| + ||b|| - ||a+b||   (>= 0 by triangle inequality)
    = 0 when a, b are collinear/same-direction (sum-pair is lossless), large when
    the motion turns within the pair (sum-pair skips the curve → lossy). Summed
    per continuous key (each normed separately to avoid unit mixing). Independent
    of N (computable from a single sample); orthogonal to entropy (reliability).
    """
    out = np.zeros(8, dtype=np.float32)
    for key, v in samples_main_dict.items():
        if key in discrete_keys:
            continue
        a_all = np.asarray(v[exec_idx])              # (16, D)
        if a_all.ndim < 2 or a_all.shape[0] < 16:
            continue
        for k in range(8):
            a = a_all[2 * k].astype(np.float64)
            b = a_all[2 * k + 1].astype(np.float64)
            out[k] += float(np.linalg.norm(a) + np.linalg.norm(b) - np.linalg.norm(a + b))
    return out


def _sum_pair_compress(v_main, is_disc):
    """Compress a (16, D) action into (8, D) by sum-pair (continuous) or
    later-of-pair (discrete)."""
    out = np.empty((8, *v_main.shape[1:]), dtype=v_main.dtype)
    for k in range(8):
        a, b = v_main[2 * k], v_main[2 * k + 1]
        out[k] = b if is_disc else (a + b)
    return out


def aac_cliff_h_star(scores, xi=1):
    """AAC paper's per-chunk h* selection (eq. 5):
       h* = max(argmax_h(avg_score[0..h] - avg_score[0..h-1]), xi)

    Returns h* in [xi, len(scores)]. Compress first h* pairs (analog of "execute
    first h* actions" in AAC where chunk size is determined per chunk). No τ.
    """
    s = np.asarray(scores, dtype=np.float64)
    H = len(s)
    avgs = np.cumsum(s) / np.arange(1, H + 1)            # avg_score[0..h-1] for h=1..H
    diffs = avgs[1:] - avgs[:-1]                         # diff at h=1..H-1
    if len(diffs) == 0:
        return max(0, xi)
    h_star = int(np.argmax(diffs)) + 1                   # convert to "first h pairs"
    return max(h_star, xi)


def selective_compress(samples_main_dict, scores, tau, decision_rule="per_pair",
                       discrete_keys=DISCRETE_KEYS, exec_sample_idx=0,
                       m8_dict=None, aac_xi=1, step_entropies=None,
                       compress_src="main_sumpair"):
    """Build variable-length action dict by compressing pairs based on score+rule.

    Threshold-based rules (need τ from calibration):

    Mode α (decision_rule='per_pair'): each pair k independently. Compress iff
        score_k < τ.

    Mode β (decision_rule='prefix_cliff'): contiguous prefix [0, k*) where
        k* = first k with score_k >= τ.

    Mode γ (decision_rule='chunk_binary'): all-or-nothing per chunk. Compress
        all 8 iff mean(scores) < τ. Uses m8_dict directly when compressing.

    Threshold-FREE rule (AAC paper, no τ; uses ξ floor on min compressed pairs):

    Mode AAC (decision_rule='aac_cliff'): per-chunk argmax-diff cliff detection
        on cumulative average score. h* = argmax_h(avg_s[0..h] - avg_s[0..h-1]),
        clamped to >= ξ. Compress first h* pairs.

    Compression for continuous dims sums adjacent pairs; discrete dims take
    the later value.
    """
    main_dict = {k: v[exec_sample_idx] for k, v in samples_main_dict.items()}
    if decision_rule == "per_pair":
        compress_idx = sorted(np.where(scores < tau)[0].tolist())
    elif decision_rule == "prefix_cliff":
        k_star = 8
        for k in range(8):
            if scores[k] >= tau:
                k_star = k
                break
        compress_idx = list(range(0, k_star))
    elif decision_rule == "chunk_binary":
        chunk_score = float(np.asarray(scores).mean())
        if chunk_score < tau:
            if compress_src == "m8" and m8_dict is not None:
                out = {}
                for key, v_main in main_dict.items():
                    if key in m8_dict and m8_dict[key].shape[0] >= 8:
                        out[key] = m8_dict[key][:8]
                    else:
                        out[key] = _sum_pair_compress(v_main, key in discrete_keys)
                return out, list(range(8))
            compress_idx = list(range(8))
        else:
            compress_idx = []
    elif decision_rule == "aac_cliff":
        # If step_entropies (16) provided, run AAC at full step granularity per
        # paper, then convert h* (in steps) to pair count via h*/2.
        if step_entropies is not None and len(step_entropies) == 16:
            h_step = aac_cliff_h_star(step_entropies, xi=2 * aac_xi)
            h_star = h_step // 2          # pair count
        else:
            h_star = aac_cliff_h_star(scores, xi=aac_xi)
        compress_idx = list(range(0, h_star))
    elif decision_rule == "aac_chunk_binary":
        # AAC binarized: compress all 8 iff h* >= K (K = aac_xi). Use step-level
        # h* if available; else pair-level. K is in pair units in both cases for
        # consistency (compare h_star_pair >= K).
        if step_entropies is not None and len(step_entropies) == 16:
            h_step = aac_cliff_h_star(step_entropies, xi=0)
            h_star = h_step // 2
        else:
            h_star = aac_cliff_h_star(scores, xi=0)
        if h_star >= aac_xi:
            if compress_src == "m8" and m8_dict is not None:
                out = {}
                for key, v_main in main_dict.items():
                    if key in m8_dict and m8_dict[key].shape[0] >= 8:
                        out[key] = m8_dict[key][:8]
                    else:
                        out[key] = _sum_pair_compress(v_main, key in discrete_keys)
                return out, list(range(8))
            compress_idx = list(range(8))
        else:
            compress_idx = []
    else:
        raise ValueError(f"unknown decision_rule {decision_rule}")
    compress_set = set(compress_idx)

    out = {}
    for key, v_main in main_dict.items():
        if v_main.shape[0] < 16:
            out[key] = v_main
            continue
        is_disc = key in discrete_keys
        use_m8 = (compress_src == "m8" and m8_dict is not None
                  and key in m8_dict and m8_dict[key].shape[0] >= 8)
        steps = []
        for k in range(8):
            a = v_main[2 * k]
            b = v_main[2 * k + 1]
            if k in compress_set:
                if use_m8:
                    steps.append(m8_dict[key][k])
                else:
                    steps.append(b if is_disc else a + b)
            else:
                steps.append(a)
                steps.append(b)
        out[key] = np.stack(steps)
    return out, compress_idx


def step_action(chunk_dict, idx, env):
    """Pull single action at index idx from variable-length chunk."""
    one = {}
    for k, v in chunk_dict.items():
        v = np.asarray(v)
        if v.ndim == 1:
            one[k] = np.asarray([[v[idx]]])
        else:
            one[k] = np.asarray([v[idx]])
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
    # Selective compression args
    p.add_argument("--score-mode", choices=["self_agree", "entropy", "hybrid"],
                   default="self_agree")
    p.add_argument("--tau", type=float, default=None,
                   help="Absolute threshold on per-pair score. Used by both decision rules. "
                        "Calibrate via --dump-scores-only first.")
    p.add_argument("--decision-rule",
                   choices=["per_pair", "prefix_cliff", "chunk_binary",
                            "aac_cliff", "aac_chunk_binary"],
                   default="aac_cliff",
                   help="per_pair (α, deprecated per user request): each pair < tau. "
                        "prefix_cliff (β τ-based): contiguous prefix until first score>=tau. "
                        "chunk_binary (γ τ-based): all-or-nothing iff mean(score) < tau. "
                        "aac_cliff (AAC paper, NO τ): h* = argmax-diff cliff, compress first "
                        "h* pairs with --aac-xi as floor. "
                        "aac_chunk_binary (AAC binary, NO τ): compress all 8 iff h* >= --aac-xi "
                        "(here aac_xi is reused as the integer K threshold on h*).")
    p.add_argument("--aac-xi", type=int, default=1,
                   help="For aac_cliff: minimum number of pairs to compress (paper ξ, default 1). "
                        "For aac_chunk_binary: integer K such that compress all 8 iff h* >= K.")
    p.add_argument("--aac-compress-src",
                   choices=["main_sumpair", "m8"],
                   default="main_sumpair",
                   help="Where compressed (8-step) action values come from when a pair is "
                        "compressed. 'main_sumpair' (default): sum-pair the executed main "
                        "sample (trajectory-consistent, no m8 needed). 'm8': use the m8 decoder "
                        "output. Applies to all compressing decision rules.")
    p.add_argument("--dump-scores-only", action="store_true",
                   help="Calibration mode: don't compress (execute raw main 16), but compute "
                        "scores per chunk and dump them to prediction.txt for τ calibration.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on entropy in hybrid (1-alpha for self_agree). Hybrid uses "
                        "per-chunk min-max normalized scores so alpha is balanced.")
    p.add_argument("--curvature-weight", type=float, default=0.0,
                   help="If >0, augment the aac_cliff score with per-pair sum-pair geometric "
                        "loss (||a||+||b||-||a+b||): combined = norm(entropy) + w*norm(curvature), "
                        "run pair-level cliff. High-curvature (turning) pairs raise the score so "
                        "the cliff stops before them (don't compress sharp turns). 0 = entropy only.")
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

    # Resume from any existing prediction.txt (matches scripts/robocasa_service.py).
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

    # Logging
    chunk_lens = []          # one length per chunk (8..16)
    all_pair_scores = []     # per-pair score per chunk (for histogram)
    all_step_entropies = []  # per-STEP entropy (16 values) per chunk if score_mode includes entropy
    moe_picks = []           # router pick per chunk (None if not MoE)

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            # Already evaluated in a prior run; skip ahead.
            continue
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            obs["video.left_view"]  = np.flip(obs["video.left_view"],  axis=1)
            obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
            obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)

            raw = client.get_action(obs)
            picked = get_moe_picked(raw)
            moe_picks.append(picked)

            # Router-gated path: non-main expert picked.
            if picked is not None and picked != 0:
                # Build n8/expert chunk dict from raw (skip _m8 suffix keys)
                expert_d = {}
                m8_or_m4_d = {}     # piggybacked compressed companion (m4 if picked==3)
                for k, v in raw.items():
                    if not k.startswith("action."):
                        continue
                    v = np.asarray(v)
                    if k.endswith(M8_SUFFIX):
                        m8_or_m4_d[k[:-len(M8_SUFFIX)]] = v
                    else:
                        expert_d[k] = v

                # picked == 3 (n8 raw, 4 pairs) AND server returned m4 companion
                # → run AAC chunk_binary on n8↔m4 self_agree score (pair count = 4)
                if picked == 3 and m8_or_m4_d:
                    # Per-pair self_agree: ||n8[2k] + n8[2k+1] - m4[k]|| for k=0..3
                    s4 = np.zeros(4, dtype=np.float32)
                    for key, v_n8 in expert_d.items():
                        if v_n8.ndim == 1:
                            v_n8 = v_n8[..., None]
                        if v_n8.shape[0] < 8 or key not in m8_or_m4_d:
                            continue
                        v_m4 = m8_or_m4_d[key]
                        if v_m4.ndim == 1:
                            v_m4 = v_m4[..., None]
                        is_disc = key in DISCRETE_KEYS
                        for k_ in range(4):
                            a, b = v_n8[2 * k_], v_n8[2 * k_ + 1]
                            if is_disc:
                                s4[k_] += float(not np.allclose(b, v_m4[k_]))
                            else:
                                s4[k_] += float(np.linalg.norm(a + b - v_m4[k_]))

                    # Apply same decision rule (scaled to 4 pairs).
                    # Also support τ-based chunk_binary on self_agree (no AAC).
                    if args.decision_rule == "chunk_binary" and args.tau is not None:
                        if float(s4.mean()) < args.tau:
                            chunk = {}
                            for key, v_n8 in expert_d.items():
                                if v_n8.ndim == 1: v_n8 = v_n8[..., None]
                                chunk[key] = m8_or_m4_d.get(key, v_n8)
                        else:
                            chunk = expert_d
                    elif args.decision_rule == "aac_chunk_binary":
                        K_n8 = max(1, args.aac_xi // 2)
                        h_star = aac_cliff_h_star(s4, xi=0)
                        if h_star >= K_n8:
                            chunk = {}
                            for key, v_n8 in expert_d.items():
                                if v_n8.ndim == 1: v_n8 = v_n8[..., None]
                                if key in m8_or_m4_d:
                                    chunk[key] = m8_or_m4_d[key]
                                else:
                                    chunk[key] = v_n8     # fallback
                        else:
                            chunk = expert_d
                    elif args.decision_rule == "aac_cliff":
                        # Compress first h* pairs of n8 → mix of m4 and n8 actions
                        xi_n8 = max(1, args.aac_xi // 2)
                        h_star = aac_cliff_h_star(s4, xi=xi_n8)
                        chunk = {}
                        for key, v_n8 in expert_d.items():
                            if v_n8.ndim == 1: v_n8 = v_n8[..., None]
                            if key not in m8_or_m4_d or v_n8.shape[0] < 8:
                                chunk[key] = v_n8
                                continue
                            v_m4 = m8_or_m4_d[key]
                            if v_m4.ndim == 1: v_m4 = v_m4[..., None]
                            is_disc = key in DISCRETE_KEYS
                            steps = []
                            for k_ in range(4):
                                if k_ < h_star:
                                    steps.append(v_m4[k_])     # compressed
                                else:
                                    steps.append(v_n8[2 * k_])
                                    steps.append(v_n8[2 * k_ + 1])
                            chunk[key] = np.stack(steps)
                    else:
                        chunk = expert_d
                else:
                    chunk = expert_d

                H_exec = next(iter(chunk.values())).shape[0]
                chunk_lens.append(H_exec)
                for j in range(H_exec):
                    obs, reward, terminated, truncated, info = step_action(chunk, j, env)
                    done = terminated or truncated
                    t += 1
                    pbar.update(1)
                    if done or t >= args.max_episode_steps:
                        break
                continue

            # Main picked (or non-MoE): split into main samples + m8 → AAC compression
            samples_main_d, m8_d = split_main_m8(raw)
            if args.score_mode == "self_agree":
                scores = per_pair_disagreement(samples_main_d, m8_d)
            elif args.score_mode == "entropy":
                if next(iter(samples_main_d.values())).shape[0] < 2:
                    raise RuntimeError(
                        "score_mode=entropy requires N>=2 samples; start server with --n_samples >= 2"
                    )
                step_E_arr = per_step_entropy_aac(samples_main_d)   # (16,)
                all_step_entropies.append(step_E_arr.tolist())
                scores = np.zeros(8, dtype=np.float32)
                for k in range(8):
                    scores[k] = step_E_arr[2 * k] + step_E_arr[2 * k + 1]
            elif args.score_mode == "hybrid":
                if next(iter(samples_main_d.values())).shape[0] < 2:
                    raise RuntimeError(
                        "score_mode=hybrid requires N>=2 samples; start server with --n_samples >= 2"
                    )
                step_E_arr = per_step_entropy_aac(samples_main_d)   # (16,)
                all_step_entropies.append(step_E_arr.tolist())
                pair_E = np.array([step_E_arr[2*k] + step_E_arr[2*k+1] for k in range(8)], dtype=np.float32)
                s_ent = normalize_score(pair_E)
                s_agr = normalize_score(per_pair_disagreement(samples_main_d, m8_d))
                scores = args.alpha * s_ent + (1.0 - args.alpha) * s_agr
            else:
                raise ValueError(f"unknown score_mode {args.score_mode}")
            all_pair_scores.append(scores.tolist())
            # Per-step entropy for AAC (only available when score_mode in {entropy,hybrid})
            step_E_for_aac = step_E_arr if args.score_mode in ("entropy", "hybrid") else None

            # Curvature augmentation (orthogonal axis): combine the reliability
            # score with per-pair sum-pair geometric loss. High-curvature pairs
            # (turning) get a higher score → pair-level cliff stops before them.
            if args.curvature_weight > 0 and args.decision_rule == "aac_cliff":
                pair_curv = per_pair_curvature(samples_main_d)             # (8,)
                scores = (normalize_score(np.asarray(scores, dtype=np.float32))
                          + args.curvature_weight * normalize_score(pair_curv))
                step_E_for_aac = None   # force pair-level cliff on the combined score

            if args.dump_scores_only:
                chunk = {k: v[0] for k, v in samples_main_d.items()}
                compressed_idx = []
            elif args.decision_rule in ("aac_cliff", "aac_chunk_binary"):
                # AAC threshold-free; tau not needed.
                # Pass step_entropies when available so AAC runs at full step
                # granularity (per paper); otherwise it falls back to pair-level.
                chunk, compressed_idx = selective_compress(
                    samples_main_d, scores, tau=None,
                    decision_rule=args.decision_rule,
                    m8_dict=m8_d, aac_xi=args.aac_xi,
                    step_entropies=step_E_for_aac,
                    compress_src=args.aac_compress_src,
                )
            elif args.tau is not None:
                chunk, compressed_idx = selective_compress(
                    samples_main_d, scores, tau=args.tau,
                    decision_rule=args.decision_rule,
                    m8_dict=m8_d,
                    compress_src=args.aac_compress_src,
                )
            else:
                raise RuntimeError(
                    "Must pass --tau, --decision-rule aac_cliff, or --dump-scores-only."
                )
            H_exec = next(iter(chunk.values())).shape[0]
            chunk_lens.append(H_exec)

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
        print(f"  ep {i}: success={succ}, action_steps={t}, mean exec len so far={np.mean(chunk_lens):.1f}")

    env.close()
    sr = float(np.mean(stats["is_success"]))
    mean_len = float(np.mean(chunk_lens)) if chunk_lens else 0.0
    flat_scores = np.array(all_pair_scores).flatten() if all_pair_scores else np.array([])
    print(f"\nFinal: success rate = {sr:.4f}  ({len(stats['is_success'])} ep)")
    print(f"Mean chunk exec length = {mean_len:.2f} (over {len(chunk_lens)} chunks)")
    if flat_scores.size > 0:
        qs = np.quantile(flat_scores, [0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"Score quantiles (p10/25/50/75/90) = {qs.tolist()}")
    with open(pred_path, "a") as f:
        f.write(f"is_success: {sr:.4f}\n")
        f.write(f"mean_exec_chunk_len: {mean_len:.4f}\n")
        f.write(f"score_mode: {args.score_mode}\n")
        f.write(f"tau: {args.tau}\n")
        f.write(f"decision_rule: {args.decision_rule}\n")
        f.write(f"dump_scores_only: {args.dump_scores_only}\n")
        if flat_scores.size > 0:
            qs = np.quantile(flat_scores, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
            f.write(f"score_quantiles_p10_25_50_75_90: {qs}\n")
            f.write(f"score_min_max: [{float(flat_scores.min())}, {float(flat_scores.max())}]\n")
            # Save full score history for offline τ calibration
            f.write(f"all_pair_scores: {json.dumps(all_pair_scores)}\n")
        if all_step_entropies:
            f.write(f"all_step_entropies: {json.dumps(all_step_entropies)}\n")
        if any(p is not None for p in moe_picks):
            from collections import Counter
            pick_counts = Counter(moe_picks)
            f.write(f"moe_pick_counts: {dict(pick_counts)}\n")


if __name__ == "__main__":
    main()
