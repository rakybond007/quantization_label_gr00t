"""Fixed-ratio action compression eval client for robocasa.

Inference-time intervention only (no retraining): receive a 16-step chunk from
the standard GR00T-N1.5 baseline server (head=main), aggregate the chunk into
blocks of size K, and execute the compressed chunk step-by-step.

Aggregator (per action key):
  - continuous keys: block-sum (delta actions compose by sum)
  - discrete keys  : block-last (gripper / control_mode latch)

If 16 is not divisible by K (e.g. K=3 -> 16 = 5*3 + 1), the first 15 actions
form 5 compressed blocks and the last action is kept raw, yielding 6 actions
to execute. K=2 -> 8 actions; K=3 -> 6 actions.
"""
import argparse
import re
import os
import json
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
    DISCRETE_KEYS, add_to, flatten, _to_ntd, step_action,
)



def patch_clip_bounds(env, scale):
    """Scale robosuite controller clip bounds (input AND output ranges) by
    `scale` — raises the per-step displacement cap; per-unit scale unchanged.
    EXPERIMENTAL: changes benchmark physics (separate results track)."""
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
            for a in ("input_max", "input_min", "output_max", "output_min"):
                if hasattr(c, a) and getattr(c, a) is not None:
                    setattr(c, a, getattr(c, a) * scale)
                    hit = True
            n += int(hit)
    return n



def patch_dynamics(env, torque_scale, kp_scale):
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


def vark_compress_chunk(chunk_dict, kmax, bound, discrete_keys=DISCRETE_KEYS, floor2=False,
                        return_blocks=False):
    """Magnitude-aware VARIABLE-K block merge over a {key: (T,D)} chunk.

    The robosuite controller clips each env-step input to [-1,1] per dim, so a
    fixed-K delta sum beyond that silently loses displacement. Here consecutive
    steps merge greedily while (a) block < kmax steps, (b) the running sum of
    EVERY continuous key stays within `bound` on all dims, (c) no discrete
    (gripper-like) key transitions inside the block. Continuous keys sum,
    discrete keys latch to the block's last value.
    """
    keys = list(chunk_dict.keys())
    arrs = {k: np.asarray(v) if np.asarray(v).ndim >= 2 else np.asarray(v)[..., None]
            for k, v in chunk_dict.items()}
    T = next(iter(arrs.values())).shape[0]
    cont = [k for k in keys if k not in discrete_keys]
    disc = [k for k in keys if k in discrete_keys]
    out = {k: [] for k in keys}
    blocks = []  # (start, end) fine-index span of each output row
    i = 0
    while i < T:
        acc = {k: arrs[k][i].astype(float).copy() for k in cont}
        last = {k: arrs[k][i] for k in disc}
        j = i + 1
        if floor2 and j < T:
            # floor-2: unconditional pair merge (fixed-K2 semantics)
            acc = {k: acc[k] + arrs[k][j] for k in cont}
            last = {k: arrs[k][j] for k in disc}
            j += 1
        while j < T and (j - i) < kmax:
            if any(np.abs(arrs[k][j] - arrs[k][j - 1]).max() > 0.5 for k in disc):
                break                      # gripper-like transition: block ends
            cands = {k: acc[k] + arrs[k][j] for k in cont}
            if any(np.abs(c).max() > bound for c in cands.values()):
                break                      # would clip at the controller
            acc = cands
            last = {k: arrs[k][j] for k in disc}
            j += 1
        for k in cont:
            out[k].append(acc[k])
        for k in disc:
            out[k].append(last[k])
        blocks.append((i, j))
        i = j
    res = {k: np.stack(v) for k, v in out.items()}
    return (res, blocks) if return_blocks else res


def compress_chunk(chunk_dict, K, discrete_keys=DISCRETE_KEYS, return_blocks=False):
    """Apply fixed-K block aggregation per action key.

    chunk_dict: {key: (T, D)} where T = base horizon (16).
    Returns {key: (T_out, D)} with T_out = (T // K) + (T mod K) raw tail.
    With return_blocks=True also returns the (start, end) fine-index span of
    each output row (shared across keys).
    """
    out = {}
    blocks = None
    for key, v in chunk_dict.items():
        v = np.asarray(v)
        if v.ndim < 2:
            v = v[..., None]
        T = v.shape[0]
        num_full = T // K
        rem = T - num_full * K
        is_disc = key in discrete_keys
        agg_blocks = []
        for i in range(num_full):
            block = v[i * K:(i + 1) * K]
            if is_disc:
                agg_blocks.append(block[-1])  # latch
            else:
                agg_blocks.append(block.sum(axis=0))  # delta sum
        if rem > 0:
            for j in range(num_full * K, T):
                agg_blocks.append(v[j])  # raw remainder
        out[key] = np.stack(agg_blocks)
        if blocks is None:
            blocks = [(i * K, (i + 1) * K) for i in range(num_full)]
            blocks += [(j, j + 1) for j in range(num_full * K, T)]
    return (out, blocks) if return_blocks else out


def collect_chunk(action_dict):
    """Pull main chunk from server reply. Server returns single-sample chunks."""
    out = {}
    for k, v in action_dict.items():
        if not k.startswith("action."):
            continue
        if k.endswith("_m8"):  # in case server returns m8 companion (we ignore)
            continue
        v = np.asarray(v)
        # Normalize to (T, D)
        if v.ndim == 1:
            v = v[..., None]
        elif v.ndim == 3:
            v = v[0]  # if N samples returned, take sample 0
        out[k] = v
    return out


# 지연 계측: 정책 1콜(디노이징 포함) vs 게이트 1콜. 현재 둘은 순차 실행이므로
# 게이트 비용은 전부 가산이다. 동시 실행 설계의 예산을 정하려면 실측이 필요하다.
import time as _time
_POLICY_MS=[]
_GATE_MS=[]


def action_rule_block(sub, clip=1.0):
    """Labeling-time action rules, recomputed on the policy's planned chunk.

    Returns (block?, reason). Mirrors robocasa_descriptors.computed_risk so the
    gate applies the same criteria the teacher applied when the label was made.
    The student sees images only (so it can run concurrently with the action
    head's denoising); these rules restore the action information it cannot see.
    Pure arithmetic on a 16xD array — no measurable latency.
    """
    import numpy as _np
    key = next((k for k in sub if "action" in k or "delta" in k), None)
    a = _np.asarray(sub[key] if key else next(iter(sub.values())), dtype=float)
    if a.ndim != 2 or a.shape[0] < 2:
        return False, ""
    g = a[:, -1]
    if _np.abs(_np.diff(g)).max() > 0.5:
        return True, "grip_transition"
    d = a[:, 5:8] if a.shape[1] >= 8 else a[:, :3]
    m = _np.linalg.norm(d, axis=1)
    if ((g > 0.5) & (m < 0.12)).mean() > 0.5:
        return True, "precise_hold"
    if len(d) > 1:
        v1, v2 = d[:-1], d[1:]
        m1, m2 = _np.linalg.norm(v1, axis=1), _np.linalg.norm(v2, axis=1)
        cos = _np.sum(v1 * v2, axis=1) / _np.maximum(m1 * m2, 1e-9)
        if _np.any((m1 > 0.10) & (m2 > 0.10) & (cos < 0)):
            return True, "reversal"
    if a.shape[1] >= 11:
        merged = a[0:-1:2, 5:11] + a[1::2, 5:11]
        if _np.mean(_np.abs(merged) > clip) > 0.20:
            return True, "clip_excess"
    return False, ""


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
    p.add_argument("--compress-k", type=int, default=2,
                   help="Block size K for fixed-ratio compression (K=2 -> 8 actions, K=3 -> 6 actions).")
    p.add_argument("--judge-url", type=str, default="",
                   help="If set, query this VLM gate per chunk to decide whether to "
                        "quantize (compress with K) or run raw (K=1). Empty = always compress with K.")
    p.add_argument("--judge-instruction", type=str, default="",
                   help="Override task description for the gate. Empty = use the env's "
                        "per-episode language instruction (recommended), else env_name.")
    p.add_argument("--action-rules", type=int, default=0,
                   help="Apply the same action-chunk rules used at labeling time to the "
                        "policy's planned chunk, and hard-gate on them. The student sees "
                        "images only (so it can run concurrently with denoising); these "
                        "rules restore the action information it cannot see. Pure "
                        "arithmetic on a 16xD array, so it adds no measurable latency.")
    p.add_argument("--judge-actions", type=int, default=0,
                   help="1이면 정책이 생성한 액션 청크를 judge 프롬프트에 함께 전달 (액션 인지형 evolve용).")
    p.add_argument("--gate-k3-threshold", type=float, default=0.0,
                   help="If >0, chunks with gate confidence >= this use K=3 merge "
                        "(confidence ladder: fine < tau < K2 < tau3 < K3).")
    p.add_argument("--judge-threshold", type=float, default=0.5,
                   help="Quantize when VLM confidence P(safe-to-compress) >= this value.")
    p.add_argument("--judge-guidance", type=str, default="",
                   help="Extra learned guidance injected into the VLM gate system prompt "
                        "(string, or @path to a text file). Used to evolve the prompt across evals.")
    p.add_argument("--gate-subchunk", type=int, default=0,
                   help="If >0, split each predicted chunk into sub-blocks of this many raw "
                        "steps and re-judge the gate per sub-block (more frequent decisions, "
                        "fresh obs per sub-block -> partial lookahead). 0 = one decision per chunk.")
    # TTL skip policy (same as the LIBERO client): reuse the previous decision
    # while ttl>0 unless a gripper transition inside the fresh (sub-)chunk — a
    # predictive, free signal from the policy's own output — forces a real call.
    p.add_argument("--gate-ttl-max", type=int, default=0, help="TTL policy off when 0")
    p.add_argument("--gate-ttl-lo", type=float, default=0.15)
    p.add_argument("--gate-ttl-hi", type=float, default=0.30)
    p.add_argument("--gate-gripper-trigger", type=int, default=1)
    p.add_argument("--dyn-scale", type=float, default=1.0,
                   help="EXPERIMENTAL: scale actuator torque caps and OSC kp by this factor.")
    p.add_argument("--clip-scale", type=float, default=1.0,
                   help="EXPERIMENTAL: scale controller clip bounds (input+output) by this "
                        "factor; raises per-step displacement cap. Changes benchmark physics.")
    p.add_argument("--vark-floor2", type=int, default=0,
                   help="floor-2 variant: always merge pairs, extend to 3..kmax within bound.")
    p.add_argument("--vark-bound", type=float, default=0.0,
                   help="If >0, magnitude-aware VARIABLE-K merge (compress-k = max block, "
                        "block sum kept within this bound per dim to avoid controller clipping). "
                        "0 = fixed-K block sum.")
    p.add_argument("--compensate", type=str, default="none", choices=["none", "model", "scalar"],
                   help="Dynamics compensation for merged deltas: 'model' inverts the sysid "
                        "ARX(1) OSC response (d' = (fine-target - a*v_now)/b, v_now from "
                        "observed EE state); 'scalar' rescales d' = c*d_sum (ablation).")
    p.add_argument("--sysid-json", type=str, default="analysis/sysid_osc_response.json",
                   help="Coefficients JSON from sysid_osc_response.py (for --compensate model, "
                        "and the default scalar c = 1/implied_K2_reach_rate).")
    p.add_argument("--scalar-c", type=float, default=None,
                   help="Override scalar compensation gain c (default from sysid json).")
    p.add_argument("--policy-backend", type=str, default="n15", choices=["n15", "n17"],
                   help="Which policy server to talk to. 'n15' (default) = the "
                        "GR00T-N1.5 server from scripts/inference_service.py, via "
                        "gr00t.eval.robocasa_simulation.SimulationInferenceClient. "
                        "'n17' = the GR00T-N1.7 PolicyServer (gr00t/eval/run_gr00t_server.py "
                        "--use-sim-policy-wrapper), via the standalone msgpack/zmq "
                        "adapter in n17_policy_client.py -- the N1.7 gr00t package is "
                        "NEVER imported here (this env is pinned to numpy 1.23.5 / "
                        "transformers 4.51.3). Compression, gating and logging are "
                        "identical for both backends.")
    args = p.parse_args()

    if args.policy_backend == "n17":
        from n17_policy_client import N17PolicyClient
        client = N17PolicyClient(host=args.host, port=args.port)
    else:
        client = SimulationInferenceClient(host=args.host, port=args.port)
    print(f"[policy] backend={args.policy_backend} {args.host}:{args.port}", flush=True)
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
    if args.clip_scale != 1.0:
        _n = patch_clip_bounds(env, args.clip_scale)
        print(f"[clip] controller clip bounds x{args.clip_scale} ({_n} controllers patched)", flush=True)
    if args.dyn_scale != 1.0:
        _n = patch_dynamics(env, args.dyn_scale, args.dyn_scale)
        print(f"[dyn] torque/kp x{args.dyn_scale} ({_n} targets patched)", flush=True)

    stats = defaultdict(list)
    pred_path = f"{args.video_dir}/prediction.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                # numpy 가 "[ True]" (앞 공백) / "[False]" 로 쓰므로 split()[0] 은
                # 성공 줄에서 "[" 가 되고 strip 하면 빈 문자열이 된다 — 성공만 조용히
                # 사라져 재개 시 다시 돌고, 요약 줄("is_success: 0.7600")까지 에피소드로
                # 세어진다. 에피소드 줄만, 대괄호·공백에 관대하게 읽는다.
                m = re.match(r"episode\s+\d+\s+is_success:\s*\[?\s*(True|False)\s*\]?", line.strip())
                if not m:
                    continue
                add_to(stats, flatten({"is_success": m.group(1)}))

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

    K = int(args.compress_k)
    CLIP_LIMIT = max(1.0, float(args.clip_scale))
    assert K >= 1
    chunk_lens = []

    # Dynamics compensation for merged EE-translation deltas (model-based
    # inversion of the OSC achieved-mode response, or scalar rescale ablation).
    EE_KEY = "action.end_effector_position"
    compensator = None
    comp_clip_total = 0.0   # summed |clip excess| in action units
    comp_clip_steps = 0     # merged steps where clipping occurred
    rule_blocks = 0; rule_reasons = {}
    merge_clip_steps = 0; merge_clip_excess = 0.0; merge_steps = 0
    comp_steps = 0          # merged steps compensated
    if args.compensate != "none":
        from osc_compensation import make_compensator
        compensator = make_compensator(args.compensate, args.sysid_json, args.scalar_c, clip_limit=max(1.0, args.clip_scale))
        print(f"[compensate] mode={args.compensate} sysid={args.sysid_json} "
              f"scalar_c={args.scalar_c}", flush=True)

    def _obs_ee(o):
        # BASE-frame EE pos: same frame the OSC delta command acts in (world
        # frame would alias axes whenever the base yaw != 0).
        v = o.get("state.end_effector_position_relative")
        return None if v is None else np.asarray(v, dtype=float).reshape(-1, 3)[-1]

    # Optional zero-shot VLM gate: per chunk get confidence P(safe-to-compress);
    # quantize (K) when conf >= threshold, else raw (K=1).
    gate = None
    gate_yes = 0
    gate_total = 0
    gate_calls = 0          # real VLM calls (gate_total includes TTL reuses)
    gate_confs = []
    gate_log = None
    if args.judge_url == "internal":
        # Internal policy-embedded gate (plan C quantizability token): the
        # GR00T server (head=main_gate) returns "_gate_prob" with each chunk;
        # no external judge server / HTTP call is made.
        gate = "internal"
        print(f"[gate] INTERNAL policy gate ON K_quantize={K} "
              f"threshold={args.judge_threshold}", flush=True)
        _gc = f"{args.video_dir}/gate_conf.csv"
        _gc_new = (not os.path.exists(_gc)) or os.path.getsize(_gc) == 0
        gate_log = open(_gc, "a")  # append: survive preemption/resume
        if _gc_new:
            gate_log.write("episode,step,conf,quantize,called,instruction\n")
    elif args.judge_url:
        from vlm_gate import VLMGate
        gate = VLMGate(args.judge_url)
        print(f"[gate] VLM gate ON url={args.judge_url} K_quantize={K} "
              f"threshold={args.judge_threshold} ttl_max={args.gate_ttl_max}", flush=True)
        _gc = f"{args.video_dir}/gate_conf.csv"
        _gc_new = (not os.path.exists(_gc)) or os.path.getsize(_gc) == 0
        gate_log = open(_gc, "a")  # append: survive preemption/resume (don't truncate)
        if _gc_new:
            gate_log.write("episode,step,conf,quantize,called,instruction\n")
    gate_guidance = ""
    if args.judge_guidance:
        g = args.judge_guidance
        gate_guidance = open(g[1:]).read() if (g.startswith("@") and os.path.exists(g[1:])) else g
        print(f"[gate] guidance ({len(gate_guidance)} chars): {gate_guidance[:160]!r}", flush=True)

    def _instr_from_obs(obs, fallback):
        v = obs.get("annotation.human.action.task_description")
        while isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            v = v[0]
        return str(v) if isinstance(v, str) and v else fallback

    def _flipped(o):
        """Non-mutating copy with the 3 camera views flipped (as GR00T expects)."""
        fo = dict(o)
        for k in ("video.left_view", "video.right_view", "video.wrist_view"):
            fo[k] = np.flip(o[k], axis=1)
        return fo

    for i in trange(args.n_episodes, desc=args.env_name):
        obs, info = env.reset()
        if i < len(stats["is_success"]):
            continue
        _ee_prev = _obs_ee(obs)
        _v_now = np.zeros(3)   # EE velocity proxy: last per-step displacement (m)
        ep_instruction = args.judge_instruction or _instr_from_obs(obs, args.env_name)
        if gate is not None:
            print(f"[gate] ep {i} instruction={ep_instruction!r}", flush=True)
        # Per-episode TTL-policy state
        _g_ttl = 0
        _g_last_conf = None
        _g_last_q = None
        _g_prev_grip = None
        done = False; t = 0
        pbar = tqdm(total=args.max_episode_steps, desc=f"ep {i}", leave=False)
        while not done and t < args.max_episode_steps:
            _t0 = _time.perf_counter()
            raw = client.get_action(_flipped(obs))    # GR00T sees flipped views
            _POLICY_MS.append((_time.perf_counter() - _t0) * 1e3)
            chunk_raw = collect_chunk(raw)            # {key: (T=16, D)}
            if compensator is not None:
                # fresh chunk = fresh policy observation: the virtual fine
                # path restarts from the true current state
                compensator.begin_chunk(_v_now)
            T = next(iter(chunk_raw.values())).shape[0]
            # Sub-chunk gating: re-judge every S raw steps with fresh obs. S=T
            # (or gate off) -> one decision per full chunk (original behaviour).
            S = args.gate_subchunk if (gate is not None and args.gate_subchunk > 0) else T
            pos = 0
            while pos < T and not done and t < args.max_episode_steps:
                sub = {k: v[pos:pos + S] for k, v in chunk_raw.items()}
                if gate is not None:
                    # TTL skip policy: gripper transition in the fresh sub-block
                    # forces a call; else reuse the previous decision while ttl>0.
                    _gk = next((k for k in sub if "gripper" in k), None)
                    call = True
                    if args.gate_ttl_max > 0 and _g_last_q is not None:
                        grip_evt = False
                        if args.gate_gripper_trigger and _gk is not None:
                            gseq = np.asarray(sub[_gk]).reshape(len(sub[_gk]), -1)[:, -1]
                            if _g_prev_grip is not None:
                                gseq = np.concatenate([[_g_prev_grip], gseq])
                            grip_evt = bool(np.abs(np.diff(gseq)).max() > 0.5) if len(gseq) > 1 else False
                        if not grip_evt and _g_ttl > 0:
                            call = False
                    if call:
                        if gate == "internal":
                            # p_quant produced by the policy itself alongside
                            # the chunk currently being executed.
                            conf = float(np.asarray(raw.get("_gate_prob", 0.0)).ravel()[0])
                        else:
                            fo = _flipped(obs)        # fresh views (reflect steps so far)
                            views = [fo["video.left_view"], fo["video.right_view"], fo["video.wrist_view"]]
                            _instr = ep_instruction
                            if args.judge_actions:
                                # 정책이 이미 만들어 둔 청크를 judge에게 함께 전달 (배포 가능: 로컬에서 읽음)
                                try:
                                    _cols = [k for k in sub.keys() if k not in DISCRETE_KEYS]
                                    _arr = np.concatenate(
                                        [np.asarray(sub[k], dtype=float).reshape(len(sub[k]), -1) for k in sorted(sub.keys())],
                                        axis=1)[:16]
                                    _instr = (f"{ep_instruction}\nPlanned actions for the next {len(_arr)} control "
                                              f"steps (last number of each step is the gripper command):\n"
                                              + json.dumps(np.round(_arr, 2).tolist()))
                                except Exception:
                                    pass
                            _g0 = _time.perf_counter()
                            res = gate.judge(views, _instr, gate_guidance)
                            _GATE_MS.append((_time.perf_counter() - _g0) * 1e3)
                            conf = float(res.get("confidence", 0.0))
                        q = conf >= args.judge_threshold
                        if args.action_rules:
                            # min(student, rules): a rule that fires blocks outright;
                            # otherwise the student decides. Multiplying instead would
                            # double-count, since the label the student learned already
                            # carried these same action terms.
                            try:
                                _blk, _why = action_rule_block(sub)
                                if _blk:
                                    q = False; rule_blocks += 1
                                    rule_reasons[_why] = rule_reasons.get(_why, 0) + 1
                            except Exception:
                                pass
                        _g_last_conf, _g_last_q = conf, q
                        if args.gate_ttl_max > 0:
                            _d = abs(conf - args.judge_threshold)
                            _g_ttl = (0 if _d < args.gate_ttl_lo
                                      else 1 if _d < args.gate_ttl_hi
                                      else args.gate_ttl_max)
                    else:
                        conf, q = _g_last_conf, _g_last_q
                        _g_ttl -= 1
                    if _gk is not None:
                        _g_prev_grip = float(np.asarray(sub[_gk]).reshape(len(sub[_gk]), -1)[-1, -1])
                    gate_confs.append(conf)
                    gate_yes += int(q); gate_total += 1; gate_calls += int(call)
                    k_eff = K if q else 1
                    if q and args.gate_k3_threshold > 0 and conf >= args.gate_k3_threshold:
                        k_eff = 3
                    if gate_log is not None:
                        gate_log.write(f"{i},{t},{conf:.4f},{int(q)},{int(call)},{ep_instruction!r}\n")
                        gate_log.flush()
                else:
                    k_eff = K
                blocks = None
                if k_eff > 1 and args.vark_bound > 0:
                    sub_exec, blocks = vark_compress_chunk(
                        sub, k_eff, args.vark_bound, floor2=args.vark_floor2 > 0,
                        return_blocks=True)
                elif k_eff > 1:
                    sub_exec, blocks = compress_chunk(sub, k_eff, return_blocks=True)
                    try:
                        for _k, _v in sub_exec.items():
                            if _k in DISCRETE_KEYS: continue
                            _a = np.asarray(_v, dtype=float)
                            _over = np.abs(_a) - np.abs(np.clip(_a, -CLIP_LIMIT, CLIP_LIMIT))
                            if _over.max() > 1e-9:
                                merge_clip_steps += int((_over.max(axis=-1) > 1e-9).sum())
                                merge_clip_excess += float(_over.sum())
                            merge_steps += _a.shape[0]
                    except Exception:
                        pass
                else:
                    sub_exec = sub
                H_exec = next(iter(sub_exec.values())).shape[0]
                chunk_lens.append(H_exec)
                for j in range(H_exec):
                    if (compensator is not None and blocks is not None
                            and EE_KEY in sub and blocks[j][1] - blocks[j][0] > 1):
                        s0, e0 = blocks[j]
                        fine = np.asarray(sub[EE_KEY], dtype=float).reshape(-1, 3)[s0:e0]
                        d_c, exc = compensator.merged_command(_v_now, fine)
                        sub_exec[EE_KEY] = np.asarray(sub_exec[EE_KEY], dtype=float)
                        sub_exec[EE_KEY][j] = d_c
                        comp_steps += 1
                        _e = float(np.abs(exc).sum())
                        if _e > 0:
                            comp_clip_total += _e
                            comp_clip_steps += 1
                    obs, reward, terminated, truncated, info = step_action(sub_exec, j, env)
                    _ee_cur = _obs_ee(obs)
                    if _ee_cur is not None and _ee_prev is not None:
                        _v_now = _ee_cur - _ee_prev
                    _ee_prev = _ee_cur
                    done = terminated or truncated
                    t += 1
                    pbar.update(1)
                    if done or t >= args.max_episode_steps:
                        break
                pos += S
        pbar.close()
        succ = info.get("is_success", False)
        add_to(stats, flatten({"is_success": succ}))
        with open(pred_path, "a") as f:
            f.write(f"episode {i} is_success: {succ} action_steps: {t}\n")
        print(f"  ep {i}: success={succ}, action_steps={t}, exec_chunk_len={H_exec}, K={K}")

    if gate_log is not None:
        gate_log.close()

    def _is_true(x):
        arr = np.asarray(x).ravel()
        return bool(arr[0]) if arr.size else False
    succ_arr = [1 if _is_true(x) else 0 for x in stats["is_success"]]
    succ_rate = float(np.mean(succ_arr)) if succ_arr else 0.0
    with open(pred_path, "a") as f:
        f.write(f"is_success: {succ_rate:.4f}\n")
        f.write(f"compress_k: {K}\n")
        if args.vark_bound > 0:
            f.write(f"vark_bound: {args.vark_bound}\n")
        if args.vark_floor2:
            f.write(f"vark_floor2: 1\n")
        if args.clip_scale != 1.0:
            f.write(f"clip_scale: {args.clip_scale}\n")
        if args.dyn_scale != 1.0:
            f.write(f"dyn_scale: {args.dyn_scale}\n")
        f.write(f"exec_chunk_len_mean: {np.mean(chunk_lens) if chunk_lens else 0:.2f}\n")
        if args.action_rules:
            f.write(f"action_rule_blocks: {rule_blocks} {rule_reasons}\n")
        if compensator is not None:
            f.write(f"compensate: {args.compensate}\n")
            f.write(f"compensate_steps: {comp_steps}\n")
            f.write(f"merge_clip_steps: {merge_clip_steps}/{merge_steps} "
                    f"merge_clip_excess: {merge_clip_excess:.3f}\n")
            f.write(f"compensate_clip_steps: {comp_clip_steps} "
                    f"clip_excess_total: {comp_clip_total:.3f}\n")
        if gate is not None:
            rate = (gate_yes / gate_total) if gate_total else 0.0
            f.write(f"gate_quantize_rate: {rate:.4f} ({gate_yes}/{gate_total})\n")
            f.write(f"gate_threshold: {args.judge_threshold}\n")
            f.write(f"gate_call_rate: {gate_calls / max(gate_total, 1):.4f} ({gate_calls}/{gate_total})\n")
            if args.gate_ttl_max > 0:
                f.write(f"gate_ttl: max={args.gate_ttl_max} lo={args.gate_ttl_lo} "
                        f"hi={args.gate_ttl_hi} gripper_trigger={args.gate_gripper_trigger}\n")
            if gate_confs:
                cs = np.asarray(gate_confs)
                f.write(f"gate_conf_mean: {cs.mean():.4f} min: {cs.min():.4f} "
                        f"max: {cs.max():.4f} p50: {np.median(cs):.4f}\n")
    gate_msg = ""
    if gate is not None and gate_total:
        cs = np.asarray(gate_confs) if gate_confs else np.zeros(1)
        gate_msg = (f"  gate_quantize_rate={gate_yes}/{gate_total}={gate_yes/gate_total:.2f}"
                    f"  conf[mean={cs.mean():.2f},min={cs.min():.2f},max={cs.max():.2f}]")
    import numpy as _np2
    _lat = ""
    if _POLICY_MS:
        _lat += f"  policy_ms[mean={_np2.mean(_POLICY_MS):.1f},p50={_np2.median(_POLICY_MS):.1f},n={len(_POLICY_MS)}]"
    if _GATE_MS:
        _lat += f"  gate_ms[mean={_np2.mean(_GATE_MS):.1f},p50={_np2.median(_GATE_MS):.1f},n={len(_GATE_MS)}]"
    print(f"[latency]{_lat}", flush=True)
    print(f"[done] succ={succ_rate:.4f}  K={K}  "
          f"exec_chunk_len_mean={np.mean(chunk_lens) if chunk_lens else 0:.2f}{gate_msg}")


if __name__ == "__main__":
    main()
