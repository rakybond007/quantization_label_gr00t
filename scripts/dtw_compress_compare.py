"""Does our trained compressed decoder beat naively compressing the base model?

At sampled GT trajectory points in robocasa we compare, for each compressed
decoder k in {m8 (k=1), m4 (k=2)}:
  - OURS : our MoE expert decoder k's prediction (force-picked).
  - BASE : the base GR00T-N1.5 model's raw 16-step prediction, then *naively*
           block-compressed with the same rule used to build decoder k's target.
both against the GT compressed target (compress_gt of the GT chunk).

We also record the router's natural pick under the conf=0.7 confidence gate, so
the analysis can condition on "points where the router actually chose a
compressed decoder". Per (point, k) we store MSE (full + continuous-only),
cosine similarity, and DTW distance vs the GT target.

Sharding: --shard i/N processes points with (global_index % N == i), so two GPUs
can split the work. Resumable: appends one JSON line per (point,k); on restart
already-done (ep,t,k) keys are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
import scripts.router_segment_analysis as rsa  # reuse compress_gt/build_obs/etc.

RC_ROOT = Path("/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300")
OUR_HF = "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"
BASE_CKPT = "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"

DISCRETE_DIMS = rsa.DISCRETE_DIMS          # (4, 11)
CONT_DIMS = [d for d in range(rsa.ACTION_DIM) if d not in DISCRETE_DIMS]
DECODER_T = rsa.DECODER_T                   # [16,8,4,8]
COMPRESSED_KS = [1, 2]                      # m8, m4
H = rsa.H                                   # 16
CONF_TAU = 0.7


# ----------------------- metrics -----------------------
def mse(a, b):
    return float(np.mean((a - b) ** 2))


def cosine_sim(a, b):
    av, bv = a.flatten(), b.flatten()
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(av @ bv / (na * nb))


def dtw_dist(a, b):
    """Multivariate DTW with euclidean local cost. a:(Ta,D) b:(Tb,D)."""
    Ta, Tb = a.shape[0], b.shape[0]
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[Ta, Tb])


def all_metrics(pred, gt):
    """pred, gt: (T, 12) aligned same T."""
    return {
        "mse": mse(pred, gt),
        "mse_cont": mse(pred[:, CONT_DIMS], gt[:, CONT_DIMS]),
        "cosine_cont": cosine_sim(pred[:, CONT_DIMS], gt[:, CONT_DIMS]),
        "dtw_cont": dtw_dist(pred[:, CONT_DIMS], gt[:, CONT_DIMS]),
    }


# ----------------------- point sampling -----------------------
def camel_task(tasks):
    for t in tasks:
        if isinstance(t, str) and " " not in t and t and t[0].isupper() and t != "Valid":
            return t
    return None


def nl_prompt(tasks):
    for t in tasks:
        if isinstance(t, str) and " " in t:
            return t
    return tasks[0] if tasks else ""


def build_point_list(n_ep_per_task, ts_per_ep, ts_start=10, ts_stride=12):
    """Deterministic list of (ep_idx, t, prompt). Grouped by CamelCase task."""
    from collections import defaultdict, OrderedDict
    task_eps = OrderedDict()
    ep_meta = {}
    with open(RC_ROOT / "meta/episodes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ct = camel_task(d.get("tasks", []))
            if ct is None:
                continue
            task_eps.setdefault(ct, [])
            if len(task_eps[ct]) < n_ep_per_task:
                task_eps[ct].append(d["episode_index"])
                ep_meta[d["episode_index"]] = (nl_prompt(d["tasks"]), d.get("length", 0))
    pts = []
    for ct in sorted(task_eps):
        for ep in task_eps[ct]:
            prompt, length = ep_meta[ep]
            n_taken = 0
            t = ts_start
            while t + H <= length and n_taken < ts_per_ep:
                pts.append((ep, t, prompt, ct))
                t += ts_stride
                n_taken += 1
    return pts


# ----------------------- model loading -----------------------
def load_base(device="cuda"):
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    cfg = DATA_CONFIG_MAP["single_panda_gripper"]
    pol = Gr00tPolicy(
        model_path=BASE_CKPT,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(backbone_model_type="eagle"),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=4,
        backbone_model_type="eagle",
    )
    pol.inference_head = "main"
    return pol


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--n-ep-per-task", type=int, default=15)
    ap.add_argument("--ts-per-ep", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1, help="cap points (smoke)")
    args = ap.parse_args()

    i_shard, n_shard = map(int, args.shard.split("/"))
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line); done.add((d["ep"], d["t"], d["k"]))
            except Exception:
                pass
    print(f"[shard {i_shard}/{n_shard}] resume: {len(done)} rows done", flush=True)

    pts = build_point_list(args.n_ep_per_task, args.ts_per_ep)
    pts = [p for idx, p in enumerate(pts) if idx % n_shard == i_shard]
    if args.limit > 0:
        pts = pts[:args.limit]
    print(f"[shard {i_shard}/{n_shard}] {len(pts)} points to process", flush=True)

    print("[load] our MoE model", flush=True)
    ours = rsa.load_model(OUR_HF)              # inference_head='moe', discrete dims set
    ohead = ours.model.action_head
    print("[load] base GR00T model", flush=True)
    base = load_base()

    n_written = 0
    for pi, (ep, t, prompt, ct) in enumerate(pts):
        if all((ep, t, k) in done for k in COMPRESSED_KS):
            continue
        try:
            df = rsa.load_episode_df(RC_ROOT, ep)
            gt16 = np.stack([np.asarray(df.iloc[t + i]["action"], dtype=np.float32) for i in range(H)])
            state = np.asarray(df.iloc[t]["observation.state"], dtype=np.float32)
            frames = {v: rsa.grab_frame(RC_ROOT, ep, v, t)
                      for v in ("left_view", "right_view", "wrist_view")}
            obs = rsa.build_obs(state, frames, prompt)
        except Exception as e:
            print(f"  [skip ep{ep} t{t}] data: {e}", flush=True)
            continue

        # ---- router natural pick under conf-gate (seeded for reproducibility) ----
        torch.manual_seed(ep * 100000 + t)
        ohead.moe_force_pick = -1
        ohead.moe_inference_stochastic = True
        ohead.moe_confidence_threshold = CONF_TAU
        try:
            rout = ours.get_action(obs)
        except Exception as e:
            print(f"  [ep{ep} t{t}] router infer fail: {e}", flush=True); continue
        probs = rout.get("_moe_probs")
        probs = list(np.asarray(probs).flatten()[:4].astype(float)) if probs is not None else [float("nan")] * 4
        picked = int(np.asarray(rout.get("_moe_picked", [-1])).flatten()[0]) if "_moe_picked" in rout else -1

        # ---- base raw 16-step prediction (once per point) ----
        ohead.moe_inference_stochastic = False
        base.inference_head = "main"
        try:
            base_out = base.get_action(obs)
            base_raw = rsa.flatten_action_dict(base_out)[:H]
        except Exception as e:
            print(f"  [ep{ep} t{t}] base infer fail: {e}", flush=True); continue

        # ---- per compressed decoder k: ours (force) vs base-compressed vs GT ----
        for k in COMPRESSED_KS:
            if (ep, t, k) in done:
                continue
            Tk = DECODER_T[k]
            gt_k = rsa.compress_gt(gt16, k)[:Tk]
            # ours: force-pick decoder k
            ohead.moe_force_pick = k
            ohead.moe_inference_stochastic = False
            try:
                our_out = ours.get_action(obs)
                our_pred = rsa.flatten_action_dict(our_out)[:Tk]
            except Exception as e:
                print(f"  [ep{ep} t{t} k{k}] our force infer fail: {e}", flush=True); continue
            ohead.moe_force_pick = -1
            # base naive-compressed to decoder k's target shape/rule
            base_comp = rsa.compress_gt(base_raw, k)[:Tk]

            if our_pred.shape[0] < Tk or base_comp.shape[0] < Tk or gt_k.shape[0] < Tk:
                continue
            m_our = all_metrics(our_pred, gt_k)
            m_base = all_metrics(base_comp, gt_k)
            rec = {
                "ep": int(ep), "t": int(t), "k": int(k),
                "decoder": rsa.DECODER_NAMES[k], "task": ct,
                "router_probs": [float(x) for x in probs],
                "router_pick": picked,
                "router_chose_this": bool(picked == k),
                "ours": m_our, "base": m_base,
                # winner by each metric (lower better for mse/dtw, higher for cosine)
                "ours_better_mse": bool(m_our["mse_cont"] < m_base["mse_cont"]),
                "ours_better_dtw": bool(m_our["dtw_cont"] < m_base["dtw_cont"]),
                "ours_better_cos": bool(m_our["cosine_cont"] > m_base["cosine_cont"]),
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n"); f.flush()
            n_written += 1

        if pi % 25 == 0:
            print(f"  [{pi}/{len(pts)}] ep{ep} t{t} pick={picked} written={n_written}", flush=True)

    print(f"[shard {i_shard}/{n_shard}] DONE, wrote {n_written} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
