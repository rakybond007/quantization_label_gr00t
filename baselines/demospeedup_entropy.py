"""Per-frame action entropy from a GR00T policy, following DemoSpeedup (arXiv:2506.05064).

DemoSpeedup's claim is that a policy's own uncertainty already marks where a
demonstration can be accelerated: frames where sampled action chunks *agree* (low
entropy) are precision moments, frames where they *disagree* (high entropy) are casual
stretches that survive downsampling.

That is the same judgement this project asks a VLM teacher to make, obtained for free
from the policy we already have. Whether the two agree is a measurement, not an opinion,
and this script produces the left-hand side of it. Sign convention matches the gate:
high entropy = compressible = high p_yes.

Estimator (paper Eq. 1-2). For target step t, pool predictions of t made from the K most
recent observations, across N noise draws each:

    S_t   = { a_ij[t] : i = 1..N, j = t-K+1..t }        |S_t| = N*K
    p(x)  = (1 / (|S_t| h sqrt(2pi))) * sum_s exp( -||x - s||^2 / (2 h^2) )
    H_t   = - sum_{s in S_t} p(s) log p(s)

Two honest notes on the implementation:
  * The paper uses ACT/DP latents; GR00T is flow-matching, so a fresh noise draw per
    call is the equivalent of their z_i. Each call to get_action re-draws it.
  * Entropy is computed on the position+rotation delta dims only. The gripper is a
    near-binary dim whose KDE is degenerate, and pooling it with continuous dims would
    let it dominate the bandwidth. It is reported separately as gripper_disagreement.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np


def kde_entropy(samples: np.ndarray, bandwidth: float) -> float:
    """Resubstitution KDE entropy of a sample set, shape (M, D)."""
    M, D = samples.shape
    if M < 2:
        return float("nan")
    d2 = ((samples[:, None, :] - samples[None, :, :]) ** 2).sum(-1)     # (M, M)
    h = bandwidth
    dens = np.exp(-d2 / (2 * h * h)).sum(1) / (M * h * np.sqrt(2 * np.pi) ** D)
    dens = np.maximum(dens, 1e-300)
    return float(-(dens * np.log(dens)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-config", default="single_panda_gripper")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20, help="how many trajectories")
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--episode-stride", type=int, default=1,
                    help="take every Nth trajectory. The 24 RoboCasa tasks sit in "
                         "contiguous blocks of 300 episodes, so consecutive sampling "
                         "yields one task and no per-task correlation is possible.")
    ap.add_argument("--n-samples", type=int, default=8, help="noise draws per observation (N)")
    ap.add_argument("--ensemble-k", type=int, default=8, help="past observations pooled (K)")
    ap.add_argument("--bandwidth", type=float, default=0.1)
    ap.add_argument("--pos-key", default="action.end_effector_position")
    ap.add_argument("--rot-key", default="action.end_effector_rotation")
    ap.add_argument("--grip-key", default="action.gripper_close")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    dc = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    ds = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="decord",
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    tmeta = ds.trajectory_lengths if hasattr(ds, "trajectory_lengths") else None
    n_traj = len(ds.trajectory_ids)
    print(f"dataset: {len(ds)} steps, {n_traj} trajectories", flush=True)

    rows = []
    t0 = time.time()
    ep_ids = list(ds.trajectory_ids)[args.episode_offset :: args.episode_stride][: args.episodes]
    for ei, tid in enumerate(ep_ids):
        T = int(ds.trajectory_lengths[tid]) if tmeta is not None else None
        if T is None:
            raise RuntimeError("dataset does not expose trajectory_lengths")
        # chunks[f] = (N, H, D_pos+D_rot+1) predictions made from frame f
        chunks: dict[int, np.ndarray] = {}
        for f in range(T):
            step = ds.get_step_data(tid, f)
            draws = []
            for _ in range(args.n_samples):
                a = policy.get_action(step)
                draws.append(np.concatenate(
                    [np.asarray(a[args.pos_key]).reshape(-1, 3),
                     np.asarray(a[args.rot_key]).reshape(-1, 3),
                     np.asarray(a[args.grip_key]).reshape(-1, 1)], axis=1))
            chunks[f] = np.stack(draws)                       # (N, H, 7)
            if f >= args.ensemble_k - 1:
                # pool every prediction of step f that the last K frames made
                pool = []
                for j in range(max(0, f - args.ensemble_k + 1), f + 1):
                    off = f - j
                    if off < chunks[j].shape[1]:
                        pool.append(chunks[j][:, off, :])      # (N, 7)
                S = np.concatenate(pool, axis=0)               # (N*K', 7)
                rows.append({
                    "episode_index": int(tid),
                    "frame_index": int(f),
                    "entropy": kde_entropy(S[:, :6], args.bandwidth),
                    "entropy_pos": kde_entropy(S[:, :3], args.bandwidth),
                    "entropy_rot": kde_entropy(S[:, 3:6], args.bandwidth),
                    "gripper_disagreement": float(S[:, 6].std()),
                    "n_pool": int(S.shape[0]),
                })
            # keep memory bounded
            drop = f - args.ensemble_k - 20
            chunks.pop(drop, None)
        print(f"[{ei+1}/{len(ep_ids)}] traj {tid}: T={T} rows={len(rows)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    summary = {"rows": len(df), "episodes": len(ep_ids), "n_samples": args.n_samples,
               "ensemble_k": args.ensemble_k, "bandwidth": args.bandwidth,
               "entropy_mean": float(df.entropy.mean()), "entropy_sd": float(df.entropy.std()),
               "seconds": round(time.time() - t0, 1), "out": args.out}
    with open(args.out.replace(".parquet", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
