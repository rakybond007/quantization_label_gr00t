"""Dump raw action-chunk samples so entropy variants can be compared without a GPU.

DemoSpeedup's entropy has three knobs whose right values are not obvious for RoboCasa:
the KDE bandwidth (the reference hardcodes 1 and leaves its own Scott's-rule estimator
as dead code), which action dimensions enter the kernel (RoboCasa mixes continuous eef
deltas with {-1,+1} discrete dims and four all-zero base_motion dims; ALOHA has neither
problem), and the temporal-ensemble depth K. Recomputing entropy on the GPU for each
combination would be wasteful, so this writes the raw samples once.

Saved per episode: (T, N, H, D) float16 in the model's NORMALIZED action space -- the
space the reference estimator operates in -- from which any (bandwidth, dims, K) can be
evaluated offline.

Speed: one transforms+backbone pass per F frames, and one action-head forward for all
F*N draws. Verified against N separate policy.get_action calls: 11.5x faster with the
sample spread matching to 1.3% and per-frame means to 0.023 (bench_entropy.py).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


def stack_obs(chunk):
    """Batch F observations, mirroring policy.unsqueeze_dict_values' type handling.
    The list branch matters: the language annotation arrives as ['task string'] and must
    become np.array(['task string']); passing the bare string changes the conditioning."""
    out = {}
    for k in chunk[0]:
        vals = [s[k] for s in chunk]
        v0 = vals[0]
        if isinstance(v0, np.ndarray):
            out[k] = np.stack(vals)
        elif isinstance(v0, list):
            out[k] = np.array([v[0] if len(v) == 1 else v for v in vals])
        elif isinstance(v0, torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = np.array(vals)
    return out


def rep(bf, n, batch):
    """repeat_interleave the batch axis only: [f0]*n, [f1]*n, ... so a reshape to
    (F, n, ...) recovers which draw belongs to which frame. Tiling would interleave
    frames; repeating non-batch axes would corrupt them."""
    return BatchFeature(data={
        k: (v.repeat_interleave(n, dim=0)
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == batch else v)
        for k, v in bf.items()})



class EpisodeVideoCache:
    """Decode each episode's videos once instead of once per frame.

    gr00t.utils.video.get_frames_by_timestamps constructs a fresh decord.VideoReader on
    every call AND rescans every frame timestamp to map one requested timestamp to an
    index. Called per frame per camera, that is 3*T reader constructions and O(T^2)
    timestamp work per episode -- which is where the dump was spending most of its time
    (0.196 s/frame measured, against 0.070 for the sampling itself).

    Here each camera is decoded once into memory (T*256*256*3 bytes ~ 13 MB per view for
    a 200-frame episode) and its timestamp table is kept, so a lookup becomes an argmin
    over a cached array. Semantics are copied from LeRobotSingleDataset.get_video and
    get_frames_by_timestamps, including the clamp that pads at episode boundaries.
    """

    def __init__(self, ds, video_keys):
        self.ds, self.video_keys = ds, video_keys
        self.tid, self.frames, self.ts = None, {}, {}

    def prime(self, tid):
        import decord
        if self.tid == tid:
            return
        self.frames, self.ts = {}, {}
        for key in self.video_keys:
            path = self.ds.get_video_path(tid, key.replace("video.", "")).as_posix()
            vr = decord.VideoReader(path)
            n = len(vr)
            self.ts[key] = np.asarray(vr.get_frame_timestamp(range(n)))[:, :1]
            self.frames[key] = vr.get_batch(range(n)).asnumpy()
            del vr
        self.tid = tid

    def get_video(self, trajectory_id, key, base_index):
        step_indices = self.ds.delta_indices[key] + base_index
        ti = self.ds.get_trajectory_index(trajectory_id)
        step_indices = np.clip(step_indices, 0, self.ds.trajectory_lengths[ti] - 1)
        vts = self.ds.curr_traj_data["timestamp"].to_numpy()[step_indices]
        idx = np.abs(self.ts[key] - vts).argmin(axis=0)
        return self.frames[key][idx]


def kde_entropy_reference(x, bandwidth):
    """KDE.kde_entropy from lingxiao-guo/DemoSpeedup, verbatim: no normalising constant,
    -mean(log(density + 1e-8)). Not -sum(p log p), which is a different functional."""
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    dens = np.exp(-d2 / (2 * bandwidth ** 2)).sum(1) / x.shape[0]
    return float(-np.log(dens + 1e-8).mean())


def episode_entropy(samples, dims, bandwidth, K):
    """samples (T, N, H, D) -> (T,). Pools every draw of step t made by the K most recent
    chunks -- the reference's all_time_samples[:, t] temporal ensemble."""
    T, N, H, D = samples.shape
    out = np.full(T, np.nan)
    for t in range(T):
        pool = [samples[j, :, t - j, dims] for j in range(max(0, t - K + 1), t + 1) if t - j < H]
        if pool:
            out[t] = kde_entropy_reference(np.concatenate(pool, 0).astype(np.float64), bandwidth)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-config", default="single_panda_gripper")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--episode-stride", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=10, help="reference get_samples default")
    ap.add_argument("--frame-batch", type=int, default=8, help="frames per backbone pass")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emit", choices=("samples", "entropy"), default="samples",
                    help="samples: raw draws for offline method work (large). "
                         "entropy: the DemoSpeedup statistic only (tiny) -- the production mode.")
    ap.add_argument("--dims", default="0:12", help="real action dims; GR00T pads to 32 and the "
                    "pad slots carry ~20x the draw-spread of any real dim, so they must be excluded")
    ap.add_argument("--bandwidth", default="1", help="reference KDE.kde_entropy hardcodes 1")
    ap.add_argument("--ensemble-k", type=int, default=8)
    args = ap.parse_args()

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE

    dc = DATA_CONFIG_MAP[args.data_config]
    p = Gr00tPolicy(model_path=args.model_path, modality_config=dc.modality_config(),
                    modality_transform=dc.transform(), embodiment_tag=args.embodiment_tag,
                    denoising_steps=args.denoising_steps,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    ds = LeRobotSingleDataset(dataset_path=args.dataset_path,
                              modality_configs=p.get_modality_config(), video_backend="decord",
                              transforms=None, embodiment_tag=args.embodiment_tag)
    os.makedirs(args.out_dir, exist_ok=True)
    N, F = args.n_samples, args.frame_batch

    def draw(chunk):
        f = len(chunk)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            bi, ai = p.model.prepare_input(p.apply_transforms(stack_obs(chunk)))
            bo = p.model.backbone(bi)
            pred = p.model.action_head.get_action(rep(bo, N, f), rep(ai, N, f))["action_pred"]
        a = pred.float().cpu().numpy()                 # (f*N, H, D) normalised space
        return a.reshape(f, N, *a.shape[1:])

    ep_ids = list(ds.trajectory_ids)[args.episode_offset::args.episode_stride][:args.episodes]

    vkeys = p.get_modality_config()["video"].modality_keys
    cache = EpisodeVideoCache(ds, vkeys)
    ds.get_video = cache.get_video          # route frame reads through the cache

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)

    def load_episode(tid):
        """Decode + assemble every step of one episode. Runs off the main thread so the
        next episode is being decoded while the GPU samples the current one."""
        ds.get_trajectory_data(tid) if hasattr(ds, "get_trajectory_data") else ds.get_step_data(tid, 0)
        cache.prime(tid)
        T_ = int(ds.trajectory_lengths[tid])
        return T_, [ds.get_step_data(tid, f) for f in range(T_)]

    t0, written = time.time(), []
    pending = pool.submit(load_episode, ep_ids[0]) if ep_ids else None
    for i, tid in enumerate(ep_ids):
        T, steps = pending.result()
        pending = pool.submit(load_episode, ep_ids[i + 1]) if i + 1 < len(ep_ids) else None
        t_gpu = time.time()
        out = np.concatenate([draw(steps[s:s + F]) for s in range(0, T, F)], axis=0)
        gpu_s = time.time() - t_gpu
        if args.emit == "samples":
            path = os.path.join(args.out_dir, f"ep{int(tid):06d}.npy")
            np.save(path, out.astype(np.float16))   # uncompressed: zlib was pure CPU cost
        else:
            a, b = (int(v) for v in args.dims.split(":"))
            ent = episode_entropy(out.astype(np.float32), slice(a, b), float(args.bandwidth),
                                  args.ensemble_k)
            path = os.path.join(args.out_dir, f"ep{int(tid):06d}_entropy.npz")
            np.savez(path, entropy=ent.astype(np.float32), episode_index=int(tid),
                     frame_index=np.arange(len(ent), dtype=np.int32))
        written.append({"episode_index": int(tid), "T": T, "shape": list(out.shape),
                        "gpu_seconds": round(gpu_s, 1)})
        print(f"[{i+1}/{len(ep_ids)}] ep {tid}: T={T} shape={out.shape} "
              f"gpu={gpu_s:.1f}s total={(time.time()-t0):.0f}s", flush=True)
    pool.shutdown()

    summary = {"episodes": written, "n_samples": N, "frame_batch": F,
               "space": "normalised (model action_pred, pre-unnormalisation)",
               "frames": sum(w["T"] for w in written),
               "seconds": round(time.time() - t0, 1),
               "gpu_seconds": round(sum(w["gpu_seconds"] for w in written), 1),
               "gpu_fraction": round(sum(w["gpu_seconds"] for w in written) / max(time.time() - t0, 1e-9), 3),
               "s_per_frame": round((time.time() - t0) / max(sum(w["T"] for w in written), 1), 4)}
    with open(os.path.join(args.out_dir, f"_summary_off{args.episode_offset}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
