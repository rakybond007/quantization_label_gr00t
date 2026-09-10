"""Compare DemoSpeedup entropy variants offline, on samples already dumped.

The reference (lingxiao-guo/DemoSpeedup) fixes three things that do not obviously
transfer from ALOHA to RoboCasa/LIBERO:

  estimator  -mean(log p)   -- not -sum(p log p). Both copies of KDE.kde_entropy agree.
  bandwidth  1              -- hardcoded, overwriting its own Scott's-rule estimate on
                               the line above. ALOHA's 14 dims are all continuous joint
                               positions; RoboCasa's are not.
  dims       all of them    -- fine for ALOHA. GR00T pads its action to 32, and the pad
                               slots have ~22x the draw-spread of any real dim, so
                               "all" would measure padding noise.

So the estimator is copied exactly, and bandwidth/dims are swept. Selection is on
estimator sanity -- does entropy vary at all, is it not dominated by constant or padded
dims -- NOT on agreement with our own gate labels, which would be circular.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def kde_entropy_reference(x: np.ndarray, bandwidth: float) -> float:
    """Exactly KDE.kde_entropy: no normalising constant, -mean(log(density + 1e-8))."""
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    dens = np.exp(-d2 / (2 * bandwidth ** 2)).sum(1) / x.shape[0]
    return float(-np.log(dens + 1e-8).mean())


def scott(x: np.ndarray) -> float:
    n, d = x.shape
    return float(max(x.std(axis=0).mean(), 1e-6) * n ** (-1.0 / (d + 4)))


def episode_entropy(samples: np.ndarray, dims: slice, bandwidth, K: int) -> np.ndarray:
    """samples: (T, N, H, D) -> (T,) entropy, pooling the K most recent chunks that
    predicted each step (the reference's all_time_samples[:, t] ensemble)."""
    T, N, H, D = samples.shape
    out = np.full(T, np.nan)
    for t in range(T):
        pool = [samples[j, :, t - j, dims] for j in range(max(0, t - K + 1), t + 1) if t - j < H]
        if not pool:
            continue
        S = np.concatenate(pool, axis=0).astype(np.float64)
        h = scott(S) if bandwidth == "scott" else float(bandwidth)
        out[t] = kde_entropy_reference(S, h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-glob", required=True)
    ap.add_argument("--ensemble-k", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=12)
    ap.add_argument("--dim-sets", default="0:12,0:7,0:6")
    ap.add_argument("--bandwidths", default="1,0.1,scott")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.samples_glob))[: args.max_episodes]
    if not files:
        raise SystemExit(f"no sample files at {args.samples_glob}")
    eps = []
    for f in files:
        a = np.load(f)
        eps.append((os.path.basename(f), (a["samples"] if hasattr(a, "files") else a).astype(np.float32)))
    print(f"{len(eps)} episodes, shape {eps[0][1].shape} (T, N, H, D)\n")

    D = eps[0][1].shape[-1]
    print("per-slice draw-spread (mean sd across the N draws) -- what each dim contributes:")
    allsd = np.concatenate([s.std(axis=1).reshape(-1, D) for _, s in eps]).mean(0)
    for name, sl in (("eef_position 0:3", slice(0, 3)), ("eef_rotation 3:6", slice(3, 6)),
                     ("gripper 6:7", slice(6, 7)), ("base_motion 7:11", slice(7, 11)),
                     ("control_mode 11:12", slice(11, 12)), ("padding 12:", slice(12, D))):
        if sl.stop is not None and sl.start >= D:
            continue
        print(f"   {name:20} {allsd[sl].mean():.5f}")

    # The estimator saturates at both ends: a kernel far wider than the point cloud makes
    # every density ~1 so entropy -> 0, and one far narrower makes each point see only
    # itself so density -> 1/M and entropy -> log(M). Either way frames stop being
    # distinguishable. `sat` places the mean on that 0..log(M) scale; useful settings sit
    # away from both ends, and `sd` says how much the frames actually separate.
    M = eps[0][1].shape[1] * args.ensemble_k
    logM = float(np.log(M))
    rows = []
    print(f"\nM = N*K = {M}, so entropy is bounded in [0, log M = {logM:.3f}]")
    print(f"\n{'dims':8} {'bandwidth':>10} {'ent mean':>10} {'ent sd':>9} {'sat':>7} {'nan%':>6}")
    print("-" * 56)
    for dspec in args.dim_sets.split(","):
        a, b = (int(v) for v in dspec.split(":"))
        for bw in args.bandwidths.split(","):
            bwv = bw if bw == "scott" else float(bw)
            vals = np.concatenate([episode_entropy(s, slice(a, b), bwv, args.ensemble_k)
                                   for _, s in eps])
            fin = vals[np.isfinite(vals)]
            sat = fin.mean() / logM if fin.size else float("nan")
            rows.append({"dims": dspec, "bandwidth": bw, "mean": float(fin.mean()),
                         "sd": float(fin.std()), "saturation": float(sat),
                         "nan_frac": float(1 - fin.size / vals.size)})
            flag = "  <- collapsed" if (sat < 0.05 or sat > 0.90) else ""
            print(f"{dspec:8} {bw:>10} {fin.mean():10.4f} {fin.std():9.4f} {sat:7.3f} "
                  f"{100*(1-fin.size/vals.size):6.1f}{flag}")
    print("\n  sat = mean / log(M). Near 0 -> kernel far too wide (all densities ~1);")
    print("  near 1 -> far too narrow (each point sees only itself). Both destroy the ranking.")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
