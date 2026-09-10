"""Turn per-frame entropy into precision / casual segments, following DemoSpeedup.

Reference: robobase/robobase/utils.py :: hdbscan_with_custom_merge +
remove_outliers_isolation_forest. Reproduced step for step:

  1. standardise entropy
  2. IsolationForest(contamination=0.1) outlier removal -- outliers are *interpolated
     from neighbours*, not dropped, so the series keeps its length
  3. standardise again
  4. cluster (standardised_index, standardised_entropy) with HDBSCAN(min_cluster_size=5)
  5. split any cluster larger than 25 into chunks of 25
  6. label each cluster precision or casual

One quirk of step 6 is worth naming rather than silently copying or silently fixing.
The reference writes `if np.mean(cluster_points[:, 1] < 1)`, i.e. *the fraction of
points below 1 sd*, used as a truth value. That is false only when the fraction is
exactly zero, so a cluster is called casual only if **every** point sits at or above
+1 sd. It is a conservative rule, and possibly a typo for `np.mean(...) < 1`, but it is
what produced the published results, so it is what `--rule reference` does. `--rule mean`
offers the other reading for comparison. Whichever is used, the assertion at the end has
to hold: casual segments must have higher mean entropy than precision segments. If the
mapping is ever inverted, the demonstrations get accelerated exactly where they must not
be, and nothing downstream would reveal it.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

PRECISION, CASUAL = 0, 1


def _standardise(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def remove_outliers_isolation_forest(data, contamination=0.1, seed=0):
    """Reference behaviour: outliers are replaced by the mean of their nearest inliers
    (or the single available neighbour at the ends), so length is preserved."""
    from sklearn.ensemble import IsolationForest
    pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(
        data.reshape(-1, 1))
    d = data.copy()
    n = len(d)
    if pred[0] == -1:
        j = 1
        while j < n and pred[j] == -1:
            j += 1
        if j < n:
            d[0] = d[j]
    if pred[-1] == -1:
        j = n - 2
        while j >= 0 and pred[j] == -1:
            j -= 1
        if j >= 0:
            d[-1] = d[j]
    for i in range(1, n - 1):
        if pred[i] == -1:
            a = i - 1
            while a >= 0 and pred[a] == -1:
                a -= 1
            b = i + 1
            while b < n and pred[b] == -1:
                b += 1
            if a >= 0 and b < n:
                d[i] = (d[a] + d[b]) / 2
            elif a >= 0:
                d[i] = d[a]
            elif b < n:
                d[i] = d[b]
    return d


def split_large_clusters(labels, max_size=25):
    labels = labels.copy()
    nxt = labels.max() + 1
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) > max_size:
            for i in range(0, len(idx), max_size):
                labels[idx[i:i + max_size]] = nxt
                nxt += 1
    return labels


def segment(entropy, rule="reference", min_cluster_size=5, max_size=25, seed=0):
    import hdbscan
    e = np.asarray(entropy, dtype=np.float64)
    ok = np.isfinite(e)
    if ok.sum() < min_cluster_size * 2:
        return np.full(len(e), PRECISION, dtype=np.int8)
    e = np.interp(np.arange(len(e)), np.flatnonzero(ok), e[ok])  # fill the K-1 warmup NaNs

    en = _standardise(e)
    en = remove_outliers_isolation_forest(en, seed=seed)
    en = _standardise(en)
    ix = _standardise(np.arange(len(en), dtype=np.float64))
    X = np.stack((ix, en), axis=-1)

    lab = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(X).labels_
    lab = split_large_clusters(lab, max_size=max_size)

    out = np.full(len(en), PRECISION, dtype=np.int8)
    for c in np.unique(lab[lab >= 0]):
        pts = X[lab == c][:, 1]
        casual = (np.mean(pts < 1) == 0) if rule == "reference" else (pts.mean() >= 1)
        out[lab == c] = CASUAL if casual else PRECISION
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=("reference", "mean"), default="reference")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    files = sorted(glob.glob(args.entropy_glob))
    if not files:
        raise SystemExit(f"no entropy files at {args.entropy_glob}")

    labels, stats = {}, []
    for f in files:
        d = np.load(f)
        ent = d["entropy"]
        lab = segment(ent, rule=args.rule)
        tid = int(d["episode_index"])
        labels[str(tid)] = lab
        e = ent[np.isfinite(ent)]
        m_p = float(np.nanmean(ent[lab == PRECISION])) if (lab == PRECISION).any() else np.nan
        m_c = float(np.nanmean(ent[lab == CASUAL])) if (lab == CASUAL).any() else np.nan
        stats.append({"episode_index": tid, "T": int(len(lab)),
                      "casual_frac": float((lab == CASUAL).mean()),
                      "mean_entropy_precision": m_p, "mean_entropy_casual": m_c})

    np.savez_compressed(args.out, **{k: v for k, v in labels.items()})
    cf = np.array([s["casual_frac"] for s in stats])
    mp = np.array([s["mean_entropy_precision"] for s in stats])
    mc = np.array([s["mean_entropy_casual"] for s in stats])
    both = np.isfinite(mp) & np.isfinite(mc)
    print(f"episodes {len(stats)}  frames {sum(s['T'] for s in stats)}")
    print(f"casual fraction: mean {cf.mean():.3f}  p10 {np.percentile(cf,10):.3f}  "
          f"p90 {np.percentile(cf,90):.3f}  (episodes with no casual segment: {(cf==0).sum()})")
    print(f"mean entropy  precision {np.nanmean(mp):.4f} | casual {np.nanmean(mc):.4f}")
    higher = (mc[both] > mp[both]).mean() if both.any() else float("nan")
    print(f"casual entropy > precision entropy in {100*higher:.1f}% of episodes")
    assert both.sum() == 0 or higher > 0.95, (
        "label mapping looks inverted -- casual segments must be the high-entropy ones, "
        "or the demonstrations get accelerated exactly where precision is required")
    summ = args.out.replace(".npz", "_summary.json")
    json.dump({"rule": args.rule, "episodes": len(stats),
               "casual_frac_mean": float(cf.mean()),
               "entropy_precision": float(np.nanmean(mp)),
               "entropy_casual": float(np.nanmean(mc)),
               "casual_higher_frac": float(higher), "per_episode": stats},
              open(summ, "w"), indent=2)
    print(f"wrote {args.out} and {summ}")


if __name__ == "__main__":
    main()
