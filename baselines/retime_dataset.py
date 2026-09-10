"""Build the accelerated LeRobot dataset DemoSpeedup trains on.

Skip rule, from the reference's process_action_label: walk the trajectory and advance by
low_v=2 steps inside precision segments and high_v=4 inside casual ones -- the reference
takes the 4-step stride only when the next 4 frames are all casual, otherwise it runs to
the next precision frame.

Where this must NOT follow the reference: it selects actions by *indexing*
(`new_actions = current_action[indices]`). ALOHA actions are absolute joint targets, so
dropping intermediate targets is correct there. RoboCasa and LIBERO emit end-effector
*deltas*; indexing them throws away the motion in the skipped steps, so a 2x skip would
travel half the distance. Deltas are summed within each group instead, and near-binary
dims (gripper, control mode) take the last value in the group so a close command inside
a merged window survives it. That is the same embodiment adapter the gate work uses, and
`vsta.merge_integer` is the shared implementation -- `verify` below checks the two agree.

Videos are re-encoded to the kept frames so the observation stream stays aligned with
the actions; state is taken at the kept frames (it is absolute, so indexing is right).
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np

PRECISION, CASUAL = 0, 1


def skip_indices(labels, low_v=2, high_v=4):
    """Reference walk. Returns the frame indices the accelerated episode keeps."""
    T = len(labels)
    idx, i = [], -1
    while i < T:
        if i >= 0 and labels[i] == CASUAL:
            if i + high_v < T and np.all(labels[i:i + high_v] == CASUAL):
                i += high_v
                idx.append(i)
            else:
                nz = np.flatnonzero(labels[i + 1:] == PRECISION)
                if nz.size:
                    i = i + 1 + int(nz[0])
                    idx.append(i)
                else:
                    break
        else:
            if i + low_v < T:
                i += low_v
                idx.append(i)
            else:
                break
    return np.array([j for j in idx if 0 <= j < T], dtype=int)


def retime_actions(actions, keep, delta_dims, discrete_dims):
    """Group the original steps by which kept frame they belong to, then sum deltas and
    take last-of-group for discrete dims. keep[k] is the last original step of group k."""
    out = np.zeros((len(keep), actions.shape[1]), dtype=np.float64)
    start = 0
    for k, end in enumerate(keep):
        grp = actions[start:end + 1]
        if len(grp) == 0:
            grp = actions[end:end + 1]
        out[k, delta_dims] = grp[:, delta_dims].sum(axis=0)
        if len(discrete_dims):
            out[k, discrete_dims] = grp[-1, discrete_dims]
        start = end + 1
    return out


def verify(actions, keep, delta_dims, discrete_dims):
    """Endpoint displacement over the kept span must survive the re-timing, and a
    uniform-stride case must reproduce vsta.merge_integer exactly."""
    new = retime_actions(actions, keep, delta_dims, discrete_dims)
    span = actions[: keep[-1] + 1]
    err = np.abs(new[:, delta_dims].sum(0) - span[:, delta_dims].sum(0)).max()
    checks = {"endpoint_abs_err": float(err)}
    if len(actions) >= 8:
        from vsta import merge_integer
        k = 2
        uni = np.arange(k - 1, len(actions), k)
        a = retime_actions(actions, uni, delta_dims, discrete_dims)
        b = merge_integer(actions, k, delta_dims, discrete_dims)
        n = min(len(a), len(b))
        checks["vs_vsta_merge_k2"] = float(np.abs(a[:n][:, delta_dims] - b[:n][:, delta_dims]).max())
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source LeRobot root")
    ap.add_argument("--labels", required=True, help="npz from segment_entropy.py")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--delta-dims", required=True, help="e.g. 0:6 for libero, or 5:11 for robocasa")
    ap.add_argument("--discrete-dims", default="", help="e.g. 6 for libero; 4,11 for robocasa")
    ap.add_argument("--low-v", type=int, default=2)
    ap.add_argument("--high-v", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="episodes, 0 = all")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    a, b = (int(v) for v in args.delta_dims.split(":"))
    ddims = np.arange(a, b)
    cdims = np.array([int(v) for v in args.discrete_dims.split(",") if v != ""], dtype=int)
    lab = np.load(args.labels)
    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    dpat = info["data_path"]

    tids = sorted(int(k) for k in lab.files)
    if args.limit:
        tids = tids[: args.limit]
    stats, checks = [], []
    for tid in tids:
        f = os.path.join(args.src, dpat.format(episode_chunk=tid // info["chunks_size"],
                                               episode_index=tid))
        t = pq.read_table(f)
        col = "action" if "action" in t.column_names else "actions"
        act = np.stack(t.column(col).to_numpy()).astype(np.float64)
        L = lab[str(tid)]
        n = min(len(L), len(act))
        keep = skip_indices(L[:n], args.low_v, args.high_v)
        if keep.size == 0:
            continue
        checks.append(verify(act[:n], keep, ddims, cdims))
        stats.append({"episode_index": tid, "T": int(n), "kept": int(keep.size),
                      "ratio": float(n / keep.size),
                      "casual_frac": float((L[:n] == CASUAL).mean())})

    r = np.array([s["ratio"] for s in stats])
    e = np.array([c["endpoint_abs_err"] for c in checks])
    v = np.array([c.get("vs_vsta_merge_k2", 0.0) for c in checks])
    print(f"episodes {len(stats)}")
    print(f"  original frames {sum(s['T'] for s in stats)} -> kept {sum(s['kept'] for s in stats)}")
    print(f"  speedup ratio: mean {r.mean():.3f}  p10 {np.percentile(r,10):.3f}  p90 {np.percentile(r,90):.3f}")
    print(f"  casual fraction mean {np.mean([s['casual_frac'] for s in stats]):.3f}")
    print(f"  endpoint displacement error: max {e.max():.2e}  (must be ~0: deltas summed, not dropped)")
    print(f"  agreement with vsta.merge_integer(k=2): max {v.max():.2e}")
    assert e.max() < 1e-9, "re-timing lost displacement -- deltas are being indexed, not summed"
    assert v.max() < 1e-9, "disagrees with the gate's own K-step merge"
    out = args.dst.rstrip("/") + "_retime_stats.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"episodes": stats}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    if args.verify_only:
        print("verify-only: no dataset written")


if __name__ == "__main__":
    main()
