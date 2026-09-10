"""Does the policy's own uncertainty point where the VLM teacher points?

DemoSpeedup, AutoSpeed and AAC all use a teacher-free signal — the policy's action
entropy — to decide where a trajectory can be compressed. This project pays a VLM
teacher to decide the same thing. The two have never been put on the same index.

They can be: the committed phase5 labels carry (episode_index, frame_index) for 247,887
RoboCasa chunks, and `demospeedup_entropy.py` writes entropy on that same index.

What this reports, and why each number is here:
  * Spearman(entropy, p_yes_soft) — the headline. Positive means the two agree that a
    chunk is compressible. The soft label is the one the students actually trained on;
    the binary p_yes has a 29.51% tied block that makes rank correlation meaningless.
  * The same against the VLM answers alone (q_A..q_D) and against the computed flags
    alone. If entropy tracks the computed flags but not the VLM answers, then entropy is
    re-deriving arithmetic we already do for free, and the VLM layer is what it cannot
    replace. If it tracks the VLM answers too, the labelling budget is in question.
  * Per-task correlation, because a pooled correlation over tasks with different
    difficulty can be produced entirely by between-task variance. This is the same trap
    `qgate labelcheck` documents.
"""
from __future__ import annotations

import argparse, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan"), int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-glob", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(args.entropy_glob))],
                    ignore_index=True)
    lab = pd.read_parquet(args.labels)
    print(f"entropy rows {len(ent)}  labels rows {len(lab)}")

    df = ent.merge(lab, on=["episode_index", "frame_index"], how="inner")
    print(f"joined rows {len(df)}  episodes {df.episode_index.nunique()}  tasks {df.task.nunique()}")
    if len(df) < 30:
        print("TOO FEW JOINED ROWS -- the labels are on a stride-8 grid; entropy must "
              "cover the same frames. Nothing to conclude.")
        return

    label_cols = [c for c in ("p_yes_soft", "p_yes", "p_raw_soft", "quantize") if c in df]
    vlm_cols = [c for c in ("q_A", "q_B", "q_C", "q_D") if c in df]
    flag_cols = [c for c in df.columns if c.startswith("c_") and not c.endswith("_soft")]
    ent_cols = [c for c in ("entropy", "entropy_pos", "entropy_rot", "gripper_disagreement") if c in df]

    out = {"joined_rows": int(len(df)), "episodes": int(df.episode_index.nunique()),
           "tasks": int(df.task.nunique()), "pooled": {}, "per_task": {}}

    print("\n=== pooled Spearman (entropy vs ...) ===")
    for e in ent_cols:
        print(f"  [{e}]")
        for c in label_cols + vlm_cols + flag_cols:
            r, n = rho(df[e].to_numpy(float), df[c].to_numpy(float))
            out["pooled"][f"{e}|{c}"] = r
            print(f"     {c:26} rho={r:+.4f}  n={n}")

    print("\n=== per task: entropy vs p_yes_soft (pooled correlation can be pure "
          "between-task variance) ===")
    key = "p_yes_soft" if "p_yes_soft" in df else label_cols[0]
    per = []
    for t, g in df.groupby("task"):
        r, n = rho(g["entropy"].to_numpy(float), g[key].to_numpy(float))
        per.append((t, r, n))
    for t, r, n in sorted(per, key=lambda x: -(x[1] if np.isfinite(x[1]) else -9)):
        out["per_task"][t] = r
        print(f"  {str(t)[:44]:46} rho={r:+.4f} n={n}")
    fin = [r for _, r, _ in per if np.isfinite(r)]
    if fin:
        out["per_task_median"] = float(np.median(fin))
        print(f"\n  median per-task rho = {np.median(fin):+.4f} over {len(fin)} tasks")
        print(f"  pooled rho          = {out['pooled'].get(f'entropy|{key}', float('nan')):+.4f}")
        print("  (if pooled >> median, the pooled number is task difficulty, not agreement)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
