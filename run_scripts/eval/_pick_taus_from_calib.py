"""Read calibration prediction.txt files for all (ckpt × score) settings,
compute per-pair score quantiles AND chunk-level mean score quantiles, and
emit τ candidates per decision rule.

Output: bash export lines that can be sourced before submit_selective_full.sh.
"""
import glob, json, os, sys
import numpy as np

base = "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/robocasa/_smoke_selective"

CKPTS = ["mh_m8", "per_expert_moe"]
SCORES = ["self_agree", "entropy", "hybrid"]

print("# Auto-derived τ candidates from calibration smoke")
print("# Per-pair score quantile → τ for per_pair (Mode α) and prefix_cliff (Mode β)")
print("# Chunk-mean score quantile → τ for chunk_binary (Mode γ)")
print()

for ckpt in CKPTS:
    for score in SCORES:
        n = "1" if score == "self_agree" else "10"
        cal_dir = os.path.join(base, f"{ckpt}_{score}_n{n}", "calibrate")
        all_pair = []
        all_chunk_mean = []
        for d in sorted(glob.glob(os.path.join(cal_dir, "*/"))):
            p = os.path.join(d, "prediction.txt")
            if not os.path.exists(p):
                continue
            for line in open(p):
                if line.startswith("all_pair_scores:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    for chunk_scores in data:
                        all_pair.extend(chunk_scores)
                        all_chunk_mean.append(float(np.mean(chunk_scores)))
        if not all_pair:
            print(f"# {ckpt} / {score}: no calibration data found")
            continue
        a = np.array(all_pair)
        c = np.array(all_chunk_mean)
        ppq = np.quantile(a, [0.25, 0.50, 0.75])
        cmq = np.quantile(c, [0.25, 0.50, 0.75])
        print(f"# {ckpt} / {score}: per-pair p25/50/75 = "
              f"{ppq[0]:.4f} / {ppq[1]:.4f} / {ppq[2]:.4f}   "
              f"chunk-mean p25/50/75 = {cmq[0]:.4f} / {cmq[1]:.4f} / {cmq[2]:.4f}")
        # Suggest mid (p50) per rule
        upper = ckpt.upper().replace("PER_EXPERT_MOE", "MOE")
        sc_upper = score.upper()
        print(f"export TAU_{upper}_{sc_upper}_PERPAIR={ppq[1]:.4f}")
        print(f"export TAU_{upper}_{sc_upper}_PREFIX={ppq[1]:.4f}")
        print(f"export TAU_{upper}_{sc_upper}_CHUNK={cmq[1]:.4f}")
        print()
