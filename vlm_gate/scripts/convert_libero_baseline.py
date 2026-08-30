"""Convert LIBERO suite/N_results.txt baselines to evolver-compatible per-task
prediction.txt (one dir per <suite>_<taskidx>), so evolve_gate_prompt.py can use
them as --raw / --k2 references (same as robocasa)."""
import os, re, glob, sys
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
def convert(src, dst):
    n = 0
    for suite in SUITES:
        for f in sorted(glob.glob(os.path.join(src, suite, "[0-9]*_results.txt"))):
            txt = open(f).read()
            tidx = os.path.basename(f).split("_")[0]
            r = re.search(r"success rate:\s*([0-9.]+)", txt)
            N = re.search(r"Total episodes:\s*(\d+)", txt)
            if not r: continue
            rate = float(r.group(1)); nep = int(N.group(1)) if N else 50
            # per-ep records if present
            recs = re.findall(r"^\s*(\d+)\t(True|False)\t(\d+)\s*$", txt, re.M)
            td = os.path.join(dst, f"{suite}_{tidx}"); os.makedirs(td, exist_ok=True)
            with open(os.path.join(td, "prediction.txt"), "w") as o:
                if recs:
                    for (idx, ok, st) in recs:
                        o.write(f"episode {idx} is_success: [{' True' if ok=='True' else 'False'}] action_steps: {st}\n")
                else:
                    nsucc = round(rate * nep)
                    for i in range(nep):
                        ok = i < nsucc
                        # no per-ep step data -> omit action_steps (evolver succ_steps -> None, graceful)
                        o.write(f"episode {i} is_success: [{' True' if ok else 'False'}]\n")
                o.write(f"is_success: {rate:.4f}\n")
            n += 1
    return n
src_raw = os.path.expanduser("~/multigpu_workspace/Isaac-GR00T/output/libero/baseline_v2_with_action_steps")
src_k2  = os.path.expanduser("~/multigpu_workspace/Isaac-GR00T/output/libero/baseline_bs32_hf_quantize_K2")
nr = convert(src_raw, "output/libero/baseline_raw")
nk = convert(src_k2,  "output/libero/baseline_K2")
print(f"converted raw={nr} tasks, K2={nk} tasks")
