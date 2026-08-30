"""Aggregate dexjoco dual-arm eval: success rate + success-only rollout length.

Step count is read from the per-episode video frame count (the sync client writes
one frame per env step), since dexjoco eval does not log action_steps to a file.
"""
import subprocess
from pathlib import Path

ROOT = Path("/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/dexjoco")
MODELS = [
    ("baseline 50k", "dual_arm_multitask_baseline_50k"),
    ("baseline 60k", "dual_arm_multitask_baseline_60k"),
    ("MoE conf0.7 50k", "dual_arm_multitask_moe4_v1_balance_50k_stochastic_conf0p7"),
    ("MoE conf0.7 60k", "dual_arm_multitask_moe4_v1_balance_60k_stochastic_conf0p7"),
]
TASKS = ["bimanual_assembly", "bimanual_hanoi", "bimanual_microwave_cook",
         "bimanual_photograph", "bimanual_unlock_ipad"]


def nframes(mp4):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(mp4)],
            capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def succ_steps(task_dir):
    """mean frame count over success episodes (steps), minus the reset frame."""
    steps = []
    for ep in sorted(task_dir.glob("episode_*_success")):
        ego = ep / "ego.mp4"
        if ego.exists():
            f = nframes(ego)
            if f > 1:
                steps.append(f - 1)  # first frame is the reset frame
    return steps


def main():
    # per (model, task): (n_succ, n_total, mean_steps)
    grid = {}
    for label, d in MODELS:
        for t in TASKS:
            td = ROOT / d / t
            if not td.is_dir():
                grid[(label, t)] = (0, 0, None)
                continue
            total = len(list(td.glob("episode_*_success"))) + len(list(td.glob("episode_*_failure")))
            steps = succ_steps(td)
            n_succ = len(list(td.glob("episode_*_success")))
            mean = sum(steps) / len(steps) if steps else None
            grid[(label, t)] = (n_succ, total, mean)

    # ---- success rate table ----
    print("\n================ SUCCESS RATE (succ/50) ================")
    hdr = f"{'task':<22}" + "".join(f"{lab:>18}" for lab, _ in MODELS)
    print(hdr)
    for t in TASKS:
        row = f"{t.replace('bimanual_',''):<22}"
        for lab, _ in MODELS:
            ns, tot, _ = grid[(lab, t)]
            row += f"{f'{ns}/{tot}':>18}"
        print(row)
    # averages
    row = f"{'AVG success %':<22}"
    for lab, _ in MODELS:
        ns = sum(grid[(lab, t)][0] for t in TASKS)
        tot = sum(grid[(lab, t)][1] for t in TASKS)
        row += f"{f'{100*ns/tot:.1f}%':>18}"
    print(row)

    # ---- success-only rollout length table ----
    print("\n========== SUCCESS-ONLY ROLLOUT LENGTH (mean steps) ==========")
    print(hdr)
    for t in TASKS:
        row = f"{t.replace('bimanual_',''):<22}"
        for lab, _ in MODELS:
            _, _, m = grid[(lab, t)]
            row += f"{(f'{m:.1f}' if m else '-'):>18}"
        print(row)
    # overall mean step (success-only, micro-averaged over success episodes)
    row = f"{'AVG steps (succ)':<22}"
    for lab, d in MODELS:
        allsteps = []
        for t in TASKS:
            allsteps += succ_steps(ROOT / d / t)
        row += f"{(f'{sum(allsteps)/len(allsteps):.1f}' if allsteps else '-'):>18}"
    print(row)


if __name__ == "__main__":
    main()
