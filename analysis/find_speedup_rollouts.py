"""Find robocasa rollout pairs (same task+episode) where OUR method (conf0.7 MoE)
finishes in far fewer action steps than the GR00T baseline, with BOTH succeeding.

robocasa eval uses a fixed per-task seed, so episode_i has the same initial state
across runs -> per-episode step counts are directly comparable.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/robocasa")
BASE = ROOT / "baseline_full_v2_with_action_steps"
# OURS dir overridable via argv[1] (different conf0.7 router variants exist; some
# have a broken 1-frame video-recording bug, so we additionally require a real
# multi-frame OURS video below).
OURS = ROOT / (sys.argv[1] if len(sys.argv) > 1 else
               "moe4_v1_b_only_no_metaq_router_qformer_1q_balance_conf0p7")

LINE = re.compile(r"episode\s+(\d+)\s+is_success:\s+\[\s*(True|False)\]\s+action_steps:\s+(\d+)")


def nframes(mp4):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(mp4)],
            capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def parse(pred):
    """task/prediction.txt -> {ep: (success_bool, steps)}"""
    out = {}
    for ln in pred.read_text().splitlines():
        m = LINE.search(ln)
        if m:
            out[int(m.group(1))] = (m.group(2) == "True", int(m.group(3)))
    return out


def main():
    rows = []  # (task, ep, base_steps, our_steps, gap, ratio)
    for task_dir in sorted(BASE.glob("*/")):
        task = task_dir.name
        bp = BASE / task / "prediction.txt"
        op = OURS / task / "prediction.txt"
        if not (bp.exists() and op.exists()):
            continue
        b, o = parse(bp), parse(op)
        for ep in sorted(set(b) & set(o)):
            (bs, bn), (os_, on) = b[ep], o[ep]
            if not (bs and os_):          # both must succeed
                continue
            # both mp4 must exist
            bv = BASE / task / f"{task}-episode_{ep}.mp4"
            ov = OURS / task / f"{task}-episode_{ep}.mp4"
            if not (bv.exists() and ov.exists()):
                continue
            gap = bn - on
            ratio = bn / on if on > 0 else 0.0
            rows.append((task, ep, bn, on, gap, ratio))

    # Rank by absolute step gap, then keep only those whose OURS (and baseline)
    # video is a real multi-frame recording (filters the 1-frame bug).
    candidates = sorted([r for r in rows if r[4] > 0], key=lambda r: -r[4])
    by_gap = []
    print(f"checking videos of {len(candidates)} gap>0 candidates for usable recordings...")
    for r in candidates:
        task, ep = r[0], r[1]
        ov = OURS / task / f"{task}-episode_{ep}.mp4"
        bv = BASE / task / f"{task}-episode_{ep}.mp4"
        if nframes(ov) > 20 and nframes(bv) > 20:
            by_gap.append(r)
        if len(by_gap) >= 25:
            break
    by_ratio = sorted([r for r in by_gap if r[3] >= 100], key=lambda r: -r[5])

    def show(title, lst, n=15):
        print(f"\n===== {title} =====")
        print(f"{'task':<22}{'ep':>4}{'base':>7}{'ours':>7}{'gap':>7}{'x':>7}")
        for task, ep, bn, on, gap, ratio in lst[:n]:
            print(f"{task:<22}{ep:>4}{bn:>7}{on:>7}{gap:>7}{ratio:>7.2f}")

    print(f"total both-success comparable episodes: {len(rows)}")
    show("TOP by absolute step gap (ours much faster)", by_gap)
    show("TOP by ratio (base/ours, min 100 ours steps)", by_ratio)

    # Print copy-paste mp4 path pairs for the top gap cases.
    print("\n===== mp4 path pairs (top 10 by gap) =====")
    for task, ep, bn, on, gap, ratio in by_gap[:10]:
        print(f"# {task} ep{ep}: baseline {bn} -> ours {on} steps ({ratio:.2f}x faster)")
        print(f"  BASE: {BASE / task / f'{task}-episode_{ep}.mp4'}")
        print(f"  OURS: {OURS / task / f'{task}-episode_{ep}.mp4'}")


if __name__ == "__main__":
    main()
