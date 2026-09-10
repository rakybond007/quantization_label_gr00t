"""Aggregate closed-loop results for one method, on either benchmark.

Reads through /s3ckpt because /ckpt is per-AZ and the array elements land on whichever
partition the scheduler picks. Reports the count alongside every number: a task with a
handful of finished episodes has a success rate quantised coarsely enough to invent
differences, and the repo's own history has a per-task regression reported from 1 of 5
episodes against 30 of 50.
"""
import argparse, glob, os, re
import numpy as np

RC_TASKS = ["TurnSinkSpout","TurnOnStove","TurnOnSinkFaucet","TurnOnMicrowave","TurnOffStove",
            "TurnOffSinkFaucet","TurnOffMicrowave","PnPStoveToCounter","PnPSinkToCounter",
            "PnPMicrowaveToCounter","PnPCounterToStove","PnPCounterToSink","PnPCounterToMicrowave",
            "PnPCounterToCab","PnPCabToCounter","OpenSingleDoor","OpenDrawer","OpenDoubleDoor",
            "CoffeeSetupMug","CoffeeServeMug","CoffeePressButton","CloseSingleDoor","CloseDrawer",
            "CloseDoubleDoor"]
SUITES = ["libero_spatial","libero_object","libero_goal","libero_10"]


def robocasa(base, min_ep):
    rows = []
    for t in RC_TASKS:
        p = os.path.join(base, t, "prediction.txt")
        if not os.path.exists(p):
            rows.append((t, 0, np.nan, np.nan)); continue
        s, k = [], []
        for ln in open(p):
            m = re.search(r"is_success:\s*\[\s*(True|False)\s*\].*action_steps:\s*(\d+)", ln)
            if m:
                s.append(m.group(1) == "True"); k.append(int(m.group(2)))
        ok = [x for x, g in zip(k, s) if g]
        rows.append((t, len(s), sum(s)/len(s) if s else np.nan,
                     sum(ok)/len(ok) if ok else np.nan))
    print(f"{'task':24} {'n':>4} {'SR':>7} {'steps(succ)':>12}")
    print("-" * 50)
    for t, n, sr, st in rows:
        flag = "" if n >= min_ep else "  <- incomplete"
        print(f"{t:24} {n:4} {sr:7.3f} {st:12.1f}{flag}")
    done = [(sr, st) for _, n, sr, st in rows if n >= min_ep and np.isfinite(sr)]
    if done:
        srs = [x for x, _ in done]; sts = [y for _, y in done if np.isfinite(y)]
        print("-" * 50)
        print(f"{'MACRO over complete':24} {len(done):4} {np.mean(srs):7.3f} {np.mean(sts):12.1f}")
        print(f"  standard error of the mean success rate: {np.std(srs)/np.sqrt(len(srs)):.4f}")
    miss = [t for t, n, _, _ in rows if n < min_ep]
    if miss:
        print(f"  incomplete or missing ({len(miss)}): {', '.join(miss)}")


def libero(base, min_ep):
    print(f"{'suite':18} {'tasks':>6} {'episodes':>9} {'SR':>7} {'steps(succ)':>12}")
    print("-" * 56)
    allsr, allst, tot = [], [], 0
    for s in SUITES:
        srs, sts, neps = [], [], 0
        for f in sorted(glob.glob(os.path.join(base, s, "*_results.txt"))):
            rec = []
            for ln in open(f):
                m = re.match(r"\s+(\d+)\t(True|False)\t(\d+)", ln)
                if m:
                    rec.append((m.group(2) == "True", int(m.group(3))))
            if len(rec) < min_ep:
                continue
            neps += len(rec)
            srs.append(sum(r[0] for r in rec)/len(rec))
            ok = [k for g, k in rec if g]
            if ok:
                sts.append(sum(ok)/len(ok))
        if srs:
            print(f"{s:18} {len(srs):6} {neps:9} {np.mean(srs):7.3f} "
                  f"{(np.mean(sts) if sts else np.nan):12.1f}")
            allsr += srs; allst += sts; tot += neps
        else:
            print(f"{s:18} {0:6} {0:9} {'-':>7} {'-':>12}")
    if allsr:
        print("-" * 56)
        print(f"{'OVERALL':18} {len(allsr):6} {tot:9} {np.mean(allsr):7.3f} {np.mean(allst):12.1f}")
        print(f"  standard error of the mean success rate: {np.std(allsr)/np.sqrt(len(allsr)):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="method directory under the eval root")
    ap.add_argument("--bench", choices=("robocasa", "libero"), required=True)
    ap.add_argument("--min-episodes", type=int, default=50,
                    help="tasks with fewer finished episodes are excluded from the macro")
    a = ap.parse_args()
    print(f"{a.bench}  {a.base}\n")
    (robocasa if a.bench == "robocasa" else libero)(a.base, a.min_episodes)
