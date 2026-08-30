"""Offline replay of the confidence-TTL skip policy on recorded gate_conf.csv.

For each (task, episode) we walk the recorded judge calls in order. Under the
policy, a call is SKIPPED when ttl>0; the reused (stale) decision is compared to
what the fresh call actually decided (recorded) -> "flip" = reuse would have
disagreed. This gives, per threshold set: call reduction vs decision fidelity.
The gripper trigger cannot be replayed (no action data in the csv), so real
in-run fidelity should be >= what we measure here (the trigger forces fresh
calls exactly where flips concentrate).
"""
import csv, glob, os, sys
from collections import defaultdict

TAU = 0.5

def load(gate_dir):
    eps = defaultdict(list)  # (task, ep) -> [conf,...] in step order
    for f in sorted(glob.glob(os.path.join(gate_dir, "*", "gate_conf.csv"))):
        task = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            r = csv.reader(fh)
            head = next(r, None)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    ep, step, conf = int(row[0]), int(row[1]), float(row[2])
                except ValueError:
                    continue
                eps[(task, ep)].append((step, conf))
    for k in eps:
        eps[k].sort()
    return eps

def replay(eps, lo, hi, tmax):
    calls = opps = flips = skips = 0
    quant_real = quant_pol = 0
    for k, seq in eps.items():
        ttl = 0; last_q = None
        for (_, conf) in seq:
            opps += 1
            fresh_q = conf >= TAU
            quant_real += int(fresh_q)
            if ttl > 0 and last_q is not None:
                skips += 1
                ttl -= 1
                if fresh_q != last_q:
                    flips += 1
                quant_pol += int(last_q)
            else:
                calls += 1
                last_q = fresh_q
                d = abs(conf - TAU)
                ttl = 0 if d < lo else (1 if d < hi else tmax)
                quant_pol += int(fresh_q)
    return dict(opps=opps, calls=calls, skip=100*skips/max(opps,1),
                flip=100*flips/max(skips,1) if skips else 0.0,
                flip_of_all=100*flips/max(opps,1),
                qr_real=100*quant_real/max(opps,1), qr_pol=100*quant_pol/max(opps,1))

if __name__ == "__main__":
    runs = {
        "gemma(best=cyc1)":  "output/libero/libero_gemma_cycle1/gate",
        "cosmos(best=cyc3)": "output/libero/libero_cosmos_cycle3/gate",
    }
    grids = [(0.15, 0.30, 3), (0.15, 0.30, 5), (0.10, 0.25, 3), (0.20, 0.35, 3),
             (0.05, 0.15, 3), (0.15, 0.30, 8), (0.02, 0.10, 4)]
    base = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
    for name, rel in runs.items():
        d = os.path.join(base, rel)
        eps = load(d)
        n = sum(len(v) for v in eps.values())
        print(f"\n### {name}  ({len(eps)} episodes, {n} judge opportunities)")
        print(f"{'lo':>5} {'hi':>5} {'tmax':>4} | {'skip%':>6} {'flip%(of skipped)':>17} {'flip%(of all)':>13} | {'quant% real->policy':>20}")
        for (lo, hi, tmax) in grids:
            r = replay(eps, lo, hi, tmax)
            print(f"{lo:>5} {hi:>5} {tmax:>4} | {r['skip']:>6.1f} {r['flip']:>17.1f} {r['flip_of_all']:>13.2f} | {r['qr_real']:>8.1f} -> {r['qr_pol']:.1f}")
