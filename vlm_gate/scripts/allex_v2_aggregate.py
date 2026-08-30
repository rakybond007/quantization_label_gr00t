"""Collect the sharded v2 label files, apply the deterministic label layer,
and write the final per-chunk records + a ratio-distribution summary.

The labelling client stores K_pre (what the two prompt stages asked for).  The
hard blocks (allex_v2_common.label_risk_v2, the v1 allex_postprocess rule with
this dataset's own thresholds) need the PREVIOUS chunks - rotation accumulated
while holding - so they are applied here, per episode, in order: a blocked chunk
is pinned to K = 1 (no compression).
"""
import glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_ratio import compress_episode, realised_ratio, ALLOWED_RATIOS
from allex_v2_common import (stage1_confidence, ceiling_from_stage2, final_ratio,
                             label_risk_v2 as label_risk)

OUTDIR = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v2")
PAT = sys.argv[1] if len(sys.argv) > 1 else f"{OUTDIR}/labels_*.jsonl"
REC = f"{OUTDIR}/records.jsonl"
SUM = f"{OUTDIR}/summary.json"
CHUNK = 16

by_ep = collections.defaultdict(dict)
nfiles = 0
for p in sorted(glob.glob(PAT)):
    if os.path.basename(p) == "records.jsonl":
        continue
    nfiles += 1
    for l in open(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        by_ep[r["ep"]][r["f"]] = r          # later files win on duplicates

# p / K_max / K are RECOMPUTED here from the stored slot probabilities and
# descriptors, so allex_v2_common stays the single source of truth: the scoring
# can be retuned without paying for the VLM calls again.  The stored K_pre from
# the labelling run is kept as K_pre_run for comparison.
for ep in by_ep:
    for r in by_ep[ep].values():
        r["K_pre_run"] = r.get("K_pre")
        r["p"] = stage1_confidence([r[f"s1_{q}"] for q in "ABCD"], r)
        r["K_max"] = ceiling_from_stage2(r["task"], *[r[f"s2_{q}"] for q in "ABCD"])
        r["K_pre"] = final_ratio(r["p"], r["K_max"])

recs = []
for ep in sorted(by_ep):
    fs = sorted(by_ep[ep])
    ds = [by_ep[ep][f] for f in fs]
    risk = label_risk(ds)
    for r, blocked in zip(ds, risk):
        r["blocked"] = int(blocked)
        r["K"] = 1.0 if blocked else float(r["K_pre"])
        recs.append(r)

with open(REC, "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------- summary
per_task = collections.defaultdict(lambda: collections.Counter())
p_by_task = collections.defaultdict(list)
kmax_by_task = collections.defaultdict(list)
for r in recs:
    per_task[r["task"]][r["K"]] += 1
    p_by_task[r["task"]].append(r["p"])
    kmax_by_task[r["task"]].append(r["K_max"])

summary = {"n_files": nfiles, "n_chunks": len(recs),
           "n_episodes": len(by_ep), "n_blocked": sum(r["blocked"] for r in recs),
           "tasks": {}}
for t in sorted(per_task):
    c = per_task[t]; n = sum(c.values())
    summary["tasks"][t] = {
        "n_chunks": n,
        "dist": {str(k): round(100.0 * c.get(k, 0) / n, 2) for k in ALLOWED_RATIOS},
        "counts": {str(k): c.get(k, 0) for k in ALLOWED_RATIOS},
        "mean_K": round(float(np.mean([r["K"] for r in recs if r["task"] == t])), 3),
        "mean_p": round(float(np.mean(p_by_task[t])), 3),
        "p_p10_p90": [round(float(np.percentile(p_by_task[t], 10)), 3),
                      round(float(np.percentile(p_by_task[t], 90)), 3)],
        "mean_K_max": round(float(np.mean(kmax_by_task[t])), 3),
        "blocked_pct": round(100.0 * np.mean([r["blocked"] for r in recs if r["task"] == t]), 2),
    }

# realised episode-level compression under the chosen schedule
tot_in = tot_out = 0
for ep in sorted(by_ep):
    fs = sorted(by_ep[ep])
    ratios = [by_ep[ep][f]["K"] for f in fs]
    n = fs[-1] + CHUNK
    idx = compress_episode(n, ratios, chunk=CHUNK)
    tot_in += n; tot_out += len(idx)
summary["realised_overall_ratio"] = round(tot_in / max(1, tot_out), 3)
summary["overall_dist"] = {
    str(k): round(100.0 * sum(1 for r in recs if r["K"] == k) / max(1, len(recs)), 2)
    for k in ALLOWED_RATIOS}
json.dump(summary, open(SUM, "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"-> {REC} ({len(recs)} chunks), {SUM}")
