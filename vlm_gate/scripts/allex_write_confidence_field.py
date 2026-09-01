"""Write our per-chunk labelling into the dataset as a confidence field.

The recordings under action_quantization/<v>/ carry one directory per labeller,
differing only in the trailing name, and a labeller publishes its result as
`meta/<column>_quantization_confidence.json`. David's file is the reference:
a `column` name, a `description` that states what 0 and 1 mean, `created_at`,
the `params` that produced it, summary stats, and the per-unit rows.

Two things differ here and the file says so rather than hiding them.

David's unit is a SEGMENT — one alpha per subtask occurrence, normalised to
[0,1]. Ours is a 16-step CHUNK, so the rows are chunks and a segment contains
many of them. Anything consuming this should key on (episode_index, frame) and
treat the value as covering that chunk.

David's scale is a speed-up ratio normalised by a pooled alpha_max. Ours is a
ratio K on a fixed grid {1, 2, 2.5, 3}, so `hojin` is reported the same way —
clip((K-1)/(K_grid_max-1)) — with the raw K kept alongside, because the grid is
the thing the robot actually executes and normalising it away loses that.

    python allex_write_confidence_field.py <records.jsonl> <dataset-dir> [column]
"""
import collections
import datetime as dt
import json
import os
import sys

REC = sys.argv[1]
DSD = sys.argv[2]
COL = sys.argv[3] if len(sys.argv) > 3 else "hojin"
GRID = (1.0, 2.0, 2.5, 3.0)
CHUNK = 16

rows = []
for line in open(REC):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
if not rows:
    raise SystemExit(f"no rows in {REC}")

# subtask segments, so each chunk can name the occurrence it falls in
segs = collections.defaultdict(list)
sp = os.path.join(DSD, "meta", "subtasks.jsonl")
if os.path.exists(sp):
    for line in open(sp):
        s = json.loads(line)
        segs[s["episode_index"]].append(s)
    for v in segs.values():
        v.sort(key=lambda s: s["start_frame"])


def seg_of(ep, f):
    for i, s in enumerate(segs.get(ep, [])):
        if s["start_frame"] <= f < s["end_frame"]:
            return i, s.get("label", "")
    return None, ""


gmax = max(GRID)
out, kdist = [], collections.Counter()
for r in rows:
    K = float(r.get("K", 1.0))
    kdist[K] += 1
    si, lab = seg_of(r["ep"], r["f"])
    out.append({
        "episode_index": r["ep"], "start_frame": r["f"],
        "end_frame": min(r["f"] + CHUNK, r["f"] + CHUNK),
        "seg": si, "label": lab or r.get("task", ""),
        "K": K, "K_max": round(float(r.get("K_max", 0)), 4),
        "p": round(float(r.get("p", 0)), 6),
        COL: round(min(1.0, max(0.0, (K - 1.0) / (gmax - 1.0))), 6),
    })

n = len(out)
mean = sum(r[COL] for r in out) / n
doc = {
    "column": COL,
    "description": (
        "normalized quantization confidence per 16-step CHUNK: 0 = 원배속, "
        f"1 = 최대 배속. {COL} = clip((K-1)/({gmax:g}-1), 0, 1) where K is the "
        "ratio the two-stage VLM gate assigned to that chunk, on the grid "
        f"{{{', '.join(f'{g:g}' for g in GRID)}}}. Unit is a chunk, not a segment: "
        "a subtask occurrence contains many rows. Raw K is kept alongside because "
        "the grid is what the robot executes."),
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "params": {
        "chunk": CHUNK,
        "grid": list(GRID),
        "method": "two-stage VLM gate: subtask prior ceiling -> stage-2 checks "
                  "clamp it per chunk -> stage-1 confidence places K inside it, "
                  "K = snap(1 + p*(K_max-1))",
        "merge_limit_rad": float(os.environ.get("ALLEX_MERGE_LIMIT", 0.385)),
        "rot_accum_limit_deg": float(os.environ.get("ALLEX_ROT_LIMIT", 55.0)),
        "gap_rate_limit_m_per_step": float(os.environ.get("ALLEX_GAP_LIMIT", 0.0065)),
        "limits_source": "calibrated on THIS recording (allex_v2_calibrate.py); "
                         "constants are not carried between recordings",
        "dataset": DSD,
    },
    "k_stats": {
        "n_chunks": n,
        "distribution_pct": {f"{k:g}": round(100 * c / n, 2) for k, c in sorted(kdist.items())},
        "mean_K": round(sum(float(k) * c for k, c in kdist.items()) / n, 4),
    },
    "overall_mean": round(mean, 4),
    "chunks": out,
}

dst = os.path.join(DSD, "meta", f"{COL}_quantization_confidence.json")
with open(dst, "w") as fh:
    json.dump(doc, fh, ensure_ascii=False)
print(f"{n} chunks -> {dst}")
print(f"  K distribution : {doc['k_stats']['distribution_pct']}")
print(f"  mean K         : {doc['k_stats']['mean_K']}")
print(f"  overall_mean   : {doc['overall_mean']}")
