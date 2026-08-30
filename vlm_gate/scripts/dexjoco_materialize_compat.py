"""Write the *cheap* half of the old per-episode LeRobot layout, in place.

Route: ADAPT, not convert (see docs/DEXJOCO_DATA.md).  Video bytes are never
duplicated -- frames come out of the packed mp4s on demand via
`dexjoco_lerobot_reader.DexjocoDataset`.  But the metadata + action tables are
tiny (~7 MB per task), so we DO materialize those in the legacy names, which
makes action/instruction consumers such as `cosmos_1call_v6.py` work unchanged:

    meta/episodes.jsonl                       {"episode_index","tasks","length"}
    meta/tasks.jsonl                          {"task_index","task"}
    data/chunk-000/episode_{ep:06d}.parquet   action / observation.state / ...

All three names are ADDITIVE -- the v3.0 layout uses
`meta/episodes/chunk-000/file-000.parquet`, `meta/tasks.parquet` and
`data/chunk-000/file-000.parquet`, so nothing is overwritten and the dataset
still loads as a v3.0 LeRobot dataset afterwards.

There is deliberately NO per-episode mp4 and no v2-style `video_path`: writing
one would mean re-encoding 5.2 GB of AV1.  Scripts that want images must use the
reader or the pre-built tiles (`dexjoco_make_tiles.py`).

    python dexjoco_materialize_compat.py                 # all six tasks
    python dexjoco_materialize_compat.py water_plant     # one task
    python dexjoco_materialize_compat.py --force water_plant
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dexjoco_lerobot_reader import DEFAULT_ROOT, TASKS, DexjocoDataset  # noqa: E402

DATA_COLS = ["action", "observation.state", "timestamp", "frame_index",
             "episode_index", "index", "task_index"]


def materialize(root, force=False):
    import pandas as pd
    ds = DexjocoDataset(root)
    meta = os.path.join(ds.root, "meta")
    ep_jsonl = os.path.join(meta, "episodes.jsonl")
    tk_jsonl = os.path.join(meta, "tasks.jsonl")

    with open(tk_jsonl, "w") as f:
        for ti, task in sorted(ds.tasks.items()):
            f.write(json.dumps({"task_index": int(ti), "task": task}) + "\n")

    with open(ep_jsonl, "w") as f:
        for ep in ds.episode_ids():
            e = ds.episodes[ep]
            tasks = e["tasks"] or ([ds.instruction(ep)] if ds.instruction(ep) else [])
            f.write(json.dumps({"episode_index": ep, "tasks": tasks,
                                "length": e["length"]}) + "\n")

    n_written = 0
    for ep in ds.episode_ids():
        e = ds.episodes[ep]
        out_dir = os.path.join(ds.root, "data", f"chunk-{ep // ds.chunks_size:03d}")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"episode_{ep:06d}.parquet")
        if os.path.exists(out) and not force:
            continue
        cols = {}
        for c in DATA_COLS:
            v = ds.column(ep, c)
            cols[c] = list(v) if v.ndim > 1 else v
        pd.DataFrame(cols).to_parquet(out, index=False)
        n_written += 1
    return ds, n_written, ep_jsonl, tk_jsonl


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    tasks = args or TASKS
    for t in tasks:
        root = t if os.path.isdir(t) else os.path.join(DEFAULT_ROOT, t)
        ds, n, ep_jsonl, tk_jsonl = materialize(root, force=force)
        # verify by artifact, not by exit code
        n_ep = sum(1 for _ in open(ep_jsonl))
        pq_dir = os.path.join(ds.root, "data", "chunk-000")
        n_pq = len([x for x in os.listdir(pq_dir) if x.startswith("episode_")])
        import pandas as pd
        probe = pd.read_parquet(os.path.join(pq_dir, "episode_000007.parquet"))
        a = np.stack(probe["action"].to_numpy()).astype(np.float32)
        ok = np.array_equal(a, ds.actions(7))
        print(f"[{ds.name}] episodes.jsonl={n_ep} tasks.jsonl={sum(1 for _ in open(tk_jsonl))} "
              f"episode_*.parquet={n_pq} (new {n}) ep7 action={a.shape} {a.dtype} "
              f"roundtrip_ok={ok}", flush=True)


if __name__ == "__main__":
    main()
