"""Fold the six per-task DexJoCo tile manifests into ONE flat manifest.

RoboCasa has a single episode-index space, so its manifest can be a bare list of
`ep0007_f080.png`.  DexJoCo does not: each of the six tasks numbers its episodes
0..99 independently, so a bare tile name is ambiguous.  Every line here is
therefore `<task>/<tile-file-name>` -- still one tile per line, still a flat text
file, but the task is carried in the line so nothing can silently merge.

    python dexjoco_tiles_manifest.py                       # default tile root
    python dexjoco_tiles_manifest.py --tiles <root> --out <file>

The per-task manifests are written by `dexjoco_make_tiles.py --out <root>`; this
reads those files, never the directories (a recursive listing of ~13k tiles on
the shared mount is exactly the kind of thing that should not be done casually).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dexjoco_label_common import DEFAULT_TILES, DEFAULT_MANIFEST, TASK_ORDER  # noqa: E402


def build(tiles_root, out_path):
    lines = []
    missing = []
    for task in TASK_ORDER:
        man = os.path.join(tiles_root, task, "tiles_manifest.txt")
        if not os.path.exists(man):
            missing.append(task)
            continue
        names = sorted({l.strip() for l in open(man) if l.strip()})
        lines += [f"{task}/{n}" for n in names]
        print(f"[{task}] {len(names)} tiles", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"manifest {len(lines)} lines -> {out_path}")
    if missing:
        print(f"WARNING: no per-task manifest for {', '.join(missing)}")
    return len(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default=DEFAULT_TILES)
    p.add_argument("--out", default=DEFAULT_MANIFEST)
    a = p.parse_args()
    raise SystemExit(0 if build(a.tiles, a.out) else 1)
