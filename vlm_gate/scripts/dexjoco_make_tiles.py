"""DexJoCo 2-view (base + wrist) tiles for VLM labelling.

The DexJoCo counterpart of `allex_make_tiles.py` / the RoboCasa tile step.
Frames are pulled straight out of the PACKED per-camera mp4s -- no per-episode
video is ever written.  One tile = the two camera views concatenated
horizontally, exactly as the allex two-camera case.

  out/<task>/tiles/ep{ep:04d}_f{f:05d}.png      (640 x 1280 x 3 for 640x640 cams)
  out/<task>/tiles_manifest.txt                 one file name per line

Frame indices are EPISODE-LOCAL (0 .. length-1), like every other labelling
script.  Note the frame field is 5 digits (episodes here reach 1053 frames, so
RoboCasa's 3-digit `_fNNN` convention would truncate) -- use `parse_tile_name`.

    python dexjoco_make_tiles.py water_plant --stride 60
    python dexjoco_make_tiles.py water_plant --stride 60 --episodes 0-4
    python dexjoco_make_tiles.py --all --stride 60
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dexjoco_lerobot_reader import DEFAULT_ROOT, TASKS, DexjocoDataset  # noqa: E402

BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DEFAULT_OUT = f"{BASE}/output/_gate_distill/dexjoco"


def parse_tile_name(nm):
    """'ep0007_f00060.png' -> (7, 60)"""
    stem = os.path.basename(nm).rsplit(".", 1)[0]
    ep, f = stem.split("_f")
    return int(ep[2:]), int(f)


def parse_eps(spec, all_eps):
    if not spec:
        return all_eps
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in set(all_eps)]


def make(task, stride, out_root, episodes=None, tail=16, resize=None, overwrite=False):
    root = task if os.path.isdir(task) else os.path.join(DEFAULT_ROOT, task)
    ds = DexjocoDataset(root)
    out_dir = os.path.join(out_root, ds.name, "tiles")
    os.makedirs(out_dir, exist_ok=True)
    man = os.path.join(out_root, ds.name, "tiles_manifest.txt")
    eps = parse_eps(episodes, ds.episode_ids())
    names = []
    for ep in eps:
        n = ds.length(ep)
        want = list(range(0, max(1, n - tail), stride))
        # one seek + forward decode per (episode, view) -- two decodes per episode
        got = [ds.frames(ep, want, v) for v in ds.views]
        made = 0
        for f in want:
            nm = f"ep{ep:04d}_f{f:05d}.png"
            p = os.path.join(out_dir, nm)
            if os.path.exists(p) and not overwrite:
                names.append(nm)
                continue
            ims = [g.get(f) for g in got]
            if any(x is None for x in ims):
                continue
            im = np.concatenate(ims, axis=1)
            img = Image.fromarray(im)
            if resize:
                img = img.resize((resize * len(ims), resize), Image.BILINEAR)
            img.save(p)
            names.append(nm)
            made += 1
        print(f"[{ds.name}] ep{ep}: {made}/{len(want)} tiles", flush=True)
    with open(man, "w") as fh:
        fh.write("\n".join(sorted(set(names))) + "\n")
    n_files = len([x for x in os.listdir(out_dir) if x.endswith(".png")])
    print(f"[{ds.name}] tiles on disk = {n_files} -> {out_dir}\n"
          f"[{ds.name}] manifest {len(set(names))} lines -> {man}")
    return out_dir, man


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--episodes", default=None, help="e.g. 0-4 or 0,7,19")
    p.add_argument("--tail", type=int, default=16, help="frames at the end to skip (chunk horizon)")
    p.add_argument("--resize", type=int, default=0, help="per-view square size; 0 = native")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    tasks = TASKS if a.all else [a.task]
    if tasks == [None]:
        p.error("give a task name or --all")
    for t in tasks:
        make(t, a.stride, a.out, episodes=a.episodes, tail=a.tail,
             resize=(a.resize or None), overwrite=a.overwrite)


if __name__ == "__main__":
    main()
