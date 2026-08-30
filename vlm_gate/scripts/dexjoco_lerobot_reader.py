"""Reader for the *packed* (LeRobot v3.0) DexJoCo datasets.

The six single-arm DexJoCo tasks were downloaded in the newer LeRobot layout:

    meta/info.json        data_path  = data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
                          video_path = videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    meta/episodes/chunk-000/file-000.parquet   <- the episode index (NOT episodes.jsonl)
    meta/tasks.parquet                          <- task strings (NOT tasks.jsonl)

i.e. *all* 100 episodes of a task live inside ONE parquet and the per-camera
videos are concatenated into a handful of mp4s.  Our labelling scripts
(`cosmos_1call_v6.py`, `allex_make_tiles.py`, ...) assume the older per-episode
layout (`data/chunk-000/episode_000007.parquet`, one mp4 per episode).

This module bridges the two WITHOUT copying any video bytes.  Everything the
old layout gave us is recoverable, because `meta/episodes/*.parquet` carries an
explicit per-episode index:

    dataset_from_index / dataset_to_index                  -> row slice in the packed parquet
    videos/<video_key>/file_index                          -> which mp4
    videos/<video_key>/from_timestamp / to_timestamp       -> where inside that mp4
    tasks (list[str]) , length

NOTE (verified on water_plant): video timestamps RESET at every file_index, and
the two cameras are split into files DIFFERENTLY (front: 50+50 episodes,
wrist: 97+3).  Never assume the two views share a file boundary.

Usage
-----
    from dexjoco_lerobot_reader import DexjocoDataset
    ds = DexjocoDataset(".../dexjoco_lerobot_datasets/water_plant")
    ds.num_episodes                  # 100
    ds.views                         # ['observation.images.front', 'observation.images.wrist']
    ds.instruction(7)                # 'Grasp the watering can and apply water to the plant.'
    a = ds.actions(7)                # (T, 22) float32
    im = ds.frame(7, 0, ds.views[0]) # (640, 640, 3) uint8 RGB
    tile = ds.tile(7, 0)             # (640, 1280, 3) uint8, views side by side
"""
import json
import os
from collections import OrderedDict

import numpy as np


def _read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pandas()


class DexjocoDataset:
    """Random access to one packed-layout DexJoCo task directory."""

    def __init__(self, root, cache_files=2):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root)
        with open(os.path.join(self.root, "meta", "info.json")) as f:
            self.info = json.load(f)
        self.fps = int(self.info["fps"])
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.features = self.info["features"]
        # camera keys are NOT the same across tasks: five tasks use
        # observation.images.front, click_mouse uses observation.images.ego_right.
        self.views = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self.views.sort(key=lambda k: (k.endswith("wrist"), k))  # base-ish view first

        self.episodes = self._load_episode_index()
        self.num_episodes = len(self.episodes)
        self.tasks = self._load_tasks()
        self._data_cache = OrderedDict()   # file_index -> {col: np.ndarray}
        self._cache_files = cache_files
        # local row offset of each data file (dataset_*_index is global)
        self._data_base = {}
        for e in self.episodes.values():
            fi = e["data_file_index"]
            b = self._data_base.get(fi)
            self._data_base[fi] = e["from_index"] if b is None else min(b, e["from_index"])

    # ------------------------------------------------------------------ meta
    def _episodes_meta_files(self):
        d = os.path.join(self.root, "meta", "episodes")
        out = []
        for chunk in sorted(os.listdir(d)):
            cd = os.path.join(d, chunk)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if f.endswith(".parquet"):
                    out.append(os.path.join(cd, f))
        return out

    def _load_episode_index(self):
        cols = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                "dataset_from_index", "dataset_to_index"]
        for v in self.views:
            cols += [f"videos/{v}/chunk_index", f"videos/{v}/file_index",
                     f"videos/{v}/from_timestamp", f"videos/{v}/to_timestamp"]
        frames = [_read_parquet(p, columns=cols) for p in self._episodes_meta_files()]
        import pandas as pd
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        eps = {}
        for r in df.itertuples(index=False):
            d = dict(zip(df.columns, r))
            ep = int(d["episode_index"])
            tasks = list(d["tasks"]) if d["tasks"] is not None else []
            eps[ep] = {
                "episode_index": ep,
                "length": int(d["length"]),
                "tasks": [str(t) for t in tasks],
                "data_chunk_index": int(d["data/chunk_index"]),
                "data_file_index": int(d["data/file_index"]),
                "from_index": int(d["dataset_from_index"]),
                "to_index": int(d["dataset_to_index"]),
                "video": {v: {"chunk_index": int(d[f"videos/{v}/chunk_index"]),
                              "file_index": int(d[f"videos/{v}/file_index"]),
                              "from_timestamp": float(d[f"videos/{v}/from_timestamp"]),
                              "to_timestamp": float(d[f"videos/{v}/to_timestamp"])}
                          for v in self.views},
            }
        return dict(sorted(eps.items()))

    def _load_tasks(self):
        """meta/tasks.parquet stores the task STRING as the (pandas) index and
        `task_index` as the only column -- the inverse of the old tasks.jsonl."""
        p = os.path.join(self.root, "meta", "tasks.parquet")
        df = _read_parquet(p)
        out = {}
        if "task_index" in df.columns:
            idx = df.index
            for task, ti in zip(idx, df["task_index"].tolist()):
                out[int(ti)] = str(task)
        return out

    def episode_ids(self):
        return list(self.episodes.keys())

    def length(self, ep):
        return self.episodes[int(ep)]["length"]

    def instruction(self, ep):
        """Natural-language instruction for an episode.

        Prefers meta/episodes' own `tasks` list; falls back to tasks.parquet via
        the task_index of the episode's first data row.
        """
        e = self.episodes[int(ep)]
        cand = [t for t in e["tasks"] if isinstance(t, str) and len(t.split()) > 1]
        if cand:
            return cand[0]
        ti = self.column(ep, "task_index")
        if ti is not None and len(ti):
            return self.tasks.get(int(ti[0]), "")
        return ""

    # ------------------------------------------------------------------ data
    def _load_data_file(self, file_index, chunk_index=0, columns=None):
        key = (chunk_index, file_index, tuple(columns) if columns else None)
        if key in self._data_cache:
            self._data_cache.move_to_end(key)
            return self._data_cache[key]
        path = os.path.join(self.root, self.info["data_path"].format(
            chunk_index=chunk_index, file_index=file_index))
        df = _read_parquet(path, columns=list(columns) if columns else None)
        cols = {}
        for c in df.columns:
            v = df[c].to_numpy()
            if v.dtype == object:                 # list/array valued column
                v = np.stack([np.asarray(x) for x in v])
            cols[c] = v
        self._data_cache[key] = cols
        while len(self._data_cache) > self._cache_files:
            self._data_cache.popitem(last=False)
        return cols

    def column(self, ep, name):
        """One column of one episode, as a numpy array (T, ...)."""
        e = self.episodes[int(ep)]
        cols = self._load_data_file(e["data_file_index"], e["data_chunk_index"], columns=[name])
        base = self._data_base[e["data_file_index"]]
        return cols[name][e["from_index"] - base: e["to_index"] - base]

    def actions(self, ep):
        """(T, 22) float32 absolute [arm_pos(3) | arm_rot rotvec(3) | hand(16)]."""
        return np.asarray(self.column(ep, "action"), dtype=np.float32)

    def states(self, ep):
        """(T, 23) float32."""
        return np.asarray(self.column(ep, "observation.state"), dtype=np.float32)

    def episode_frame(self, ep, name="frame_index"):
        return self.column(ep, name)

    # ----------------------------------------------------------------- video
    def video_path(self, ep, view):
        v = self.episodes[int(ep)]["video"][view]
        return os.path.join(self.root, self.info["video_path"].format(
            video_key=view, chunk_index=v["chunk_index"], file_index=v["file_index"]))

    def frames(self, ep, frame_indices, view):
        """{frame_index: (H,W,3) uint8 RGB} for episode-local frame indices.

        One open + one seek + a forward decode to the last wanted frame.  The
        episode's slice of the packed mp4 is located by from_timestamp, which is
        reset per file_index -- so this is exact, not approximate.
        """
        import av
        want = sorted({int(f) for f in frame_indices})
        if not want:
            return {}
        e = self.episodes[int(ep)]
        v = e["video"][view]
        t0 = v["from_timestamp"]
        # target presentation timestamps, in seconds, inside the packed file
        targets = {f: t0 + f / self.fps for f in want}
        tol = 0.5 / self.fps
        out = {}
        path = self.video_path(ep, view)
        with av.open(path) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            seek_to = max(0.0, targets[want[0]] - 1e-6)
            c.seek(int(seek_to / st.time_base), stream=st)
            tlast = targets[want[-1]]
            pending = list(want)
            for fr in c.decode(video=0):
                ts = float(fr.pts * st.time_base)
                if ts > tlast + tol:
                    break
                while pending and ts >= targets[pending[0]] - tol:
                    f = pending.pop(0)
                    if abs(ts - targets[f]) <= tol:
                        out[f] = fr.to_ndarray(format="rgb24")
                    else:                      # requested frame already passed
                        continue
                if not pending:
                    break
        return out

    def frame(self, ep, f, view):
        return self.frames(ep, [f], view).get(int(f))

    def tile(self, ep, f, views=None):
        """Horizontal concat of the views at one frame -- the DexJoCo analogue
        of allex_make_tiles.py's two-view tile.  None if any view is missing."""
        views = views or self.views
        ims = [self.frame(ep, f, v) for v in views]
        if any(x is None for x in ims):
            return None
        return np.concatenate(ims, axis=1)


def open_task(datasets_root, task):
    return DexjocoDataset(os.path.join(datasets_root, task))


TASKS = ["water_plant", "hammer_nail", "pick_bucket",
         "pinch_tongs", "fold_glasses", "click_mouse"]

DEFAULT_ROOT = ("/sjw_alinlab/home/hojin2/quantization_agent_workspace/assets/datasets/"
                "dexjoco_lerobot/dexjoco_lerobot_datasets")


def _selfcheck(tasks=None):
    """`python dexjoco_lerobot_reader.py [task ...]` -- verify by artifact."""
    ok = True
    for t in (tasks or TASKS):
        root = t if os.path.isdir(t) else os.path.join(DEFAULT_ROOT, t)
        ds = DexjocoDataset(root)
        ep = ds.episode_ids()[7]
        a, s = ds.actions(ep), ds.states(ep)
        im = ds.frame(ep, 0, ds.views[0])
        til = ds.tile(ep, 0)
        good = (ds.num_episodes == 100 and a.shape == (ds.length(ep), 22)
                and a.dtype == np.float32 and s.shape[1] == 23
                and im is not None and im.dtype == np.uint8 and im.ndim == 3
                and til is not None and til.shape[1] == im.shape[1] * len(ds.views)
                and len(ds.instruction(ep).split()) > 1)
        ok &= good
        print(f"[{ds.name}] eps={ds.num_episodes} views={ds.views} "
              f"action={a.shape}{a.dtype} state={s.shape} frame={im.shape}{im.dtype} "
              f"tile={til.shape} instr={ds.instruction(ep)!r} OK={good}", flush=True)
    return ok


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(0 if _selfcheck(_sys.argv[1:] or None) else 1)
