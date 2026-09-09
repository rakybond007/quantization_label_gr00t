"""Metadata-driven reader for the raw RoboCasa LeRobot format."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

VIEWS = ("left_view", "right_view", "wrist_view")
OFFSETS = (0, 8, 16)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Reader:
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root/"meta/info.json").read_text())
        self.modality = json.loads((self.root/"meta/modality.json").read_text())
        self.episodes = {r["episode_index"]: r for r in map(json.loads, (self.root/"meta/episodes.jsonl").read_text().splitlines())}
        self.fingerprint = {n: digest(self.root/"meta"/n) for n in ("info.json", "modality.json", "episodes.jsonl")}
        self.cached_ep = None
        self.videos = None
        self.decoded = None

    def path(self, ep, video=None):
        values = dict(episode_index=ep, episode_chunk=ep//self.info["chunks_size"])
        if video is not None:
            values["video_key"] = self.modality["video"][video]["original_key"]
        return self.root/self.info["video_path" if video else "data_path"].format(**values)

    def load(self, ep, video=False):
        if ep != self.cached_ep:
            table = pq.read_table(self.path(ep), columns=["action", "episode_index", "index", "timestamp"])
            if len(table) != self.episodes[ep]["length"] or not np.all(table["episode_index"].to_numpy() == ep):
                raise ValueError(f"episode identity/length mismatch: {ep}")
            self.actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            self.indices = table["index"].to_numpy()
            self.timestamps = table["timestamp"].to_numpy()
            if not np.all(np.diff(self.indices) == 1) or self.actions.shape != (len(table), 12):
                raise ValueError(f"invalid raw rows: {ep}")
            if not np.isfinite(self.actions).all():
                raise ValueError(f"nonfinite actions: {ep}")
            self.cached_ep, self.videos, self.decoded = ep, None, None
        if video and self.videos is None:
            from decord import VideoReader, cpu
            self.videos = [VideoReader(str(self.path(ep, v)), ctx=cpu(0), num_threads=1) for v in VIEWS]
            # These source videos have extra frames. Match the training reader's
            # timestamp convention, not a guessed +10 warmup or tail offset.
            self.video_indices = []
            for vr in self.videos:
                times = vr.get_frame_timestamp(np.arange(len(vr)))[:, 0]
                if np.any(np.diff(times) <= 0):
                    raise ValueError(f"nonmonotone video timestamps: {ep}")
                right = np.searchsorted(times, self.timestamps).clip(0, len(times)-1)
                left = (right-1).clip(0, len(times)-1)
                ids = np.where(abs(times[left]-self.timestamps) <= abs(times[right]-self.timestamps), left, right)
                if np.max(abs(times[ids]-self.timestamps)) > .5/self.info["fps"] + 1e-5:
                    raise ValueError(f"video does not cover action timestamps: {ep}")
                self.video_indices.append(ids)
            # Dense stride1 otherwise repeatedly seeks backward for t/t+8/t+16.
            # Keep at most one decoded episode and cap it at 512 MiB per worker.
            estimated = len(self.actions)*3*256*256*3
            if estimated <= 512*(1 << 20):
                self.decoded = [vr.get_batch(ids).asnumpy() for vr, ids in zip(self.videos, self.video_indices)]

    def field(self, actions, key):
        m = self.modality["action"][key]
        return actions[:, m["start"]:m["end"]]

    def facts(self, ep, frame):
        self.load(ep)
        w = self.actions[frame:frame+16]
        if len(w) != 16 or frame+16 >= len(self.actions):
            raise ValueError("incomplete observation window")
        d = self.field(w, "end_effector_position")
        mag = np.linalg.norm(d, axis=1)
        g = self.field(w, "gripper_close")[:, 0]
        mode = self.field(w, "control_mode")[:, 0]
        if not np.isin(g, [0, 1]).all() or not np.isin(mode, [0, 1]).all():
            raise ValueError("discrete command outside metadata range")
        x = dict(command_delta_mean=float(mag.mean()), command_delta_max=float(mag.max()),
                 gripper_start=int(g[0]), gripper_end=int(g[-1]),
                 gripper_change=bool(np.any(g[1:] != g[:-1])),
                 mode_change=bool(np.any(mode[1:] != mode[:-1])),
                 command_direction_reversal=bool(np.any(np.sum(d[1:]*d[:-1], axis=1) < -1e-8)))
        txt = ("RECORDED COMMAND FACTS for 16 intervals (command units, not measured movement): "
               f"mean/peak translation-command norm {mag.mean():.5f}/{mag.max():.5f}; "
               f"gripper command start/end {g[0]:.0f}/{g[-1]:.0f} (0=open, 1=close); "
               f"gripper command changes {x['gripper_change']}; control mode changes {x['mode_change']}; "
               f"successive translation commands reverse direction {x['command_direction_reversal']}. "
               "These facts do not measure contact, grasp security, force or tracking success.")
        return x, txt

    def image(self, ep, frame):
        self.load(ep, video=True)
        canvas = Image.new("RGB", (768, 840), "white")
        draw = ImageDraw.Draw(canvas)
        for col, vr in enumerate(self.videos):
            indices = [frame+k for k in OFFSETS]
            batch = (self.decoded[col][indices] if self.decoded is not None else
                     vr.get_batch(self.video_indices[col][indices]).asnumpy())
            for row, (offset, im) in enumerate(zip(OFFSETS, batch)):
                draw.text((col*256+5, row*280+5), f"{('START','MIDDLE','END')[row]} t+{offset} {VIEWS[col]}", fill="black")
                canvas.paste(Image.fromarray(im).convert("RGB").resize((256, 256)), (col*256, row*280+24))
        return canvas
