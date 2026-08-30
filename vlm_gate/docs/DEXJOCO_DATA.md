# DexJoCo LeRobot datasets — layout, chosen route, and how to reproduce

The six single-arm DexJoCo tasks (`water_plant`, `hammer_nail`, `pick_bucket`,
`pinch_tongs`, `fold_glasses`, `click_mouse`; 100 episodes each, 5.2 GB total)
arrive in the **packed LeRobot v3.0 layout**, which our labelling scripts do not
speak. This file records the difference, the route taken, and the exact steps to
get from a fresh HuggingFace download to labelling-ready data on another machine.

Dataset root (`$DSROOT`):

    $WS/assets/datasets/dexjoco_lerobot/dexjoco_lerobot_datasets/<task>

---

## 1. The layout difference

| | old (v2.x, what our scripts assume) | new (v3.0, what we downloaded) |
|---|---|---|
| actions | `data/chunk-000/episode_000007.parquet`, one file per episode | `data/chunk-000/file-000.parquet`, **all 100 episodes in one file** |
| video | `videos/chunk-000/<view>/episode_000007.mp4` | `videos/<view>/chunk-000/file-00K.mp4`, **episodes concatenated** |
| episode index | `meta/episodes.jsonl` | `meta/episodes/chunk-000/file-000.parquet` |
| tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| info paths | `episode_chunk` / `episode_index` format keys | `chunk_index` / `file_index` format keys |

`meta/info.json` for these datasets:

```
"data_path" : "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
"fps": 30, "chunks_size": 1000, "total_episodes": 100, "codebase_version": "v3.0"
```

Nothing is lost by the packing: `meta/episodes/.../file-000.parquet` carries an
explicit per-episode index, one row per episode:

* `dataset_from_index` / `dataset_to_index` — the row slice inside the packed data parquet
* `videos/<video_key>/file_index` — which mp4 the episode lives in
* `videos/<video_key>/from_timestamp` / `to_timestamp` — where inside that mp4
* `tasks` (list[str]), `length`, plus per-episode `stats/*`

### Traps found while verifying (water_plant)

1. **Video timestamps reset at every `file_index`.** `from_timestamp` is relative
   to its own mp4, not to the concatenation of all of them.
2. **The two cameras are split into files differently.** front: 50 + 50 episodes
   (14164 + 13581 frames); wrist: 97 + 3 episodes (26875 + 870 frames). Never
   assume the views share a file boundary.
3. **The camera key is not the same across tasks.** Five tasks use
   `observation.images.front` + `observation.images.wrist`; **`click_mouse` uses
   `observation.images.ego_right`** + `observation.images.wrist`. Read the keys
   from `info.json`, never hard-code "base"/"front".
4. **Video is AV1** (`libdav1d` via PyAV 12.3). It decodes and *seeks* fine, but
   only in an env with a dav1d-enabled PyAV — `envs/quant_gate` has it.
5. **Episodes exceed 999 frames** (`pick_bucket` max 1053), so RoboCasa's
   3-digit `_fNNN` tile-name convention truncates. Tiles here use `_fNNNNN`.
6. `meta/tasks.parquet` stores the **task string as the pandas index** and
   `task_index` as the only column — the inverse of the old `tasks.jsonl`.

---

## 2. Route chosen: **ADAPT (b), plus a cheap metadata materialization**

**Why not convert.** The only expensive thing in these datasets is video (5.1 of
the 5.2 GB), and it is AV1. Splitting the packed mp4s into 600 per-episode mp4s
means either re-encoding (hours of CPU, generation loss) or stream-copying at
keyframe boundaries (fragile, and still +5 GB). And it buys nothing: the packed
mp4s **seek**. A random episode's first frame decodes in ~0.5 s, five scattered
frames of a 300-frame episode in ~1.7 s — the same order as opening a small
per-episode mp4. Conversion would spend disk and hours to reproduce an index
that `meta/episodes/*.parquet` already contains exactly.

**Why materialize the metadata anyway.** The *non-video* half of the old layout
is tiny (~7 MB per task) and is what most consumers actually touch:
`cosmos_1call_v6.py` only needs `meta/episodes.jsonl` + per-episode action
parquets, and it reads images from pre-built tiles, not from video. So we write
those three legacy names **in place and additively** — they do not collide with
any v3.0 name, and the dataset still loads as a v3.0 LeRobot dataset afterwards:

    meta/episodes.jsonl                       {"episode_index", "tasks", "length"}
    meta/tasks.jsonl                          {"task_index", "task"}
    data/chunk-000/episode_{ep:06d}.parquet   action / observation.state / timestamp /
                                              frame_index / episode_index / index / task_index

Net cost ≈ 60 MB for all six tasks (measured: `data/` dirs went from ~58 MB to
117 MB); no video byte is duplicated.

**What is deliberately NOT produced:** per-episode mp4s, and a v2-style
`video_path`. Any consumer that wants pixels goes through the reader or through
pre-built tiles. `extract_gate_backbone_features.py` is the one existing script
that formats `info["video_path"]` itself — it needs a small patch to call
`DexjocoDataset.frames(ep, idxs, view)` instead (its `read_frames` helper has the
same signature/return shape, so it is a one-line swap plus the `chunk`/`view`
plumbing). That patch is left for the consumer, not applied here.

---

## 3. Files written (all under `$WS/vlm_gate/`)

| file | role |
|---|---|
| `scripts/dexjoco_lerobot_reader.py` | `DexjocoDataset`: `instruction(ep)`, `actions(ep)`, `states(ep)`, `column(ep, name)`, `frames(ep, idxs, view)`, `frame(ep, f, view)`, `tile(ep, f)`, `video_path(ep, view)`. Run it directly for a 6-task self-check. |
| `scripts/dexjoco_materialize_compat.py` | writes `episodes.jsonl` / `tasks.jsonl` / `episode_*.parquet` in place, and verifies a round-trip |
| `scripts/dexjoco_make_tiles.py` | 2-view horizontal tiles + manifest, straight from the packed mp4s (the DexJoCo analogue of `allex_make_tiles.py`) |
| `docs/DEXJOCO_DATA.md` | this file |

Tiles land in `$WS/vlm_gate/output/_gate_distill/dexjoco/<task>/tiles/` as
`ep{ep:04d}_f{f:05d}.png` (frame indices are **episode-local**), with
`../tiles_manifest.txt` listing them. `dexjoco_make_tiles.parse_tile_name()`
parses the name back to `(ep, f)`.

---

## 4. Verified numbers

Interpreter used throughout: `/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate/bin/python`
(pyarrow 14.0.1, pandas 2.2.3, numpy 1.26.4, PyAV 12.3.0 with libdav1d).

| task | episodes | frames | min/max ep length | views |
|---|---|---|---|---|
| water_plant | 100 | 27745 | 203 / 394 | front, wrist |
| hammer_nail | 100 | 21571 | 149 / 522 | front, wrist |
| pick_bucket | 100 | 43300 | 226 / 1053 | front, wrist |
| pinch_tongs | 100 | 40065 | 286 / 678 | front, wrist |
| fold_glasses | 100 | 53632 | 280 / 842 | front, wrist |
| click_mouse | 100 | 32680 | 190 / 772 | **ego_right**, wrist |

* action: `(T, 22) float32`; layout per `dexjoco_descriptors.py` is
  `[0:3] arm_pos | [3:6] arm_rot rotvec | [6:22] hand (16 DoF)`, all **absolute**.
* observation.state: `(T, 23) float32`.
* decoded frame: `(640, 640, 3) uint8` RGB; tile: `(640, 1280, 3) uint8`.
* fps 30 for all six.
* Round-trips checked by artifact: reader slice == direct parquet slice
  (`np.array_equal`, ep 7 of water_plant, `(308, 22)`); the tile's left/right
  halves == `reader.frame(ep, f, view)` exactly; `ep3_f0` from the reader ==
  the same frame obtained by a manual PyAV seek.

### Instructions / task descriptions

Two independent sources, both verified to agree:

1. `meta/episodes/*.parquet` column `tasks` — a `list[str]` per episode.
   `DexjocoDataset.instruction(ep)` takes the first entry with >1 word.
2. `meta/tasks.parquet` — the task **string is the pandas index**, `task_index`
   is the column; joined against the episode's `task_index` in the data parquet
   as a fallback. Each of the six datasets has exactly one task, e.g.
   `0 -> "Grasp the watering can and apply water to the plant."`.

`materialize_compat` writes source 1 into `episodes.jsonl` and source 2 (inverted
back to `{"task_index", "task"}`) into `tasks.jsonl`.

---

## 5. Fresh HuggingFace download -> labelling-ready, on another machine

```bash
export WS=<workspace>
export PY=<conda>/envs/quant_gate/bin/python      # needs pyarrow, pandas, PyAV w/ dav1d
export DSROOT=$WS/assets/datasets/dexjoco_lerobot/dexjoco_lerobot_datasets

# 1. download (HF_HUB_DISABLE_XET=1: xet has bitten us on this cluster before)
HF_HUB_DISABLE_XET=1 huggingface-cli download <repo> --repo-type dataset --local-dir $DSROOT

# 2. sanity-check the packed layout (prints eps / shapes / dtypes / instruction per task)
$PY $WS/vlm_gate/scripts/dexjoco_lerobot_reader.py

# 3. materialize the legacy metadata + per-episode action parquets (~60 MB, ~4 min)
$PY $WS/vlm_gate/scripts/dexjoco_materialize_compat.py

# 4. build labelling tiles (start with one task, check the PNGs, then --all)
$PY $WS/vlm_gate/scripts/dexjoco_make_tiles.py water_plant --stride 60 --episodes 0-3
$PY $WS/vlm_gate/scripts/dexjoco_make_tiles.py --all --stride 60
```

Step 3 is idempotent (skips existing `episode_*.parquet` unless `--force`);
step 4 is resumable (skips existing PNGs unless `--overwrite`) and rewrites the
manifest each run.

Judge success by artifacts, not exit codes: step 3 must print
`episodes.jsonl=100 ... episode_*.parquet=100 ... roundtrip_ok=True` per task;
step 4 must print a non-zero `tiles on disk` and a manifest line count, and the
PNGs must be two 640x640 views side by side.

Tile budget at `--stride 60`: ~1.4 GB for all six tasks at native 640x640
(measured 383 KB/tile; 219k frames / 60 ~= 3.6k tiles). Use `--resize 256` if the
judge takes 256x256 input, which cuts that by roughly an order of magnitude. `--stride` and `--tail` (frames dropped at the
end of an episode, default 16 = the chunk horizon) are the two knobs.
