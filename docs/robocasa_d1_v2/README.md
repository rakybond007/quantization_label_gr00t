# RoboCasa D1 v2

Implementation of the four-question / five-grade + U proposal. No per-task
quantile allocation. `IMPLEMENTATION_PLAN.md` documents the assumptions; the
executed prompt is `vlm_gate/prompts/robocasa_d1_v2.txt`.

Status on 2026-09-10: CPU policy/inference parity and small-head soft-label training
checks passed. A GPU pilot and real two-GPU GR00T smoke are queued, not yet passed.
There is no new production label release yet. Synthetic fixture labels are tests,
not Cosmos annotations and must never be published as a benchmark label dataset.

## Labeling

Use a working Cosmos3-Nano environment to serve `vlm_gate/scripts/vlm_gate_cosmos.py`
on one GPU per server. Pin model snapshot
`7a312c868bcce8e40b3eb40861300a9d0ba3fde1` for the initial pilot. The client needs
numpy, pyarrow, Pillow and decord. Existing data and labels are inputs only.

```bash
python vlm_gate/scripts/label_robocasa_d1.py prepare \
  --dataset /data/robocasa_mg_gr00t_300 --pilot --out /runs/d1/pilot_manifest.json

python vlm_gate/scripts/label_robocasa_d1.py label \
  --dataset /data/robocasa_mg_gr00t_300 \
  --manifest /runs/d1/pilot_manifest.json \
  --legacy /data/robocasa-ratio-labels-contact/labels/robocasa.parquet \
  --model-revision 7a312c868bcce8e40b3eb40861300a9d0ba3fde1 \
  --url http://127.0.0.1:8760 --batch-size 4 \
  --out /runs/d1/pilot.jsonl --save-images /runs/d1/pilot_images
```

The default pilot is192 windows:24 tasks x2 episodes x4 positions. Inspect the
rubric against images, including U, before full labeling. Do not modify thresholds
to get a preferred speed histogram. Compare at least16 windows with batch1.
Full: prepare without `--pilot`, then launch two clients with `--num-shards 2`
and `--shard 0`/`1`, each on its own server/port. No GPU training batch inference
is made from these inference batch tests.

Resume uses the same raw output path. Prompt/model/reader/policy/server/manifest
fingerprints must match. Batch size may change. Shard count must currently remain
fixed; no completed batch is discarded. Changing shard count requires explicit
offline redistribution, not silently changing CLI flags. Only an incomplete final
JSONL record can be recovered, with an audit record.

Each input is t/t+8/t+16 in three views. Match video frames by parquet timestamps
(the inspected videos have10 additional frames; no guessed +10 offset is applied).
Last16 rows of each episode cannot provide the full context and are masked. There
are1,958,257 full-context starts in the inspected7,200-episode dataset.
The contact release's fixed decisions are preserved with their source, including
its proxy fallback; the new probabilities describe NEW A-D questions.

## Build and apply

```bash
python vlm_gate/scripts/build_robocasa_d1_labels.py \
  --dataset /data/robocasa_mg_gr00t_300 --manifest /runs/d1/full_manifest.json \
  --raw /runs/d1/raw/shard_0.jsonl /runs/d1/raw/shard_1.jsonl \
  --out /runs/robocasa-D1-Astra/labels/robocasa_training.parquet

python vlm_gate/scripts/apply_robocasa_d1_labels.py \
  --dataset /data/robocasa_mg_gr00t_300 \
  --labels /runs/robocasa-D1-Astra/labels/robocasa_training.parquet \
  --out /data/robocasa_d1_sidecar --dry-run

python vlm_gate/scripts/apply_robocasa_d1_labels.py \
  --dataset /data/robocasa_mg_gr00t_300 \
  --labels /runs/robocasa-D1-Astra/labels/robocasa_training.parquet \
  --out /data/robocasa_d1_sidecar
```

Application only needs numpy/pyarrow and the adjacent `.manifest.json`; the apply
script can be copied standalone. It verifies metadata hashes and row identities.
Use an empty output directory. `--mode parquet` copies base parquets, preserves
all original columns, adds carrier metadata/statistics and symlinks the original
video directory. That symlink requires the original videos on the same server;
transport them or recreate the symlink when moving the copied dataset.

Canonical probabilities `gp_A..D` are six-element arrays in order1,2,3,4,5,U.
The training ratio carrier is `[class_index,valid]` (2 floats, class0..3).
`grade_prob_label` is24 probabilities followed by4 validity flags (28 floats).
U is a valid sixth category, not an invalid question. Semantic-unknown ratio rows
may have ratio-valid0 while all four question targets remain valid. Tail rows
mask both. Grade expectations are never categorical CE indices.

The old generic apply script and old five-question hard-grade loader are NOT
compatible with this schema. Use the F-level soft-grade config described below.

## Training on another server

Use F-level branch `hj`, in its own environment with the repo installed editable:
`python -m pip install -e /path/to/F_level`. Follow that repo's existing FLARE
environment instructions for dependencies. Run from the F-level checkout:

```bash
DATASET_PATH=/data/robocasa_mg_gr00t_300 \
RATIO_DIR=/data/robocasa_d1_sidecar \
FLEVEL_ENV=/envs/flevel \
BASE_MODEL_PATH=/models/GR00T-N1.5-3B \
OUTPUT_ROOT=/runs/robocasa_d1 \
BATCH_SIZE=32 GRAD_ACCUM=1 \
bash papers/reproducing/FLARE/run_scripts/train_robocasa_d1.sh
```

Default: two GPUs, per-GPU32, accum1, global64,60k steps. If memory requires
per-GPU16, set GRAD_ACCUM=2 to retain global64. This launcher works without Slurm;
CUDA_VISIBLE_DEVICES should expose the intended two GPUs.
For `--mode parquet`, set DATASET_PATH and RATIO_DIR to the same copied output.

The model trains all four action decoders, ratio CE, and four six-category soft
grade heads (coefficient0.1). Carriers are stripped BEFORE action padding/loss.
FLARE predicted future latents plus instruction features feed the readout; actual
future demo images are teacher inputs only during training.

Serving uses the learned ratio classifier capped by the instruction's registered
task cap. Unknown-grade bounds that disagree trigger fine fallback. Checkpoints
carry the instruction-cap map; unregistered instructions require an explicit
`_flevel_task_cap` request field. Output grade index6 corresponds to U, as declared
in `_flevel_grade_categories`. Teacher contact overrides are learned supervision,
not oracle contact information available at deployment.

## Local verification

`python vlm_gate/scripts/test_robocasa_d1.py` checks5,184 annotation/inference
cases (requires the sibling F_level_hj checkout for parity), malformed token/prob
rejection, contact handling and interrupted JSONL recovery.
`docs/robocasa_d1_v2/check_policy_spec.py` is a standalone arithmetic specification.
F-level's `smoke_d1_soft_grades.py` checks a tiny actual head's soft CE, U/masks,
all-decoder gradients and state reload. `smoke_d1_dataset.py` checks the real
dataset/transform/FLARE collation path without loading model weights.
Passing these does not imply the GPU pilot or full training has passed.
