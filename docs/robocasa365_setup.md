# RoboCasa365 Setup for GR00T N1.5 Fine-Tune + Eval

## What "RoboCasa365" Refers To

**RoboCasa365** is the v1.0 expansion of the original RoboCasa benchmark, accompanying an ICLR 2026 paper (Nasiriany et al.). It is *not* a separate codebase — it is the v1.0 release of the existing `robocasa/robocasa` GitHub repo (released ~Feb 2026). The "365" denotes the total task count.

| Aspect | Original RoboCasa (v0.2) | **RoboCasa365 (v1.0)** |
|---|---|---|
| Atomic tasks | 25 | **65** |
| Composite tasks | 75 | **300** |
| **Total** | 100 | **365** |
| Kitchen scenes | ~120 | 2,500+ |
| 3D objects | ~2,500 | 3,200+ |
| Human demos | tens of hours | 600+ hours |
| Synthetic / robot data | small | 1,600+ hours (MimicGen) |

References:
- Paper: https://arxiv.org/abs/2603.04356 (ICLR 2026)
- Project page: https://robocasa.ai/
- HF dataset: https://huggingface.co/datasets/nvidia/robocasa365-datasets

## State of the Cloned Codebase

The cloned `multigpu_workspace/robocasa/` repo on disk **already contains the v1.0 / 365 task surface**:
- `robocasa/environments/kitchen/single_stage/` — 8 files defining ~37 atomic kitchen task classes (coffee, doors, drawer, microwave, navigate, pnp, sink, stove)
- `robocasa/environments/kitchen/multi_stage/` — 20 activity dirs (baking, boiling, brewing, chopping_food, clearing_table, defrosting_food, frying, making_toast, meat_preparation, mixing_and_blending, reheating_food, restocking_supplies, sanitize_surface, serving_food, setting_the_table, snack_preparation, steaming_food, tidying_cabinets_and_drawers, washing_dishes, washing_fruits_and_vegetables) with ~85 multi-stage task classes
- Total environment classes counted in the cloned repo: **123** (includes intermediate base classes + some `Kitchen` mixins; the 65/300 split lives at the task-instance level)

So no upstream pull is required for the simulator side — `robocasa` cloned at `/sjw_alinlab2/home/hojin2/multigpu_workspace/robocasa/` already supports the 365-task surface. The gap is **demonstration data**.

## Existing Local Demo Data (24-atomic-task subset)

Path: `/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300`
- Source: `kimtaey/robocasa_mg_gr00t_300` on HF
- 7,200 episodes, 300 demos × 24 atomic tasks (LeRobot-formatted, GR00T-ready)
- Tasks (24): `CloseDoubleDoor, CloseDrawer, CloseSingleDoor, CoffeePressButton, CoffeeServeMug, CoffeeSetupMug, OpenDoubleDoor, OpenDrawer, OpenSingleDoor, PnPCabToCounter, PnPCounterToCab, PnPCounterToMicrowave, PnPCounterToSink, PnPCounterToStove, PnPMicrowaveToCounter, PnPSinkToCounter, PnPStoveToCounter, TurnOffMicrowave, TurnOffSinkFaucet, TurnOffStove, TurnOnMicrowave, TurnOnSinkFaucet, TurnOnStove, TurnSinkSpout`
- Covers the original v0.2 PnP / Door / Drawer / Coffee / Faucet families; missing the 41 new atomic tasks added in v1.0 (blender, kettle, oven, etc.) and **all 300 composite tasks**.

`kimtaey` HF profile only hosts `_30 / _100 / _300` variants (all 24-task atomic subsets, max 2.07M rows in metadata). No `_full` or `_365` variant exists yet.

## Recommended Full-365 Dataset

**Repo ID:** `nvidia/robocasa365-datasets` (HuggingFace, gated/auth required — `WebFetch` returned 401)

Estimated size (from paper): 600 hr human + 1,600 hr synthetic ≈ 2,200 hr ≈ **multi-TB raw**. Likely sharded by task; do **not** bulk-download.

Suggested staging command (run after auth, **DO NOT run yet**):
```bash
# Inspect first
huggingface-cli download nvidia/robocasa365-datasets --repo-type dataset \
    --local-dir /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/nvidia/robocasa365 \
    --include "atomic/**"  # start with atomic-only subset

# Or for a single task:
huggingface-cli download nvidia/robocasa365-datasets --repo-type dataset \
    --include "atomic/PickPlaceCounterToCabinet/**" \
    --local-dir /sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/nvidia/robocasa365
```
After download, verify it is in LeRobot v2.0+ format (look for `meta/episodes.jsonl`, `data/chunk-*/episode_*.parquet`, `videos/chunk-*/...`). If raw HDF5 / robomimic-format, conversion via `robocasa.scripts.convert_robomimic_to_lerobot` (or equivalent in Isaac-GR00T's `gr00t/utils/data_conversion.py`) will be required before `gr00t_finetune.py` can ingest it.

## Sbatch Templates

Created at `Isaac-GR00T/run_scripts/train/robocasa365/`:
- `finetune_robocasa365_baseline.sh` — mirror of `robocasa/finetune_gr00t_n1_5_baseline.sh`. 2 GPUs, batch 32 (global 64), 60k steps, `single_panda_gripper` data config.
- `finetune_robocasa365_mh_m8_econsist.sh` — mirror of `robocasa/finetune_gr00t_n1_5_mh_m8_econsist.sh`. Adds multi-horizon (factors 2,4) + merged-8 head + ensemble-consistency loss + discrete-fix on dims 6,11.

Both currently point at the **24-task subset** (`kimtaey/robocasa_mg_gr00t_300`) with a clearly marked `# TODO: swap to full 365-task LeRobot dataset` line. Output checkpoints land under `ckpt/robocasa365/groot/...` so they don't collide with the existing 24-task runs.

## Sim Eval

The eval client is unchanged from the existing 24-task pipeline. After training:

```bash
# Terminal 1 (alin_vla / gr00t env): policy server
python Isaac-GR00T/scripts/inference_service.py --server \
    --port 8970 \
    --model_path Isaac-GR00T/ckpt/robocasa365/groot/groot_n1_5_bs64_baseline/checkpoint-60000 \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment

# Terminal 2 (robocasa_alinvla env): per-task client
python Isaac-GR00T/scripts/robocasa_service.py --client \
    --port 8970 \
    --env_name PnPCounterToCab \
    --video_dir output/groot/robocasa365/<run_name>/<step>/PnPCounterToCab \
    --max_episode_steps 750 \
    --n_episodes 50 \
    --generative_textures
```

For the full 365-task sweep, extend the `TASK_NAMES` array in `run_scripts/eval/eval_gr00t_robocasa.sh` with the 41 new atomic + 300 composite task class names from `multigpu_workspace/robocasa/robocasa/environments/kitchen/{single_stage,multi_stage}/`. Recommend starting with a subset (e.g. all 65 atomic) since 365 × 50 episodes = 18,250 sim rollouts.

## Open Items / Blockers

1. **Full 365 dataset is not yet on disk.** `nvidia/robocasa365-datasets` requires HF auth + (likely) format conversion. No `kimtaey/robocasa_mg_gr00t_full` exists. Until then the templates train on the 24-task subset.
2. **Composite tasks need a multi-task language-conditioned config.** `single_panda_gripper` data config has been used for the 24 atomic tasks; composite tasks may need richer language annotations (the dataset includes them per RoboCasa365 paper) and possibly a new `data_config` if observation modalities differ.
3. **Eval task list** in `eval_gr00t_robocasa.sh` is hard-coded to 24. Extend to 65 atomic (and optionally composite) once the v1.0 task class names are confirmed runnable in the cloned repo.
