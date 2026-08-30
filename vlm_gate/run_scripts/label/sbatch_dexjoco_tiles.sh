#!/bin/bash
#SBATCH --job-name=dexjoco_two_view_tile_generation_all_six_single_arm_tasks
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p cpu
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --comment=MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1_tiles
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# LAUNCH:
#   MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1 \
#     sbatch run_scripts/label/sbatch_dexjoco_tiles.sh
# then, once all six are done:
#   ~/miniconda3/envs/quant_gate_eval/bin/python scripts/dexjoco_tiles_manifest.py
# One array task per DexJoCo task. Tiles are two views (base + wrist) side by
# side, 320 px per view (640x640 native, downscaled by 2), sampled every 16th
# frame -- one non-overlapping 16-step descriptor window per tile, ~13.3k tiles
# over the six tasks. Existing tiles are kept, so a requeue resumes.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
TASKS=(click_mouse fold_glasses hammer_nail pick_bucket pinch_tongs water_plant)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/dexjoco_make_tiles.py "$T" \
  --stride 16 --resize 320 --tail 16 \
  --out $BASE/output/_gate_distill/dexjoco_v1
