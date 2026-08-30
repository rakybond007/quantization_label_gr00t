#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=out/%j-dexjoco_convert.out
#SBATCH --error=out/%j-dexjoco_convert.err
# Convert one DexJoCo task from v3.0 to v2.0 (CPU work, GPU unused).
# Usage: sbatch _convert_one_task.sh <task_name> [regime]
set -u
TASK=${1:?need task name}
REGIME=${2:-dexjoco_lerobot_datasets}
BASE=$HOME/multigpu_workspace/Isaac-GR00T
SRC=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/full/$REGIME/$TASK
DST=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/$TASK
mkdir -p out
SRC_MOD="$SRC/meta/modality.json"
if [ ! -f "$SRC_MOD" ]; then
    cp /sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/_dl/dexjoco_lerobot_datasets/hammer_nail/meta/modality.json "$SRC_MOD"
    echo "[i] copied single-arm modality.json template into $SRC/meta/"
fi
$HOME/miniconda3/envs/gr00t/bin/python $BASE/scripts/convert_lerobot_v30_to_v20.py --src "$SRC" --dst "$DST"
