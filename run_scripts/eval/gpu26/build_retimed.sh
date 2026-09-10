#!/bin/bash
#SBATCH --job-name=build_demospeedup_retimed_lerobot_dataset_with_videos
#SBATCH --qos=eval
#SBATCH --time=3:45:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/build_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/build_%A_%a.out
set -uo pipefail
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) SRC=/s3data/robocasa_mg_gr00t_300/v1; DELTA=5:11; DISC=4,11 ;;
  libero)   SRC=/s3data/libero_gr00t_delta/v1;    DELTA=0:6;  DISC=6    ;;
esac
DST=/ckpt/$USER/datasets/${BENCH}_demospeedup
# /ckpt is per-AZ; the segmentation job may have written on another partition, so read
# the labels back through the S3 mirror, which every node sees.
LABELS=/s3ckpt/$USER/baselines/demospeedup/labels_$BENCH.npz
mkdir -p "$DST"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export NO_ALBUMENTATIONS_UPDATE=1
cd "$HOME/quant_label_workspace/quantization_label_gr00t/baselines"
echo "[$(date '+%T')] node=$(hostname) bench=$BENCH shard=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS:-1}"
python -u build_retimed_dataset.py --src "$SRC" --labels "$LABELS" --dst "$DST" \
  --delta-dims "$DELTA" --discrete-dims "$DISC" \
  --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards ${NSHARDS:-1}
echo "[$(date '+%T')] done"
