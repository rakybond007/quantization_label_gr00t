#!/bin/bash
#SBATCH --job-name=demospeedup_action_entropy_over_robocasa_frames_vs_gate_labels
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=3:30:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/entropy_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/entropy_%A_%a.out
set -uo pipefail
# DemoSpeedup's per-frame action entropy on the RoboCasa policy we already have.
# The point is not to reproduce DemoSpeedup end to end; it is to get the teacher-free
# signal onto the same (episode_index, frame_index) index as the committed phase5 gate
# labels, so the two can be correlated instead of argued about.
DATA=/s3data/robocasa_mg_gr00t_300/v1
OUT=/ckpt/$USER/baselines/demospeedup_entropy
mkdir -p "$OUT"
SHARD=${SLURM_ARRAY_TASK_ID:-0}
EP_PER_SHARD=${EP_PER_SHARD:-10}
N_SAMPLES=${N_SAMPLES:-8}
# RoboCasa's 24 tasks are contiguous blocks of 300 episodes, so shards must stride
# across the dataset or every shard lands inside one task.
STRIDE=${STRIDE:-60}
OFFSET_STEP=${OFFSET_STEP:-15}
K=${ENSEMBLE_K:-8}

CKPT=$(find "$HOME/.cache/huggingface/hub/models--prehj--GR00T-N1.5-robocasa-baseline/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
cd "$HOME/quant_label_workspace/quantization_label_gr00t"
echo "[$(date '+%T')] node=$(hostname) shard=$SHARD ckpt=$CKPT"
python -u baselines/demospeedup_entropy.py \
  --dataset-path "$DATA" --model-path "$CKPT" \
  --data-config single_panda_gripper --embodiment-tag new_embodiment \
  --episodes $EP_PER_SHARD --episode-offset $((SHARD * OFFSET_STEP)) --episode-stride $STRIDE \
  --n-samples $N_SAMPLES --ensemble-k $K \
  --out "$OUT/entropy_${TAG:-s}${SHARD}.parquet"
echo "[$(date '+%T')] rows written:"; ls -la "$OUT"/entropy_${TAG:-s}${SHARD}.parquet 2>/dev/null
