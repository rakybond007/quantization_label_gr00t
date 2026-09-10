#!/bin/bash
#SBATCH --job-name=demospeedup_finetune_on_entropy_retimed_demonstrations
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dstrain_%j.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dstrain_%j.out
set -uo pipefail
# DemoSpeedup (arXiv:2506.05064) stage 3: train on the entropy-accelerated demonstrations.
# Recipe is pinned to each benchmark's existing control checkpoint so the re-timed data is
# the only variable -- same steps, same global batch, same action_horizon (16).
#   robocasa control: prehj/GR00T-N1.5-robocasa-baseline        60k, global 64
#   libero   control: prehj/GR00T-N1.5-libero-baseline-bs32-60k 60k, global 32
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) CFG=single_panda_gripper; TAG=new_embodiment; PER_GPU=32 ;;   # 32 x 2 = 64
  libero)   CFG=libero;               TAG=libero;         PER_GPU=16 ;;   # 16 x 2 = 32
esac
# The source dataset lives in /s3data; the accelerated copy derived from it is staged in
# /ckpt per the storage contract.
SRC=/s3data/${SRC_NAME:?}
DATA=/ckpt/$USER/datasets/${BENCH}_demospeedup
CKPT=/ckpt/$USER/${BENCH}/demospeedup_$(date +%Y%m%d_%H%M%S)   # fresh dir: the trainer
# resumes from whatever sits in output_dir and resume is broken on this stack
mkdir -p "$CKPT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export HF_TOKEN=$(cat "$HOME/quant_label_workspace/hf_token.txt")
export HF_HUB_DISABLE_XET=1 NO_ALBUMENTATIONS_UPDATE=1
export WANDB_PROJECT=GR00T-demospeedup WANDB_MODE=${WANDB_MODE:-offline}
cd "$HOME/quant_label_workspace/quantization_label_gr00t"

echo "[$(date '+%F %T')] node=$(hostname) bench=$BENCH data=$DATA (derived from $SRC)"
python -c "
import json;d=json.load(open('$DATA/meta/info.json'))
print('  retimed dataset:',d['total_episodes'],'episodes',d['total_frames'],'frames')
import os;print('  stats.json present:',os.path.exists('$DATA/meta/stats.json'),'(False = will be recomputed)')"

python scripts/gr00t_finetune.py \
    --dataset-path "$DATA" --output-dir "$CKPT" \
    --data-config $CFG --embodiment-tag $TAG \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-$BENCH-demospeedup \
    --batch-size $PER_GPU --num-gpus 2 \
    --max-steps ${MAX_STEPS:-60000} --save-steps 10000 \
    --dataloader-num-workers 16 --report-to wandb
echo "[$(date '+%F %T')] checkpoints:"; ls -d "$CKPT"/checkpoint-* 2>/dev/null
