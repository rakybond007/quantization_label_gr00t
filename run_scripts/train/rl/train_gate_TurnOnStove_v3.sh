#!/bin/bash
#SBATCH --job-name=gate_rl_TurnOnStove_v3
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="GRPO gate v3: merge-friendly prior + strong length reward"
#SBATCH --partition=batch
#SBATCH --output=out/%j-gate_rl_TurnOnStove_v3.out
#SBATCH --error=out/%j-gate_rl_TurnOnStove_v3.err

# v3 differences from v2:
#   - init_logit: -2.0 -> +0.5  (P(merge) starts ~0.62, not 0.12)
#   - prior_logit: -2.0 -> +0.5  (KL anchor pulls toward merge-friendly state)
#   - kl_weight: 0.05 -> 0.02   (less anchoring, more freedom)
#   - alpha_len: 1.0 -> 1.5     (stronger short-success bonus)
#   - entropy_weight: 0.01 -> 0.02 (slight exploration boost)

set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
SAVE_DIR="$BASE_DIR/ckpt/rl/gate_baseline_TurnOnStove_v3"
mkdir -p out "$SAVE_DIR"

cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/robocasa_gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-TurnOnStove_K8_300iter_v3_mergePrior}"
ITERS="${ITERS:-300}"
GROUP="${GROUP:-8}"
MAX_STEPS="${MAX_STEPS:-1500}"

echo "[i] task=TurnOnStove iters=$ITERS group=$GROUP max_env_steps=$MAX_STEPS"
echo "    init_logit=+0.5 prior_logit=+0.5 kl_weight=0.02 alpha_len=1.5 entropy=0.02"

"$CONDA_PATH/envs/robocasa_gr00t/bin/python" "$BASE_DIR/scripts/train_gate_rl.py" \
    --model-path "$CKPT" \
    --env-name TurnOnStove \
    --iters $ITERS \
    --group-size $GROUP \
    --max-env-steps $MAX_STEPS \
    --log-video-every 10 \
    --save-path "$SAVE_DIR" \
    --init-logit 0.5 \
    --prior-logit 0.5 \
    --kl-weight 0.02 \
    --alpha-len 1.5 \
    --alpha-merge 0.0 \
    --entropy-weight 0.02 \
    --wandb-project GR00T-RL-gate \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --gpu 0
