#!/bin/bash
#SBATCH --job-name=gate_rl_TurnOnStove
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="GRPO gate training on TurnOnStove (frozen baseline GR00T-N1.5)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-gate_rl_TurnOnStove.out
#SBATCH --error=out/%j-gate_rl_TurnOnStove.err

# Single-task RL: train per-chunk merge gate via GRPO on TurnOnStove.
# - frozen baseline GR00T-N1.5 finetuned on Robocasa as policy
# - gate predicts P(merge 16->8) from VL backbone features
# - K=8 group rollouts per iteration (GRPO group-relative advantage)
# - sparse reward: success/fail + small length / merge bonuses

set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
SAVE_DIR="$BASE_DIR/ckpt/rl/gate_baseline_TurnOnStove"
mkdir -p out "$SAVE_DIR"

cd "$BASE_DIR"

# --- Env ---
export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/robocasa_gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-TurnOnStove_K8_300iter}"
ITERS="${ITERS:-300}"
GROUP="${GROUP:-8}"
MAX_STEPS="${MAX_STEPS:-1500}"

echo "[i] task=TurnOnStove iters=$ITERS group=$GROUP max_env_steps=$MAX_STEPS"

"$CONDA_PATH/envs/robocasa_gr00t/bin/python" "$BASE_DIR/scripts/train_gate_rl.py" \
    --model-path "$CKPT" \
    --env-name TurnOnStove \
    --iters $ITERS \
    --group-size $GROUP \
    --max-env-steps $MAX_STEPS \
    --log-video-every 10 \
    --save-path "$SAVE_DIR" \
    --wandb-project GR00T-RL-gate \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --gpu 0
