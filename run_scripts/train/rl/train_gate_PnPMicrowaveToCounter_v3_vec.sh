#!/bin/bash
#SBATCH --job-name=gate_rl_PnPMicrowaveToCounter_v3_vec
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="GRPO gate v3 vec on PnPMicrowaveToCounter: K=N=32 parallel"
#SBATCH --partition=batch
#SBATCH --output=out/%j-gate_rl_PnPMicrowaveToCounter_v3_vec.out
#SBATCH --error=out/%j-gate_rl_PnPMicrowaveToCounter_v3_vec.err

# Same v3 settings as TurnOnStove vec script, applied to PnPMicrowaveToCounter
# (baseline ~24%, contact-rich PnP — different challenge surface).

set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
SAVE_DIR="$BASE_DIR/ckpt/rl/gate_baseline_PnPMicrowaveToCounter_v3_vec"
mkdir -p out "$SAVE_DIR"

cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/robocasa_gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

export MUJOCO_GL=egl

WANDB_RUN_NAME="${WANDB_RUN_NAME:-PnPMicrowaveToCounter_K32_vec_v3}"
ITERS="${ITERS:-300}"
GROUP="${GROUP:-32}"
NENVS="${NENVS:-32}"
MAX_STEPS="${MAX_STEPS:-1500}"

echo "[i] task=PnPMicrowaveToCounter iters=$ITERS group=$GROUP n_envs=$NENVS"

"$CONDA_PATH/envs/robocasa_gr00t/bin/python" "$BASE_DIR/scripts/train_gate_rl.py" \
    --model-path "$CKPT" \
    --env-name PnPMicrowaveToCounter \
    --iters $ITERS \
    --group-size $GROUP \
    --n-envs $NENVS \
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
