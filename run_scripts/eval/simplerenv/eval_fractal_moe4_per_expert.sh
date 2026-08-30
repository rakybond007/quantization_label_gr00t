#!/bin/bash
#SBATCH --job-name=eval_simpler_fractal_moe4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-3
#SBATCH --output=out/%A_%a-eval_simpler_fractal_moe4.out
#SBATCH --error=out/%A_%a-eval_simpler_fractal_moe4.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="Eval SimplerEnv Fractal/Google MoE per_expert: head=moe + variable-horizon"

# 4 main google tasks → 4 array jobs (no variant aggregation for first pass).
set -u

PORT=$((9500 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/simplerenv/fractal/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/simplerenv/fractal_moe4_per_expert"
SIMPLER_PYTHON="$BASE_DIR/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_LOADER_LAYERS_DISABLE='*'
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

TASK_NAMES=(
  "google_robot_pick_coke_can"
  "google_robot_move_near"
  "google_robot_open_drawer"
  "google_robot_close_drawer"
)
TASK="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"
ENV_NAME="simpler_env_google/$TASK"

echo "[i] Array $SLURM_ARRAY_TASK_ID | task=$TASK | port=$PORT"

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config simplerenv_fractal \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe \
    --discrete-action-dims 6 \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SPID=$!
sleep 60
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>/dev/null && break
    sleep 5
done

ODIR="$OUTPUT_BASE/$TASK"
mkdir -p "$ODIR"

EVAL_CMD="$SIMPLER_PYTHON -u $BASE_DIR/scripts/simplerenv_service_moe.py \
    --port $PORT --host localhost \
    --env_name $ENV_NAME \
    --video_dir $ODIR \
    --seed 42 \
    --n_episodes $N_EPISODES \
    --max_episode_steps 300"

if command -v xvfb-run &>/dev/null; then
    xvfb-run -a $EVAL_CMD >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
elif command -v Xvfb &>/dev/null; then
    _DISP=$(( (SLURM_JOB_ID * 10 + SLURM_ARRAY_TASK_ID) % 100 + 100 ))
    Xvfb :$_DISP -screen 0 1024x768x24 -nolisten tcp &>/dev/null &
    _XVFB_PID=$!
    sleep 1
    DISPLAY=:$_DISP $EVAL_CMD >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
    kill $_XVFB_PID 2>/dev/null
else
    MUJOCO_GL=egl $EVAL_CMD >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
fi

kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
