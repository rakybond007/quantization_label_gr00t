#!/bin/bash
#SBATCH --job-name=eval_mh_robocasa
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="Multi-horizon eval on Robocasa (head selectable)"
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_mh.out
#SBATCH --error=out/%A_%a-eval_mh.err
#SBATCH --time=2-00:00:00

# Generic full eval. The HEAD env var selects which decoder to use.
# Submit per head (use --export so HEAD reaches the SLURM job):
#   sbatch --export=ALL,HEAD=main         run_scripts/eval/eval_multi_horizon_full.sh
#   sbatch --export=ALL,HEAD=f2           run_scripts/eval/eval_multi_horizon_full.sh
#   sbatch --export=ALL,HEAD=f4           run_scripts/eval/eval_multi_horizon_full.sh
#   sbatch --export=ALL,HEAD=ensemble     run_scripts/eval/eval_multi_horizon_full.sh
#   sbatch --export=ALL,HEAD=ensemble_fix run_scripts/eval/eval_multi_horizon_full.sh
#
# ensemble_fix: WLS combine across main+f2+f4 for continuous dims, but for
# DISCRETE dims (binary gripper_close, control_mode for single_panda_gripper)
# we overwrite with main's value because averaging binary signals is meaningless.

set -u

HEAD="${HEAD:-main}"
PORT_BASE="${PORT_BASE:-8830}"
PORT=$((PORT_BASE + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_multi_horizon/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/robocasa/multi_horizon_full/$HEAD"
mkdir -p out "$OUTPUT_BASE"

cd "$BASE_DIR"

# --- Environment variables ---
export NO_ALBUMENTATIONS_UPDATE=1
# CUDA libs from the gr00t env's pip-installed nvidia packages
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[i] Array $SLURM_ARRAY_TASK_ID | head=$HEAD | port=$PORT"

# --- Server (gr00t env) ---
# Discrete dims for single_panda_gripper: 6=gripper_close, 11=control_mode.
# Always passed; only used by 'ensemble_fix' head (other heads ignore).
"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head "$HEAD" \
    --discrete-action-dims 6 11 &
SPID=$!
sleep 60

# --- 24 tasks split across 8 array jobs (3 tasks each) ---
TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
SELECTED=()
[ $SLURM_ARRAY_TASK_ID -lt 8 ] && SELECTED+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
[ $((SLURM_ARRAY_TASK_ID + 8)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
[ $((SLURM_ARRAY_TASK_ID + 16)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")

echo "[i] Tasks: ${SELECTED[*]}"

# --- Clients (robocasa_gr00t env), parallel within array job ---
MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"
    mkdir -p "$ODIR"
    "$CONDA_PATH/envs/robocasa_gr00t/bin/python" "$BASE_DIR/scripts/robocasa_service.py" --client \
        --port $PORT --host localhost \
        --env_name "$TASK" \
        --video_dir "$ODIR" \
        --seed 42 \
        --n_episodes $N_EPISODES \
        --max_episode_steps 1500 \
        --generative_textures \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
