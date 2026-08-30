#!/bin/bash
#SBATCH --job-name=eval_robocasa_moe3_per_expert_main_m8_m4_router_pick_as_is_no_compression_60k_chkpt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_moe3_as_is.out
#SBATCH --error=out/%A_%a-eval_robocasa_moe3_as_is.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="Robocasa MoE3 (main+m8+m4) per_expert eval — head=moe (router pick as-is, no compression heuristic)"

# 24 tasks split into 8 array jobs × 3 tasks each.
set -u

PORT=$((10050 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe3_per_expert_body/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/robocasa/moe3_per_expert_stochastic_temp0p3"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[i] Array $SLURM_ARRAY_TASK_ID | robocasa MoE3 as_is | port=$PORT"

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe --moe-stochastic --moe-inference-temp 0.3 \
    --discrete-action-dims 6 11 \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SPID=$!
sleep 60
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>/dev/null && break
    sleep 5
done

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

MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"
    mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_moe.py" \
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

for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
