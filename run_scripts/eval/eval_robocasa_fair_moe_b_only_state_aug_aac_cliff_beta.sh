#!/bin/bash
#SBATCH --job-name=eval_robocasa_fair_moe_b_only_state_aug_60k_aac_cliff_entropy_xi1_N10_beta_t_24t_50ep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_fair_moe_b_only_state_aug_aac_beta.out
#SBATCH --error=out/%A_%a-eval_robocasa_fair_moe_b_only_state_aug_aac_beta.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="fair_moe + state_aug @ 60k: head=moe_selective N=10 + AAC cliff (entropy xi=1) + Beta-sampled timestep schedule"

set -u

AAC_XI=1
N_SAMPLES=10

PORT=$((9610 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_fair_moe_b_only_state_aug/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/robocasa/fair_moe_b_only_state_aug_aac_cliff_beta_xi${AAC_XI}_N${N_SAMPLES}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples $N_SAMPLES \
    --beta_sample_t \
    --discrete-action-dims 6 11 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
# Wait for "Server is ready", but fast-exit if server dies (lesson from past zombies)
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready in 5 min"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; fi

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

MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"
    mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_selective.py" \
        --port $PORT --host localhost \
        --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps 1500 --generative_textures \
        --score-mode entropy --decision-rule aac_cliff --aac-xi $AAC_XI \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
