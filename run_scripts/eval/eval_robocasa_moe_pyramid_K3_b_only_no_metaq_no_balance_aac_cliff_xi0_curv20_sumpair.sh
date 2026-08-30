#!/bin/bash
#SBATCH --job-name=eval_robocasa_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance_60k_aac_cliff_xi0_curv20_sumpair_moe_selective_24t_50ep_arr8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_moe_pyramid_K3_aac_cliff_xi0_curv20_sumpair.out
#SBATCH --error=out/%A_%a-eval_robocasa_moe_pyramid_K3_aac_cliff_xi0_curv20_sumpair.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="robocasa MoE pyramid K=3 {raw16, merged8(2x), merged4(4x)} @60k + AAC cliff head=moe_selective N=10 entropy xi=0 + curvature 2.0 main_sumpair."

set -u
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance/checkpoint-60000"
PORT=$((9940 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50
N_SAMPLES=10

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUTPUT_BASE="$BASE_DIR/output/robocasa/moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance_aac_cliff_xi0_curv20_sumpair"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe_selective --n_samples $N_SAMPLES \
    --discrete-action-dims 6 11 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; fi

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
        --score-mode entropy --decision-rule aac_cliff \
        --aac-xi 0 --aac-compress-src main_sumpair --curvature-weight 2.0 \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
