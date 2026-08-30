#!/bin/bash
#SBATCH --job-name=eval_robocasa_moe4_per_expert_metaq_no_length_cost_as_is_60k_chkpt_resub_3tasks
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --output=out/%j-eval_robocasa_metaq_no_length_cost_resub_3tasks.out
#SBATCH --error=out/%j-eval_robocasa_metaq_no_length_cost_resub_3tasks.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="Resubmit: 3 remaining tasks (OpenDrawer, PnPSinkToCounter, TurnSinkSpout) for no_length_cost @ 60k after 318729/318702 zmq port conflict zombies."

# Resub of the 3 tasks that did not finish in the original array job 318700/318729
# (both array_task_id=0 zombies hit port 9400 'Address already in use' and hung).
# Single-job version with a fresh port (9450) and a distinct log suffix to avoid
# any leftover races.
set -u
PORT=9450
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_metaq_n8_no_length_cost/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/robocasa/moe4_per_expert_metaq_no_length_cost_as_is"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server-resub.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_metaq.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!

# Wait up to 5 min for "Server is ready" AND detect early server death so we
# don't fall through to a doomed client like the original script did.
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then
        READY=1; break
    fi
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "[ERR] inference server died before becoming ready"; tail -30 "$SERVER_LOG"; exit 1
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "[ERR] inference server not ready within 5 min"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1
fi

SELECTED=("OpenDrawer" "PnPSinkToCounter" "TurnSinkSpout")
MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"
    mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_moe.py" \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps 1500 --generative_textures \
        >& "$ODIR/eval-resub.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
