#!/bin/bash
#SBATCH --job-name=resub_PnPCounterToSink_metaq_n16_v1_60k_chkpt_single_job_no_array_no_siblings
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --output=out/%j-resub_PnPCounterToSink_metaq_n16_v1.out
#SBATCH --error=out/%j-resub_PnPCounterToSink_metaq_n16_v1.err
#SBATCH --time=06:00:00
#SBATCH --comment="Single-job resub PnPCounterToSink for metaq_n16_v1. Separate job (not array) so slurm schedules independently to avoid co-located GL contention."

set -u
PORT=9700
N_EPISODES=50
TASK=PnPCounterToSink
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
mkdir -p out
cd "$BASE_DIR"

CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('prehj/gr00t-n1.5-robocasa-metaq-n16-v1', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/robocasa/moe4_per_expert_metaq_n16_v1_as_is"
ODIR="$OUTPUT_BASE/$TASK"
mkdir -p "$ODIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$ODIR/server-resub.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_metaq.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; fi

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures --no_record_video \
    >& "$ODIR/eval-resub.log"
kill "$SPID" 2>/dev/null
