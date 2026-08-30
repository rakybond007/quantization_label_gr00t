#!/bin/bash
#SBATCH --job-name=eval_robocasa_router_instr_end_stochastic_RESUME_PnPCounterToSink_no_record_video
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --output=out/%j-eval_router_instr_end_stoch_resume_pnpsink.out
#SBATCH --error=out/%j-eval_router_instr_end_stoch_resume_pnpsink.err
#SBATCH --time=4:00:00
#SBATCH --comment="Resume PnPCounterToSink for router_instr_end stochastic (uses --no_record_video to bypass mujoco framebuffer crash that killed prior run after 1ep)."

set -u
TASK="PnPCounterToSink"
PORT=10070
N_EPISODES=50
HF_REPO=prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-router-instr-end-60k

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/robocasa/moe4_v1_b_only_no_metaq_router_instr_end_hf_stochastic"
mkdir -p out "$OUTPUT_BASE/$TASK"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/$TASK/server-resume.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 --moe-stochastic \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 60); do
    if grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$OUTPUT_BASE/$TASK" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    --no_record_video \
    >& "$OUTPUT_BASE/$TASK/eval-resume.log"
RC=$?
kill "$SPID" 2>/dev/null
echo "[i] done rc=$RC"
exit $RC
