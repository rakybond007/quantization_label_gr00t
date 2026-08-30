#!/bin/bash
#SBATCH --job-name=resub_PnPCounterToSink_4variants_n16v1_n32v1_fair_moe_v2_metaq_v2_n8_60k_chkpt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-3
#SBATCH --output=out/%A_%a-resub_PnPCounterToSink.out
#SBATCH --error=out/%A_%a-resub_PnPCounterToSink.err
#SBATCH --time=06:00:00
#SBATCH --comment="Resub PnPCounterToSink only (50 ep) for 4 variants that hit mujoco.FatalError 'framebuffer not complete' during ep 0->1 reset in array slot 3."

set -u
PORT=$((9680 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50
TASK=PnPCounterToSink

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
mkdir -p out
cd "$BASE_DIR"

case $SLURM_ARRAY_TASK_ID in
    0)
        CKPT_FETCH="from huggingface_hub import snapshot_download; print(snapshot_download('prehj/gr00t-n1.5-robocasa-metaq-n16-v1', repo_type='model'))"
        SERVICE=scripts/inference_service_metaq.py
        OUT_DIR_NAME=moe4_per_expert_metaq_n16_v1_as_is
        ;;
    1)
        CKPT_FETCH="from huggingface_hub import snapshot_download; print(snapshot_download('prehj/gr00t-n1.5-robocasa-metaq-n32-v1', repo_type='model'))"
        SERVICE=scripts/inference_service_metaq.py
        OUT_DIR_NAME=moe4_per_expert_metaq_n32_v1_as_is
        ;;
    2)
        CKPT_FETCH="from huggingface_hub import snapshot_download; print(snapshot_download('prehj/gr00t-n1.5-robocasa-fair-moe-v2-b-only', repo_type='model'))"
        SERVICE=scripts/inference_service_fair_moe_v2.py
        OUT_DIR_NAME=fair_moe_v2_b_only_as_is
        ;;
    3)
        CKPT_FETCH="from huggingface_hub import snapshot_download; print(snapshot_download('prehj/gr00t-n1.5-robocasa-metaq-v2-n8-b-only', repo_type='model'))"
        SERVICE=scripts/inference_service_metaq_v2.py
        OUT_DIR_NAME=metaq_v2_n8_b_only_as_is
        ;;
    *) echo "[ERR] unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "$CKPT_FETCH")"
OUTPUT_BASE="$BASE_DIR/output/robocasa/$OUT_DIR_NAME"
ODIR="$OUTPUT_BASE/$TASK"
mkdir -p "$ODIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$ODIR/server-resub.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/$SERVICE" --server \
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

# Run PnPCounterToSink ALONE — no parallel siblings — to avoid the
# array_task_id=3 framebuffer-not-complete race we previously hit.
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
    --max_episode_steps 1500 --generative_textures \
    >& "$ODIR/eval-resub.log"
kill "$SPID" 2>/dev/null
