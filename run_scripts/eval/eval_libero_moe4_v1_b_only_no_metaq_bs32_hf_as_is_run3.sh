#!/bin/bash
#SBATCH --job-name=eval_libero_moe4_v1_K4_b_only_no_metaq_bs32_HF_60k_chkpt_as_is_router_pick_4suite_10arr_50ep_run3
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_moe4_v1_b_only_no_metaq_hf.out
#SBATCH --error=out/%A_%a-eval_libero_moe4_v1_b_only_no_metaq_hf.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="libero MoE4 v1 (K=4 u16/c8/c4/n8) + B only + no metaq HF ckpt @60k, as_is router pick, all 4 suites."

set -u
HF_REPO=prehj/GR00T-N1.5-libero-moe4-v1-K4-b-only-no-metaq-bs32-60k
PORT=$((8990 + SLURM_ARRAY_TASK_ID))

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/libero/moe4_v1_b_only_no_metaq_bs32_hf_as_is_run3"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"

SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/serve_policy_fair_moe.py" \
    --port=$PORT --model-path=$CKPT --head=moe --n-samples=1 \
    --discrete-action-dims 6 \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
READY=0
for i in $(seq 1 90); do
    if grep -q "Creating server" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died early"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; fi
sleep 20

TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH="$HOME/multigpu_workspace/openpi/packages/openpi-client/src" "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
        --args.task-suite-name $TASK_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID \
        --args.port=$PORT \
        --args.decision_mode=as_is \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SPID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
