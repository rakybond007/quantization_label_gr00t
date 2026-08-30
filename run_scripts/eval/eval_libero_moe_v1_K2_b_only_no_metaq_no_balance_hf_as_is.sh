#!/bin/bash
#SBATCH --job-name=eval_libero_moe_v1_K2_raw16_m8_b_only_no_balance_HF_60k_as_is_4suite_10arr_50ep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_moe_v1_K2_b_only_no_balance.out
#SBATCH --error=out/%A_%a-eval_libero_moe_v1_K2_b_only_no_balance.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="libero MoE v1 K=2 (raw16, merged8) b-only no-metaq NO-BALANCE HF @60k, head=moe as_is, 4 suites."

set -u
HF_REPO=prehj/GR00T-N1.5-libero-moe-v1-K2-raw16-merged8-b-only-no-metaq-no-balance-60k
PORT=$((9120 + SLURM_ARRAY_TASK_ID))

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/libero/moe_v1_K2_raw16_m8_b_only_no_metaq_no_balance_hf_as_is"
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
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; kill "$SPID" 2>/dev/null; exit 1; }
sleep 20

OPENPI_SRC="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH="$OPENPI_SRC" "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
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
