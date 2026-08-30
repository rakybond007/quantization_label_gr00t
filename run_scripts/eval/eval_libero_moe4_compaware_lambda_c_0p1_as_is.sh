#!/bin/bash
#SBATCH --job-name=eval_libero_moe4_compaware_lambda_c_0p1_as_is_60k_chkpt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_moe4_compaware_as_is.out
#SBATCH --error=out/%A_%a-eval_libero_moe4_compaware_as_is.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="Libero MoE4 (main+m8+m4+n8) compression-aware lambda_c=0.1 — as_is eval"

set -u
HEAD=moe; MODE=as_is
PORT=$((10000 + 180 + SLURM_ARRAY_TASK_ID))

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_moe4_per_expert_body_compression_aware_lambda_c_0p1/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/libero/moe4_compaware_lambda_c_0p1_as_is"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[i] libero MoE4 compaware as_is (head=$HEAD)"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/serve_policy.py" \
    --port=$PORT --model-path=$CKPT --head=$HEAD \
    --discrete-action-dims 6 > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SERVE_PID=$!
sleep 90

TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
        --args.task-suite-name $TASK_NAME --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID --args.port=$PORT \
        --args.decision_mode=$MODE \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SERVE_PID" 2>/dev/null
