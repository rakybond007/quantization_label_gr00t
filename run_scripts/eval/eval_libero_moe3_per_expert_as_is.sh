#!/bin/bash
#SBATCH --job-name=eval_libero_moe3_per_expert_main_m8_m4_router_pick_as_is_no_compression_60k_chkpt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_moe3_as_is.out
#SBATCH --error=out/%A_%a-eval_libero_moe3_as_is.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="Libero MoE3 (main+m8+m4) per_expert eval — head=moe (router pick as-is, no compression heuristic)"

set -u
HEAD=moe
N=1
MODE=as_is
PORT=$((9810 + SLURM_ARRAY_TASK_ID))

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_moe3_per_expert_body/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/libero/moe3_per_expert_${MODE}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[i] Array $SLURM_ARRAY_TASK_ID | libero MoE3 $MODE (head=$HEAD N=$N) | port=$PORT"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/serve_policy.py" \
    --port=$PORT --model-path=$CKPT --head=$HEAD --n-samples=$N \
    --discrete-action-dims 6 \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SERVE_PID=$!
sleep 90

TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
        --args.task-suite-name $TASK_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID \
        --args.port=$PORT \
        --args.decision_mode=$MODE \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
kill "$SERVE_PID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
