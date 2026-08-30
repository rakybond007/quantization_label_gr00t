#!/bin/bash
#SBATCH --job-name=eval_libero_baseline_vanilla_global_bs32_HF_60k_chkpt_4suite_10arr_50ep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_baseline_bs32_hf.out
#SBATCH --error=out/%A_%a-eval_libero_baseline_bs32_hf.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="Eval libero baseline bs32 HF (60k) on 4 task suites x 10 tasks x 50 ep"

# Adapted from run_scripts/eval/eval_gr00t_libero.sh — paths/env updated for our setup.
# Each array task: 1 GPU runs the policy server, 4 LIBERO suites are evaluated in
# parallel as separate clients (libero conda env). task_idx within each suite =
# SLURM_ARRAY_TASK_ID, so array=0..9 covers all 10 tasks per suite.
# Total: 10 array jobs × 4 suites × 50 ep = 2000 episodes.

set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
HF_REPO=prehj/GR00T-N1.5-libero-baseline-bs32-60k
CKPT="$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")"
OUTPUT_BASE="$BASE_DIR/output/libero/baseline_bs32_hf"
mkdir -p out "$OUTPUT_BASE"

cd "$BASE_DIR"

# CUDA libs from gr00t env
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

PORT=$((8260 + SLURM_ARRAY_TASK_ID))
echo "[i] Array $SLURM_ARRAY_TASK_ID | libero baseline | port=$PORT"

"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/serve_policy.py" \
    --port=$PORT \
    --model-path=$CKPT \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SERVE_PID=$!
sleep 60

TASK_NAMES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH="$HOME/multigpu_workspace/openpi/packages/openpi-client/src" "$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t.py" \
        --args.task-suite-name $TASK_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$SLURM_ARRAY_TASK_ID \
        --args.port=$PORT \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

kill "$SERVE_PID" 2>/dev/null
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
