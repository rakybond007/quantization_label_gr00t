#!/bin/bash
# Benchmark inference cost across 4 server configurations on per_expert_moe ckpt
# (and one separate baseline ckpt). Sequential server start/stop, dummy obs.
#
# Output: ms per inference call, mean / p50 / p95 per setting.
set -u
BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
PORT=8700
NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

OUT=$BASE/output/_bench_inference
mkdir -p $OUT

start_server() {
    local CKPT=$1; local HEAD=$2; local NSAMPLES=$3; local LABEL=$4
    echo "[$(date '+%T')] === START server: $LABEL (head=$HEAD N=$NSAMPLES) ==="
    PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
        --port $PORT \
        --model_path "$CKPT" \
        --data_config single_panda_gripper \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        --head $HEAD \
        --n_samples $NSAMPLES \
        --discrete-action-dims 6 11 \
        > $OUT/server_${LABEL}.log 2>&1 &
    SPID=$!
    for i in $(seq 1 60); do
        grep -q "Server is ready" $OUT/server_${LABEL}.log 2>/dev/null && break
        sleep 5
    done
    grep -q "Server is ready" $OUT/server_${LABEL}.log || { echo "[ERROR] server not ready"; tail -20 $OUT/server_${LABEL}.log; return 1; }
    echo "[$(date '+%T')]   server ready"
}

run_bench() {
    local LABEL=$1
    $CONDA/envs/gr00t/bin/python -u $BASE/scripts/_bench_inference_cost.py \
        --port $PORT --n-warmup 5 --n-trials 50 --label "$LABEL" \
        2>&1 | tee $OUT/bench_${LABEL}.log
}

stop_server() {
    [ -n "${SPID:-}" ] && { kill $SPID 2>/dev/null; sleep 5; }
    SPID=""
}

trap "stop_server" EXIT INT TERM

BASELINE_CKPT="$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
MOE_CKPT="$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"

# 1. baseline GR00T (no MoE) — head=main
start_server "$BASELINE_CKPT" main 1 baseline_main && run_bench baseline_main; stop_server

# 2. MoE per_expert no compression — head=moe (router + picked expert only)
start_server "$MOE_CKPT" moe 1 moe_no_comp && run_bench moe_no_comp; stop_server

# 3. MoE + AAC (entropy, N=10) — head=moe_selective with N samples
start_server "$MOE_CKPT" moe_selective 10 moe_aac && run_bench moe_aac; stop_server

# 4. MoE + self_consistency (self_agree, N=1) — head=moe_selective with N=1
start_server "$MOE_CKPT" moe_selective 1 moe_selfagree && run_bench moe_selfagree; stop_server

echo "[$(date '+%T')] === ALL DONE ==="
echo
echo "Summary (mean ms per inference):"
for L in baseline_main moe_no_comp moe_aac moe_selfagree; do
    grep "mean" $OUT/bench_${L}.log 2>/dev/null | head -1 | xargs -I{} echo "  $L  → {}"
done
