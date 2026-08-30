#!/bin/bash
# Smoke libero MoE 4-mode eval. 1 task (libero_10 task 0) × 3 ep per mode.
# Usage: bash _smoke_libero_moe.sh [as_is|aac_cliff|aac_chunk_binary|self_agree]
set -u
MODE=${1:?'need mode: as_is | aac_cliff | aac_chunk_binary | self_agree'}

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/libero/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
PORT=8800

case "$MODE" in
    as_is)              HEAD=moe;            N=1;  EXTRA_ARGS=""; ;;
    aac_cliff)          HEAD=moe_selective;  N=10; EXTRA_ARGS="--args.aac_xi 1"; ;;
    aac_chunk_binary)   HEAD=moe_selective;  N=10; EXTRA_ARGS="--args.aac_xi 4"; ;;
    self_agree)         HEAD=moe_selective;  N=1;  EXTRA_ARGS="--args.self_agree_tau 0.5"; ;;
    *) echo "[ERROR] unknown mode $MODE"; exit 1 ;;
esac

OUT="$BASE_DIR/output/libero/_smoke_moe/$MODE"
mkdir -p "$OUT"

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[$(date '+%T')] === SMOKE: libero $MODE  (head=$HEAD N=$N) ==="
"$CONDA_PATH/envs/gr00t/bin/python" "$BASE_DIR/scripts/serve_policy.py" \
    --port=$PORT --model-path=$CKPT --head=$HEAD --n-samples=$N \
    --discrete-action-dims 6 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM

for i in $(seq 1 90); do
    grep -q "Creating server" "$OUT/server.log" 2>/dev/null && break
    sleep 5
done
sleep 30   # extra wait for full bind
echo "[$(date '+%T')]   server started, running client..."

"$CONDA_PATH/envs/libero/bin/python" "$BASE_DIR/gr00t/eval/libero/eval_taskwise_gr00t_moe.py" \
    --args.task-suite-name libero_10 \
    --args.video-out-path "$OUT" \
    --args.task_idx=0 \
    --args.port=$PORT \
    --args.num_trials_per_task=3 \
    --args.decision_mode=$MODE \
    $EXTRA_ARGS \
    >& "$OUT/eval.log"

succ=$(grep "Total success rate" "$OUT/0_results.txt" 2>/dev/null | tail -1)
err=$(grep -c "Traceback\|Caught exception" "$OUT/eval.log" 2>/dev/null)
picks=$(grep "Pick counts" "$OUT/0_results.txt" 2>/dev/null | tail -1)
kinds=$(grep "Chunk kind counts" "$OUT/0_results.txt" 2>/dev/null | tail -1)
echo "[$(date '+%T')]   $MODE → $succ | $picks | $kinds | errors=$err"
kill $SPID 2>/dev/null
