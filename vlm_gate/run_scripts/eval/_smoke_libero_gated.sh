#!/bin/bash
# Smoke: VLM-gated action quantization on LIBERO (base GR00T-N1.5).
#   GPU0: libero policy server (serve_policy.py, quant_gate env, websocket)
#   GPU1: VLM judge (gemma | cosmos)
#   client: gr00t/eval/libero/eval_taskwise_gr00t_quantize.py (libero env) with
#           --judge-url -> per-chunk gate (quantize K vs raw) + evolver output.
# NOT sbatch. Run inside the 2-GPU interactive alloc (srun --overlap).
# Usage: JUDGE_BACKEND=gemma|cosmos _smoke_libero_gated.sh [suite] [task_idx] [n_ep] [K]
set -u
SUITE="${1:-libero_object}"
TASK_IDX="${2:-0}"
N_EP="${3:-3}"
K="${4:-2}"
JUDGE_BACKEND="${JUDGE_BACKEND:-gemma}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OPENPI="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
HF_REPO="prehj/GR00T-N1.5-libero-baseline-bs32-60k"
PORT="${PORT:-9571}"; JUDGE_PORT="${JUDGE_PORT:-8151}"
OUT="$BASE_DIR/output/libero/_smoke_gated_${JUDGE_BACKEND}"
GATE_OUT="$OUT/gate"; SUITE_OUT="$OUT/$SUITE"
GUIDE="${GUIDANCE_FILE:-$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt}"  # naive seed
mkdir -p "$OUT" "$GATE_OUT" "$SUITE_OUT"; cd "$PRIV"
export NO_ALBUMENTATIONS_UPDATE=1

CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[smoke] libero ckpt=$CKPT"

timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null || true; sleep 1
cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---- GPU0: libero policy server ----
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
echo "[smoke] (GPU0) serve_policy :$PORT"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$CONDA/envs/quant_gate/bin/python" "$PRIV/scripts/serve_policy.py" \
  --port=$PORT --model-path="$CKPT" --embodiment_tag=libero --head main > "$OUT/server.log" 2>&1 &
SPID=$!

# ---- GPU1: judge ----
echo "[smoke] (GPU1) judge=$JUDGE_BACKEND :$JUDGE_PORT"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
    "$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" \
    --serve --model nvidia/Cosmos3-Nano --port $JUDGE_PORT --host 127.0.0.1 > "$OUT/judge.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    "$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" \
    --serve --model google/gemma-4-12b-it --port $JUDGE_PORT --host 127.0.0.1 > "$OUT/judge.log" 2>&1 &
fi
JPID=$!

# ---- wait for both (policy: poll port; judge: JUDGE READY) ----
echo "[smoke] waiting for servers..."
for i in $(seq 1 180); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3>&- ; break; }; kill -0 $SPID 2>/dev/null || { echo "[ERR] policy died"; tail -40 "$OUT/server.log"; exit 1; }; sleep 5; done
echo "[smoke] policy server up."
for i in $(seq 1 180); do grep -q "JUDGE READY" "$OUT/judge.log" 2>/dev/null && break; kill -0 $JPID 2>/dev/null || { echo "[ERR] judge died"; tail -40 "$OUT/judge.log"; exit 1; }; sleep 5; done
echo "[smoke] judge up. running gated eval ($SUITE task $TASK_IDX, $N_EP ep, K=$K)..."

# ---- client: gated libero eval ----
PYTHONPATH="$OPENPI:${PYTHONPATH:-}" MUJOCO_EGL_DEVICE_ID=0 \
  "$CONDA/envs/libero/bin/python" "$PRIV/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py" \
  --args.task-suite-name "$SUITE" --args.task_idx=$TASK_IDX --args.port=$PORT --args.host=127.0.0.1 \
  --args.num_trials_per_task=$N_EP --args.compress_k=$K --args.video-out-path "$SUITE_OUT" \
  --args.judge-url "http://127.0.0.1:$JUDGE_PORT" --args.judge-threshold 0.5 \
  --args.judge-guidance "@$GUIDE" --args.gate-out-dir "$GATE_OUT" 2>&1 | tail -25
echo "[smoke] DONE. evolver output:"
cat "$GATE_OUT/${SUITE}_${TASK_IDX}/prediction.txt" 2>/dev/null
