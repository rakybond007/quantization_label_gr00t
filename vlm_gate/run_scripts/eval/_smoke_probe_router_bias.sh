#!/bin/bash
# Controlled probe: does obs['_compress_bias'] move the MoE router? Same obs,
# swept bias, watch moe_probs. NOT sbatch — run inside the interactive alloc.
set -u
TASK="${1:-OpenDrawer}"
CONF="${2:-0.7}"   # confp07 routing to match eval; probs are deterministic anyway
JUDGE_MODEL=""
HF_REPO="prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"; CONDA="$HOME/miniconda3"
PORT=11500
OUT="$BASE_DIR/output/robocasa/_probe_router_bias"; mkdir -p "$OUT"; cd "$BASE_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SLD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[probe] ckpt=$CKPT"

cleanup(){ kill ${SPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

SLOG="$OUT/server.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SLD" PYTHONUNBUFFERED=1 \
"$CONDA/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" --data_config single_panda_gripper \
    --embodiment_tag new_embodiment --denoising_steps 4 --head moe \
    --discrete-action-dims 6 11 --moe-stochastic --moe-confidence-threshold $CONF > "$SLOG" 2>&1 &
SPID=$!
for i in $(seq 1 150); do grep -q "Server is ready" "$SLOG" 2>/dev/null && break; kill -0 $SPID 2>/dev/null || { echo "[ERR] server died"; tail -30 "$SLOG"; exit 1; }; sleep 5; done
echo "[probe] server ready"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/probe_router_bias.py" \
    --env_name "$TASK" --port $PORT --host 127.0.0.1 --n_probes 6 2>&1 | tee "$OUT/probe.log"
echo "[probe] DONE"
