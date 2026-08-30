#!/bin/bash
# Smoke: MoE-v1 VLA + VLM-gate router bias on robocasa.
#   GPU0: MoE policy server (inference_service_fair_moe, head=moe)
#   GPU1: VLM judge (Gemma 4 12B, vlm_gate.py)
#   client: robocasa_service_moe.py with --judge-url -> when VLM conf >= tau,
#           add --bias to the router logits of the compressed decoders (m8/m4).
# NOT for sbatch. Run on the 2-GPU interactive allocation (tmux 0:0).
# Usage: _smoke_robocasa_moe_vlm_router.sh [tasks] [episodes] [bias] [tau]
set -u
TASKS="${1:-OpenDrawer CoffeePressButton}"
N_EPISODES="${2:-2}"
BIAS="${3:-2.0}"
TAU="${4:-0.5}"
JUDGE_MODEL="google/gemma-4-12b-it"
HF_REPO="prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"
GUIDANCE="@$HOME/quantization_agent_workspace/vlm_gate/analysis/_evolver/guidance_versions/v9_manual.txt"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA="$HOME/miniconda3"
PORT=11000; JUDGE_PORT=11100
OUT="$BASE_DIR/output/robocasa/_smoke_moe_vlm_router"
mkdir -p "$OUT"; cd "$BASE_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "[smoke] resolving MoE ckpt..."
CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
echo "[smoke] ckpt=$CKPT"

cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

# GPU0: MoE server
SLOG="$OUT/server.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    --moe-stochastic --moe-confidence-threshold 0.7 > "$SLOG" 2>&1 &
SPID=$!

# GPU1: VLM judge
JLOG="$OUT/judge.log"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
"$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
    --model "$JUDGE_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$JLOG" 2>&1 &
JPID=$!

wait_ready() { for i in $(seq 1 150); do grep -qE "$2" "$1" 2>/dev/null && return 0; kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }; sleep 5; done; echo "[ERR] $4 not ready"; tail -40 "$1"; return 1; }
wait_ready "$SLOG" "Server is ready" "$SPID" "MoE server" || exit 1
echo "[smoke] MoE server ready"
wait_ready "$JLOG" "JUDGE READY" "$JPID" "VLM judge" || exit 1
echo "[smoke] judge ready"

for TASK in $TASKS; do
    ODIR="$OUT/$TASK"; mkdir -p "$ODIR"
    echo "==================== $TASK ===================="
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
    "$CONDA/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/robocasa_service_moe.py" \
        --port $PORT --host 127.0.0.1 --env_name "$TASK" --video_dir "$ODIR" \
        --seed 42 --n_episodes "$N_EPISODES" --max_episode_steps 400 --generative_textures \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold "$TAU" --bias "$BIAS" \
        --bias-mode "${BIAS_MODE:-signed}" --judge-guidance "$GUIDANCE" 2>&1 | tee "$ODIR/eval.log"
    echo "[smoke] $TASK router picks + gate:"; tail -4 "$ODIR/prediction.txt" 2>/dev/null
done
echo "[smoke] ALL DONE"
