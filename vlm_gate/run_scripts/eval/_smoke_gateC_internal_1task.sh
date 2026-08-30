#!/bin/bash
# srun debug smoke: gateC ckpt loads with gate via from_pretrained + p_quant varies.
set -u
CKPT_DIR="/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/vla_gateC_60k/checkpoint-60000"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
PORT=$((13000 + RANDOM % 800))
OUT="$BASE_DIR/output/robocasa/_smoke_gateC_internal"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$BASE_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
cleanup(){ kill ${SPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} 2>/dev/null; }
trap cleanup EXIT

SERVER_LOG="$OUT/server.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main_gate > "$SERVER_LOG" 2>&1 &
SPID=$!
for i in $(seq 1 120); do
    grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { echo "[ERR] server died"; tail -60 "$SERVER_LOG"; exit 1; }
    sleep 5
done
grep -q "Server is ready" "$SERVER_LOG" || { echo "[ERR] not ready"; tail -60 "$SERVER_LOG"; exit 1; }
echo "== server up. quant_gate attach line:"
grep -i "quant_gate" "$SERVER_LOG" || echo "[WARN] no attach line in server log"

TASK=TurnOnStove
ODIR="$OUT/$TASK"; mkdir -p "$ODIR"
PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_compress.py" \
    --port $PORT --host localhost --env_name "$TASK" \
    --video_dir "$ODIR" --seed 42 --n_episodes 2 \
    --max_episode_steps 400 --generative_textures \
    --compress-k 2 --judge-url internal --judge-threshold 0.5 --gate-subchunk 0 \
    2>&1 | tail -20

echo "== gate_conf.csv summary =="
"$CONDA_PATH/envs/quant_gate_eval/bin/python" - "$ODIR/gate_conf.csv" <<'EOF'
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1])
print(f"n={len(df)} p: min={df.conf.min():.3f} max={df.conf.max():.3f} "
      f"mean={df.conf.mean():.3f} std={df.conf.std():.3f} nunique={df.conf.nunique()}")
print("quantize_rate:", df.quantize.mean().round(3))
for ep, g in df.groupby("episode"):
    print(f"ep{ep}: n={len(g)} p_std={g.conf.std():.4f} p=[{g.conf.min():.3f},{g.conf.max():.3f}] qrate={g.quantize.mean():.2f}")
assert df.conf.nunique() > 1, "CONSTANT GATE PROB — collapsed"
print("SMOKE PASS: p_quant varies")
EOF
