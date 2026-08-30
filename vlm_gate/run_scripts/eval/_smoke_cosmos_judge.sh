#!/bin/bash
# Smoke: nvidia/Cosmos3-Nano VLM judge via the Transformers reasoner path
# (in-process Cosmos3OmniForConditionalGeneration, NO vLLM).
#   GPU0: vlm_gate_cosmos.py --serve loads the reasoner, exposes POST /judge
#   test: --ping a dummy image -> expect continuous confidence in [0,1]
# NOT sbatch. Run inside the interactive GPU alloc (srun --overlap).
set -u
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
MODEL="nvidia/Cosmos3-Nano"
GATE_PORT="${GATE_PORT:-8123}"
OUT="$BASE_DIR/output/_smoke_cosmos"; mkdir -p "$OUT"; cd "$BASE_DIR"
GATE_LOG="$OUT/gate.log"

timeout 10 fuser -k "${GATE_PORT}/tcp" 2>/dev/null || true
sleep 1
cleanup(){ kill ${GPID:-} 2>/dev/null; timeout 10 fuser -k "${GATE_PORT}/tcp" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[smoke] launching Cosmos3-Nano reasoner judge on GPU0 (loads ~16B, a few min) ..."
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
  "$VENV/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
    --model "$MODEL" --port "$GATE_PORT" --host 127.0.0.1 > "$GATE_LOG" 2>&1 &
GPID=$!
ok=0
for i in $(seq 1 144); do
  grep -q "JUDGE READY" "$GATE_LOG" 2>/dev/null && { ok=1; break; }
  kill -0 $GPID 2>/dev/null || { echo "[ERR] judge died during load"; tail -60 "$GATE_LOG"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "[ERR] judge not ready in time"; tail -60 "$GATE_LOG"; exit 1; }
echo "[smoke] judge ready."

"$VENV/bin/python" - <<PY
from PIL import Image; import numpy as np
Image.fromarray((np.random.rand(256,256,3)*255).astype('uint8')).save("$OUT/dummy.png")
print("[smoke] dummy image written")
PY

echo "[smoke] pinging with dummy image + instruction ..."
PYTHONPATH="$BASE_DIR/scripts" "$VENV/bin/python" "$BASE_DIR/scripts/vlm_gate_cosmos.py" \
  --ping "$OUT/dummy.png" --url "http://127.0.0.1:$GATE_PORT" \
  --instruction "the robot arm is reaching across the counter toward a mug" 2>&1 | tee "$OUT/ping.out"
echo "[smoke] DONE"
