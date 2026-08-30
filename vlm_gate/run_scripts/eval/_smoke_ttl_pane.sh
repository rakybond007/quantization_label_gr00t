#!/bin/bash
# Pane smoke for the TTL skip policy: policy + gemma judge (compiled) + TTL client
# all on ONE GPU (the tmux 2:0 debug alloc). 1 suite x 1 task x 2 episodes.
set -u
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OPENPI="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
OUT="$BASE_DIR/output/libero/_ttl_smoke"
rm -rf "$OUT"; mkdir -p "$OUT/gate"
PORT=9911; JPORT=19911
export NO_ALBUMENTATIONS_UPDATE=1
CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('prehj/GR00T-N1.5-libero-baseline-bs32-60k'))")
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$CONDA/envs/quant_gate/bin/python" \
  "$PRIV/scripts/serve_policy.py" --port=$PORT --model-path="$CKPT" \
  --embodiment_tag=libero --head main > "$OUT/server.log" 2>&1 &
SPID=$!
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JUDGE_COMPILE=1 \
  "$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" \
  --serve --model google/gemma-4-12b-it --port $JPORT --host 127.0.0.1 > "$OUT/judge.log" 2>&1 &
JPID=$!
for i in $(seq 1 120); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3>&-; break; }; kill -0 $SPID 2>/dev/null || { echo "[ERR] policy died"; tail -20 "$OUT/server.log"; exit 1; }; sleep 5; done
for i in $(seq 1 150); do grep -q "JUDGE READY" "$OUT/judge.log" 2>/dev/null && break; kill -0 $JPID 2>/dev/null || { echo "[ERR] judge died"; tail -20 "$OUT/judge.log"; exit 1; }; sleep 5; done
echo "[smoke] servers ready"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$OPENPI:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
  "$CONDA/envs/libero/bin/python" "$PRIV/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py" \
  --args.task-suite-name libero_spatial --args.task_idx=0 \
  --args.port=$PORT --args.host=127.0.0.1 --args.num_trials_per_task=2 \
  --args.compress_k=2 --args.video-out-path "$OUT/vid" \
  --args.judge-url "http://127.0.0.1:$JPORT" --args.judge-threshold 0.5 \
  --args.judge-guidance "@$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt" \
  --args.gate-out-dir "$OUT/gate" \
  --args.gate-ttl-max 3 --args.gate-ttl-lo 0.15 --args.gate-ttl-hi 0.30 \
  > "$OUT/client.log" 2>&1
RC=$?
echo "[smoke] client rc=$RC"
echo "----- gate outputs -----"
D="$OUT/gate/libero_spatial_0"
echo "== prediction.txt =="; cat "$D/prediction.txt" 2>/dev/null
echo "== gate_conf.csv (head) =="; head -5 "$D/gate_conf.csv" 2>/dev/null
echo "== sidecar (fields) =="; head -1 "$D/ep_records.jsonl" 2>/dev/null | cut -c1-200
echo "SMOKE_RC=$RC"
