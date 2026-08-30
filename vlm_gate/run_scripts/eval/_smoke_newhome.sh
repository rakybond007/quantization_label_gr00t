#!/bin/bash
# NEW-HOME migration smoke: one <bench> x <backend> combo on a 2-GPU alloc.
# Everything resolves from $HOME (run with HOME=/sjw_alinlab/home/hojin2).
#   usage: _smoke_newhome.sh <libero|robocasa> <gemma|cosmos>
set -u
BENCH="$1"; BACKEND="$2"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OPENPI="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
OUT="$BASE_DIR/output/_newhome_smoke/${BENCH}_${BACKEND}"
rm -rf "$OUT"; mkdir -p "$OUT/gate"
PORT=9931; JPORT=19931
export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
GUIDE="$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt"

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

# ---- GPU0: policy server ----
if [ "$BENCH" = libero ]; then
  CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('prehj/GR00T-N1.5-libero-baseline-bs32-60k'))")
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$CONDA/envs/quant_gate/bin/python" \
    "$PRIV/scripts/serve_policy.py" --port=$PORT --model-path="$CKPT" \
    --embodiment_tag=libero --head main > "$OUT/server.log" 2>&1 &
else
  CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$CONDA/envs/quant_gate/bin/python" -u \
    "$BASE_DIR/scripts/inference_service.py" --server --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head main > "$OUT/server.log" 2>&1 &
fi
SPID=$!

# ---- GPU1: judge ----
if [ "$BACKEND" = cosmos ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
    "$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" \
    --serve --model nvidia/Cosmos3-Nano --port $JPORT --host 127.0.0.1 > "$OUT/judge.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JUDGE_COMPILE=1 \
    "$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" \
    --serve --model google/gemma-4-12b-it --port $JPORT --host 127.0.0.1 > "$OUT/judge.log" 2>&1 &
fi
JPID=$!

for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3>&-; break; }; kill -0 $SPID 2>/dev/null || { echo "[SMOKE:$BENCH/$BACKEND] ERR policy died"; tail -15 "$OUT/server.log"; exit 1; }; sleep 5; done
for i in $(seq 1 150); do grep -q "JUDGE READY" "$OUT/judge.log" 2>/dev/null && break; kill -0 $JPID 2>/dev/null || { echo "[SMOKE:$BENCH/$BACKEND] ERR judge died"; tail -15 "$OUT/judge.log"; exit 1; }; sleep 5; done
echo "[SMOKE:$BENCH/$BACKEND] servers ready"

# ---- client (2 episodes, TTL on) ----
if [ "$BENCH" = libero ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$OPENPI:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
    "$CONDA/envs/libero/bin/python" "$PRIV/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py" \
    --args.task-suite-name libero_spatial --args.task_idx=0 \
    --args.port=$PORT --args.host=127.0.0.1 --args.num_trials_per_task=2 \
    --args.compress_k=2 --args.video-out-path "$OUT/vid" \
    --args.judge-url "http://127.0.0.1:$JPORT" --args.judge-threshold 0.5 \
    --args.judge-guidance "@$GUIDE" --args.gate-out-dir "$OUT/gate" \
    --args.gate-ttl-max 3 > "$OUT/client.log" 2>&1
  RC=$?
  D="$OUT/gate/libero_spatial_0"
else
  PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
    "$CONDA/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/robocasa_service_compress.py" \
    --port $PORT --host localhost --env_name CoffeePressButton \
    --video_dir "$OUT/task" --seed 42 --n_episodes 2 --max_episode_steps 700 \
    --generative_textures --compress-k 2 \
    --judge-url "http://127.0.0.1:$JPORT" --judge-threshold 0.5 \
    --judge-guidance "@$GUIDE" --gate-subchunk 8 --gate-ttl-max 3 > "$OUT/client.log" 2>&1
  RC=$?
  D="$OUT/task"
fi
echo "[SMOKE:$BENCH/$BACKEND] client rc=$RC"
grep -E "^episode |gate_quantize_rate|gate_call_rate" "$D/prediction.txt" 2>/dev/null | head -5 | sed "s/^/[SMOKE:$BENCH\/$BACKEND]   /"
[ $RC -eq 0 ] && grep -q "gate_call_rate" "$D/prediction.txt" 2>/dev/null && echo "[SMOKE:$BENCH/$BACKEND] ✅ PASS" || echo "[SMOKE:$BENCH/$BACKEND] ❌ FAIL"
