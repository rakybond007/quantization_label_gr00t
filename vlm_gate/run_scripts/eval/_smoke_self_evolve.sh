#!/bin/bash
# Short INTERACTIVE self-evolve loop (eval -> evolve -> eval) to validate the
# whole mechanism incl. evolver v2 accept/reject. Base GR00T + Gemma judge, few
# tasks x 1 ep. Servers load once; client + evolve run per cycle on the worker
# (worker has network, so `claude -p` evolve works here). NOT the full 1200-ep
# sbatch loop — this is a fast end-to-end smoke.
# Usage: _smoke_self_evolve.sh [cycles] ["task list"] [episodes] [max_steps]
set -u
CYCLES="${1:-2}"
TASKS="${2:-OpenDrawer CloseDrawer CoffeePressButton CloseSingleDoor}"
N_EP="${3:-1}"
MAX_STEPS="${4:-250}"
K="${K:-2}"

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA="$HOME/miniconda3"
CKPT="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
JUDGE_MODEL="google/gemma-4-12b-it"
PORT="${PORT:-10940}"; JUDGE_PORT="${JUDGE_PORT:-8140}"
RUN="$BASE_DIR/output/robocasa/_se_smoke"
GUIDE="$BASE_DIR/analysis/_evolver/se_smoke_guidance.txt"      # evolving guidance (live file untouched)
SEED_GUIDE="$BASE_DIR/run_scripts/eval/vlm_gate_guidance.txt"
mkdir -p "$RUN"; cd "$BASE_DIR"
cp "$SEED_GUIDE" "$GUIDE"
rm -f analysis/_evolver/best_state.json                        # fresh gating state for this smoke

export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null || true; sleep 1
cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; timeout 10 fuser -k "${PORT}/tcp" "${JUDGE_PORT}/tcp" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[se] (GPU0) base GR00T server :$PORT"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
  "$CONDA/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
  --port $PORT --model_path "$CKPT" --data_config single_panda_gripper --embodiment_tag new_embodiment \
  --denoising_steps 4 --head main > "$RUN/server.log" 2>&1 &
SPID=$!
echo "[se] (GPU1) Gemma judge :$JUDGE_PORT"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  "$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
  --model "$JUDGE_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$RUN/judge.log" 2>&1 &
JPID=$!
wait_ready(){ for i in $(seq 1 150); do grep -qE "$2" "$1" 2>/dev/null && return 0; kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -30 "$1"; return 1; }; sleep 5; done; echo "[ERR] $4 timeout"; tail -30 "$1"; return 1; }
wait_ready "$RUN/server.log" "Server is ready" "$SPID" "GR00T" || exit 1
wait_ready "$RUN/judge.log" "JUDGE READY" "$JPID" "Gemma" || exit 1
echo "[se] servers ready"

prev=""
for c in $(seq 1 "$CYCLES"); do
  OUT="$RUN/cycle$c"; mkdir -p "$OUT"
  echo "[se] ========== cycle $c : EVAL =========="
  for T in $TASKS; do
    mkdir -p "$OUT/$T"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
      "$CONDA/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/robocasa_service_compress.py" \
      --port $PORT --host 127.0.0.1 --env_name "$T" --video_dir "$OUT/$T" \
      --seed 42 --n_episodes "$N_EP" --max_episode_steps "$MAX_STEPS" --generative_textures \
      --compress-k "$K" --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-guidance "$GUIDE" \
      > "$OUT/$T/eval.log" 2>&1
    echo "[se]   $T -> $(grep -E 'is_success:' "$OUT/$T/prediction.txt" 2>/dev/null | head -1)"
  done
  echo "[se] ========== cycle $c : EVOLVE (v2) =========="
  args=(--gate "$OUT" --guidance-file "$GUIDE")
  [ -n "$prev" ] && args+=(--prev-gate "$prev")
  python3 "$BASE_DIR/scripts/evolve_gate_prompt.py" "${args[@]}" 2>&1 \
    | grep -E "GATING|running-best|AGGREGATE READ|CHANGE|\[written\]" | head -10
  prev="$OUT"
done
echo "[se] DONE $CYCLES cycles. audit: analysis/_evolver/evolution_log.jsonl"
rm -f analysis/_evolver/best_state.json   # don't leave smoke state for real runs
