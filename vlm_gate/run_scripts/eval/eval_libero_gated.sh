#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_libero_base_gr00t_n1_5_vlm_gated_quantize_k2_4suite_10arr_50ep
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_gated.out
#SBATCH --error=out/%A_%a-eval_libero_gated.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="LIBERO base GR00T-N1.5 + VLM-gated K2 quantization. GPU0=policy, GPU1=judge (gemma|cosmos). array=task_idx 0-9, 4 suites parallel/job (same as the working K2 baseline) x 50 ep."

# VLM-gated LIBERO eval. Each array task (task_idx 0-9): GPU0 serves the libero
# policy; GPU1 serves the judge; the 4 suites run as parallel clients that query
# the shared judge per chunk (quantize K vs raw). Writes evolver-compatible
# prediction.txt + gate_conf.csv under <gate_out>/<suite>_<task_idx>/.
#   JUDGE_BACKEND=gemma|cosmos  (default gemma)
set -u
JUDGE_BACKEND="${JUDGE_BACKEND:-gemma}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OPENPI="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
HF_REPO="prehj/GR00T-N1.5-libero-baseline-bs32-60k"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((9600 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((19600 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
K=2; TAU="${TAU:-0.5}"
# TTL skip policy (0 = off). Replay-calibrated per-backend defaults:
# cosmos confs cluster near tau (needs tight bands), gemma's sit at extremes.
GATE_TTL_MAX="${GATE_TTL_MAX:-0}"
VARK_BOUND="${VARK_BOUND:-0}"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  GATE_TTL_LO="${GATE_TTL_LO:-0.05}"; GATE_TTL_HI="${GATE_TTL_HI:-0.15}"
else
  GATE_TTL_LO="${GATE_TTL_LO:-0.15}"; GATE_TTL_HI="${GATE_TTL_HI:-0.30}"
fi
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/libero/${JUDGE_BACKEND}_gated}"
GATE_OUT="$OUTPUT_BASE/gate"
mkdir -p out "$OUTPUT_BASE" "$GATE_OUT"; cd "$PRIV"
export NO_ALBUMENTATIONS_UPDATE=1

CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

# ---- GPU0: libero policy server ----
NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$CONDA/envs/quant_gate/bin/python" "$PRIV/scripts/serve_policy.py" \
  --port=$PORT --model-path="$CKPT" --embodiment_tag=libero --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---- GPU1: judge ----
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
    "$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" \
    --serve --model nvidia/Cosmos3-Nano --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JUDGE_COMPILE=1 \
    "$CONDA/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" \
    --serve --model google/gemma-4-12b-it --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
fi
JPID=$!

# ---- wait for both ----
for i in $(seq 1 200); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3>&-; break; }; kill -0 $SPID 2>/dev/null || { echo "[ERR] policy died"; tail -40 "$SERVER_LOG"; exit 1; }; sleep 5; done
for i in $(seq 1 200); do grep -q "JUDGE READY" "$JUDGE_LOG" 2>/dev/null && break; kill -0 $JPID 2>/dev/null || { echo "[ERR] judge died"; tail -40 "$JUDGE_LOG"; exit 1; }; sleep 5; done
echo "[i] servers ready (array $SLURM_ARRAY_TASK_ID, judge=$JUDGE_BACKEND)"

# ---- 4 suites as parallel clients (task_idx = array id), shared policy+judge ----
# EGL: pin each client to ONE visible GPU (GPU0) via CUDA_VISIBLE_DEVICES=0.
# LIBERO's OffScreenRenderEnv (mujoco EGL) races during init when 2 GPUs are
# visible (the gpu:2 alloc we need for the judge) -> 4 parallel clients hit
# EGL_NOT_INITIALIZED and die. The working K2 baseline ran on gpu:1 (one GPU
# visible), so we reproduce that single-GPU visibility for the renderers here.
# (Do NOT use MUJOCO_EGL_DEVICE_ID — that pins the EGL *enumeration* and is the
# thing that crashed earlier; CUDA_VISIBLE_DEVICES limits *visibility* instead.)
# The judge stays on GPU1, reached over HTTP (clients need no GPU1 visibility).
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for SUITE in "${SUITES[@]}"; do
    ODIR="$OUTPUT_BASE/$SUITE"; mkdir -p "$ODIR"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$OPENPI:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
      "$CONDA/envs/libero/bin/python" "$PRIV/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py" \
      --args.task-suite-name "$SUITE" --args.task_idx=$SLURM_ARRAY_TASK_ID \
      --args.port=$PORT --args.host=127.0.0.1 --args.num_trials_per_task=$N_EPISODES \
      --args.compress_k=$K --args.video-out-path "$ODIR" \
      --args.judge-url "http://127.0.0.1:$JUDGE_PORT" --args.judge-threshold $TAU \
      --args.judge-guidance "@$GUIDANCE_FILE" --args.gate-out-dir "$GATE_OUT" \
      --args.gate-ttl-max $GATE_TTL_MAX --args.vark-bound $VARK_BOUND --args.gate-ttl-lo $GATE_TTL_LO --args.gate-ttl-hi $GATE_TTL_HI \
      >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done (judge=$JUDGE_BACKEND)."
