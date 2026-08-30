#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_libero_base_gr00t_n1_5_naive_quantize_K_4suite_10arr_50ep
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-9
#SBATCH --output=out/%A_%a-eval_libero_naiveK.out
#SBATCH --error=out/%A_%a-eval_libero_naiveK.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="LIBERO base GR00T-N1.5 + NAIVE always-K block quantization (no VLM gate). K via env. Evolver-compatible baseline layout."

# Naive fixed-K baseline (judge disabled -> client compresses every chunk).
# env: K (default 3), N_EPISODES (default 50), OUTPUT_BASE.
# Writes evolver layout OUTPUT_BASE/<suite>_<idx>/prediction.txt via gate-out-dir.
set -u
K="${K:-3}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
OPENPI="$HOME/multigpu_workspace/openpi/packages/openpi-client/src"
HF_REPO="prehj/GR00T-N1.5-libero-baseline-bs32-60k"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((9600 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
VARK_BOUND="${VARK_BOUND:-0}"
CLIP_SCALE="${CLIP_SCALE:-1}"
DYN_SCALE="${DYN_SCALE:-1}"
VARK_FLOOR2="${VARK_FLOOR2:-0}"
case "$VARK_BOUND" in ""|0|0.0|0.00) VARTAG="K${K}";; *) VARTAG="varK${K}";; esac
case "$CLIP_SCALE" in ""|1|1.0) :;; *) VARTAG="clipK${K}";; esac
case "$DYN_SCALE" in ""|1|1.0) :;; *) VARTAG="dynK${K}";; esac
case "$VARK_FLOOR2" in ""|0) :;; *) VARTAG="fvarK${K}";; esac
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/libero/baseline_${VARTAG}}"
mkdir -p out "$OUTPUT_BASE"; cd "$PRIV"
export NO_ALBUMENTATIONS_UPDATE=1

CKPT=$("$CONDA/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")

cleanup(){ kill ${SPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} 2>/dev/null; }
trap cleanup EXIT

NV="$CONDA/envs/quant_gate/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$CONDA/envs/quant_gate/bin/python" "$PRIV/scripts/serve_policy.py" \
  --port=$PORT --model-path="$CKPT" --embodiment_tag=libero --head main > "$SERVER_LOG" 2>&1 &
SPID=$!
for i in $(seq 1 200); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3>&-; break; }; kill -0 $SPID 2>/dev/null || { echo "[ERR] policy died"; tail -40 "$SERVER_LOG"; exit 1; }; sleep 5; done
echo "[i] policy ready (array $SLURM_ARRAY_TASK_ID, naive K=$K)"

SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")
MAIN_PIDS=()
for SUITE in "${SUITES[@]}"; do
    ODIR="$OUTPUT_BASE/vid_$SUITE"; mkdir -p "$ODIR"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$OPENPI:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
      "$CONDA/envs/libero/bin/python" "$PRIV/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py" \
      --args.task-suite-name "$SUITE" --args.task_idx=$SLURM_ARRAY_TASK_ID \
      --args.port=$PORT --args.host=127.0.0.1 --args.num_trials_per_task=$N_EPISODES \
      --args.compress_k=$K --args.video-out-path "$ODIR" \
      --args.gate-out-dir "$OUTPUT_BASE" --args.vark-bound $VARK_BOUND --args.vark-floor2 $VARK_FLOOR2 --args.clip-scale $CLIP_SCALE --args.dyn-scale $DYN_SCALE \
      >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done (naive K=$K)."
