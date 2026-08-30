#!/bin/bash
# Smoke selective compression: start server (head=selective) on a ckpt, sweep τ
# values via robocasa_service_selective.py client on a few hard tasks.
#
# Usage:
#   # Calibration (no compression, just dump scores):
#   bash _smoke_selective.sh mh_m8 self_agree 1 calibrate
#
#   # Threshold sweep (one client run per τ):
#   bash _smoke_selective.sh mh_m8 self_agree 1 tau:0.05 tau:0.15 tau:0.30
#
#   # Entropy / hybrid (need N>=2 server samples):
#   bash _smoke_selective.sh per_expert_moe entropy 10 tau:0.001 tau:0.005 tau:0.020
set -u

CKPT_TAG=${1:?"need ckpt tag: mh_m8 | per_expert_moe"}
SCORE=${2:-self_agree}
NSAMPLES=${3:-1}
shift 3 || true
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(calibrate)

BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
PORT=8400
N_EP=10
TASKS=("CoffeeSetupMug" "TurnOffStove")

case "$CKPT_TAG" in
    mh_m8)
        CKPT="$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_discfix/checkpoint-60000"
        ;;
    per_expert_moe)
        CKPT="$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
        ;;
    *)
        echo "[ERROR] unknown ckpt tag $CKPT_TAG"; exit 1
        ;;
esac

OUT_BASE="$BASE/output/robocasa/_smoke_selective/${CKPT_TAG}_${SCORE}_n${NSAMPLES}"
mkdir -p "$OUT_BASE"

NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[$(date '+%T')] === SMOKE: $CKPT_TAG / $SCORE / N=$NSAMPLES / args=${ARGS[*]} ==="
echo "[$(date '+%T')] Server log: $OUT_BASE/server.log"

PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
    --port $PORT \
    --model_path "$CKPT" \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head selective \
    --n_samples "$NSAMPLES" \
    --discrete-action-dims 6 11 \
    > "$OUT_BASE/server.log" 2>&1 &
SPID=$!
trap "echo '[trap] kill $SPID'; kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null; exit" EXIT INT TERM

for i in $(seq 1 120); do
    grep -q "Server is ready" "$OUT_BASE/server.log" 2>/dev/null && break
    sleep 5
done
if ! grep -q "Server is ready" "$OUT_BASE/server.log" 2>/dev/null; then
    echo "[ERROR] Server not ready in 10 min"; tail -30 "$OUT_BASE/server.log"; exit 1
fi
echo "[$(date '+%T')] Server ready (port $PORT)"

run_client() {
    local LABEL=$1 EXTRA=$2
    for TASK in "${TASKS[@]}"; do
        ODIR="$OUT_BASE/${LABEL}/$TASK"
        mkdir -p "$ODIR"
        echo "[$(date '+%T')]   $LABEL task=$TASK -> $ODIR"
        PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u \
            $BASE/scripts/robocasa_service_selective.py \
            --port $PORT --host localhost \
            --env_name "$TASK" \
            --video_dir "$ODIR" \
            --seed 42 \
            --n_episodes $N_EP \
            --max_episode_steps 1500 \
            --generative_textures \
            --score-mode $SCORE \
            $EXTRA \
            > "$ODIR/eval.log" 2>&1
        succ=$(grep -E "^episode .* True" "$ODIR/prediction.txt" 2>/dev/null | wc -l)
        ep=$(grep -c "^episode" "$ODIR/prediction.txt" 2>/dev/null)
        meanlen=$(grep "mean_exec_chunk_len" "$ODIR/prediction.txt" 2>/dev/null | tail -1 | tr -d '\n')
        quants=$(grep "score_quantiles" "$ODIR/prediction.txt" 2>/dev/null | tail -1 | tr -d '\n')
        echo "[$(date '+%T')]     -> succ=$succ/$ep  $meanlen  $quants"
    done
}

for arg in "${ARGS[@]}"; do
    case "$arg" in
        calibrate)
            run_client "calibrate" "--dump-scores-only" ;;
        tau:*)
            tau_val=${arg#tau:}
            label="tau${tau_val//./p}"
            run_client "$label" "--tau $tau_val" ;;
        k:*)
            k_val=${arg#k:}
            label="kpref${k_val}"
            run_client "$label" "--k-prefix $k_val" ;;
        *)
            echo "[ERROR] unknown arg $arg (expected 'calibrate' | 'tau:VALUE' | 'k:VALUE')"; exit 1 ;;
    esac
done

kill $SPID 2>/dev/null
echo "[$(date '+%T')] DONE smoke $CKPT_TAG / $SCORE / N=$NSAMPLES"
