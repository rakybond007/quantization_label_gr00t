#!/bin/bash
# Master smoke driver: run all remaining hyperparameter-search smokes for
# selective compression. Restart-safe — skips runs whose prediction.txt is
# already complete (has 'mean_exec_chunk_len' line).
#
# Usage: bash _smoke_master.sh
#
# Per ckpt × server-N: starts ONE server, runs all clients that need it.
# 4 server configs total (mh_m8 N=1, mh_m8 N=10, moe N=1, moe N=10).
set -u

BASE=$HOME/multigpu_workspace/Isaac-GR00T
CONDA=$HOME/miniconda3
TASKS=("CoffeeSetupMug" "TurnOffStove")
N_EP=10
PORT=8400

NVIDIA_PKG_DIR="$CONDA/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

# ---------- helpers ----------
ckpt_path() {
    case "$1" in
        mh_m8)         echo "$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_mh_m8_discfix/checkpoint-60000" ;;
        per_expert_moe) echo "$BASE/ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000" ;;
    esac
}

run_done() {
    # $1 = output dir for one task. Returns 0 if prediction.txt already finished.
    [ -f "$1/prediction.txt" ] && grep -q "mean_exec_chunk_len" "$1/prediction.txt"
}

start_server() {
    local CKPT_TAG=$1 N=$2
    local SERVER_DIR="$BASE/output/robocasa/_smoke_selective/_server_${CKPT_TAG}_n${N}"
    mkdir -p "$SERVER_DIR"
    local CKPT
    CKPT=$(ckpt_path "$CKPT_TAG")
    echo "[$(date '+%T')] === START SERVER: $CKPT_TAG N=$N ==="
    PYTHONUNBUFFERED=1 $CONDA/envs/gr00t/bin/python -u $BASE/scripts/inference_service.py --server \
        --port $PORT \
        --model_path "$CKPT" \
        --data_config single_panda_gripper \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        --head selective \
        --n_samples "$N" \
        --discrete-action-dims 6 11 \
        > "$SERVER_DIR/server.log" 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do
        grep -q "Server is ready" "$SERVER_DIR/server.log" 2>/dev/null && break
        sleep 5
    done
    if ! grep -q "Server is ready" "$SERVER_DIR/server.log" 2>/dev/null; then
        echo "[ERROR] Server not ready"; tail -30 "$SERVER_DIR/server.log"
        kill $SPID 2>/dev/null; return 1
    fi
    echo "[$(date '+%T')]   server ready (PID=$SPID, port=$PORT)"
    return 0
}

stop_server() {
    [ -n "${SPID:-}" ] && { kill $SPID 2>/dev/null; sleep 3; }
    SPID=""
}

run_client() {
    local CKPT_TAG=$1 SCORE=$2 LABEL=$3 EXTRA=$4
    local CLIENT_SCORE=$SCORE
    [ "$SCORE" = "kpref" ] && CLIENT_SCORE=self_agree
    local OUT_BASE="$BASE/output/robocasa/_smoke_selective/${CKPT_TAG}_${SCORE}_n${SERVER_N}"
    # Allow override of N_EP and TASKS via env vars (for validation runs)
    local USE_N_EP=${VAL_NEP:-$N_EP}
    local USE_TASKS_LIST
    if [ -n "${VAL_TASKS:-}" ]; then
        USE_TASKS_LIST=($VAL_TASKS)
    else
        USE_TASKS_LIST=("${TASKS[@]}")
    fi
    for TASK in "${USE_TASKS_LIST[@]}"; do
        local ODIR="$OUT_BASE/${LABEL}/$TASK"
        if run_done "$ODIR"; then
            echo "[$(date '+%T')]   [skip] $CKPT_TAG/$SCORE/$LABEL/$TASK (already complete)"
            continue
        fi
        mkdir -p "$ODIR"
        echo "[$(date '+%T')]   $CKPT_TAG/$SCORE/$LABEL/$TASK"
        PYTHONUNBUFFERED=1 $CONDA/envs/robocasa_gr00t/bin/python -u \
            $BASE/scripts/robocasa_service_selective.py \
            --port $PORT --host localhost \
            --env_name "$TASK" \
            --video_dir "$ODIR" \
            --seed 42 \
            --n_episodes $USE_N_EP \
            --max_episode_steps 1500 \
            --generative_textures \
            --score-mode "$CLIENT_SCORE" \
            $EXTRA \
            > "$ODIR/eval.log" 2>&1
        local succ ep meanlen
        succ=$(grep -E "^episode .* True" "$ODIR/prediction.txt" 2>/dev/null | wc -l)
        ep=$(grep -c "^episode" "$ODIR/prediction.txt" 2>/dev/null)
        meanlen=$(grep "mean_exec_chunk_len" "$ODIR/prediction.txt" 2>/dev/null | tail -1 | tr -d '\n')
        echo "[$(date '+%T')]     -> succ=$succ/$ep   $meanlen"
    done
}

# ---------- run plan ----------
# Per ckpt × server N, declare all (label, args) pairs.
# Mode α calibration: --dump-scores-only
# Mode α τ sweep: --tau X
# Mode β K sweep: --k-prefix X (only run with N=1 server since β doesn't use scores)

# τ candidates from mh_m8/self_agree calibration (8400 samples): p25/50/75 ≈
#   0.099 / 0.329 / 0.657. Re-calibrated per (ckpt, score) below.

# Plan format below: for each (ckpt, server_N): list of (score_mode, label, extra-args)
# Calibration first (so τ candidates can be derived if missing). For now hard-coded
# τ values from current understanding; rerun calibration → adjust if needed.

run_smoke_for_server() {
    local CKPT_TAG=$1 N=$2
    SERVER_N=$N
    start_server "$CKPT_TAG" "$N" || return 1
    trap "stop_server; exit" EXIT INT TERM

    # Phase 1: Calibration (no compression, just dump scores) — verify per-step
    # entropy distribution + per-position pattern.
    # Phase 2: AAC mode validation (1 task × 3 ep) — verify aac_cliff and
    # aac_chunk_binary code paths execute end-to-end without errors.
    local VAL_TASK="CoffeeSetupMug"
    if [ "$N" = "1" ]; then
        run_client "$CKPT_TAG" self_agree calibrate "--dump-scores-only"
        # Validation smokes (small N to catch errors)
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" self_agree val_aac_cliff \
            "--decision-rule aac_cliff --aac-xi 1"
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" self_agree val_aac_binary \
            "--decision-rule aac_chunk_binary --aac-xi 4"
    else
        run_client "$CKPT_TAG" entropy calibrate "--dump-scores-only"
        run_client "$CKPT_TAG" hybrid calibrate "--dump-scores-only"
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" entropy val_aac_cliff \
            "--decision-rule aac_cliff --aac-xi 1"
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" entropy val_aac_binary \
            "--decision-rule aac_chunk_binary --aac-xi 4"
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" hybrid val_aac_cliff \
            "--decision-rule aac_cliff --aac-xi 1"
        VAL_NEP=3 VAL_TASKS="$VAL_TASK" run_client "$CKPT_TAG" hybrid val_aac_binary \
            "--decision-rule aac_chunk_binary --aac-xi 4"
    fi

    stop_server
}

# Iterate: 4 server configs total. Each is restartable (skips already-done tasks).
for COMBO in "mh_m8 1" "mh_m8 10" "per_expert_moe 1" "per_expert_moe 10"; do
    set -- $COMBO
    CKPT_TAG=$1; N=$2
    echo ""
    echo "[$(date '+%T')] ##### COMBO: $CKPT_TAG N=$N #####"
    run_smoke_for_server "$CKPT_TAG" "$N" || {
        echo "[$(date '+%T')] [error] COMBO $CKPT_TAG N=$N failed; continuing"
    }
done

echo "[$(date '+%T')] ===== ALL SMOKE COMBOS DONE ====="
