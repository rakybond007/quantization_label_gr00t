#!/bin/bash
# Adopt a pending eval job of a dead self-evolve loop, run its evolve step, then
# continue the remaining cycles via the normal loop script.
#   usage: _resume_loop_driver.sh <bench:libero|robocasa> <BACKEND> <TAG> <JID> <CYCLE> <TOTAL> [PREV_OUT]
set -u
BENCH="$1"; BACKEND="$2"; TAG="$3"; JID="$4"; CYC="$5"; TOTAL="$6"; PREV="${7:-}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PY="$HOME/miniconda3/bin/python"
cd "$BASE_DIR"
GUIDE="$BASE_DIR/analysis/_evolver/${TAG}_guidance.txt"
VER_DIR="$BASE_DIR/analysis/_evolver/guidance_versions_${TAG}"
LOG="$BASE_DIR/analysis/_evolver/evolution_log_${TAG}.jsonl"
BEST="$BASE_DIR/analysis/_evolver/best_state_${TAG}.json"

echo "[resume-$TAG] adopting job $JID (cycle $CYC/$TOTAL)"
until ! squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 120; done
echo "[resume-$TAG] cycle $CYC eval done."

if [ "$BENCH" = libero ]; then
    OUT="$BASE_DIR/output/libero/${TAG}_cycle${CYC}"
    tot=0; for d in "$OUT"/gate/*/; do n=$(grep -c "^episode " "$d/prediction.txt" 2>/dev/null); tot=$((tot + ${n:-0})); done
    args=(--gate "$OUT/gate" --root-dir "$BASE_DIR/output/libero" --raw baseline_raw --k2 baseline_K2)
    [ -n "$PREV" ] && args+=(--prev-gate "$PREV/gate")
    MIN=2000
else
    OUT="$BASE_DIR/output/robocasa/${TAG}_cycle${CYC}"
    tot=0; for d in "$OUT"/*/; do n=$(grep -c "^episode " "$d/prediction.txt" 2>/dev/null); tot=$((tot + ${n:-0})); done
    args=(--gate "$OUT" --root-dir "$BASE_DIR/output/robocasa" --raw baseline_full_v2_with_action_steps --k2 baseline_compress_K2)
    [ -n "$PREV" ] && args+=(--prev-gate "$PREV")
    MIN=1200
fi
echo "[resume-$TAG] cycle $CYC episodes=$tot"
if [ "$tot" -lt "$MIN" ]; then
    echo "[resume-$TAG] ❌ coverage short ($tot < $MIN) — HALT before evolve"; exit 1
fi

echo "[resume-$TAG] evolve cycle $CYC"
"$PY" "$BASE_DIR/scripts/evolve_gate_prompt.py" "${args[@]}" \
    --guidance-file "$GUIDE" --ver-dir "$VER_DIR" --log "$LOG" --best-state "$BEST" \
    || { echo "[resume-$TAG] evolve failed"; exit 1; }

NEXT=$((CYC + 1)); REMAIN=$((TOTAL - CYC))
if [ "$REMAIN" -le 0 ]; then echo "[resume-$TAG] all $TOTAL cycles done."; exit 0; fi
echo "[resume-$TAG] continuing cycles $NEXT..$TOTAL via loop script"
LOOPSH="$BASE_DIR/run_scripts/eval/self_evolve_loop_${BENCH}.sh"
JUDGE_BACKEND="$BACKEND" GATE_TTL_MAX=3 FRESH=0 N_EPISODES=50 exec bash "$LOOPSH" "$REMAIN" "$NEXT"
