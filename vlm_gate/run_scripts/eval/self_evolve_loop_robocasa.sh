#!/bin/bash
# RoboCasa self-evolve loop from the NAIVE guidance, judge = gemma|cosmos.
# Each cycle: sbatch the unified 24-task gated eval (eval_robocasa_gated.sh) with
# the current guidance, wait for the array, then evolve (composite baseline-
# anchored gating vs raw + always-K2). Per-(benchmark,model) isolation of
# guidance/versions/log/gating-state so runs never collide.
#
# Run on the LOGIN node. Usage: JUDGE_BACKEND=gemma|cosmos self_evolve_loop_robocasa.sh <cycles> [start]
#   env: N_EPISODES (default 50), FRESH=1 to reseed naive + reset gating.
#        GATE_TTL_MAX>0 enables the TTL skip policy and isolates the run under
#        TAG robocasa_<backend>_ttl.
set -u
N="${1:-5}"; START="${2:-1}"
BACKEND="${JUDGE_BACKEND:-gemma}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PY="$HOME/miniconda3/bin/python"
SBATCH="$BASE_DIR/run_scripts/eval/eval_robocasa_gated.sh"
NAIVE_SEED="$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt"
export N_EPISODES="${N_EPISODES:-50}"
export GATE_TTL_MAX="${GATE_TTL_MAX:-0}"
cd "$BASE_DIR"; mkdir -p analysis/_evolver out

# per-(benchmark, model) isolated paths
TAG="robocasa_${BACKEND}"
[ "$GATE_TTL_MAX" -gt 0 ] 2>/dev/null && TAG="${TAG}_ttl"
GUIDE="$BASE_DIR/analysis/_evolver/${TAG}_guidance.txt"
VER_DIR="$BASE_DIR/analysis/_evolver/guidance_versions_${TAG}"
LOG="$BASE_DIR/analysis/_evolver/evolution_log_${TAG}.jsonl"
BEST="$BASE_DIR/analysis/_evolver/best_state_${TAG}.json"
mkdir -p "$VER_DIR"

if [ "$START" -le 1 ] || [ "${FRESH:-0}" = 1 ]; then
    cp "$NAIVE_SEED" "$GUIDE"; rm -f "$BEST"
    echo "[loop-$TAG] seeded NAIVE guidance -> $GUIDE ; reset gating state"
fi
[ -f "$GUIDE" ] || { echo "[loop-$TAG] ERROR: $GUIDE missing"; exit 1; }

prev=""
for c in $(seq "$START" $((START + N - 1))); do
    OUT="$BASE_DIR/output/robocasa/${TAG}_cycle${c}"
    echo "[loop-$TAG] ===== cycle $c : eval (24 tasks x ${N_EPISODES}ep, judge=$BACKEND, ttl=$GATE_TTL_MAX) ====="
    cp "$GUIDE" "analysis/_evolver/${TAG}_guidance_cycle${c}_input.txt" 2>/dev/null || true
    jid=$(JUDGE_BACKEND="$BACKEND" sbatch --parsable \
        --export=ALL,JUDGE_BACKEND="$BACKEND",OUTPUT_BASE="$OUT",GUIDANCE_FILE="$GUIDE",N_EPISODES="$N_EPISODES",GATE_TTL_MAX="$GATE_TTL_MAX"${GATE_TTL_LO:+,GATE_TTL_LO="$GATE_TTL_LO"}${GATE_TTL_HI:+,GATE_TTL_HI="$GATE_TTL_HI"} "$SBATCH")
    echo "[loop-$TAG] submitted array job $jid -> $OUT"
    until ! squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 120; done
    echo "[loop-$TAG] cycle $c eval done."

    tot=0; for d in "$OUT"/*/; do n=$(grep -c "^episode " "$d/prediction.txt" 2>/dev/null); tot=$((tot + ${n:-0})); done
    echo "[loop-$TAG] cycle $c episodes=$tot"

    echo "[loop-$TAG] ===== cycle $c : evolve (composite gating vs raw+K2) ====="
    args=(--gate "$OUT" --guidance-file "$GUIDE"
          --root-dir "$BASE_DIR/output/robocasa"
          --raw baseline_full_v2_with_action_steps --k2 baseline_compress_K2
          --ver-dir "$VER_DIR" --log "$LOG" --best-state "$BEST")
    [ -n "$prev" ] && args+=(--prev-gate "$prev")
    "$PY" "$BASE_DIR/scripts/evolve_gate_prompt.py" "${args[@]}" || { echo "[loop-$TAG] evolve failed"; exit 1; }
    prev="$OUT"
done
echo "[loop-$TAG] done $N cycles. log: $LOG ; guidance: $GUIDE"
