#!/bin/bash
# Self-evolving guidance loop with the COSMOS judge, from the NAIVE seed prompt.
# Each cycle: sbatch the full 24-task Cosmos-gated eval with the current guidance,
# wait for the array to finish, then let the Claude evolver (v2 accept/reject,
# raw + always-K2 baselines as references) rewrite the guidance. Repeat.
#
# Run on the LOGIN node (submits sbatch + polls + evolves; evolve needs network).
# Usage: self_evolve_loop_cosmos.sh <num_cycles> [start_cycle]
#   env: N_EPISODES (default 50), FRESH=1 to reseed naive + reset gating state.
set -u
N="${1:-3}"
START="${2:-1}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PY="$HOME/miniconda3/bin/python"
SBATCH="$BASE_DIR/run_scripts/eval/eval_robocasa_cosmos_tau0p5.sh"
AUTO_GUIDE="$BASE_DIR/analysis/_evolver/cosmos_auto_guidance.txt"
NAIVE_SEED="$BASE_DIR/analysis/_evolver/_run1_archive/guidance_cycle1_input.txt"
export N_EPISODES="${N_EPISODES:-50}"
cd "$BASE_DIR"
mkdir -p analysis/_evolver out

# Fresh start (cycle 1): seed the evolving guidance with the NAIVE prompt and
# clear the v2 running-best so the loop truly begins from naive + raw/K2 refs.
if [ "$START" -le 1 ] || [ "${FRESH:-0}" = 1 ]; then
    cp "$NAIVE_SEED" "$AUTO_GUIDE"
    rm -f analysis/_evolver/best_state.json
    echo "[loop-cosmos] seeded NAIVE guidance -> $AUTO_GUIDE ; reset gating state"
fi
[ -f "$AUTO_GUIDE" ] || { echo "[loop-cosmos] ERROR: $AUTO_GUIDE missing"; exit 1; }

prev=""
for c in $(seq "$START" $((START + N - 1))); do
    OUT="$BASE_DIR/output/robocasa/cosmos_auto_cycle${c}"
    echo "[loop-cosmos] ===== cycle $c : eval (24 tasks x ${N_EPISODES}ep, cosmos judge) ====="
    cp "$AUTO_GUIDE" "analysis/_evolver/cosmos_guidance_cycle${c}_input.txt" 2>/dev/null || true
    jid=$(sbatch --parsable --export=ALL,OUTPUT_BASE="$OUT",GUIDANCE_FILE="$AUTO_GUIDE",N_EPISODES="$N_EPISODES" "$SBATCH")
    echo "[loop-cosmos] submitted array job $jid -> $OUT"

    # wait for the whole array to leave the queue
    until ! squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 120; done
    echo "[loop-cosmos] cycle $c eval done."

    tot=0; for d in "$OUT"/*/; do tot=$((tot + $(grep -c "^episode " "$d/prediction.txt" 2>/dev/null || echo 0))); done
    echo "[loop-cosmos] cycle $c episodes=$tot"

    echo "[loop-cosmos] ===== cycle $c : evolve (v2 accept/reject vs raw+K2) ====="
    args=(--gate "$OUT" --guidance-file "$AUTO_GUIDE")
    [ -n "$prev" ] && args+=(--prev-gate "$prev")
    "$PY" "$BASE_DIR/scripts/evolve_gate_prompt.py" "${args[@]}" || { echo "[loop-cosmos] evolve failed"; exit 1; }
    prev="$OUT"
done
echo "[loop-cosmos] done $N cycles. audit: analysis/_evolver/evolution_log.jsonl ; guidance: $AUTO_GUIDE"
