#!/bin/bash
# Self-evolving guidance loop: chain  eval -> evolve -> eval  for N cycles, no human.
# Each cycle: run the full gate eval with the current guidance, then let the Claude
# evolver rewrite the guidance from the results, then repeat.
#
# Run on the LOGIN node (it submits sbatch + polls). Long-running (hours/cycle).
# Usage: self_evolve_loop.sh <num_cycles> [start_cycle]
set -u
N="${1:-3}"
START="${2:-1}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
PY="$HOME/miniconda3/bin/python"
SBATCH="$BASE_DIR/run_scripts/eval/eval_robocasa_vlm_gate_gemma4_tau0p5_mv_guide.sh"
# Dedicated evolving guidance file (kept separate from the manual v9 live file).
AUTO_GUIDE="${AUTO_GUIDE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance_auto.txt}"
cd "$BASE_DIR"
mkdir -p analysis/_evolver
[ -f "$AUTO_GUIDE" ] || { echo "[loop] ERROR: $AUTO_GUIDE missing (seed it with the naive initial prompt first)"; exit 1; }

prev=""
for c in $(seq "$START" $((START + N - 1))); do
    OUT="$BASE_DIR/output/robocasa/vlm_gate_auto_cycle${c}"
    echo "[loop] ===== cycle $c : eval with current guidance ====="
    cp "$AUTO_GUIDE" "analysis/_evolver/guidance_cycle${c}_input.txt" 2>/dev/null || true
    jid=$(sbatch --parsable --export=ALL,OUTPUT_BASE="$OUT",GUIDANCE_FILE="$AUTO_GUIDE" "$SBATCH")
    echo "[loop] submitted job $jid -> $OUT"

    # wait for the whole array job to leave the queue
    until ! squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 120; done
    echo "[loop] cycle $c eval done."

    # verify completeness (1200 ep) before evolving
    tot=0; for d in "$OUT"/*/; do tot=$((tot + $(grep -c "^episode " "$d/prediction.txt" 2>/dev/null || echo 0))); done
    echo "[loop] cycle $c episodes=$tot"

    echo "[loop] ===== cycle $c : evolve guidance ====="
    args=(--gate "$OUT" --guidance-file "$AUTO_GUIDE"); [ -n "$prev" ] && args+=(--prev-gate "$prev")
    "$PY" "$BASE_DIR/scripts/evolve_gate_prompt.py" "${args[@]}" || { echo "[loop] evolve failed"; exit 1; }
    prev="$OUT"
done
echo "[loop] done $N cycles. Audit: analysis/_evolver/evolution_log.jsonl"
