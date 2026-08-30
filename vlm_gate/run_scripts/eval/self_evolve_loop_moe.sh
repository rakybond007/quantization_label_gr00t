#!/bin/bash
# Self-evolving guidance loop for the MoE router-bias gate: chain
#   eval (MoE + signed gate) -> evolve -> eval  for N cycles, no human.
# Runs on the LOGIN node (submits sbatch + polls). Long-running (hours/cycle).
# Usage: self_evolve_loop_moe.sh <num_cycles> [start_cycle]
set -u
N="${1:-3}"; START="${2:-1}"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"; PY="$HOME/miniconda3/bin/python"
SBATCH="$BASE_DIR/run_scripts/eval/eval_robocasa_moe_vlm_router.sh"
ROUTER="$BASE_DIR/output/robocasa/moe_router_confp07_ctrl"   # gate-OFF baseline
AUTO_GUIDE="${AUTO_GUIDE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance_moe_auto.txt}"
cd "$BASE_DIR"; mkdir -p analysis/_evolver
[ -f "$AUTO_GUIDE" ] || { echo "[loop-moe] ERROR: $AUTO_GUIDE missing"; exit 1; }

prev=""
for c in $(seq "$START" $((START + N - 1))); do
    OUT="$BASE_DIR/output/robocasa/moe_gate_auto_cycle${c}"
    echo "[loop-moe] ===== cycle $c : eval (MoE + signed gate, s=0.5) ====="
    cp "$AUTO_GUIDE" "analysis/_evolver/guidance_moe_cycle${c}_input.txt" 2>/dev/null || true
    jid=$(sbatch --parsable \
        --export=ALL,MOE_CONF=0.7,BIAS=0.5,BIAS_MODE=signed,GUIDANCE_FILE="$AUTO_GUIDE",OUTPUT_BASE="$OUT" \
        "$SBATCH")
    echo "[loop-moe] submitted job $jid -> $OUT"
    until ! squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 120; done
    echo "[loop-moe] cycle $c eval done."

    tot=0; for d in "$OUT"/*/; do tot=$((tot + $(grep -c "^episode " "$d/prediction.txt" 2>/dev/null || echo 0))); done
    echo "[loop-moe] cycle $c episodes=$tot"

    echo "[loop-moe] ===== cycle $c : evolve guidance ====="
    args=(--router "$ROUTER" --gate "$OUT" --guidance-file "$AUTO_GUIDE")
    [ -n "$prev" ] && args+=(--prev-gate "$prev")
    "$PY" "$BASE_DIR/scripts/evolve_gate_prompt_moe.py" "${args[@]}" || { echo "[loop-moe] evolve failed"; exit 1; }
    prev="$OUT"
done
echo "[loop-moe] done $N cycles. Audit: analysis/_evolver/evolution_log_moe.jsonl"
