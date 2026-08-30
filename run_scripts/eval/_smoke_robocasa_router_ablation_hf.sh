#!/bin/bash
# Smoke 2 router-input ablation ckpts on robocasa (HF). as_is, no beta.
set -u
BASE="$HOME/multigpu_workspace/Isaac-GR00T"
SUM="$BASE/output/robocasa/_smoke_router_ablation_summary.txt"
mkdir -p "$BASE/output/robocasa"; : > "$SUM"

echo "[$(date '+%T')] === SMOKE 1/2: router_instr_end ===" | tee -a "$SUM"
bash "$BASE/run_scripts/eval/_smoke_eval_hf_generic.sh" \
    "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-router-instr-end-60k" \
    "router_instr_end" "scripts/inference_service_fair_moe.py" 9700 0 \
    2>&1 | tee -a "$SUM"

echo "[$(date '+%T')] === SMOKE 2/2: router_vl_self_attn ===" | tee -a "$SUM"
bash "$BASE/run_scripts/eval/_smoke_eval_hf_generic.sh" \
    "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-router-vl-self-attn-60k" \
    "router_vl_self_attn" "scripts/inference_service_fair_moe.py" 9700 0 \
    2>&1 | tee -a "$SUM"

echo "[$(date '+%T')] === ROUTER_SMOKE_ALL_DONE ===" | tee -a "$SUM"
grep -E "PASS|FAIL" "$SUM"
