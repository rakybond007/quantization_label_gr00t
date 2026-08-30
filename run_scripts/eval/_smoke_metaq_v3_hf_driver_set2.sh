#!/bin/bash
# Smoke 2 more HF metaq_v3 ckpts sequentially on 1 GPU (tmux 0:1).
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
SUMMARY="$BASE_DIR/output/robocasa/_smoke_metaq_v3_hf_set2_summary.txt"
mkdir -p "$BASE_DIR/output/robocasa"; : > "$SUMMARY"

run_one() {
    local REPO=$1; local TAG=$2; local PORT=$3
    bash "$BASE_DIR/run_scripts/eval/_smoke_eval_hf_generic.sh" \
        "$REPO" "$TAG" "scripts/inference_service_metaq_v3.py" "$PORT" 0 \
        2>&1 | tee -a "$SUMMARY"
}

echo "[$(date '+%T')] === SMOKE: metaq_v3a (K=4, B only) ===" | tee -a "$SUMMARY"
run_one "prehj/GR00T-N1.5-robocasa-metaq-v3a-n32-K4-b-only-60k" "metaq_v3a_n32_b_only" 9880
echo "[$(date '+%T')] === SMOKE: metaq_v3b (K=6, B+D) ===" | tee -a "$SUMMARY"
run_one "prehj/GR00T-N1.5-robocasa-metaq-v3b-n32-K6-b-d-lc-0p10-60k" "metaq_v3b_n32_b_d" 9881

echo "[$(date '+%T')] === ALL DONE ===" | tee -a "$SUMMARY"
echo "=== PASS/FAIL summary ===" | tee -a "$SUMMARY"
grep -E "PASS|FAIL" "$SUMMARY" | tee -a "$SUMMARY"
