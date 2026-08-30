#!/bin/bash
# Drive 2 sequential smokes on 1 GPU (tmux 0:1):
#   1) robocasa MoE4 v1 + B only + no metaq (HF)
#   2) libero    MoE4 v1 + B only + no metaq (HF)
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
SUMMARY="$BASE_DIR/output/robocasa/_smoke_moe4_v1_no_metaq_hf_summary.txt"
mkdir -p "$BASE_DIR/output/robocasa" "$BASE_DIR/output/libero"
: > "$SUMMARY"

echo "[$(date '+%T')] === SMOKE 1/2: robocasa MoE4 v1 B-only no metaq ===" | tee -a "$SUMMARY"
bash "$BASE_DIR/run_scripts/eval/_smoke_eval_hf_generic.sh" \
    "prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k" \
    "robocasa_moe4_v1_no_metaq" \
    "scripts/inference_service_fair_moe.py" \
    9700 0 \
    2>&1 | tee -a "$SUMMARY"

echo "[$(date '+%T')] === SMOKE 2/2: libero MoE4 v1 B-only no metaq ===" | tee -a "$SUMMARY"
bash "$BASE_DIR/run_scripts/eval/_smoke_libero_fair_moe_hf.sh" \
    "prehj/GR00T-N1.5-libero-moe4-v1-K4-b-only-no-metaq-60k" \
    "libero_moe4_v1_no_metaq" \
    8801 \
    2>&1 | tee -a "$SUMMARY"

echo "[$(date '+%T')] === ALL DONE ===" | tee -a "$SUMMARY"
echo "=== PASS/FAIL summary ===" | tee -a "$SUMMARY"
grep -E "PASS|FAIL" "$SUMMARY" | tee -a "$SUMMARY"
