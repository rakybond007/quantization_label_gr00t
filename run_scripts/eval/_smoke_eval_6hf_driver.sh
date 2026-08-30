#!/bin/bash
# Driver: 6 HF smokes, 2-at-a-time on GPU 0/1 in tmux 0:1.
# Models from the user's resub upload:
#   1 n16_v1   prehj/gr00t-n1.5-robocasa-metaq-n16-v1       metaq v1
#   2 n16_v2   prehj/gr00t-n1.5-robocasa-metaq-n16-v2       metaq v2
#   3 n32_v1   prehj/gr00t-n1.5-robocasa-metaq-n32-v1       metaq v1
#   4 n32_v2   prehj/gr00t-n1.5-robocasa-metaq-n32-v2       metaq v2
#   5 fair_moe_v2_b_only   prehj/gr00t-n1.5-robocasa-fair-moe-v2-b-only  fair_moe v2
#   6 metaq_v2_n8_b_only   prehj/gr00t-n1.5-robocasa-metaq-v2-n8-b-only  metaq v2
set -u
SCRIPT="$HOME/multigpu_workspace/Isaac-GR00T/run_scripts/eval/_smoke_eval_hf_generic.sh"
SUMMARY="$HOME/multigpu_workspace/Isaac-GR00T/output/robocasa/_smoke_6hf_summary.txt"
mkdir -p "$(dirname "$SUMMARY")"
: > "$SUMMARY"

run_pair() {
    local r1="$1" t1="$2" s1="$3" p1="$4"
    local r2="$5" t2="$6" s2="$7" p2="$8"
    echo "[$(date '+%T')] === BATCH: $t1 (gpu0/port$p1) + $t2 (gpu1/port$p2) ==="
    bash "$SCRIPT" "$r1" "$t1" "$s1" "$p1" 0 >> "$SUMMARY" 2>&1 &
    PID1=$!
    bash "$SCRIPT" "$r2" "$t2" "$s2" "$p2" 1 >> "$SUMMARY" 2>&1 &
    PID2=$!
    wait $PID1
    wait $PID2
}

# Batch 1
run_pair \
  prehj/gr00t-n1.5-robocasa-metaq-n16-v1 n16_v1 scripts/inference_service_metaq.py    9710 \
  prehj/gr00t-n1.5-robocasa-metaq-n16-v2 n16_v2 scripts/inference_service_metaq_v2.py 9711

# Batch 2
run_pair \
  prehj/gr00t-n1.5-robocasa-metaq-n32-v1 n32_v1 scripts/inference_service_metaq.py    9712 \
  prehj/gr00t-n1.5-robocasa-metaq-n32-v2 n32_v2 scripts/inference_service_metaq_v2.py 9713

# Batch 3
run_pair \
  prehj/gr00t-n1.5-robocasa-fair-moe-v2-b-only  fair_moe_v2_b_only scripts/inference_service_fair_moe_v2.py 9714 \
  prehj/gr00t-n1.5-robocasa-metaq-v2-n8-b-only  metaq_v2_n8_b_only scripts/inference_service_metaq_v2.py    9715

echo
echo "[$(date '+%T')] === ALL BATCHES DONE ==="
echo "=== PASS/FAIL summary ==="
grep -E "===.*(PASS|FAIL)" "$SUMMARY"
