#!/bin/bash
# Sequential smoke sweep on libero_10 task 0 × 20 ep, dual server on port 9876.
set -u
OUT=/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/libero/dual_sweep
mkdir -p $OUT
PY=$HOME/miniconda3/envs/libero/bin/python
SCRIPT=$HOME/multigpu_workspace/Isaac-GR00T/gr00t/eval/libero/eval_taskwise_gr00t_dual.py

declare -a CONFIGS=(
    "agree_t0p10|agreement|0.10|5|3"
    "agree_t0p30|agreement|0.30|5|3"
    "agree_t1p00|agreement|1.00|5|3"
    "combined_t0p10|combined|0.10|5|3"
    "combined_t0p20|combined|0.20|5|3"
    "combined_t0p30|combined|0.30|5|3"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name decision thresh r_main r_m8 <<< "$cfg"
    DST=$OUT/$name
    if [ -f $DST/0_results.txt ]; then
        echo "[$(date '+%T')] $name: already done, skipping"
        continue
    fi
    rm -rf $DST && mkdir -p $DST
    echo "[$(date '+%T')] === Variant: $name decision=$decision thresh=$thresh ==="
    $PY $SCRIPT \
        --args.task-suite-name libero_10 \
        --args.task_idx=0 \
        --args.port=9876 \
        --args.video-out-path $DST \
        --args.num_trials_per_task=20 \
        --args.decision=$decision \
        --args.agreement_thresh=$thresh \
        --args.replan_steps=$r_main \
        --args.replan_steps_m8=$r_m8 \
        > $DST/run.log 2>&1
    echo "[$(date '+%T')] $name: $(cat $DST/0_results.txt 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')"
done
echo "[$(date '+%T')] all done"
