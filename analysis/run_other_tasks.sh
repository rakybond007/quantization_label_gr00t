#!/bin/bash
set -u
PY=/sjw_alinlab2/home/hojin2/miniconda3/envs/gr00t/bin/python
GT=/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
CKPT=prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k
cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T

declare -A EPS
EPS[TurnSinkSpout]="3900,3901,3902,3903,3904"
EPS[CoffeePressButton]="2700,2701,2702,2703,2704"
EPS[CloseDrawer]="3300,3301,3302,3303,3304"
EPS[PnPCounterToSink]="1200,1201,1202,1203,1204"
EPS[TurnOnStove]="0,1,2,3,4"
EPS[OpenDrawer]="6000,6001,6002,6003,6004"

declare -A PROMPT
PROMPT[TurnSinkSpout]="turn the sink spout"
PROMPT[CoffeePressButton]="press the coffee machine button"
PROMPT[CloseDrawer]="close the drawer"
PROMPT[PnPCounterToSink]="pick the object and place it in the sink"
PROMPT[TurnOnStove]="turn on the stove"
PROMPT[OpenDrawer]="open the drawer"

for TASK in TurnSinkSpout CoffeePressButton CloseDrawer PnPCounterToSink TurnOnStove OpenDrawer; do
    OUT=analysis/router_segment_other/${TASK}
    mkdir -p $OUT
    echo "=== $TASK -> $OUT ==="
    NO_ALBUMENTATIONS_UPDATE=1 $PY scripts/router_segment_analysis.py collect \
        --gt-root $GT --ckpt $CKPT --task $TASK --task-prompt "${PROMPT[$TASK]}" \
        --episode-indices ${EPS[$TASK]} --n-segments 5 --subsample-stride 8 \
        --output-dir $OUT 2>&1 | tee -a $OUT/run.log
done
echo "ALL_TASKS_DONE"
