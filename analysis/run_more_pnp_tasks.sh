#!/bin/bash
set -u
PY=/sjw_alinlab2/home/hojin2/miniconda3/envs/gr00t/bin/python
GT=/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
CKPT=prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k
cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T
declare -A EPS PROMPT
EPS[PnPMicrowaveToCounter]="5100,5101,5102,5103,5104"
EPS[PnPStoveToCounter]="900,901,902,903,904"
EPS[PnPCabToCounter]="1800,1801,1802,1803,1804"
EPS[PnPSinkToCounter]="4800,4801,4802,4803,4804"
PROMPT[PnPMicrowaveToCounter]="pick the object from the microwave and place it on plate located on the counter"
PROMPT[PnPStoveToCounter]="pick the object from the pan and place it on the plate"
PROMPT[PnPCabToCounter]="pick the object from the cabinet and place it on the counter"
PROMPT[PnPSinkToCounter]="pick the object from the sink and place it on the plate located on the counter"
for TASK in PnPMicrowaveToCounter PnPStoveToCounter PnPCabToCounter PnPSinkToCounter; do
    OUT=analysis/router_segment_other/${TASK}
    mkdir -p $OUT
    echo "=== $TASK -> $OUT ==="
    NO_ALBUMENTATIONS_UPDATE=1 $PY scripts/router_segment_analysis.py collect \
        --gt-root $GT --ckpt $CKPT --task $TASK --task-prompt "${PROMPT[$TASK]}" \
        --episode-indices ${EPS[$TASK]} --n-segments 5 --subsample-stride 8 \
        --output-dir $OUT 2>&1 | tee -a $OUT/run.log
done
echo "MORE_PNP_DONE"
