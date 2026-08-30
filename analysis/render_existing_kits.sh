#!/bin/bash
# Render render_kit for all existing episode data (excluding incoming PnP variants).
set -u
PY=/sjw_alinlab2/home/hojin2/miniconda3/envs/gr00t/bin/python
GT=/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
OUT=analysis/router_segment_kits

cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T
mkdir -p $OUT

# Map (src_dir, task_short, eps...)
render_one() {
    SRC=$1; LABEL=$2; EP=$3; CAPTIONS=$4
    $PY scripts/router_segment_analysis.py render_kit \
        --inputs "$SRC:$EP:${LABEL}_ep${EP}" \
        --captions "$CAPTIONS" \
        --gt-root $GT --n-dips 1 \
        --out-root $OUT --hide-loss-gap 2>&1 | tail -1
}

# PnPCounterToSink (5 eps) — pick/place pattern
PCAP="approach|reach for object|lift / carry|over sink|release / return"
for EP in 1200 1201 1202 1203 1204; do
    render_one analysis/router_segment_other/PnPCounterToSink PnP_CounterToSink $EP "$PCAP"
done

# OpenDrawer (5 eps) — pull-then-release
ODCAP="top view|approach drawer|grasp handle|pulling|after pull"
for EP in 6000 6001 6002 6003 6004; do
    render_one analysis/router_segment_other/OpenDrawer OpenDrawer $EP "$ODCAP"
done

# CloseDrawer (5 eps) — push-then-contact
CDCAP="approach|reach drawer|push start|closing|after close"
for EP in 3300 3301 3302 3303 3304; do
    render_one analysis/router_segment_other/CloseDrawer CloseDrawer $EP "$CDCAP"
done

# TurnOnStove (5 eps) — approach + knob turn peak
TSCAP="kitchen view|stove view|knob approach|knob turn|after turn"
for EP in 0 1 2 3 4; do
    render_one analysis/router_segment_other/TurnOnStove TurnOnStove $EP "$TSCAP"
done

# CloseDoubleDoor top 5 by dynamic range
DDCAP="start|early phase|mid|late|end"
for EP in 3008 3003 3001 3007 3005; do
    render_one analysis/router_segment_closedoubledoor CloseDoubleDoor $EP "$DDCAP"
done

echo "EXISTING_KITS_DONE"
