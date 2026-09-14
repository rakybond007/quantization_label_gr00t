#!/bin/bash
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
export ALLEX_OUT="$PWD/output/allex_v5tempo_v3"
export ALLEX_CHECKS="CLAMP,LOOSE,SHOVE,FLIP,FREE"
export ALLEX_FULL=1 ALLEX_STRIDE=4 ALLEX_BATCH=32 ALLEX_NSHARDS=2
export ALLEX_SHARD="${SH:?}"
PORT=$((8400 + ALLEX_SHARD))
LOG=/tmp/judge_gap_$ALLEX_SHARD.log
OMP_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
trap 'kill %1 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q "JUDGE READY" $LOG && break; sleep 10; done
grep -q "JUDGE READY" $LOG || { echo "판정기 안 뜸"; exit 1; }
"$RUN_PY" -u scripts/allex_v3_label.py $PORT
