#!/bin/bash
# allex v2 variable-ratio labelling — 1-episode smoke.
# usage (inside an srun --gpus=1 allocation):  allex_v2_smoke.sh <PORT> <EP>
set -u
WS=$HOME/quantization_agent_workspace
BASE=$WS/vlm_gate
PORT=${1:-8720}
EP=${2:-0}
LOG=$BASE/output/allex_v2/judge_smoke_$PORT.log
mkdir -p $BASE/output/allex_v2
cd $BASE
$WS/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do
  sleep 10
  grep -q "JUDGE READY" $LOG && break
  kill -0 $JP 2>/dev/null || break
done
grep -q "JUDGE READY" $LOG || { echo "JUDGE FAILED TO START"; tail -30 $LOG; kill $JP 2>/dev/null; exit 1; }
echo "judge up on $PORT"
TAG=smoke_ep$EP $HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/allex_v2_label.py $PORT 0 1 $EP
rc=$?
kill $JP 2>/dev/null
exit $rc
