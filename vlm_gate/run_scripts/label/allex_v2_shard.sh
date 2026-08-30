#!/bin/bash
# One labelling shard: bring up the Cosmos judge, then label episodes ep%NSH==S.
# usage (inside an srun --gpus=1 allocation):  allex_v2_shard.sh <PORT> <SHARD> <NSH>
set -u
WS=$HOME/quantization_agent_workspace
BASE=$WS/vlm_gate; cd $BASE
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
PORT=$1; S=$2; NSH=$3
LOG=$BASE/output/allex_v2/judge_${NSH}_$S.log
mkdir -p $BASE/output/allex_v2
$WS/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do
  sleep 10
  grep -q "JUDGE READY" $LOG && break
  kill -0 $JP 2>/dev/null || break
done
grep -q "JUDGE READY" $LOG || { echo "JUDGE FAILED TO START"; tail -30 $LOG; kill $JP 2>/dev/null; exit 1; }
echo "judge up on $PORT (shard $S/$NSH)"
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/allex_v2_label.py $PORT $S $NSH
rc=$?
kill $JP 2>/dev/null
exit $rc
