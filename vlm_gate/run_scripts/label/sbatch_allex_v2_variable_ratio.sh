#!/bin/bash
#SBATCH --job-name=allex_v2_variable_ratio_two_stage_labeling_all_80_episodes
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# 80 episodes / 14,850 chunks, two Cosmos calls each. Sharded by episode index
# mod 8. The client is resumable (it skips (ep,f) already in its own shard file),
# so a background-partition preemption + requeue costs only the current chunk.
WS=$HOME/quantization_agent_workspace
BASE=$WS/vlm_gate; cd $BASE
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
S=$SLURM_ARRAY_TASK_ID; PORT=$((8730+S))
LOG=$BASE/output/allex_v2/judge_s$S.log
mkdir -p $BASE/output/allex_v2
$WS/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do
  sleep 10
  grep -q "JUDGE READY" $LOG && break
  kill -0 $JP 2>/dev/null || break
done
grep -q "JUDGE READY" $LOG || { echo "JUDGE FAILED TO START"; tail -30 $LOG; kill $JP 2>/dev/null; exit 1; }
echo "judge up on $PORT"
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/allex_v2_label.py $PORT $S 8
rc=$?
kill $JP 2>/dev/null
exit $rc
