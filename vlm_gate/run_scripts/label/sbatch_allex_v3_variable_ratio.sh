#!/bin/bash
#SBATCH --job-name=allex_v3_variable_ratio_two_stage_labeling_all_80_episodes_same_prompt_as_v1
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# The v3 allex recording, labelled with the SAME two-stage prompt as v1 so the
# two are directly comparable. 80 episodes / 168,448 frames, about 10.5k chunks
# at stride 16 — v1 is 237,667 frames, so v3's episodes are shorter.
#
# ALLEX_DS / ALLEX_OUT point the labeller at this recording and keep its output
# off v1's; both default to v1 when unset.
#
# Two Cosmos calls each. Sharded by episode index
# mod 8. The client is resumable (it skips (ep,f) already in its own shard file),
# so a background-partition preemption + requeue costs only the current chunk.
WS=$HOME/quantization_agent_workspace
BASE=$WS/vlm_gate; cd $BASE
export ALLEX_DS=/rlwrld2/home/david/action_quantization/v3/subtask_labeled_data_update_eef_256x256_hojin
export ALLEX_OUT=$BASE/output/allex_v3
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
S=$SLURM_ARRAY_TASK_ID; PORT=$((8760+S))
LOG=$ALLEX_OUT/judge_s$S.log
mkdir -p $ALLEX_OUT
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
