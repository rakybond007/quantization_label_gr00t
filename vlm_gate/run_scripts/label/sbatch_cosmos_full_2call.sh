#!/bin/bash
#SBATCH --job-name=cosmos_full_two_call_labeling_stride8_visual_action_split_robocasa_262k
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8400+S))
$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py --serve --port $PORT \
  > output/_gate_distill/cosmos_fullsb_judge$S.log 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" output/_gate_distill/cosmos_fullsb_judge$S.log && break; done
grep -q "JUDGE READY" output/_gate_distill/cosmos_fullsb_judge$S.log || { echo "judge 기동 실패"; exit 1; }
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/cosmos_full_2call.py $PORT $S 16
kill $JP 2>/dev/null
