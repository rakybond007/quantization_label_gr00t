#!/bin/bash
#SBATCH --job-name=allex_frontier_demo_cosmos_two_call_labeling_feasibility_six_episodes_bimanual_humanoid
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
PORT=8600
$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py --serve --port $PORT \
  > output/_gate_distill/allex_cosmos_judge.log 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" output/_gate_distill/allex_cosmos_judge.log && break; done
grep -q "JUDGE READY" output/_gate_distill/allex_cosmos_judge.log || { tail -20 output/_gate_distill/allex_cosmos_judge.log; exit 1; }
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/allex_2call_label.py $PORT
kill $JP 2>/dev/null
