#!/bin/bash
#SBATCH --job-name=cosmos_robocasa_system_prompt_polarity_ablation_neutral_and_none_stratified_1295
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p sjw_alinlab
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
for MODE in aligned; do
  PORT=$((8620 + RANDOM % 50))
  GATE_SYSTEM=$MODE $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py \
      --serve --port $PORT > output/_gate_distill/cosmos_sysabl_$MODE.log 2>&1 &
  JP=$!
  for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" output/_gate_distill/cosmos_sysabl_$MODE.log && break; done
  grep -q "JUDGE READY" output/_gate_distill/cosmos_sysabl_$MODE.log || { tail -20 output/_gate_distill/cosmos_sysabl_$MODE.log; kill $JP; continue; }
  GATE_SYSTEM=$MODE $HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/cosmos_2call_fast.py $PORT 0 1 0 strat
  kill $JP 2>/dev/null; sleep 10
done
