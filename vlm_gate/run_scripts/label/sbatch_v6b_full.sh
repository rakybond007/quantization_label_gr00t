#!/bin/bash
#SBATCH --job-name=robocasa_v6b_single_call_computed_facts_plus_scene_questions_full_stride8_262k
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p sjw_alinlab
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8800+S))
GATE_SYSTEM=aligned $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py \
  --serve --port $PORT > output/_gate_distill/v6b_judge$S.log 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" output/_gate_distill/v6b_judge$S.log && break; done
grep -q "JUDGE READY" output/_gate_distill/v6b_judge$S.log || { tail -20 output/_gate_distill/v6b_judge$S.log; exit 1; }
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/cosmos_1call_v6.py $PORT $S 16
kill $JP 2>/dev/null
