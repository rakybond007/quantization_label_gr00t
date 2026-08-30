#!/bin/bash
#SBATCH --job-name=robocasa_phase6_pilot_3k_five_axes_guidance_v5_acceptance_retest
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# 전량 248k 를 다시 라벨링하기 전에 3천 청크로 문항 자격을 시험한다.
# 동작군(누르기·돌리기·열기·닫기·집어옮기기)별 600 개씩 층화 추출.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8900+S))
GATE_SYSTEM=aligned $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u scripts/vlm_gate_cosmos.py \
  --serve --port $PORT > output/_gate_distill/p6v5_judge$S.log 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" output/_gate_distill/p6v5_judge$S.log && break; done
grep -q "JUDGE READY" output/_gate_distill/p6v5_judge$S.log || { tail -20 output/_gate_distill/p6v5_judge$S.log; exit 1; }
GUIDANCE=phase6 MANIFEST=$BASE/output/_gate_distill/pilot3k_manifest.txt \
  $HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/cosmos_1call_v6.py $PORT $S 4
kill $JP 2>/dev/null
