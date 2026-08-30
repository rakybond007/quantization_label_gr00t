#!/bin/bash
#SBATCH --job-name=libero_v1_full_chunk_labeling_cosmos_judge_five_questions_16shard
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# 샤드마다 제 판정 서버를 8900+shard 포트에 띄우고, 그 포트로 라벨러를 돌린다.
# 라벨러는 재개 가능하다 — 선점 후 재큐되면 이미 쓴 (ep,f) 는 건너뛴다.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8900+S))
LOG=output/_gate_distill/libero_v1_judge$S.log
GATE_SYSTEM=aligned $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u \
  scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" $LOG && break; done
grep -q "JUDGE READY" $LOG || { tail -20 $LOG; kill $JP 2>/dev/null; exit 1; }
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/libero_label_chunks.py $PORT $S 16
kill $JP 2>/dev/null
