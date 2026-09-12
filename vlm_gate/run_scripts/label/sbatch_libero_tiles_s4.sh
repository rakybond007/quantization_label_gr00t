#!/bin/bash
#SBATCH --job-name=libero_tile_generation_stride4_front_and_wrist_sixteen_shards_for_question_design
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# job-name 이 50자 미만이면 클러스터가 제출을 거부한다.
#
# **array 는 4 다. 16 이 아니다.** background 는 남는 자리를 쓰는 파티션이라
# 16샤드를 물어도 되지만(robocasa phase9 가 그랬다), sjw_alinlab 은 학습용이라
# 그만큼 물면 남의 학습을 밀어낸다. allex 라벨링도 alinlab 계열로 갈 때는
# array 0-1 로 줄였고, phase9 는 array 대신 GPU 4장에 일꾼 4개로 돌렸다.
# background 로 되돌릴 때만 16 으로 올린다.
# 전 프레임 타일. 정면 + 손목 두 뷰를 가로로 이어 붙여 2배 축소한다.
# 이미 있는 타일은 다시 그리지 않으므로 재큐돼도 이어서 간다.
BASE=$HOME/quantization_agent_workspace; cd $BASE
NSH=$((SLURM_ARRAY_TASK_MAX+1))   # 샤드 수는 array 크기에서 받는다
TILE_STRIDE=${TILE_STRIDE:-1} $HOME/miniconda3/envs/quant_gate/bin/python -u \
  vlm_gate/scripts/gen_libero_tiles_shard.py $SLURM_ARRAY_TASK_ID $NSH
