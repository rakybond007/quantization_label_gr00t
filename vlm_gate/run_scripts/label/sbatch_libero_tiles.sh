#!/bin/bash
#SBATCH --job-name=libero_tiles_stride1_16shard
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# 전 프레임 타일. 정면 + 손목 두 뷰를 가로로 이어 붙여 2배 축소한다.
# 이미 있는 타일은 다시 그리지 않으므로 재큐돼도 이어서 간다.
BASE=$HOME/quantization_agent_workspace; cd $BASE
TILE_STRIDE=${TILE_STRIDE:-1} $HOME/miniconda3/envs/quant_gate/bin/python -u \
  vlm_gate/scripts/gen_libero_tiles_shard.py $SLURM_ARRAY_TASK_ID 16
