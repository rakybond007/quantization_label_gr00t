#!/bin/bash
#SBATCH --job-name=libero_gate_distill_tile_generation_two_view_stride4_16shard
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# LIBERO 1693 에피소드 · 2뷰 · stride 4 -> 약 68k 타일.
# 이미 있는 타일은 다시 그리지 않으므로 선점 후 재큐해도 이어서 돈다.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/gen_libero_tiles_shard.py \
  $SLURM_ARRAY_TASK_ID 16
# 전 샤드가 끝난 뒤 평문 매니페스트를 만든다:
#   python scripts/gen_libero_tiles_shard.py merge 16
