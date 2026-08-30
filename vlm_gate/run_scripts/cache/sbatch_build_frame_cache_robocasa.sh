#!/bin/bash
#SBATCH --job-name=rebuild_gate_frame_cache_robocasa_stride8_262k_frames_three_views_128px_for_gateA
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -eu
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
CK=/rlwrld-unified-checkpoints/$USER/checkpoints
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u scripts/build_gate_frame_cache.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels $CK/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet \
  --out-dir /sjw_alinlab/home/hojin2/quantization_agent_workspace/assets/frame_cache_robocasa \
  --res 128 --shard $SLURM_ARRAY_TASK_ID --num-shards 16
