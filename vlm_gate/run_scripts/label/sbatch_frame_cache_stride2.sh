#!/bin/bash
#SBATCH --job-name=build_robocasa_frame_cache_stride2_for_ratio_head_training
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -eu
# 프레임 캐시를 stride 2 로 넓힌다. 지금은 stride 4 라 204만 라벨 중 258,809 만
# 학습에 쓸 수 있다. stride 2 면 약 102만 · 144 GB 다.
#
# 기존 stride 4 캐시는 건드리지 않는다. 새 디렉터리에 굽고, 학습이 어느 쪽을
# 쓸지는 --cache-dir 로 고른다.
W=$HOME/quantization_agent_workspace
LAB=$W/assets/labels/robocasa/ratio_contact_stride2.parquet
OUT=$W/assets/frame_cache_robocasa_stride2

$HOME/miniconda3/envs/quant_gate/bin/python $W/vlm_gate/scripts/make_stride_labels.py --stride 2
mkdir -p $OUT
$HOME/miniconda3/envs/quant_gate/bin/python $W/vlm_gate/scripts/build_gate_frame_cache_real.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LAB" --out-dir "$OUT"
echo "--- 산출물"; du -sh $OUT; ls $OUT | head -4
