#!/bin/bash
# 배속 머리 본학습.
#
#   sbatch vlm_gate/experiments/ratio_head/sbatch_ratio_head.sh
#
# 스텝 수 근거 (2026-09-09):
#   이전 학습    258,809행 · bs 64 · 10에폭 = 40,438스텝 · 본 표본 2.59M
#   지금         258,809행 · bs 256 · 30k스텝 = 29.7에폭 · 7.68M   이전의 3배
#
# **라벨이 8배가 됐어도 학습 행은 안 늘었다.** 프레임 캐시가 stride 4 라
# 204만 라벨 중 258,809 만 이미지가 있다. 그래서 배치를 512 로 키우면 59에폭이
# 되어 과하다 -- 데이터가 안 늘었는데 학습량만 늘리는 것이다.
# stride2 캐시(약 102만)가 들어오면 bs 512 로 올려 15에폭을 맞춘다.
set -eu
W=$HOME/quantization_agent_workspace
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=${RUN:-$W/vlm_gate/experiments/ratio_head/artifacts/contact_30k}
LAB=${LAB:-$W/vlm_gate/output/_gate_distill/robocasa_contact_ratio.parquet}
BS=${BS:-256}
STEPS=${STEPS:-30000}

$HOME/miniconda3/envs/quant_gate/bin/python "$HERE/train.py" \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LAB" \
  --cache-dir "$W/assets/frame_cache_robocasa" \
  --task-emb "$W/assets/robocasa_task_embeddings.npz" \
  --out-dir "$RUN" \
  --max-steps "$STEPS" --bs "$BS" --num-workers 8 "$@"
