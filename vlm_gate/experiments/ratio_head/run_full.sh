#!/bin/bash
# 배속 머리 본학습.
#
#   sbatch vlm_gate/experiments/ratio_head/sbatch_ratio_head.sh
#
# **직접 sbatch 하지 않는다.** 파티션과 GPU 수는 그 파일에 적혀 있다 --
# 학습은 sjw_alinlab 이고 background 가 아니다.
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
# 지시문 열이 있는 판을 쓴다. 없는 판이면 학습이 태스크 이름으로 떨어지고,
# 그러면 임베딩(지시문 334종)에서 24개가 전부 안 찾아진다.
LAB=${LAB:-$W/vlm_gate/output/_gate_distill/robocasa_contact_ratio_instr.parquet}
BS=${BS:-256}
STEPS=${STEPS:-30000}
# **홀드아웃을 준다.** 기본이 0 이라 첫 본학습(170707)이 전량으로 돌았고,
# 그래서 체크포인트는 나왔는데 일반화했는지 말할 근거가 없다. 학습 손실이
# 0.1486 -> 0.1274 로 내려간 것은 맞췄다는 증거가 아니다 -- 367k 파라미터가
# 258k 행을 외운 것일 수도 있다. 에피소드 단위로 자른다(train.py:113).
VAL=${VAL:-0.1}
# 캐시. stride2 판(약 102만 행)이 나왔으면 그쪽을 쓰고 배치를 512 로 올린다.
CACHE=${CACHE:-$W/assets/frame_cache_robocasa}

$HOME/miniconda3/envs/quant_gate/bin/python "$HERE/train.py" \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LAB" \
  --cache-dir "$CACHE" \
  --task-emb "$W/assets/robocasa_task_embeddings.npz" \
  --out-dir "$RUN" \
  --max-steps "$STEPS" --bs "$BS" --val-frac "$VAL" --num-workers 8 "$@"
