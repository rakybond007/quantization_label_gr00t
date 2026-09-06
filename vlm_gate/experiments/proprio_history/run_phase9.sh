#!/usr/bin/env bash
# GPU 할당 안에서 돌린다. phase9 라벨로 두 팔을 같은 조건으로 학습한다.
#   run_phase9.sh <출력 디렉터리> <arch: proprio|baseline> [추가 인자...]
# 표본 제한 인자를 주면 스모크, 안 주면 캐시가 덮는 258,809행 전량이다.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS=/sjw_alinlab/home/hojin2/quantization_agent_workspace
PY=/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python
RUN_DIR="${1:?새 출력 디렉터리를 넘겨라}"
ARCH="${2:?proprio 또는 baseline}"
shift 2
"$PY" -u "$HERE/train.py" \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/phase9_cached.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$RUN_DIR" --arch "$ARCH" "$@"
