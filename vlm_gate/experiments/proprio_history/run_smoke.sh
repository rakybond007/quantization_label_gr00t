#!/usr/bin/env bash
# Run INSIDE the GPU allocation. No CPU/memory/time/partition resource flags.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS=/sjw_alinlab/home/hojin2/quantization_agent_workspace
PY=/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python
RUN_DIR="${1:?Pass a new, non-existing output directory}"
"$PY" -u "$HERE/train.py" \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase6_softA.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$RUN_DIR" --max-tasks 8 --episodes-per-task 6 --frames-per-episode 32 \
  --epochs 6 --bs 64 --num-workers 0 --preload
"$PY" -u "$HERE/verify.py" --run-dir "$RUN_DIR"
