#!/bin/bash
# phase5 라벨로 학생 A' 학습 — 스모크 (1에폭). 산출물이 나오면 본학습 sbatch.
set -u
WS="$HOME/quantization_agent_workspace"
cd "$WS/vlm_gate"
OUT="${OUT_DIR:-$WS/assets/modules_A/robocasa_module_A_phase5_smoke}"
mkdir -p "$OUT"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase5_1call_full.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" \
  --epochs "${EPOCHS:-1}" --bs 256 --lr 3e-4 --num-workers 8
