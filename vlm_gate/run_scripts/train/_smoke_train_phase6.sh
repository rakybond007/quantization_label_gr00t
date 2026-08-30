#!/bin/bash
# REJECTED — do not run. The phase6 labels this trains on rank tasks
# uncorrelated with what compression actually costs them: Spearman against
# measured per-task K=2 damage is +0.019, against +0.420 for phase5.
# Check any new generation first:  qgate labelcheck <parquet> --reference <phase5>
# phase6(5문항) softA 라벨로 학생 학습 — 스모크 1에폭. 산출물이 나오면 본학습 sbatch.
# 계산층·집계식·학습설정을 phase5 softA 와 동일하게 두어 프롬프트만 달라지게 한다.
set -u
WS="$HOME/quantization_agent_workspace"
cd "$WS/vlm_gate"
OUT="${OUT_DIR:-$WS/assets/modules_A/robocasa_module_A_phase6_softA_smoke}"
mkdir -p "$OUT"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase6_softA.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" \
  --epochs "${EPOCHS:-1}" --bs 256 --lr 3e-4 --num-workers 8
