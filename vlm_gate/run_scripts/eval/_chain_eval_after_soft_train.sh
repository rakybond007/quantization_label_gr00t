#!/bin/bash
# 133509 (A/B 학생 학습) 완료를 기다렸다가 각각 폐루프 평가를 제출한다.
set -u
WS="$HOME/quantization_agent_workspace"; cd "$WS/vlm_gate"
JID=${1:-133509}
for i in $(seq 1 240); do
  squeue -j $JID -h -o "%T" 2>/dev/null | grep -q . || break
  sleep 60
done
echo "[chain] 학습 $JID 종료 $(date '+%m-%d %H:%M')"
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
for TAG in A B; do
  CK="$WS/assets/modules_A/robocasa_module_A_phase5_soft$TAG/gate_module_best.pt"
  if [ ! -f "$CK" ]; then echo "[chain] $TAG 체크포인트 없음 — 평가 건너뜀"; continue; fi
  echo "[chain] $TAG 평가 제출: $CK"
  sbatch --export=ALL,MODULE_CKPT=$CK,JUDGE_BACKEND=module,TAU=0.5,N_EPISODES=50,\
OUTPUT_BASE=$WS/vlm_gate/output/robocasa/phase5_soft${TAG}_module_tau0p5,\
MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR \
    --job-name=eval_robocasa_gated_phase5_soft${TAG}_student_module_tau0p5_24tasks_50ep \
    run_scripts/eval/eval_robocasa_gated.sh
done
echo "[chain] 완료"; squeue -u $USER -o "%.9i %.11P %.44j %.2t %.7M %R"
