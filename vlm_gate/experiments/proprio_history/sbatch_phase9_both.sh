#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=proprio_history_vs_baseline_on_phase9_cached_labels_full
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=out/%A_%a-p9train.out
#SBATCH --error=out/%A_%a-p9train.err
#SBATCH --comment="Train image+text baseline and the proprio/executed-action arm on the phase9 labels, same split and seed."
set -u
# 두 팔은 같은 split·seed·에폭·선택기준으로 돈다. 다른 것은 입력뿐이다.
# 0 = baseline (이미지 + 지시문), 1 = proprio (+ 현재 proprio + 직전 실행 액션 16)
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/proprio_history_phase9
E="$HOME/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
mkdir -p out
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ARCH=baseline;;
  *) ARCH=proprio;;
esac
D="$E/artifacts/phase9_${ARCH}"
rm -rf "$D"
bash "$E/run_phase9.sh" "$D" "$ARCH" --epochs 12 --bs 256 --lr 3e-4 --num-workers 8
