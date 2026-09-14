#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=phase_classification_accuracy_against_action_derived_labels_free_ground_truth
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=out/%j-ratio5.out
#SBATCH --error=out/%j-ratio5.err
set -u
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
PORT=$((25000 + SLURM_JOB_ID % 900))
LOG="analysis/ratio_prompt/judge_${SLURM_JOB_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!; trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }

rc=0
# v1a 에서 한 가지만 바꾼 두 갈래
run_arm () {  # 이름  환경변수들
    local tag=$1; shift
    echo "=== $tag ==="
    env "$@" CHECKS=ratio_checks_v1 TAG="$tag" \
        "$RUN_PY" -u scripts/ratio_frame_label.py "$PORT" dev || rc=1
    local f="analysis/ratio_prompt/${tag}.jsonl"
    local n=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "$tag: $n 행 (gp $(grep -c '\"gp\"' "$f" 2>/dev/null || echo 0))"
    [ "$n" -lt 100 ] && { echo "실패: $tag $n 행"; rc=1; }
}
CHECKS=ratio_checks_v5 TAG=ratio_v5 "$RUN_PY" -u scripts/ratio_scene_label.py "$PORT" || rc=1
N=$(wc -l < analysis/ratio_prompt/ratio_v5.jsonl 2>/dev/null || echo 0); echo "ratio_v5: $N 행"; [ "$N" -lt 600 ] && rc=1
exit $rc
