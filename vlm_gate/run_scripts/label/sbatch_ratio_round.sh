#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=ratio_prompt_round_two_versions_on_dev_split_robocasa_rollout_initial_frames
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=out/%j-ratio.out
#SBATCH --error=out/%j-ratio.err
set -u
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1

PORT=$((24000 + SLURM_JOB_ID % 900))
LOG="analysis/ratio_prompt/judge_${SLURM_JOB_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!; trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }

SPLIT="${SPLIT:-dev}"
rc=0
for V in ${VERSIONS:-ratio_checks_v1 ratio_checks_v1b}; do
    echo "=== $V ($SPLIT) ==="
    CHECKS=$V TAG="${V}_${SPLIT}" "$RUN_PY" -u scripts/ratio_frame_label.py "$PORT" "$SPLIT" || rc=1
    F="analysis/ratio_prompt/${V}_${SPLIT}.jsonl"
    N=$(wc -l < "$F" 2>/dev/null || echo 0)
    G=$(grep -c '"gp"' "$F" 2>/dev/null || echo 0)
    echo "$V: $N 행 (gp $G)"
    [ "$N" -lt 100 ] && { echo "실패: $V 이 $N 행뿐"; tail -5 "$LOG"; rc=1; }
done
exit $rc
