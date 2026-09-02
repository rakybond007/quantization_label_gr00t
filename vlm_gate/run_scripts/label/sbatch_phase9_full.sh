#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_twosided_5checks_text_no_logits_260k_16shard
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-15
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%A_%a-phase9_label.out
#SBATCH --error=out/%A_%a-phase9_label.err
#SBATCH --comment="robocasa phase9: 5 two-sided checks, text answers, no logits. EVERY frame of 7200 episodes (~2.1M), decoded from video, 16 shards, batch 8."
set -u
# The submit guard requires this and only checks the prefix; a job without it
# is rejected at sbatch time, not at run time.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
# Absolute interpreters throughout. A bare `python` is not on PATH inside a
# batch step and has silently killed finished training runs here before.
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"
PORT=$((10500 + (SLURM_ARRAY_JOB_ID % 30) * 30 + SLURM_ARRAY_TASK_ID))
LOG="/tmp/judge_p9_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

export PHASE9_OUT="$PWD/output/_gate_distill/phase9_full"
mkdir -p "$PHASE9_OUT" out

"$JUDGE_PY" -u scripts/vlm_gate_cosmos.py --serve --port "$PORT" > "$LOG" 2>&1 &
JUDGE=$!
trap 'kill $JUDGE 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "judge failed to start"; tail -20 "$LOG"; exit 1; }
echo "judge ready on $PORT (shard $SLURM_ARRAY_TASK_ID/16)"

# Resumable: the script skips chunks already in its own shard file, so a
# preemption on the background partition costs only the chunk in flight.
# Every frame, decoded from the videos -- no stride, and no new tile files.
export PHASE9_BATCH=8
"$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$SLURM_ARRAY_TASK_ID" 16
