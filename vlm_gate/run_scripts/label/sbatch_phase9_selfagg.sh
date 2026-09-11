#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_selfaggregated_single_grade_stride4_four_gpus_alinlab
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=sjw_alinlab
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-phase9_selfagg.out
#SBATCH --error=out/%j-phase9_selfagg.err
#SBATCH --comment="robocasa phase9 self-aggregated: the model is told which observations count for and against and returns ONE grade Z, instead of five per-question grades the client sums. stride 4, four GPUs on one node."
set -u
# sjw_alinlab rather than background: background is PriorityTier=1 while every
# other partition is 2, and the rlwrld account holds 56.9% of cluster usage, so
# array tasks there sit at (Priority) for hours. Within sjw_alinlab this user's
# FairShare is 0.237 against 0.05-0.15 for the heavy users, so it starts.
# Four GPUs and no more -- this is the training partition and a bigger ask
# displaces someone's run.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1

# Its own directory. The five-question labels in phase9_full/ are stride 1 and
# carry A..E; these carry Z. Sharing a directory would make the worklist read
# those rows as done and the two label sets would interleave in one file.
export PHASE9_OUT="$PWD/output/_gate_distill/phase9_selfagg"
export PHASE9_BATCH=8
export PHASE9_STRIDE=4
export SELFAGG=1
NW=4
mkdir -p "$PHASE9_OUT" out

"$RUN_PY" -u scripts/phase9_worklist.py "$NW" "$PHASE9_OUT" || exit 1

PIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12600 + (SLURM_JOB_ID % 150) * 8 + W))
  LOG="$PHASE9_OUT/judge_w${W}_${SLURM_JOB_ID}.log"
  # torch grabs every visible core per process, so four judges on a 24-core node
  # ask for 96 threads and the OMP pools spin. Capping measured free.
  CUDA_VISIBLE_DEVICES=$W OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
      --serve --port "$PORT" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "worker $W: judge on GPU $W port $PORT"
done
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT

for W in $(seq 0 $((NW-1))); do
  LOG="$PHASE9_OUT/judge_w${W}_${SLURM_JOB_ID}.log"
  for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
  grep -q "JUDGE READY" "$LOG" || { echo "judge $W failed"; tail -20 "$LOG"; exit 1; }
done
echo "all $NW judges ready"

WPIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12600 + (SLURM_JOB_ID % 150) * 8 + W))
  CUDA_VISIBLE_DEVICES=$W PHASE9_WORKLIST="$PHASE9_OUT/worklist_${NW}_${W}.json" \
    "$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$W" "$NW" \
    > "$PHASE9_OUT/worker_${W}_${SLURM_JOB_ID}.log" 2>&1 &
  WPIDS+=($!)
done
for p in "${WPIDS[@]}"; do wait "$p"; done
echo "done: $(cat "$PHASE9_OUT"/labels_*.jsonl | wc -l) rows total"
