#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_remaining_frames_four_workers_one_node_alinlab
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-phase9_alinlab.out
#SBATCH --error=out/%j-phase9_alinlab.err
#SBATCH --comment="Finish robocasa phase9 labelling. Two workers on one node, one GPU each, over a pre-computed worklist of the frames still missing."
set -u
# background is PriorityTier=1 and every other partition is 2, so the 16-way
# array sat at (Priority) for fourteen hours after being preempted -- nothing ran
# and 52% was as far as it got. This asks for four GPUs on one node in a tier-2
# partition instead of sixteen scattered array tasks. Two rather than four
# because only worker-node109 has GPUs free (3 of 8) and it is PLANNED, so a
# request for four sits behind a reservation. The worklist makes the worker
# count free to change: rerun phase9_worklist.py with a different N and the
# remaining frames are re-dealt, so more workers can be added later.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export PHASE9_OUT="$PWD/output/_gate_distill/phase9_full"
export PHASE9_BATCH=8
NW=2
mkdir -p "$PHASE9_OUT" out

# What is left, split four ways, before a single GPU is touched. Reads every
# labels_*.jsonl already there, whatever shard naming produced it.
"$RUN_PY" -u scripts/phase9_worklist.py "$NW" "$PHASE9_OUT" || exit 1

PIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12100 + (SLURM_JOB_ID % 150) * 8 + W))
  LOG="$PHASE9_OUT/judge_w${W}_${SLURM_JOB_ID}.log"
  # torch takes every core it can see per process, so N judges on one node ask
  # for N x 24 threads on 24 cores and the OMP pool spins waiting for work --
  # 155328's two judges sat at 821% CPU each. Capping costs nothing measured
  # (0.258 vs 0.257 s/frame on a free A100) and stops the oversubscription.
  CUDA_VISIBLE_DEVICES=$W OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
      --serve --port "$PORT" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "worker $W: judge on GPU $W port $PORT"
done
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT

for W in $(seq 0 $((NW-1))); do
  PORT=$((12100 + (SLURM_JOB_ID % 150) * 8 + W))
  LOG="$PHASE9_OUT/judge_w${W}_${SLURM_JOB_ID}.log"
  for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
  grep -q "JUDGE READY" "$LOG" || { echo "judge $W failed"; tail -20 "$LOG"; exit 1; }
done
echo "all $NW judges ready"

WPIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12100 + (SLURM_JOB_ID % 150) * 8 + W))
  CUDA_VISIBLE_DEVICES=$W PHASE9_WORKLIST="$PHASE9_OUT/worklist_${NW}_${W}.json" \
    "$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$W" "$NW" \
    > "$PHASE9_OUT/worker_${W}_${SLURM_JOB_ID}.log" 2>&1 &
  WPIDS+=($!)
done
for p in "${WPIDS[@]}"; do wait "$p"; done
echo "done: $(cat "$PHASE9_OUT"/labels_*.jsonl | wc -l) rows total"
