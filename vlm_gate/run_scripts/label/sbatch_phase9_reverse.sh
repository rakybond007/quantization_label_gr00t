#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_remaining_frames_reverse_pass_two_more_workers_meeting_in_middle
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-phase9_reverse.out
#SBATCH --error=out/%j-phase9_reverse.err
#SBATCH --comment="Second pair of phase9 workers walking the SAME worklist backwards while the first pair goes forwards. They meet in the middle and the reverse pass stops itself."
set -u
# The worklist is NOT regenerated. Re-dealing it would hand out episodes the
# running job is part-way through and lose their partial work; instead these
# two read worklist_2_0/1.json from the end. Each reverse worker also reads the
# forward worker's labels file at startup, so it begins past everything already
# written, and watches the forward worker's log to stop when the two meet.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export PHASE9_OUT="$PWD/output/_gate_distill/phase9_full"
export PHASE9_BATCH=8
export PHASE9_REVERSE=1
NW=2
FRONT_JOB="${FRONT_JOB:-${1:-}}"      # job id of the forward pass
[ -n "$FRONT_JOB" ] || { echo "FRONT_JOB (앞선 잡 번호) 가 필요하다"; exit 1; }

for W in $(seq 0 $((NW-1))); do
  [ -f "$PHASE9_OUT/worklist_${NW}_${W}.json" ] || {
    echo "worklist_${NW}_${W}.json 이 없다 -- 앞선 잡과 같은 분할이어야 한다"; exit 1; }
done

PIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12100 + (SLURM_JOB_ID % 150) * 8 + W))
  LOG="$PHASE9_OUT/judge_r${W}_${SLURM_JOB_ID}.log"
  # See sbatch_phase9_alinlab.sh: uncapped, each judge asks for every core and
  # the OMP pool spins. Capping measured 0.258 vs 0.257 s/frame -- free.
  CUDA_VISIBLE_DEVICES=$W OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
      --serve --port "$PORT" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "reverse worker $W: judge on GPU $W port $PORT"
done
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT

for W in $(seq 0 $((NW-1))); do
  LOG="$PHASE9_OUT/judge_r${W}_${SLURM_JOB_ID}.log"
  for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
  grep -q "JUDGE READY" "$LOG" || { echo "judge $W failed"; tail -20 "$LOG"; exit 1; }
done
echo "all $NW reverse judges ready"

WPIDS=()
for W in $(seq 0 $((NW-1))); do
  PORT=$((12100 + (SLURM_JOB_ID % 150) * 8 + W))
  CUDA_VISIBLE_DEVICES=$W \
    PHASE9_WORKLIST="$PHASE9_OUT/worklist_${NW}_${W}.json" \
    PHASE9_FRONT_LOG="$PHASE9_OUT/worker_${W}_${FRONT_JOB}.log" \
    "$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$W" "$NW" \
    > "$PHASE9_OUT/reverse_${W}_${SLURM_JOB_ID}.log" 2>&1 &
  WPIDS+=($!)
done
for p in "${WPIDS[@]}"; do wait "$p"; done
echo "reverse done: $(cat "$PHASE9_OUT"/labels_*.jsonl | wc -l) rows across all files"
