#!/bin/bash
#SBATCH --job-name=phase7_grade_pilot_binary_versus_graded_answers_same_prefill_2000_chunks
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -u
# The pilot's first home was an interactive srun on the debug partition, which is
# the wrong shape for it: debug is for driving a shell by hand, and the allocation
# sat SUSPENDED for ten hours behind other users' work while the run stalled at
# 800 of 2000 chunks. It belongs where every other labelling job runs — the
# background partition, submitted, requeued on preemption. The client is
# resumable, so a requeue costs only the chunk in flight.
WS=$HOME/quantization_agent_workspace
BASE=$WS/vlm_gate; cd "$BASE"
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
N="${N:-2000}"
PORT=$((8800 + RANDOM % 60))
LOG=$BASE/output/_gate_distill/phase7_pilot/judge.log
mkdir -p "$BASE/output/_gate_distill/phase7_pilot"

"$WS/cosmos_judge_venv/bin/python" -u scripts/vlm_gate_cosmos.py --serve --port $PORT > "$LOG" 2>&1 &
JP=$!
for i in $(seq 1 120); do
  sleep 10
  grep -q "JUDGE READY" "$LOG" && break
  kill -0 $JP 2>/dev/null || break
done
grep -q "JUDGE READY" "$LOG" || { echo "JUDGE FAILED TO START"; tail -30 "$LOG"; kill $JP 2>/dev/null; exit 1; }
echo "judge up on $PORT"

"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/phase7_grade_pilot.py "$PORT" "$N"
rc=$?
kill $JP 2>/dev/null
exit $rc
