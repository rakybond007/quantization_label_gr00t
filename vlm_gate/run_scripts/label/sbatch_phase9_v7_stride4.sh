#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_robocasa_phase9_v7_questions_stride4_with_grade_probs_background_array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%A_%a-p9v7.out
#SBATCH --error=out/%A_%a-p9v7.err
#SBATCH --comment="robocasa phase9 v7 questions, stride 4 (511k frames), grade probabilities stored. Short walltime + 1 GPU per array task so the backfill scheduler can slot them in; resumable so preemption costs only the chunk in flight."
set -u
# background 를 쓰는 이유: TRESBillingWeights GPU=0.2 로 다른 파티션의 1/5 이고 노드가
# 25개로 가장 많다. PriorityTier=1 이라 선점당하지만 worklist 로 재개하므로 진행 중
# 청크만 잃는다. walltime 3시간은 backfill 용이다 -- 대기 잡의 136개가 2일을 요청한다.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1

export PHASE9_OUT="$PWD/output/_gate_distill/phase9_v7_stride4"
export PHASE9_BATCH=8
export PHASE9_STRIDE=4
export CHECKS=phase9_checks_v7
NW=16
: "${SLURM_ARRAY_TASK_ID:=0}"
mkdir -p "$PHASE9_OUT" out

# 배열 0번만 worklist 를 만들고 나머지는 기다린다. 동시에 쓰면 서로 덮어쓴다.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  PHASE9_STRIDE=4 "$RUN_PY" -u scripts/phase9_worklist.py "$NW" "$PHASE9_OUT" || exit 1
  touch "$PHASE9_OUT/.worklist_ready"
else
  for _ in $(seq 1 180); do [ -f "$PHASE9_OUT/.worklist_ready" ] && break; sleep 10; done
  [ -f "$PHASE9_OUT/.worklist_ready" ] || { echo "worklist 가 안 생겼다"; exit 1; }
fi

# 고정 포트는 같은 노드에 두 배열 태스크가 앉으면 Errno 98 로 죽는다. 빈 포트를 찾는다.
for p in $(seq 18000 18199); do
  ss -ltn 2>/dev/null | grep -q ":$p " || { PORT=$p; break; }
done
: "${PORT:?빈 포트를 못 찾았다}"
LOG="$PHASE9_OUT/judge_${SLURM_ARRAY_TASK_ID}.log"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!
trap 'kill $J 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "judge 실패"; tail -20 "$LOG"; exit 1; }
echo "shard $SLURM_ARRAY_TASK_ID/$NW · port $PORT · CHECKS=$CHECKS · stride $PHASE9_STRIDE"

PHASE9_WORKLIST="$PHASE9_OUT/worklist_${NW}_${SLURM_ARRAY_TASK_ID}.json" \
  "$RUN_PY" -u scripts/phase9_label_full.py "$PORT" "$SLURM_ARRAY_TASK_ID" "$NW"
