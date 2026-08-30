#!/bin/bash
#SBATCH --job-name=label_robocasa_dataset_vlm_gate_quantizability_for_distill_submodule_240eps_stratified
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-label_gate_dataset.out
#SBATCH --error=out/%j-label_gate_dataset.err
#SBATCH --comment="Label robocasa LeRobot dataset frames with the VLM gate (best evolved guidance) for gate-module distillation. JUDGE_BACKEND=gemma|cosmos."

# env: JUDGE_BACKEND (gemma|cosmos), EPS_PER_TASK (default 10), EVERY (default 8)
set -u
JUDGE_BACKEND="${JUDGE_BACKEND:-cosmos}"
EPS_PER_TASK="${EPS_PER_TASK:-10}"
EVERY="${EVERY:-8}"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
DS=/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
CK="${CK_OVERRIDE:-/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_${JUDGE_BACKEND}_v1}"
GUIDE="${GUIDE:-$BASE/analysis/_evolver/_varkA/robocasa_${JUDGE_BACKEND}_ttl_best_guidance.txt}"
PORT=$((19900 + SLURM_JOB_ID % 80))
mkdir -p out "$CK/labels" "$BASE/output/_gate_distill"
cd "$BASE"
export NO_ALBUMENTATIONS_UPDATE=1

: "${SLURM_ARRAY_TASK_ID:=0}"
FULL="${FULL:-0}"
if [ "$FULL" = 1 ]; then
  # full dataset: array id별 720개 에피소드 샤드 (10분할 × 720 = 7200)
  EPS=$(python3 -c "a=$SLURM_ARRAY_TASK_ID; print(','.join(str(i) for i in range(a*720, (a+1)*720)))")
  SHARD="_shard$SLURM_ARRAY_TASK_ID"
else
  # 층화 테스트: 24개 태스크 블록(300ep) × EPS_PER_TASK개, 블록 내 30 간격
  EPS=$(python3 -c "print(','.join(str(b*300 + j*30) for b in range(24) for j in range($EPS_PER_TASK)))")
  SHARD=""
fi

cleanup(){ kill ${JP:-} 2>/dev/null; sleep 2; kill -9 ${JP:-} 2>/dev/null; }
trap cleanup EXIT

JLOG="$BASE/output/_gate_distill/judge_scale_${JUDGE_BACKEND}_$SLURM_JOB_ID.log"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE/scripts" \
    "$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python" -u "$BASE/scripts/vlm_gate_cosmos.py" \
    --serve --model nvidia/Cosmos3-Nano --port $PORT --host 127.0.0.1 > "$JLOG" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JUDGE_COMPILE=1 \
    "$HOME/miniconda3/envs/vlm_judge/bin/python" -u "$BASE/scripts/vlm_gate.py" \
    --serve --model google/gemma-4-12b-it --port $PORT --host 127.0.0.1 > "$JLOG" 2>&1 &
fi
JP=$!
for i in $(seq 1 200); do grep -q "JUDGE READY" "$JLOG" 2>/dev/null && break; kill -0 $JP 2>/dev/null || { echo "[ERR] judge died"; tail -20 "$JLOG"; exit 1; }; sleep 5; done
echo "[i] judge ready ($JUDGE_BACKEND)"

"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u "$BASE/scripts/label_gate_dataset.py" \
  --dataset-path "$DS" --judge-url "http://127.0.0.1:$PORT" \
  --guidance "@$GUIDE" --every $EVERY --episode-list "$EPS" \
  --out "$CK/labels/${LABEL_NAME:-strat240}${SHARD:-}.parquet"
echo "[i] labeling done ($JUDGE_BACKEND)."
