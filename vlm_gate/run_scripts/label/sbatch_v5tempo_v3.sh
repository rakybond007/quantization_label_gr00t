#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_merged_v5tempo_v3_grade_ladder_batched_stride4_two_gpus
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=out/%A_%a-v5tempo_v3.out
#SBATCH --error=out/%A_%a-v5tempo_v3.err
#SBATCH --comment="Label merged_v5tempo with the v3 grade-ladder questions, batched, stride 4, six cells."
set -u
# v3 방식이다. 모델이 등급을 직접 쓰고 그 자리의 1~5 분포를 같이 읽는다.
# v2 와 달리 배치가 된다 -- v2 는 YES/NO 자리를 강제하고 그 확률을 읽어서
# 판정기의 배치 경로에 그 값이 없다.
#
# 실측: 배치 32 로 2.8 청크/초. stride 4 로 약 3.5만 청크면 GPU 2대에 두 시간.
#
# 칸은 여섯이다. 주기(가져오기 -> 뒤집기 -> 넘기기)에서 물체를 뽑으므로
# Bring/Pass 도 박스와 봉투가 갈린다 -- v2 는 넷이었다.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/allex_v5tempo_v3
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
export ALLEX_OUT="$PWD/output/allex_v5tempo_v3"
export ALLEX_CHECKS="CLAMP,LOOSE,SHOVE,FLIP,FREE"
export ALLEX_FULL=1
export ALLEX_STRIDE=4
export ALLEX_BATCH=32
export ALLEX_NSHARDS=2
export ALLEX_SHARD=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$ALLEX_OUT" out
PORT=$((14000 + SLURM_ARRAY_JOB_ID % 300 + ALLEX_SHARD))
LOG="$ALLEX_OUT/judge_s${ALLEX_SHARD}.log"

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }
echo "판정기 준비됨. 샤드 $ALLEX_SHARD / $ALLEX_NSHARDS"

# 이어 붙이기다. records_s2_<SH>.jsonl 에 없는 청크만 한다.
"$RUN_PY" -u scripts/allex_v3_label.py "$PORT" || exit 1
echo "샤드 $ALLEX_SHARD 끝: $(wc -l < "$ALLEX_OUT/records_s2_${ALLEX_SHARD}.jsonl") 청크"
