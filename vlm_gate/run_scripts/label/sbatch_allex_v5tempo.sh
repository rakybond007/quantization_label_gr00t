#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_merged_v5tempo_v2_prompt_measured_ceilings_stride4_two_gpus
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=out/%A_%a-v5tempo.out
#SBATCH --error=out/%A_%a-v5tempo.err
#SBATCH --comment="Label merged_v5tempo with the v2 prompt and the measured ceilings (Rotate Box 1.5-2.0). 1280 episodes over 8 shards."
set -u
# stride 4: 구간 안에서 네 청크마다 하나만 라벨하고, 사이 청크는 나중에
# allex_v2_fill.py 가 같은 구간의 가장 가까운 라벨로 메운다. 구간 안에서 세므로
# 짧은 구간도 첫 청크는 반드시 라벨된다.
#
# 실측 1.2 청크/초. 13만 청크를 stride 4 로 줄이면 약 4.6만이고 GPU 2대면
# 대여섯 시간이다. 배치는 못 쓴다 -- 판정기의 배치 경로는 자유 서술을 파싱하고
# v2 는 강제 슬롯의 확률을 읽어서, 배치 응답에 그 확률이 없다.
export ALLEX_STRIDE=4
# 프롬프트는 v2 그대로. 상한/하한도 최신 실측 그대로다:
#   Rotate Box 1.5~2.0, Rotate PolyBag 2.5, Bring 3.0(박스)/2.0(봉투), Pass 3.0.
# 이 데이터셋만 다른 점 둘 -- 에피소드가 1280 개라 chunk-001 이 있고,
# 서브태스크 이름이 task_index 가 아니라 meta/subtasks.jsonl 에 있다.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/allex_v5tempo
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
export ALLEX_OUT="$PWD/output/allex_v5tempo"
mkdir -p "$ALLEX_OUT" out
NSH=2
SH=${SLURM_ARRAY_TASK_ID:-0}
PORT=$((13500 + SLURM_ARRAY_JOB_ID % 300 + SH))
LOG="$ALLEX_OUT/judge_s${SH}.log"

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$LOG" 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }
echo "판정기 준비됨. 샤드 $SH / $NSH"

# 이어 붙이기다. labels_s8_<SH>.jsonl 이 이미 있으면 거기 없는 청크만 한다.
"$RUN_PY" -u scripts/allex_v2_label.py "$PORT" "$SH" "$NSH" || exit 1
echo "샤드 $SH 끝: $(wc -l < "$ALLEX_OUT/labels_s${NSH}_${SH}.jsonl") 청크"
