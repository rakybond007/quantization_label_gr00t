#!/bin/bash
# 배치 폭별 처리 속도 측정. 같은 에피소드를 폭만 바꿔 돌린다.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
PORT="${PORT:-8291}"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_speed.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q "JUDGE READY" /tmp/judge_speed.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_speed.log || { echo "판정기 안 뜸"; exit 1; }
echo "판정기 준비됨"
for B in 1 8 16 32; do
  O="$PWD/output/speed_b$B"
  rm -rf "$O"; mkdir -p "$O"
  T0=$(date +%s)
  ALLEX_OUT="$O" ALLEX_BATCH=$B TAG=t "$RUN_PY" -u scripts/allex_v2_label.py "$PORT" 0 1 0 >/dev/null 2>&1
  T1=$(date +%s)
  N=$(wc -l < "$O/labels_t.jsonl" 2>/dev/null || echo 0)
  echo "배치 $B  ->  $N 청크 / $((T1-T0))초   =  $(python3 -c "print(f'{$N/max(1,$T1-$T0):.1f}')") 청크/초"
done
