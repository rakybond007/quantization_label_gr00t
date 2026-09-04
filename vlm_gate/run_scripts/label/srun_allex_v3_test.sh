#!/bin/bash
# allex v3 시험 라벨링 + 영상화. srun 안에서 도는 것을 전제로 한다.
# parquet 은 건드리지 않는다 -- records.jsonl 과 mp4 만 나온다.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
EPS="${EPS:-0,1,2,3}"
PORT="${PORT:-8251}"
export ALLEX_OUT="$PWD/output/allex_v3checks"
mkdir -p "$ALLEX_OUT" "$HOME/quantization_agent_workspace/assets/videos"

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_allex_v3.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 60); do grep -q "JUDGE READY" /tmp/judge_allex_v3.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_allex_v3.log || { echo "판정기 안 뜸"; tail -20 /tmp/judge_allex_v3.log; exit 1; }
echo "판정기 준비됨"

"$RUN_PY" -u scripts/allex_v3_label.py "$PORT" "$EPS" || exit 1
echo "--- PROMPT_METHOD 검증 5항목 ---"
"$RUN_PY" -u scripts/allex_v3_verify.py || exit 1

"$RUN_PY" -u scripts/allex_v3_render.py \
    "$HOME/quantization_agent_workspace/assets/videos/allex_v3_ratio.mp4"
