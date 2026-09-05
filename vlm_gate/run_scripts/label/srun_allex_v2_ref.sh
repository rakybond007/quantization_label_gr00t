#!/bin/bash
# v3 루프의 참고 지표를 위한 v2 라벨. 표본 청크만 돌린다.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="${ALLEX_DS:-/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin}"
export ALLEX_SAMPLE_FILE="${ALLEX_SAMPLE_FILE:-$PWD/output/allex_sample/D.json}"
export ALLEX_OUT="${ALLEX_OUT:-$PWD/output/allex_v2_ref_${SLURM_JOB_ID:-manual}}"
PORT="${PORT:-8273}"
rm -rf "$ALLEX_OUT"; mkdir -p "$ALLEX_OUT"
echo "출력: $ALLEX_OUT"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_allex_v2ref.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 60); do grep -q "JUDGE READY" /tmp/judge_allex_v2ref.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_allex_v2ref.log || { echo "판정기 안 뜸"; tail -20 /tmp/judge_allex_v2ref.log; exit 1; }
"$RUN_PY" -u scripts/allex_v2_label.py "$PORT" 0 1 || exit 1
"$RUN_PY" -u scripts/allex_v2_aggregate.py 2>/dev/null || true
ls -la "$ALLEX_OUT"
