#!/bin/bash
# merged_v5tempo 스모크. 에피소드 두 개(앞 덩이 하나, 뒤 덩이 하나)만.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
export ALLEX_OUT="${ALLEX_OUT:-$PWD/output/allex_v5tempo_smoke}"
PORT="${PORT:-8281}"
rm -rf "$ALLEX_OUT"; mkdir -p "$ALLEX_OUT"
echo "출력: $ALLEX_OUT"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_v5tempo.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 60); do grep -q "JUDGE READY" /tmp/judge_v5tempo.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_v5tempo.log || { echo "판정기 안 뜸"; tail -20 /tmp/judge_v5tempo.log; exit 1; }
TAG=smoke "$RUN_PY" -u scripts/allex_v2_label.py "$PORT" 0 1 0,1200 || exit 1
"$RUN_PY" - <<'PY'
import json, collections, os
p = os.environ["ALLEX_OUT"] + "/labels_smoke.jsonl"
rs = [json.loads(l) for l in open(p)]
print(f"\n청크 {len(rs)}개")
print("서브태스크별:", collections.Counter(r["task"] for r in rs))
for t in sorted({r["task"] for r in rs}):
    v = [r for r in rs if r["task"] == t]
    ks = collections.Counter(r["K"] for r in v)
    print(f"  {t:<15} n={len(v):4d}  " + "  ".join(f"{k:g}x {n}" for k, n in sorted(ks.items())))
print("에피소드:", sorted({r["ep"] for r in rs}))
PY
