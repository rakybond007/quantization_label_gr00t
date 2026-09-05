#!/bin/bash
# v3 방식(등급표 + 분포, 배치) 스모크 + 속도 측정. 에피소드 몇 개만.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_DS="/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo"
export ALLEX_CHECKS="CLAMP,LOOSE,SHOVE,FLIP,FREE"
export ALLEX_FULL=1
export ALLEX_STRIDE="${ALLEX_STRIDE:-4}"
export ALLEX_NSHARDS=640          # 1280 에피소드 중 두 개만 (0, 640)
export ALLEX_SHARD=0
PORT="${PORT:-8295}"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_v3smoke.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q "JUDGE READY" /tmp/judge_v3smoke.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_v3smoke.log || { echo "판정기 안 뜸"; exit 1; }
echo "판정기 준비됨"
for B in 8 16 32; do
  O="$PWD/output/v3smoke_b$B"
  rm -rf "$O"; mkdir -p "$O"
  T0=$(date +%s)
  ALLEX_OUT="$O" ALLEX_BATCH=$B "$RUN_PY" -u scripts/allex_v3_label.py "$PORT" > "$O/log" 2>&1
  T1=$(date +%s)
  N=$(wc -l < "$O/records_s640_0.jsonl" 2>/dev/null || wc -l < "$O/records.jsonl" 2>/dev/null || echo 0)
  echo "배치 $B  ->  $N 청크 / $((T1-T0))초  =  $("$RUN_PY" -c "print(f'{$N/max(1,$T1-$T0):.2f}')") 청크/초"
  tail -3 "$O/log"
done
echo "--- 칸별 ---"
"$RUN_PY" - <<'PY'
import json, collections, os
p = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/v3smoke_b32")
f = p + "/records_s640_0.jsonl"
rs = [json.loads(l) for l in open(f)]
print(f"청크 {len(rs)}, 에피소드 {sorted({r['ep'] for r in rs})}")
print(collections.Counter(r["cell"] for r in rs))
PY
